"""Read-only successor admission: recorded identities plus a fresh dispatch scan.

No process control, scientific seal, scoring permission, or graceful-cleanup claim.
"""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
_dependency_spec = importlib.util.spec_from_file_location('_ssa_retirement_evidence_for_closure', HERE / 'retirement_evidence.py')
evidence = importlib.util.module_from_spec(_dependency_spec)
_dependency_spec.loader.exec_module(evidence)
KINDS = {'NORMAL_TERMINATION', 'OPERATIONAL_ABORT'}
SHARED = {'mathworksservicehost.exe', 'mathworksservicehost', 'mathworkslicensing.exe',
          'lmgrd.exe', 'mlm.exe', 'mwlm.exe'}


def require(value, reason):
    if not value:
        raise ValueError(reason)


def verify_sources(closure):
    expected = closure.get('verifier_source_sha256')
    require(isinstance(expected, dict) and set(expected) == {'closure_verifier.py', 'retirement_evidence.py'},
            'Both verifier source dependencies must be frozen')
    require(Path(evidence.__file__).resolve() == HERE / 'retirement_evidence.py', 'Imported dependency path differs')
    for name, sha in expected.items():
        path = HERE / name
        require(hashlib.sha256(path.read_bytes()).hexdigest() == sha, 'Verifier dependency changed: ' + name)


def key(row):
    require(type(row.get('pid')) is int and row['pid'] > 0, 'Invalid identity PID')
    value = row.get('create_time')
    require(type(value) in (int, float) and math.isfinite(value) and value > 0,
            'Invalid identity creation time')
    return row['pid'], value


def load_bound(path, expected):
    evidence.reject_links(path)
    before = evidence.metadata(path)
    raw = evidence.io_path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, 'Evidence SHA mismatch: ' + str(path))
    require(evidence.metadata(path) == before and evidence.digest(path) == expected,
            'Evidence changed while read: ' + str(path))
    return json.loads(raw.decode('utf-8-sig'))


def relative_path(root, value):
    require(isinstance(value, str) and bool(value), 'Missing relative evidence path')
    part = Path(value)
    require(not part.is_absolute() and '..' not in part.parts, 'Evidence path must not escape its root')
    target = (Path(root) / part).absolute()
    require(evidence.inside(target, root), 'Evidence path escapes root')
    evidence.reject_links(target)
    return target


def reference(root, ref):
    require(isinstance(ref, dict) and set(ref) == {'path', 'sha256'}, 'Evidence reference schema differs')
    path = relative_path(root, ref['path'])
    return load_bound(path, ref['sha256']), path


def binding(record, s1_root, protocol_sha, runtime_sha, start_sha):
    require(evidence.norm(record['s1_root']) == evidence.norm(s1_root), 'Evidence is for another S1 root')
    for field, expected in [('prior_protocol_sha256', protocol_sha), ('prior_runtime_sha256', runtime_sha),
                            ('execution_started_sha256', start_sha)]:
        require(record.get(field) == expected, 'Evidence run binding differs: ' + field)


def epoch(value):
    stamp = datetime.fromisoformat(value)
    require(stamp.utcoffset() is not None, 'Evidence timestamp must include its timezone')
    return stamp.timestamp()


def snapshot(root, ref, phase, source, protocol_sha, runtime_sha):
    m, path = reference(root, ref)
    require(m['schema'] == 1 and m['phase'] == phase, 'Snapshot phase/schema differs')
    require(evidence.norm(m['source_root']) == evidence.norm(source), 'Snapshot belongs to another source')
    require(evidence.norm(m['snapshot_root']) == evidence.norm(path.parent), 'Snapshot root differs from manifest location')
    expected_status = 'LIVE_NONTRANSACTIONAL' if phase == 'pre' else 'POST_CAPTURE_NONTRANSACTIONAL'
    require(m['status'] == expected_status and m['scientific_seal'] is False
            and m['public_distribution_allowed'] is False, 'Snapshot scope differs')
    for when in ['binding_before', 'binding_after']:
        require(m[when]['passed'] is True, 'Snapshot source bindings were unresolved')
        files = m[when]['protocol_files']
        require(files['FROZEN_PROTOCOL.json'] == protocol_sha and files['RUNTIME_PROTOCOL.json'] == runtime_sha,
                'Snapshot protocol identity differs')
    rows = {}
    for row in m['files']:
        rel = row['relative_path']
        require(rel not in rows, 'Duplicate snapshot member')
        target = relative_path(path.parent / 'source_files', rel)
        require(evidence.norm(row['target']) == evidence.norm(target), 'Snapshot target path differs')
        require(evidence.norm(row['source']) == evidence.norm(Path(source) / rel), 'Snapshot source path differs')
        rows[rel] = row
    m['_rows'], m['_directory'], m['_used'] = rows, path.parent, {}
    return m


def snapshot_json(m, rel):
    row = m['_rows'][rel]
    require(row.get('copied_sha256') and row.get('snapshot_file_sha256_verified') is True,
            'Required process evidence has no verified complete byte copy')
    path = relative_path(m['_directory'] / 'source_files', rel)
    m['_used'][str(path)] = row['copied_sha256']
    return load_bound(path, row['copied_sha256'])


def from_either(pre, post, rel):
    return snapshot_json(pre if rel in pre['_rows'] else post, rel)


def recorded_identities(pre, post, source, p, start, controller_exe, window, protocol_sha, runtime_sha):
    known, controllers = {}, set()
    cases = {c['case_id']: c for c in p['cases']}
    start_epoch = datetime.fromisoformat(start['start_utc']).timestamp()
    for m in [pre, post]:
        for when in ['process_before', 'process_after']:
            observation = m[when]
            controller = observation.get('controller')
            if controller and controller.get('identity_status') == 'VERIFIED_TASK_CANDIDATE':
                ident = key(controller)
                require(controller['pid'] == start['controller_pid'] and evidence.controller_matches(
                    controller, source, runtime_sha, controller_exe, start_epoch, window),
                    'Captured controller does not bind this execution')
                controllers.add(ident)
                known[ident] = 'controller'
            for item in observation.get('task_candidates', []):
                require(item.get('identity_status') in ('VERIFIED_TASK_CANDIDATE', 'VERIFIED_TASK_ASSOCIATED_HELPER'),
                        'Captured task candidate remains unresolved')
                require(item['name'].casefold() not in SHARED, 'Shared service cannot be an owned identity')
                require(item.get('identity_checks') and all(v is True for v in item['identity_checks'].values()),
                        'Captured task identity checks differ')
                ident = key(item)
                if 'inherited_run_identity_from_pid' in item:
                    require(any(a['pid'] == item['inherited_run_identity_from_pid'] for a in item['parent_chain']),
                            'Descendant lacks its recorded MATLAB ancestry')
                    role = 'os_console_helper' if item.get('process_role') == 'OS_CONSOLE_HELPER' else 'owned_descendant'
                    known.setdefault(ident, role)
                    continue
                cwd = Path(item['cwd'])
                require(evidence.inside(cwd, Path(source) / 'runs'), 'Worker cwd is outside original runs')
                rel = (cwd / 'RUN_IDENTITY.json').relative_to(source).as_posix()
                identity = from_either(pre, post, rel)
                case = cases[identity['case_id']]
                require(identity['scientific_protocol_sha256'] == protocol_sha and
                        identity['runtime_protocol_sha256'] == runtime_sha and
                        identity['input_sha256'] == case['sha256'], 'Worker original RUN_IDENTITY differs')
                require(evidence.norm(cwd) == evidence.norm(Path(source) / 'runs' / case['case_id'] / 'repeat1'),
                        'Worker case directory differs')
                known.setdefault(ident, 'matlab_worker')
    require(len(controllers) == 1, 'One original controller creation-time identity is required')
    for rel in post['_rows']:
        if not rel.startswith('runs/') or not rel.endswith('/repeat1/launcher_receipt.json'):
            continue
        receipt = snapshot_json(post, rel)
        case = cases[receipt['case_id']]
        require(rel == 'runs/' + case['case_id'] + '/repeat1/launcher_receipt.json', 'Receipt case path differs')
        require(receipt['protocol_sha256'] == protocol_sha and receipt['runtime_protocol_sha256'] == runtime_sha
                and receipt['input_sha256'] == case['sha256'], 'Receipt run binding differs')
        if receipt.get('matlab_started') is True:
            ident = key(dict(pid=receipt['launched_pid'], create_time=receipt['process_create_time']))
            known.setdefault(ident, 'matlab_launcher')
    return known


def current_generation_matches(source, post):
    """Bind current dispatch/ledger generation without decoding predictions or X/y."""
    paths, errors = evidence.inventory(source)
    require(not errors, 'Cannot inspect the current S1 file generation')
    def relevant(rel):
        return ('/' not in rel and rel.endswith('.json')) or (
            rel.startswith('runs/') and rel.endswith(('/RUN_IDENTITY.json', '/launcher_receipt.json')))
    current = {p.relative_to(source).as_posix(): p for p in paths if relevant(p.relative_to(source).as_posix())}
    expected = {rel: row for rel, row in post['_rows'].items() if relevant(rel)}
    require(set(current) == set(expected), 'S1 dispatch/ledger files differ from post capture')
    for rel, path in current.items():
        require(evidence.digest(path) == expected[rel].get('copied_sha256'),
                'S1 dispatch/ledger changed after post capture: ' + rel)


def fresh_scan(provider, source, known, controller_exe, runtime_sha):
    live, unresolved, observations = set(), {}, []
    rows, errors = provider.minimal()
    index = {r['pid']: r for r in rows}
    require(len(index) == len(rows), 'Duplicate PID in fresh process enumeration')
    known_pids = {pid for pid, _ in known}
    for error in errors:
        # Unknown name means ownership cannot be excluded; do not guess absence.
        name = str(error.get('name', '')).casefold()
        if error.get('pid') in known_pids or not name or name.startswith(('python', 'pypy', 'matlab')):
            unresolved[str(error.get('pid', 'enumeration'))] = 'PROCESS_ENUMERATION_ACCESS_OR_IDENTITY_FAILURE'
    for ident in known:
        row = index.get(ident[0])
        same = row is not None and key(row) == ident
        if same:
            if known[ident] == 'os_console_helper':
                unresolved[str(ident[0])] = 'ORIGINAL_OS_HELPER_STILL_PRESENT_NOT_AUTOMATICALLY_IGNORABLE'
            else:
                live.add(ident[0])
        observations.append(dict(pid=ident[0], create_time=ident[1],
            observed_state=('ORIGINAL_OS_HELPER_PRESENT_UNVERIFIED' if same and known[ident] == 'os_console_helper' else
                'ORIGINAL_IDENTITY_LIVE') if same else
                ('PID_REUSED' if row is not None else 'NOT_OBSERVED')))
    expected_script = evidence.norm(Path(source) / 'run_formal.py')
    new_roots = set(live)
    for pid, row in index.items():
        name = str(row.get('name', '')).casefold()
        possible = pid in known_pids or name.startswith(('python', 'pypy')) or name in ('matlab', 'matlab.exe')
        if not possible:
            continue
        try:
            detail = provider.details(pid)
            require(key(detail) == key(row) and detail['ppid'] == row['ppid'], 'PID identity changed during scan')
            command = detail['cmdline']
            require(isinstance(command, list) and bool(command) and bool(command[0])
                    and all(isinstance(a, str) for a in command) and detail.get('exe') and detail.get('cwd'),
                    'Unusable process executable/cwd/command line')
            script = any(evidence.argument_path(a, detail['cwd']) == expected_script
                         for a in command[1:] if not a.startswith('-'))
            module = (evidence.norm(detail['cwd']) == evidence.norm(source) and
                any(command[i] == '-m' and command[i+1] == 'run_formal' for i in range(len(command)-1)))
            bridge = any(evidence.BRIDGE.search(a) for a in command)
            related_bridge = False
            if bridge:
                for arg in command:
                    match = evidence.BRIDGE.search(arg)
                    if match:
                        args = [s.replace("''", "'") for s in match.groups()]
                        related_bridge |= any(evidence.inside(v, source) for v in args[:3])
            if script or module or related_bridge:
                # A new invocation is a blocker even if it uses the wrong runtime SHA.
                live.add(pid)
                new_roots.add(pid)
            elif any((evidence.norm(str(source)) in evidence.norm(arg)
                      or evidence.norm(detail['cwd']) == evidence.norm(source)) and
                     any(marker in arg for marker in ['run_formal', 'ssa_clock_bridge', 'ssa_formal_worker', 'ssa_formal_profile'])
                     for arg in command):
                unresolved[str(pid)] = 'AMBIGUOUS_SOURCE_REFERENCING_LAUNCH'
        except Exception as exc:
            unresolved[str(pid)] = type(exc).__name__ + ': ' + str(exc)
    for pid, row in index.items():
        chain = evidence.parent_chain(pid, index)
        if row['name'].casefold() in SHARED or any(a['name'].casefold() in SHARED for a in chain):
            continue
        if any(a['pid'] in new_roots for a in chain):
            try:
                detail = provider.details(pid)
                require(key(detail) == key(row), 'Descendant PID changed')
                if row['name'].casefold() == 'conhost.exe':
                    unresolved[str(pid)] = 'OS_HELPER_OF_LIVE_PRIOR_BATCH'
                else:
                    live.add(pid)
            except Exception as exc:
                unresolved[str(pid)] = type(exc).__name__ + ': ' + str(exc)
    return dict(live_previous_batch_processes=len(live), unverified_previous_batch_processes=len(unresolved),
        live_pids=sorted(live), unresolved=unresolved, recorded_identity_observations=observations,
        fresh_process_observation=True, observed_utc=datetime.now(timezone.utc).isoformat())


def verify_closed_for_successor(s1_root, closure_path, closure_sha, *, process_provider=None, evidence_root=None):
    """Read-only admission. Injected providers are for fake-metadata tests only."""
    result = dict(passed=False, fresh_process_observation=False, closure_kind=None,
        prior_protocol_sha256=None, live_previous_batch_processes=0,
        unverified_previous_batch_processes=1, scope='PROCESS_EXIT_ADMISSION_ONLY',
        graceful_application_cleanup_certified=False, scientific_scoring_authorized=False, errors=[])
    try:
        source, closure_path = evidence.logical_path(s1_root).absolute(), evidence.logical_path(closure_path).absolute()
        root = Path(evidence_root or HERE).absolute()
        require(evidence.inside(closure_path, root) and not evidence.inside(closure_path, source),
                'Closure must be in the independent evidence root')
        evidence.reject_links(source)
        evidence.reject_links(closure_path)
        closure = load_bound(closure_path, closure_sha)
        verify_sources(closure)
        require(type(closure.get('schema')) is int and closure['schema'] == 1
                and closure.get('record_type') == 'SSA_PRIOR_BATCH_CLOSURE', 'Closure schema differs')
        kind = closure['closure_kind']
        require(kind in KINDS, 'Unknown closure kind')
        result['closure_kind'] = kind
        protocol_sha = evidence.digest(source / 'FROZEN_PROTOCOL.json')
        runtime_sha = evidence.digest(source / 'RUNTIME_PROTOCOL.json')
        start_sha = evidence.digest(source / 'EXECUTION_STARTED.json')
        binding(closure, source, protocol_sha, runtime_sha, start_sha)
        result['prior_protocol_sha256'] = protocol_sha
        checked, p = evidence.verify_bindings(source, runtime_sha)
        require(checked['passed'] is True, 'Original frozen bindings are unresolved')
        start = load_bound(source / 'EXECUTION_STARTED.json', start_sha)
        require(start['protocol_sha256'] == protocol_sha and start['runtime_protocol_sha256'] == runtime_sha,
                'Original start record differs')
        pre = snapshot(root, closure['pre_snapshot'], 'pre', source, protocol_sha, runtime_sha)
        post = snapshot(root, closure['post_snapshot'], 'post', source, protocol_sha, runtime_sha)
        require(pre['_directory'] != post['_directory'], 'Pre and post must be distinct snapshots')
        require(post['all_initial_files_observed_stable'] is True and not post['unresolved']
                and not any(post['inventory_changes'].values()), 'Post byte capture remains incomplete or unstable')
        for m in [pre, post]:
            require(m['_rows']['EXECUTION_STARTED.json']['copied_sha256'] == start_sha,
                    'Snapshot refers to a different original execution')
        exit_record, exit_path = reference(root, closure['exit_verification'])
        binding(exit_record, source, protocol_sha, runtime_sha, start_sha)
        require(exit_record.get('record_type') == 'IDENTITY_EXIT_VERIFICATION', 'Exit record type differs')
        post_exit = post['external_references']['external_exit_verification']
        require(evidence.norm(post_exit['path']) == evidence.norm(exit_path)
                and post_exit['sha256'] == closure['exit_verification']['sha256'], 'Post capture exit reference differs')
        window = closure.get('controller_creation_window_seconds', 120)
        require(type(window) is int and 1 <= window <= 600, 'Invalid controller creation window')
        known = recorded_identities(pre, post, source, p, start, closure['controller_executable'], window,
                                    protocol_sha, runtime_sha)
        stated = [key(r) for r in closure['registered_identities']]
        require(len(stated) == len(set(stated)) and set(stated) == set(known), 'Registered identities are incomplete or include another run')
        require(epoch(exit_record['observed_utc']) >= max(created for _, created in known),
                'Exit verification predates a registered process')
        require(epoch(exit_record['observed_utc']) <= epoch(post['capture_started_utc'])
                and epoch(closure['recorded_utc']) >= epoch(post['capture_finished_utc']),
                'Exit record/post capture/closure chronology differs')
        exits = exit_record['identities']
        require(len(exits) == len(known) and {key(r) for r in exits} == set(known), 'Exit verification identities differ')
        require(all(r.get('state') in ('GONE', 'PID_REUSED') for r in exits), 'Exit verification remains unresolved')
        require(closure['old_directory_not_rewritten'] is True and closure['partial_and_missing_outputs_preserved'] is True,
                'Closure must retain original unfinished evidence')
        if kind == 'OPERATIONAL_ABORT':
            require(closure['original_batch_unfinished'] is True, 'Operational abort must be declared unfinished')
            actions, _ = reference(root, closure['actions'])
            binding(actions, source, protocol_sha, runtime_sha, start_sha)
            require(actions.get('record_type') == 'OPERATIONAL_ABORT_ACTIONS' and actions.get('reason'), 'Missing actual abort action record')
            require(isinstance(actions.get('actions'), list) and actions['actions'], 'No actual actions recorded')
            for action in actions['actions']:
                require(key(action) in known and isinstance(action.get('action'), str) and action['action']
                        and isinstance(action.get('result'), str) and action['result'], 'Action identity/result does not belong to this batch')
        else:
            require(closure['original_batch_unfinished'] is False, 'Normal termination flag differs')
            require(not (source/'CONTROLLER_FAILURE.json').exists()
                    and not (source/'UNRESOLVED_OWNED_PROCESSES.json').exists(), 'Normal closure has unresolved controller/cleanup evidence')
            finish = snapshot_json(post, 'EXECUTION_FINISHED.json')
            require(finish['status'] == 'FINISHED_WITH_COMPLETE_OUTCOME_LEDGER'
                    and finish['protocol_sha256'] == protocol_sha and finish['runtime_sha256'] == runtime_sha,
                    'No actual bound normal-termination ledger')
            require(finish['logical_outcomes'] == p['counts']['logical_outcomes'], 'Normal terminal ledger count differs')
            for name, expected in [('CV_OUTCOMES.json', p['counts']['maximum_cv_fits']),
                                   ('FINAL_OUTCOMES.json', p['counts']['logical_outcomes']),
                                   ('SELECTIONS.json', p['counts']['maximum_final_fits'])]:
                values = snapshot_json(post, name)
                require(isinstance(values, list) and len(values) == expected,
                        'Normal terminal ledger has wrong row count: ' + name)
            for rel in post['_rows']:
                if rel.endswith('/launcher_receipt.json'):
                    receipt = snapshot_json(post, rel)
                    require(receipt.get('matlab_started') is not True or receipt.get('cleanup', {}).get('complete') is True,
                            'Normal termination has unresolved receipt cleanup')
        current_generation_matches(source, post)
        scan = fresh_scan(process_provider or evidence.PsutilReader(), source, known,
                          closure['controller_executable'], runtime_sha)
        result.update(scan)
        # Bind again after the process observation; never silently admit mutated evidence.
        require(evidence.digest(closure_path) == closure_sha, 'Closure changed during fresh observation')
        verify_sources(closure)
        for name in ['pre_snapshot', 'post_snapshot', 'exit_verification'] + (['actions'] if kind == 'OPERATIONAL_ABORT' else []):
            ref = closure[name]
            require(evidence.digest(relative_path(root, ref['path'])) == ref['sha256'],
                    'Referenced evidence changed during process observation: ' + name)
        for m in [pre, post]:
            for path, expected in m['_used'].items():
                require(evidence.digest(path) == expected, 'Consumed archived identity evidence changed')
        current_generation_matches(source, post)
        result['registered_identity_count'] = len(known)
        result['passed'] = (scan['live_previous_batch_processes'] == 0
                            and scan['unverified_previous_batch_processes'] == 0)
    except Exception as exc:
        result['errors'].append(type(exc).__name__ + ': ' + str(exc))
        result['passed'] = False
        result['unverified_previous_batch_processes'] = max(1, result['unverified_previous_batch_processes'])
    return result


if __name__ == '__main__':
    print('Import verify_closed_for_successor(s1_root, closure_path, closure_sha). No scan is performed by default.')
