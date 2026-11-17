"""Local-only operational evidence. No process control and no scientific sealing."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys

CHUNK = 256 * 1024
HERE = Path(__file__).resolve().parent
CAPTURES = HERE / 'captures'
SCHEMA = 1


def utc():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(io_path(path).read_text(encoding='utf-8-sig'))


def logical_path(path):
    text = os.fspath(path)
    if os.name == 'nt' and text.startswith('\\\\?\\UNC\\'):
        text = '\\\\' + text[8:]
    elif os.name == 'nt' and text.startswith('\\\\?\\'):
        text = text[4:]
    return Path(text)


def io_path(path):
    """Extended Windows paths for I/O only; evidence retains ordinary absolute paths."""
    path = logical_path(path).absolute()
    if os.name != 'nt':
        return path
    text = str(path)
    return Path('\\\\?\\UNC\\' + text[2:] if text.startswith('\\\\') else '\\\\?\\' + text)


def norm(path):
    return os.path.normcase(os.path.abspath(os.fspath(logical_path(path))))


def inside(path, root):
    try:
        return os.path.commonpath([norm(path), norm(root)]) == norm(root)
    except ValueError:
        return False


def metadata(path):
    s = io_path(path).lstat()
    return dict(size=s.st_size, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns,
                inode=s.st_ino, device=s.st_dev)


def is_link(path):
    s = io_path(path).lstat()
    return stat.S_ISLNK(s.st_mode) or bool(getattr(s, 'st_file_attributes', 0) & 0x400)


def reject_links(path):
    path = logical_path(path).absolute()
    for ancestor in [path, *path.parents]:
        if io_path(ancestor).exists() and is_link(ancestor):
            raise ValueError('Symlink/reparse point is not allowed: ' + str(ancestor))


def digest(path):
    h = hashlib.sha256()
    with io_path(path).open('rb') as f:
        for block in iter(lambda: f.read(CHUNK), b''):
            h.update(block)
    return h.hexdigest()


def observation(path):
    """Hash at most the initially observed length; live append cannot extend work."""
    before = metadata(path)
    h, count = hashlib.sha256(), 0
    with io_path(path).open('rb') as f:
        while count < before['size']:
            block = f.read(min(CHUNK, before['size'] - count))
            if not block:
                break
            h.update(block)
            count += len(block)
    after = metadata(path)
    return dict(before=before, after=after, sha256=h.hexdigest(), bytes_hashed=count,
                stable_during_observation=before == after and count == before['size'])


def verify_bindings(source, expected_runtime_sha):
    """Only source/protocol files are hashed; external X/y files are never opened."""
    source = Path(source).absolute()
    reject_links(source)
    result = dict(checked_utc=utc(), passed=False, unresolved=[], sources=[])
    try:
        paths = {n: source / n for n in ('FROZEN_PROTOCOL.json', 'RUNTIME_PROTOCOL.json', 'FREEZE_RECEIPT.json')}
        for path in paths.values():
            reject_links(path)
        hashes = {name: digest(path) for name, path in paths.items()}
        p, r, f = (read_json(paths[n]) for n in paths)
        result['protocol_files'] = hashes
        if hashes['RUNTIME_PROTOCOL.json'] != expected_runtime_sha:
            raise ValueError('Expected runtime SHA does not match')
        if r['original_scientific_protocol_sha256'] != hashes['FROZEN_PROTOCOL.json']:
            raise ValueError('Runtime scientific-protocol binding differs')
        if f['protocol_sha256'] != hashes['FROZEN_PROTOCOL.json'] or f['runtime_sha256'] != expected_runtime_sha:
            raise ValueError('Freeze receipt binding differs')
        if f['preflight']['passed'] is not True or p['source_hashes'] != r['bound_files']:
            raise ValueError('Preflight or bound source set differs')
        for raw, expected in sorted(p['source_hashes'].items()):
            path = Path(raw)
            if not inside(path, source):
                raise ValueError('Bound executable is outside source directory')
            try:
                reject_links(path)
                seen = observation(path)
                okay = seen['stable_during_observation'] and seen['sha256'] == expected
                result['sources'].append(dict(path=str(path), expected_sha256=expected,
                    observation=seen, matched=okay))
                if not okay:
                    result['unresolved'].append(dict(path=str(path), reason='SOURCE_CHANGED_OR_HASH_MISMATCH'))
            except (OSError, ValueError) as exc:
                result['unresolved'].append(dict(path=str(path), reason=repr(exc)))
        expected_set = {norm(x) for x in p['source_hashes']}
        actual_set = {norm(x) for x in io_path(source).glob('*.py')} | {norm(x) for x in io_path(source).glob('*.m')}
        actual_set |= {norm(x) for x in io_path(source / 'private_generated').glob('*.m')}
        if expected_set != actual_set:
            result['unresolved'].append(dict(reason='EXECUTABLE_SOURCE_SET_DIFFERS',
                unexpected=sorted(actual_set - expected_set), missing=sorted(expected_set - actual_set)))
        external = {}
        for case in p['cases']:
            path, sha = case['source_input'], case['source_input_sha256']
            if path in external and external[path] != sha:
                raise ValueError('Conflicting external source hashes')
            external[path] = sha
        result['external_input_references'] = [dict(path=k, expected_sha256=v,
            copied=False, opened=False, hash_recomputed=False) for k, v in sorted(external.items())]
        result['external_input_count'] = len(external)
        if len(external) != 72:
            result['unresolved'].append(dict(reason='EXPECTED_72_EXTERNAL_SOURCE_INPUTS', observed=len(external)))
        result['passed'] = not result['unresolved']
        return result, p
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result['unresolved'].append(dict(reason=repr(exc)))
        return result, None


class PsutilReader:
    """Read-only adapter. Deliberately no terminate/suspend/signal methods."""
    def __init__(self):
        import psutil
        self.psutil = psutil

    def minimal(self):
        records, unresolved = [], []
        for process in self.psutil.process_iter():
            try:
                records.append(dict(pid=process.pid, ppid=process.ppid(),
                    create_time=process.create_time(), name=process.name()))
            except self.psutil.NoSuchProcess:
                unresolved.append(dict(pid=process.pid, reason='VANISHED_DURING_ENUMERATION'))
            except self.psutil.Error as exc:
                unresolved.append(dict(pid=process.pid, reason=type(exc).__name__))
        return records, unresolved

    def details(self, pid):
        p = self.psutil.Process(pid)
        with p.oneshot():
            return dict(pid=p.pid, ppid=p.ppid(), create_time=p.create_time(),
                name=p.name(), exe=p.exe(), cwd=p.cwd(), cmdline=p.cmdline())


def parent_chain(pid, index):
    chain, seen = [], {pid}
    ppid = index[pid]['ppid']
    while ppid and ppid in index and ppid not in seen:
        seen.add(ppid)
        row = index[ppid]
        chain.append({k: row[k] for k in ('pid', 'ppid', 'create_time', 'name')})
        ppid = row['ppid']
    return chain


def argument_path(token, cwd):
    return norm(token if os.path.isabs(token) else Path(cwd) / token)


def controller_matches(detail, source, runtime_sha, expected_exe, start_epoch, window):
    args = detail['cmdline']
    script = norm(Path(source) / 'run_formal.py')
    script_present = any(argument_path(a, detail['cwd']) == script for a in args[1:] if not a.startswith('-'))
    flag = '--execute-runtime-sha'
    runtime_present = any(args[i] == flag and i + 1 < len(args) and args[i+1] == runtime_sha
                          for i in range(len(args)))
    return (script_present and runtime_present and norm(detail['exe']) == norm(expected_exe)
            and start_epoch - window <= detail['create_time'] <= start_epoch + 2)


BRIDGE = re.compile(r"ssa_clock_bridge\('((?:''|[^'])*)','((?:''|[^'])*)','((?:''|[^'])*)','((?:''|[^'])*)'\)")


def matlab_executables(frozen_launcher):
    launcher = Path(frozen_launcher)
    paths = {norm(launcher)}
    if launcher.name.casefold() == 'matlab.exe' and launcher.parent.name.casefold() == 'bin':
        paths.add(norm(launcher.parent / 'win64' / 'MATLAB.exe'))
    return paths


def console_helper_checks(detail, owner):
    """Recognize only the observed Windows direct-child console-host shape."""
    windows = Path(os.environ.get('SystemRoot', r'C:\Windows'))
    expected = windows / 'System32' / 'conhost.exe'
    args = detail['cmdline']
    first = args[0] if args else ''
    if first.startswith('\\??\\'):
        first = first[4:]  # Strip only this exact NT prefix, never arbitrary components.
    return dict(system_executable=norm(detail['exe']) == norm(expected),
        argument_executable=bool(first) and norm(first) == norm(expected),
        command_shape=len(args) == 2 and re.fullmatch(r'0x[0-9a-fA-F]+', args[1]) is not None,
        system_cwd=norm(detail['cwd']) == norm(windows),
        direct_parent=detail['ppid'] == owner['pid'],
        bounded_creation=0 <= detail['create_time'] - owner['create_time'] <= 5)


def inspect_processes(source, protocol, runtime_sha, controller_exe, window=120, reader=None,
                      bindings_verified=False):
    """Return observations, never an exit/cleanup certificate."""
    source = Path(source).absolute()
    result = dict(observed_utc=utc(), observation_only=True, cleanup_verified=False,
        controller=None, task_candidates=[], ancestors=[], excluded_candidate_count=0, unresolved=[])
    if not bindings_verified:
        result['unresolved'].append(dict(reason='SOURCE_BINDINGS_NOT_VERIFIED'))
        result['all_observed_task_identities_resolved'] = False
        return result
    try:
        start = read_json(source / 'EXECUTION_STARTED.json')
        progress_path = source / 'STARTED_PROGRESS.json'
        result['start_record'] = start
        result['progress_record'] = read_json(progress_path) if io_path(progress_path).exists() else None
        if start['runtime_protocol_sha256'] != runtime_sha or start['protocol_sha256'] != digest(source / 'FROZEN_PROTOCOL.json'):
            raise ValueError('Execution start identity differs')
        start_epoch = datetime.fromisoformat(start['start_utc']).timestamp()
        expected_pid = start['controller_pid']
        reader = reader or PsutilReader()
        minimal, errors = reader.minimal()
        result['unresolved'].extend(errors)
        index = {r['pid']: r for r in minimal}
        if expected_pid not in index:
            result['controller'] = dict(pid=expected_pid, identity_status='NOT_OBSERVED',
                cleanup_inferred=False)
            result['unresolved'].append(dict(pid=expected_pid, reason='NOT_OBSERVED_IS_NOT_AN_EXIT_CERTIFICATE'))
        else:
            detail = reader.details(expected_pid)
            consistent = (detail['create_time'] == index[expected_pid]['create_time']
                          and detail['ppid'] == index[expected_pid]['ppid'])
            if consistent and controller_matches(detail, source, runtime_sha, controller_exe, start_epoch, window):
                detail.update(identity_status='VERIFIED_TASK_CANDIDATE', parent_chain=parent_chain(expected_pid, index))
                result['controller'] = detail
                result['ancestors'] = detail['parent_chain']
            else:
                result['controller'] = dict(pid=expected_pid, create_time=detail['create_time'],
                    identity_status='UNRESOLVED', reason='CONTROLLER_IDENTITY_MISMATCH')
                result['unresolved'].append(dict(pid=expected_pid, reason='CONTROLLER_IDENTITY_MISMATCH'))
        cases = {c['case_id']: c for c in protocol['cases']}
        matlab_exes = matlab_executables(protocol['matlab_executable'])
        for pid, minimal_row in index.items():
            if pid == expected_pid or minimal_row['name'].casefold() not in ('matlab', 'matlab.exe'):
                continue  # Shared MathWorks services and unrelated executables are excluded.
            try:
                detail = reader.details(pid)
                command = detail['cmdline']
                batches = [command[i+1] for i, token in enumerate(command[:-1]) if token == '-batch']
                match = BRIDGE.search(batches[0]) if len(batches) == 1 else None
                if not match:
                    result['excluded_candidate_count'] += 1
                    continue  # No command line from hand-opened MATLAB is retained.
                input_path, output_path, author_path, run_id = [v.replace("''", "'") for v in match.groups()]
                if not inside(output_path, source / 'runs'):
                    result['excluded_candidate_count'] += 1
                    continue
                # Only a source-owned bridge candidate may have its full command retained.
                detail['parent_chain'] = parent_chain(pid, index)
                detail['identity_status'] = 'UNRESOLVED'
                result['task_candidates'].append(detail)
                reject_links(output_path)
                identity_path = Path(output_path) / 'RUN_IDENTITY.json'
                identity = read_json(identity_path)
                case = cases[identity['case_id']]
                expected_output = source / 'runs' / case['case_id'] / 'repeat1'
                checks = dict(exe=norm(detail['exe']) in matlab_exes,
                    stable_pid=detail['create_time'] == minimal_row['create_time'] and detail['ppid'] == minimal_row['ppid'],
                    cwd=norm(detail['cwd']) == norm(expected_output),
                    output=norm(output_path) == norm(expected_output),
                    input=norm(input_path) == norm(case['input']),
                    author=norm(author_path) == norm(protocol['author_root']),
                    run_id=run_id == identity['run_id'],
                    runtime=identity['runtime_protocol_sha256'] == runtime_sha,
                    protocol=identity['scientific_protocol_sha256'] == digest(source / 'FROZEN_PROTOCOL.json'),
                    input_sha=identity['input_sha256'] == case['sha256'],
                    creation_window=max(start_epoch - 2, io_path(identity_path).stat().st_mtime - 2) <= detail['create_time'] <= datetime.now(timezone.utc).timestamp() + 2,
                    ancestry_times=all(a['create_time'] <= detail['create_time'] for a in detail['parent_chain']),
                    verified_controller_ancestry=result['controller'].get('identity_status') == 'VERIFIED_TASK_CANDIDATE'
                        and any(a['pid'] == expected_pid and a['create_time'] == result['controller']['create_time'] for a in detail['parent_chain']))
                receipt_path = Path(output_path) / 'launcher_receipt.json'
                if io_path(receipt_path).exists():
                    receipt = read_json(receipt_path)
                    checks['receipt_pid'] = receipt.get('launched_pid') == pid
                    checks['receipt_creation'] = receipt.get('process_create_time') == detail['create_time']
                detail.update(identity_checks=checks, run_identity_sha256=digest(identity_path))
                if all(checks.values()):
                    detail['identity_status'] = 'VERIFIED_TASK_CANDIDATE'
                else:
                    result['unresolved'].append(dict(pid=pid, reason='WORKER_IDENTITY_CHECK_FAILED', checks=checks))
            except Exception as exc:
                result['unresolved'].append(dict(pid=pid, reason=type(exc).__name__ + ': ' + str(exc)))
        roots = [r for r in result['task_candidates'] if r['identity_status'] == 'VERIFIED_TASK_CANDIDATE']
        covered = {r['pid'] for r in result['task_candidates']} | {expected_pid}
        shared = {'mathworksservicehost.exe', 'mathworksservicehost', 'mathworkslicensing.exe',
                  'lmgrd.exe', 'mlm.exe', 'mwlm.exe'}
        for pid, minimal_row in index.items():
            if pid in covered:
                continue
            chain = parent_chain(pid, index)
            owners = [r for r in roots if any(a['pid'] == r['pid'] and a['create_time'] == r['create_time'] for a in chain)]
            if not owners:
                continue
            if minimal_row['name'].casefold() in shared or any(a['name'].casefold() in shared for a in chain):
                result['excluded_candidate_count'] += 1
                continue
            try:
                detail = reader.details(pid)
                owner = owners[0]
                if minimal_row['name'].casefold() == 'conhost.exe':
                    direct = next((r for r in owners if r['pid'] == detail['ppid']), owner)
                    checks = console_helper_checks(detail, direct)
                    checks['stable_pid'] = (detail['create_time'] == minimal_row['create_time']
                                            and detail['ppid'] == minimal_row['ppid'])
                    detail.update(parent_chain=chain, identity_checks=checks,
                        inherited_run_identity_from_pid=direct['pid'], process_role='OS_CONSOLE_HELPER',
                        process_control_policy='NOT_A_NUMERICAL_WORKER_NO_AUTOMATIC_CONTROL',
                        identity_status='VERIFIED_TASK_ASSOCIATED_HELPER' if all(checks.values()) else 'UNRESOLVED')
                    result['task_candidates'].append(detail)
                    if not all(checks.values()):
                        result['unresolved'].append(dict(pid=pid, reason='OS_CONSOLE_HELPER_UNRESOLVED'))
                    continue
                checks = dict(stable_pid=detail['create_time'] == minimal_row['create_time']
                    and detail['ppid'] == minimal_row['ppid'],
                    creation_after_owner=detail['create_time'] >= owner['create_time'],
                    cwd_in_run=inside(detail['cwd'], owner['cwd']),
                    ancestry_times=all(a['create_time'] <= detail['create_time'] for a in chain))
                detail.update(parent_chain=chain, identity_checks=checks,
                    inherited_run_identity_from_pid=owner['pid'],
                    identity_status='VERIFIED_TASK_CANDIDATE' if all(checks.values()) else 'UNRESOLVED')
                result['task_candidates'].append(detail)
                if not all(checks.values()):
                    result['unresolved'].append(dict(pid=pid, reason='DESCENDANT_IDENTITY_UNRESOLVED'))
            except Exception as exc:
                result['unresolved'].append(dict(pid=pid, reason=type(exc).__name__ + ': ' + str(exc)))
        result['all_observed_task_identities_resolved'] = not result['unresolved']
    except Exception as exc:
        result['unresolved'].append(dict(reason=type(exc).__name__ + ': ' + str(exc)))
        result['all_observed_task_identities_resolved'] = False
    return result


def inventory(source):
    files, unresolved, stack = [], [], [Path(source)]
    while stack:
        folder = stack.pop()
        try:
            with os.scandir(io_path(folder)) as entries:
                for entry in entries:
                    path = logical_path(entry.path)
                    try:
                        if is_link(path):
                            unresolved.append(dict(path=str(path), reason='REPARSE_OR_SYMLINK_NOT_FOLLOWED'))
                        elif entry.is_dir(follow_symlinks=False):
                            stack.append(path)
                        elif entry.is_file(follow_symlinks=False):
                            files.append(path)
                        else:
                            unresolved.append(dict(path=str(path), reason='NONREGULAR_ENTRY'))
                    except OSError as exc:
                        unresolved.append(dict(path=str(path), reason=repr(exc)))
        except OSError as exc:
            unresolved.append(dict(path=str(folder), reason=repr(exc)))
    return sorted(files), unresolved


def copy_observed(path, target):
    row = dict(source=str(path), target=str(target), status='UNRESOLVED')
    try:
        reject_links(path)
        row['before'] = observation(path)
        io_path(target.parent).mkdir(parents=True, exist_ok=True)
        size = metadata(path)['size']
        h, copied = hashlib.sha256(), 0
        with io_path(path).open('rb') as source, io_path(target).open('xb') as destination:
            while copied < size:
                chunk = source.read(min(CHUNK, size - copied))
                if not chunk:
                    break
                destination.write(chunk)
                h.update(chunk)
                copied += len(chunk)
        row.update(copied_sha256=h.hexdigest(), copied_bytes=copied)
        row['after'] = observation(path)
        same = (row['before']['stable_during_observation'] and row['after']['stable_during_observation']
            and row['before']['before'] == row['after']['after']
            and row['before']['sha256'] == row['copied_sha256'] == row['after']['sha256']
            and copied == row['before']['before']['size'])
        row['status'] = 'OBSERVED_STABLE' if same else 'UNSTABLE_ACTIVE_OR_CHANGED'
        row['snapshot_file_sha256_verified'] = digest(target) == row['copied_sha256']
    except Exception as exc:
        row['error'] = type(exc).__name__ + ': ' + str(exc)
        if io_path(target).is_file():
            row.update(partial_copy_sha256=digest(target), partial_copy_bytes=io_path(target).stat().st_size)
    return row


def hash_comparison(pre, post_files):
    """Only copied byte hashes are compared, never inferred execution outcomes."""
    before = {r['relative_path']: r.get('copied_sha256', r.get('partial_copy_sha256')) for r in pre['files']}
    after = {r['relative_path']: r.get('copied_sha256', r.get('partial_copy_sha256')) for r in post_files}
    return dict(scope='FILE_HASH_DIFFERENCES_ONLY', cleanup_inferred=False,
        added=sorted(set(after) - set(before)), removed=sorted(set(before) - set(after)),
        changed=sorted(k for k in before.keys() & after.keys() if before[k] != after[k]),
        identical_hash=sorted(k for k in before.keys() & after.keys() if before[k] is not None and before[k] == after[k]),
        uncomparable=sorted(k for k in before.keys() | after.keys() if before.get(k) is None or after.get(k) is None))


def capture(source, destination, runtime_sha, controller_exe, phase, pre_manifest=None,
            exit_verification=None, window=120, capture_root=CAPTURES, process_reader=None):
    source, destination, capture_root = map(lambda p: logical_path(p).absolute(), (source, destination, capture_root))
    if io_path(destination).exists() or not inside(destination, capture_root) or destination == capture_root:
        raise ValueError('Destination must be a new child directory of the dedicated captures directory')
    if inside(destination, source) or inside(source, destination):
        raise ValueError('Source and destination must not contain one another')
    reject_links(source)
    reject_links(destination.parent)
    if phase not in ('pre', 'post'):
        raise ValueError('Explicit pre/post phase required')
    if phase == 'post' and (not pre_manifest or not exit_verification):
        raise ValueError('Post capture requires pre manifest and external exit-verification record')
    if phase == 'pre' and (pre_manifest or exit_verification):
        raise ValueError('Pre capture takes no post-only references')
    prior = None
    refs = {}
    if phase == 'post':
        for kind, path in [('pre_manifest', pre_manifest), ('external_exit_verification', exit_verification)]:
            reject_links(path)
            refs[kind] = dict(path=str(Path(path).absolute()), sha256=digest(path))
        prior = read_json(pre_manifest)
        if prior.get('schema') != SCHEMA or prior.get('phase') != 'pre' or norm(prior['source_root']) != norm(source):
            raise ValueError('Pre manifest is not for this source')
        # Reference only: this tool does not certify the supplied exit evidence.
    check_before, protocol = verify_bindings(source, runtime_sha)
    if protocol is None:
        raise ValueError('Cannot read the bound protocol; no snapshot destination created')
    io_path(destination.parent).mkdir(parents=True, exist_ok=True)
    io_path(destination).mkdir(exist_ok=False)
    manifest = dict(schema=SCHEMA, phase=phase, source_root=str(source), snapshot_root=str(destination),
        capture_started_utc=utc(), status='LIVE_NONTRANSACTIONAL' if phase == 'pre' else 'POST_CAPTURE_NONTRANSACTIONAL',
        scientific_seal=False, private_author_source_included=True, public_distribution_allowed=False,
        use='LOCAL_OPERATIONAL_EVIDENCE_ONLY', cleanup_verified=False, external_references=refs,
        binding_before=check_before, files=[], unresolved=[])
    with io_path(destination / 'CAPTURE_STARTED.json').open('x', encoding='utf-8') as f:
        json.dump(dict(schema=SCHEMA, phase=phase, source_root=str(source),
            started_utc=manifest['capture_started_utc'], status='INCOMPLETE_UNTIL_MANIFEST_EXISTS',
            scientific_seal=False, tool_sha256=digest(__file__), private_local_only=True), f, indent=2)
    manifest['process_before'] = inspect_processes(source, protocol, runtime_sha, controller_exe, window,
        process_reader, bindings_verified=check_before['passed'])
    before_files, errors = inventory(source)
    manifest['unresolved'].extend(errors)
    external_paths = {norm(r['path']) for r in check_before.get('external_input_references', [])}
    manifest['referenced_large_inputs_excluded_from_copy'] = []
    for path in before_files:
        relative = path.relative_to(source).as_posix()
        if norm(path) in external_paths:
            manifest['referenced_large_inputs_excluded_from_copy'].append(relative)
            continue
        row = copy_observed(path, destination / 'source_files' / relative)
        row['relative_path'] = relative
        manifest['files'].append(row)
    after_files, errors = inventory(source)
    manifest['unresolved'].extend(errors)
    before_names = {p.relative_to(source).as_posix() for p in before_files}
    after_names = {p.relative_to(source).as_posix() for p in after_files}
    manifest['inventory_changes'] = dict(appeared_after_initial_inventory=sorted(after_names-before_names),
        disappeared_after_initial_inventory=sorted(before_names-after_names))
    manifest['binding_after'], _ = verify_bindings(source, runtime_sha)
    manifest['process_after'] = inspect_processes(source, protocol, runtime_sha, controller_exe, window,
        process_reader, bindings_verified=manifest['binding_after']['passed'])
    if prior is not None:
        manifest['pre_post_hash_comparison'] = hash_comparison(prior, manifest['files'])
    manifest['capture_finished_utc'] = utc()
    manifest['all_initial_files_observed_stable'] = not manifest['unresolved'] and all(r['status'] == 'OBSERVED_STABLE' for r in manifest['files'])
    manifest['transactional_completeness_claimed'] = False
    with io_path(destination / 'MANIFEST.json').open('x', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    return dict(snapshot=str(destination), manifest_sha256=digest(destination / 'MANIFEST.json'),
        status=manifest['status'], files=len(manifest['files']), scientific_seal=False,
        cleanup_verified=False, public_distribution_allowed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check-bindings', action='store_true', help='Read source bindings only; no process enumeration')
    mode.add_argument('--inspect', action='store_true', help='Explicit read-only live process observations')
    mode.add_argument('--capture', action='store_true', help='Explicit private byte snapshot into a NEW dedicated directory')
    parser.add_argument('--source', type=Path)
    parser.add_argument('--expected-runtime-sha')
    parser.add_argument('--controller-exe', type=Path)
    parser.add_argument('--controller-window-seconds', type=int, default=120)
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--phase', choices=['pre', 'post'])
    parser.add_argument('--pre-manifest', type=Path)
    parser.add_argument('--exit-verification-record', type=Path)
    args = parser.parse_args()
    if not (args.check_bindings or args.inspect or args.capture):
        parser.print_help()
        return
    if not args.source or not re.fullmatch('[0-9a-f]{64}', args.expected_runtime_sha or ''):
        parser.error('Explicit source and 64-character expected runtime SHA required')
    if not 1 <= args.controller_window_seconds <= 600:
        parser.error('Controller creation window must be 1..600 seconds')
    if (args.inspect or args.capture) and not args.controller_exe:
        parser.error('Explicit expected controller executable required')
    if args.capture:
        if not args.destination or not args.phase:
            parser.error('Capture requires explicit destination and phase')
        answer = capture(args.source, args.destination, args.expected_runtime_sha, args.controller_exe,
            args.phase, args.pre_manifest, args.exit_verification_record, args.controller_window_seconds)
    else:
        answer, protocol = verify_bindings(args.source, args.expected_runtime_sha)
        if args.inspect:
            answer['process_observations'] = inspect_processes(args.source, protocol,
                args.expected_runtime_sha, args.controller_exe, args.controller_window_seconds,
                bindings_verified=answer['passed']) if protocol else None
    print(json.dumps(answer, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
