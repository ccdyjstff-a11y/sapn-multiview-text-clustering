"""Runtime-only three-phase launcher; default is read-only status.

The exact runtime SHA is required to execute. No class labels or scoring.
SSA finite-profile worker and exact new inputs are bound by this protocol.
"""
from pathlib import Path
import argparse, datetime, hashlib, importlib.util, json, os, subprocess, sys, time, traceback, uuid
sys.dont_write_bytecode = True
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import psutil
from phase_clock import PhaseClock
from owned_processes import act_owned, identity_state

ROOT = Path(__file__).resolve().parent
RUNTIME = None
RUNTIME_SHA = None
CURRENT_CASE = None
GLOBAL_DEADLINE = None

class GlobalDeadline(RuntimeError):
    pass

def check_deadline(record, stage):
    if GLOBAL_DEADLINE is not None and time.monotonic() >= GLOBAL_DEADLINE:
        record.update(global_deadline_triggered=True, global_deadline_stage=stage,
                      global_detection_overshoot_seconds=max(0.,time.monotonic()-GLOBAL_DEADLINE))
        raise GlobalDeadline('48-hour execution deadline reached: '+stage)

def sleep_poll(seconds):
    if GLOBAL_DEADLINE is not None:
        seconds=min(seconds,max(0.,GLOBAL_DEADLINE-time.monotonic()))
    time.sleep(seconds)

def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()

def dump(path, value):
    temp = Path(path).with_suffix(Path(path).suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    os.replace(temp, path)

def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def q(value):
    return "'" + str(value).replace("'", "''") + "'"

def verify_runtime():
    assert sha(ROOT / 'RUNTIME_PROTOCOL.json') == RUNTIME_SHA
    for path, digest in RUNTIME['bound_files'].items():
        assert sha(path) == digest, path
    p = read(ROOT / 'FROZEN_PROTOCOL.json')
    assert sha(ROOT / 'FROZEN_PROTOCOL.json') == RUNTIME['original_scientific_protocol_sha256']
    for path, digest in p['source_hashes'].items():
        assert sha(path) == digest, path
    # Entry and seal verify all 216 inputs; each fit verifies only its own MAT.
    if CURRENT_CASE is not None:
        assert sha(CURRENT_CASE['input']) == CURRENT_CASE['sha256'], CURRENT_CASE['input']
        assert sha(CURRENT_CASE['source_input']) == CURRENT_CASE['source_input_sha256'], CURRENT_CASE['source_input']
    return p

def assess_environment(receipt, probe, probe_sha, manifest_sha, runtime_environment):
    reasons = []
    for condition, reason in [
        (receipt.get('status') == 'PASS', 'environment launcher status is not PASS'),
        (receipt.get('exit_code') == 0, 'environment process did not exit zero'),
        (receipt.get('probe_sha256') == probe_sha, 'launcher does not bind the completed probe result'),
        (receipt.get('manifest_sha256') == manifest_sha, 'environment manifest identity mismatch'),
        (receipt.get('runtime_environment') == runtime_environment, 'runtime environment mismatch'),
        (probe.get('status') == 'PASS', 'environment function did not pass'),
        (receipt.get('dataset_loaded') is False and receipt.get('model_invoked') is False and
         probe.get('dataset_loaded') is False and probe.get('model_invoked') is False, 'environment check scope mismatch')]:
        if not condition:
            reasons.append(reason)
    return dict(passed=not reasons, reasons=reasons)

def environment_gate(p):
    e = RUNTIME['bridge_evidence']
    result, receipt = read(e['result']), read(e['receipt'])
    passed = (sha(e['result']) == e['result_sha256'] and sha(e['receipt']) == e['receipt_sha256']
        and result.get('passed') is True and result.get('class_truth_loaded') is False
        and result.get('ACC_NMI_ARI_computed') is False and receipt.get('exit_code') == 0
        and receipt.get('status') == 'RETURNED')
    return dict(passed=passed, scope='Actual completed trace bridge in the same MATLAB environment')


def remember(proc, entries):
    errors = []
    try:
        members = [proc] + proc.children(recursive=True)
    except psutil.NoSuchProcess:
        members = [proc]
    except psutil.Error as error:
        members = [proc]
        errors.append(repr(error))
    for item in members:
        try:
            entries[(item.pid, item.create_time())] = item
        except psutil.NoSuchProcess:
            pass
        except psutil.Error as error:
            errors.append(repr(error))
    return errors

def cleanup(proc, entries):
    errors, actions = [], []
    try:
        errors.extend(remember(proc, entries))
    except Exception as error:
        errors.append('discovery: ' + repr(error))
    try:
        actions = act_owned(entries, 'kill')
    except Exception as error:
        errors.append('kill: ' + repr(error))
    try:
        psutil.wait_procs(list(entries.values()), timeout=5)
    except Exception as error:
        errors.append('bounded reap: ' + repr(error))
    unresolved = []
    for (pid, created), process in entries.items():
        state, error = identity_state(process, created)
        if state in ['live', 'unverified']:
            unresolved.append(dict(pid=pid, create_time=created, state=state, error=error))
    return dict(actions=actions, unresolved_original_processes=unresolved,
                discovery_errors=errors, complete=not unresolved and not errors)

def freeze_owned_tree(proc, entries):
    """Freeze root first, then descendants until stable; no free active time."""
    began = time.monotonic()
    root_key = (proc.pid, proc.create_time())
    entries[root_key] = proc
    frozen = {root_key: proc}
    outcomes = act_owned(frozen, 'suspend')
    if any(item['result'] == 'error' for item in outcomes):
        raise RuntimeError('Could not suspend owned root: ' + repr(outcomes))
    while True:
        errors = remember(proc, entries)
        if errors:
            raise RuntimeError('Could not enumerate frozen root descendants: ' + repr(errors))
        pending = {key: value for key, value in entries.items() if key not in frozen}
        if not pending:
            return frozen, outcomes
        if time.monotonic() - began >= 5:
            raise RuntimeError('Owned tree did not become stable during bounded suspension')
        actions = act_owned(pending, 'suspend')
        outcomes.extend(actions)
        if any(item['result'] == 'error' for item in actions):
            raise RuntimeError('Could not suspend owned descendants: ' + repr(actions))
        frozen.update(pending)

def read_markers(output):
    # Read finished first: if it exists, the prior atomic started rename must
    # already have happened. Reading in the other order can create a false race.
    values = {}
    for phase in ['finished', 'started']:
        path = output / ('worker_' + phase + '.json')
        if path.exists():
            try:
                values[phase] = read(path)
            except (ValueError, OSError) as error:
                return None, None, phase + ': ' + repr(error)
    return values.get('started'), values.get('finished'), None

def observe_process(proc, output, clock, origin):
    markers = read_markers(output)
    code = proc.poll()
    if code is not None:
        markers = read_markers(output)  # Final snapshot after confirmed exit.
    now = time.monotonic() - origin
    status = clock.observe(now, *markers[:2], code is not None, code, markers[2])
    return now, code, status

def run_one(p, case, repeat):
    assert CURRENT_CASE == case, 'Per-fit verification must bind the exact dispatched case'
    output = ROOT / 'runs' / case['case_id'] / ('repeat' + str(repeat))
    # Never reuse a receipt created under another clock, protocol, or input.
    if (output / 'launcher_receipt.json').exists():
        old = read(output / 'launcher_receipt.json')
        assert old['runtime_protocol_sha256'] == RUNTIME_SHA
        assert old['protocol_sha256'] == sha(ROOT / 'FROZEN_PROTOCOL.json')
        assert old['input_sha256'] == case['sha256']
        for name, digest in old.get('output_hashes', {}).items():
            assert sha(output / name) == digest, name
        return old
    assert not output.exists(), 'Preserve any partial run; no silent restart'
    output.mkdir(parents=True)
    identity = dict(run_id=uuid.uuid4().hex, case_id=case['case_id'], repeat=repeat,
                    input_sha256=case['sha256'], scientific_protocol_sha256=sha(ROOT / 'FROZEN_PROTOCOL.json'),
                    runtime_protocol_sha256=RUNTIME_SHA)
    dump(output / 'RUN_IDENTITY.json', identity)
    record = dict(case_id=case['case_id'], repeat=repeat, start_utc=utc(),
                  protocol_sha256=identity['scientific_protocol_sha256'], runtime_protocol_sha256=RUNTIME_SHA,
                  input_sha256=case['sha256'], run_id=identity['run_id'], status='NOT_STARTED',
                  matlab_started=False, events=[], output_hashes={})
    proc = None
    entries = {}
    clock = None
    pause_members = {}
    try:
        check_deadline(record, 'before_source_verification')
        verify_runtime()
        check_deadline(record, 'after_source_verification')
        wait_started = time.monotonic()
        while psutil.virtual_memory().available < 3 * 2**30:
            check_deadline(record, 'start_memory_wait')
            waited = time.monotonic() - wait_started
            dump(ROOT / 'STARTED_PROGRESS.json', dict(case_id=case['case_id'], repeat=repeat,
                 status='WAITING_FOR_START_MEMORY', elapsed_seconds=waited, updated_utc=utc()))
            if waited >= p['runtime']['memory_wait_limit_seconds']:
                record['status'] = 'RESOURCE_START_WAIT_CAP'
                return record
            sleep_poll(p['runtime']['poll_seconds'])
        record['resource_start_wait_seconds'] = time.monotonic() - wait_started
        command = 'addpath(' + q(ROOT) + '); ssa_clock_bridge(' + ','.join(map(q, [
            case['input'], output, p['author_root'], identity['run_id']])) + ');'
        argv = [p['matlab_executable'], '-nosplash', '-nodesktop', '-singleCompThread',
                '-sd', str(output), '-batch', command]
        env = os.environ.copy()
        env.update(p['runtime_environment'])
        env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        record.update(argv=argv, runtime_environment=p['runtime_environment'])
        clock = PhaseClock(identity['run_id'], case['timeout_active_seconds'],
                           pause_limit=p['runtime']['memory_wait_limit_seconds'])
        with (output / 'stdout.log').open('wb') as log:
            verify_runtime()  # Source/current MAT recheck after the memory wait.
            check_deadline(record, 'immediately_before_Popen')
            if psutil.virtual_memory().available < 3 * 2**30:
                record['status']='RESOURCE_FELL_BEFORE_LAUNCH'
                return record
            check_deadline(record, 'after_immediate_memory_query_before_Popen')
            origin = time.monotonic()  # Popen creation overhead belongs to startup.
            proc = psutil.Popen(argv, cwd=output, env=env, stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            record.update(matlab_started=True, launched_pid=proc.pid)
            record['process_create_time']=proc.create_time()
            last_progress = -float('inf')
            while True:
                check_deadline(record, 'running_or_suspended_poll')
                errors = remember(proc, entries)
                if errors:
                    raise RuntimeError('Owned-process discovery failed: ' + repr(errors))
                now, code, status = observe_process(proc, output, clock, origin)
                check_deadline(record, 'after_exit_and_marker_observation')
                if status is not None:
                    record['status'] = status
                    record['cleanup'] = cleanup(proc, entries)
                    if not record['cleanup']['complete']:
                        record['status'] = 'OWNED_CLEANUP_INCOMPLETE'
                    break
                free = psutil.virtual_memory().available
                if not clock.paused and free < 2 * 2**30:
                    pause_members, outcomes = freeze_owned_tree(proc, entries)
                    # If a live owned process could not suspend, budget cannot
                    # legitimately be paused while it continues computing.
                    if any(item['result'] == 'error' for item in outcomes):
                        record['status'] = 'OWNED_SUSPEND_FAILED'
                        record['events'].append(dict(event='suspend_failed', outcomes=outcomes, utc=utc()))
                        record['cleanup'] = cleanup(proc, entries)
                        break
                    # Suspend calls take time while some descendants may still
                    # execute; charge it before pausing the active clock.
                    suspended_at, code, status = observe_process(proc, output, clock, origin)
                    check_deadline(record, 'after_suspend_action')
                    if status is not None:
                        record['status'] = status
                        record['cleanup'] = cleanup(proc, entries)
                        break
                    clock.set_paused(suspended_at, True)
                    record['events'].append(dict(event='memory_suspend', elapsed_seconds=now,
                                                   effective_elapsed_seconds=suspended_at,
                                                   suspend_action_active_seconds=suspended_at-now,
                                                   available_bytes=free, outcomes=outcomes))
                elif clock.paused:
                    new_members = {key: value for key, value in entries.items() if key not in pause_members}
                    if new_members:
                        # A stable fully suspended tree cannot create a live
                        # descendant. Do not grant unmetered time if that invariant fails.
                        raise RuntimeError('Unexpected new owned descendant while clock was paused')
                    if free >= 3 * 2**30:
                        # Charge the entire resume action conservatively.
                        check_deadline(record, 'before_resume_action')
                        clock.set_paused(now, False)
                        outcomes = act_owned(pause_members, 'resume')
                        check_deadline(record, 'after_resume_action')
                        if any(item['result'] == 'error' for item in outcomes):
                            record['status'] = 'OWNED_RESUME_FAILED'
                            record['cleanup'] = cleanup(proc, entries)
                            break
                        record['events'].append(dict(event='memory_resume', elapsed_seconds=now,
                                                       available_bytes=free, outcomes=outcomes))
                        pause_members = {}
                if now - last_progress >= 10:
                    dump(ROOT / 'STARTED_PROGRESS.json', dict(case_id=case['case_id'], repeat=repeat,
                         status='SUSPENDED' if clock.paused else 'RUNNING', pid=proc.pid, updated_utc=utc(),
                         available_bytes=free, clock=clock.report(), runtime_protocol_sha256=RUNTIME_SHA))
                    last_progress = now
                sleep_poll(p['runtime']['poll_seconds'])
            record['exit_code'] = proc.wait(timeout=10)
        record['clock'] = clock.report()
        if record.get('cleanup') and not record['cleanup']['complete']:
            record['status'] = 'OWNED_CLEANUP_INCOMPLETE'
        if (output / 'worker_summary.json').exists():
            worker = read(output / 'worker_summary.json')
            if record['status'] == 'COMPLETE_ZERO_EXIT_VALID_MARKERS' and record['exit_code'] == 0:
                # The original scientific gate still evaluates this summary.
                # A returned worker may have a handled numerical/domain error.
                record['worker'] = worker
            else:
                record['unqualified_worker_summary'] = worker
        if record['status'] == 'COMPLETE_ZERO_EXIT_VALID_MARKERS' and 'worker' not in record:
            record['status'] = 'MISSING_WORKER_SUMMARY_AFTER_RETURN'
        verify_runtime()
        check_deadline(record, 'after_final_source_and_input_verification')
    except BaseException as error:
        record.update(status='LAUNCHER_EXCEPTION', error=repr(error), traceback=traceback.format_exc())
        old_worker=record.pop('worker', None)
        if old_worker is not None:record['unqualified_worker_summary']=old_worker
        if isinstance(error, GlobalDeadline):record['status']='GLOBAL_EXECUTION_WALL_CAP'
        if proc is not None:
            record['cleanup'] = cleanup(proc, entries)
        if isinstance(error, KeyboardInterrupt):
            record['interrupted'] = True
    finally:
        if record.get('cleanup') and not record['cleanup']['complete']:
            record['status'] = 'OWNED_CLEANUP_INCOMPLETE'
            record.pop('worker', None)
            dump(ROOT / 'UNRESOLVED_OWNED_PROCESSES.json', record['cleanup'])
        if clock is not None:
            record['clock'] = clock.report()
        record['end_utc'] = utc()
        for name in ['RUN_IDENTITY.json', 'stdout.log', 'worker_started.json', 'worker_finished.json',
                     'worker_summary.json', 'native_outputs.mat']:
            if (output / name).exists():
                record['output_hashes'][name] = sha(output / name)
        if 'worker_summary.json' in record['output_hashes']:
            record['worker_summary_sha256'] = record['output_hashes']['worker_summary.json']
        if 'native_outputs.mat' in record['output_hashes']:
            record['native_outputs_sha256'] = record['output_hashes']['native_outputs.mat']
        if GLOBAL_DEADLINE is not None and time.monotonic()>=GLOBAL_DEADLINE:
            record['global_deadline_triggered']=True
            record['global_end_overshoot_seconds']=max(0.,time.monotonic()-GLOBAL_DEADLINE)
            if record['status']=='COMPLETE_ZERO_EXIT_VALID_MARKERS':
                record['status']='GLOBAL_EXECUTION_WALL_CAP'
                old_worker=record.pop('worker',None)
                if old_worker is not None:record['unqualified_worker_summary']=old_worker
        dump(output / 'launcher_receipt.json', record)
        dump(ROOT / 'STARTED_PROGRESS.json', dict(case_id=case['case_id'], repeat=repeat,
             status=record['status'], updated_utc=utc(), receipt=str(output / 'launcher_receipt.json')))
    return record

if __name__=='__main__':
    raise SystemExit('Use run_formal.py; this module has no independent dispatch entry.')
