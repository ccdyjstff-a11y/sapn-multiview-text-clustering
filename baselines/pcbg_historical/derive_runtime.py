"""Deterministic, checked wrapper derivation. No data or process invocation."""
from pathlib import Path
import ast,difflib
ROOT=Path(__file__).resolve().parent
V5=ROOT.parent/'native_qualification_clock_v5'
def put(name,data):
    p=ROOT/name
    if p.exists() and p.read_bytes()!=data:
        assert not (ROOT/'PROTOCOL_DRAFT.json').exists() and not (ROOT/'FROZEN_PROTOCOL.json').exists(),'Prepared sources require explicit re-audit'
    p.write_bytes(data)
def replace(text,old,new):
    assert text.count(old)==1,old
    return text.replace(old,new)
def derive():
    for name in ['phase_clock.py','owned_processes.py','test_clock.py']:
        put(name,(V5/name).read_bytes())
    main=ROOT.parent.parent.parent
    put('query_protocol.py',(main/'query_protocol.py').read_bytes())
    worker=(V5/'pcbg_qualification_worker.m').read_text(encoding='utf8')
    changed=replace(worker,'function pcbg_qualification_worker(','function pcbg_formal_worker(')
    changed=replace(changed,"    keys=fieldnames(expectedConfig);","    assert(isscalar(opts.anchor_rate) && any(opts.anchor_rate==[.7,.1,.3]), ...\n        'PCBGQ:AnchorGrid','Undeclared anchor rate');\n    expectedConfig.anchor_rate=opts.anchor_rate;\n    keys=fieldnames(expectedConfig);")
    put('pcbg_formal_worker.m',changed.encode('utf8'))
    bridge=(V5/'pcbg_clock_bridge.m').read_text(encoding='utf8')
    put('pcbg_clock_bridge.m',replace(bridge,'pcbg_qualification_worker(inputPath,outputDir,authorRoot);','pcbg_formal_worker(inputPath,outputDir,authorRoot);').encode('utf8'))
    source=(V5/'run_clock_qualification.py').read_text(encoding='utf8')
    result=source[:source.index('\ndef main():')]
    result=replace(result,'RUNTIME_SHA = None','''RUNTIME_SHA = None
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
    time.sleep(seconds)''')
    result=replace(result,"    for case in p['cases']:\n        assert sha(case['input']) == case['sha256'], case['input']", "    # Entry and seal verify all 216 inputs; each fit verifies only its own MAT.\n    if CURRENT_CASE is not None:\n        assert sha(CURRENT_CASE['input']) == CURRENT_CASE['sha256'], CURRENT_CASE['input']")
    result=replace(result,'    try:\n        verify_runtime()\n        wait_started',"    try:\n        check_deadline(record, 'before_source_verification')\n        verify_runtime()\n        check_deadline(record, 'after_source_verification')\n        wait_started")
    result=replace(result,"    output = ROOT / 'runs' / case['case_id'] / ('repeat' + str(repeat))", "    assert CURRENT_CASE == case, 'Per-fit verification must bind the exact dispatched case'\n    output = ROOT / 'runs' / case['case_id'] / ('repeat' + str(repeat))")
    result=replace(result,"        while psutil.virtual_memory().available < 3 * 2**30:\n            waited", "        while psutil.virtual_memory().available < 3 * 2**30:\n            check_deadline(record, 'start_memory_wait')\n            waited")
    result=result.replace("time.sleep(p['runtime']['poll_seconds'])","sleep_poll(p['runtime']['poll_seconds'])")
    result=replace(result,"record.update(argv=argv, runtime_environment=p['runtime_environment'], matlab_started=True)","record.update(argv=argv, runtime_environment=p['runtime_environment'])")
    result=replace(result,"            origin = time.monotonic()  # Popen creation overhead belongs to startup.","            verify_runtime()  # Source/current MAT recheck after the memory wait.\n            check_deadline(record, 'immediately_before_Popen')\n            if psutil.virtual_memory().available < 3 * 2**30:\n                record['status']='RESOURCE_FELL_BEFORE_LAUNCH'\n                return record\n            check_deadline(record, 'after_immediate_memory_query_before_Popen')\n            origin = time.monotonic()  # Popen creation overhead belongs to startup.")
    result=replace(result,"            record.update(launched_pid=proc.pid, process_create_time=proc.create_time())", "            record.update(matlab_started=True, launched_pid=proc.pid)\n            record['process_create_time']=proc.create_time()")
    result=replace(result,"            while True:\n                errors = remember", "            while True:\n                check_deadline(record, 'running_or_suspended_poll')\n                errors = remember")
    result=replace(result,"                now, code, status = observe_process(proc, output, clock, origin)\n                if status", "                now, code, status = observe_process(proc, output, clock, origin)\n                check_deadline(record, 'after_exit_and_marker_observation')\n                if status")
    result=replace(result,"                    suspended_at, code, status = observe_process(proc, output, clock, origin)\n                    if status", "                    suspended_at, code, status = observe_process(proc, output, clock, origin)\n                    check_deadline(record, 'after_suspend_action')\n                    if status")
    result=replace(result,"                        clock.set_paused(now, False)\n                        outcomes = act_owned", "                        check_deadline(record, 'before_resume_action')\n                        clock.set_paused(now, False)\n                        outcomes = act_owned")
    result=replace(result,"                        outcomes = act_owned(pause_members, 'resume')\n                        if any", "                        outcomes = act_owned(pause_members, 'resume')\n                        check_deadline(record, 'after_resume_action')\n                        if any")
    result=replace(result,"        verify_runtime()\n    except BaseException", "        verify_runtime()\n        check_deadline(record, 'after_final_source_and_input_verification')\n    except BaseException")
    result=replace(result,"        record.pop('worker', None)\n        if proc is not None:","        old_worker=record.pop('worker', None)\n        if old_worker is not None:record['unqualified_worker_summary']=old_worker\n        if isinstance(error, GlobalDeadline):record['status']='GLOBAL_EXECUTION_WALL_CAP'\n        if proc is not None:")
    result=replace(result,"        dump(output / 'launcher_receipt.json', record)","        if GLOBAL_DEADLINE is not None and time.monotonic()>=GLOBAL_DEADLINE:\n            record['global_deadline_triggered']=True\n            record['global_end_overshoot_seconds']=max(0.,time.monotonic()-GLOBAL_DEADLINE)\n            if record['status']=='COMPLETE_ZERO_EXIT_VALID_MARKERS':\n                record['status']='GLOBAL_EXECUTION_WALL_CAP'\n                old_worker=record.pop('worker',None)\n                if old_worker is not None:record['unqualified_worker_summary']=old_worker\n        dump(output / 'launcher_receipt.json', record)")
    result+="\nif __name__=='__main__':\n    raise SystemExit('Use run_formal.py; this module has no independent dispatch entry.')\n"
    put('run_clock_qualification.py',result.encode('utf8'))
    metric_source=main.parent/'SAPN_Text_v4_2_submission/experiment.py'
    raw=metric_source.read_text(encoding='utf8');tree=ast.parse(raw)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='metrics')
    body='\n'.join(raw.splitlines()[fn.lineno-1:fn.end_lineno])+'\n'
    put('metrics_reference.py',('"""Exact metrics function extracted from the original scorer; no model imports."""\nimport numpy as np\nfrom scipy.optimize import linear_sum_assignment\nfrom sklearn.metrics import normalized_mutual_info_score,adjusted_rand_score\n\n'+body).encode('utf8'))
    diff=''.join(difflib.unified_diff(source.splitlines(True),result.splitlines(True),fromfile='v5/run_clock_qualification.py',tofile='formal/run_clock_qualification.py'))
    diff+='\n'+''.join(difflib.unified_diff(worker.splitlines(True),changed.splitlines(True),fromfile='v5/pcbg_qualification_worker.m',tofile='formal/pcbg_formal_worker.m'))
    put('SOURCE_DIFF.patch',diff.encode('utf8'))
if __name__=='__main__':derive()
