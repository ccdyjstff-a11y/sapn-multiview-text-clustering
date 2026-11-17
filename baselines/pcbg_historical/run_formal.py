"""Single finite native PCBG dispatch. Default is read-only; never reads oracle."""
import argparse,datetime,json,os,sys,time
sys.dont_write_bytecode=True
import numpy as np
from formal_common import *
import query_protocol as qp
import run_clock_qualification as runtime

def order(p):
    return [next(c for c in p['conditions'] if c['dataset']==ds and c['seed']==seed and c['Q']==Q)
        for seed in [23,24,25] for Q in [160,400] for ds in FIT_DATA]
def case_map(p):return {c['case_id']:c for c in p['cases']}
def serialized_scores(a):return [[float(v) if np.isfinite(v) else None for v in row] for row in a]
def must_stop(r):
    return bool(r.get('global_deadline_triggered') or r.get('interrupted') or
        r['status'] in ['OWNED_CLEANUP_INCOMPLETE','LAUNCHER_EXCEPTION','INVALID_PHASE_MARKER',
                       'MISSING_PHASE_MARKER_ON_EXIT','STARTUP_WALL_TIME_CAP'])
def run(expected):
    no_oracle_guard()
    assert sha(ROOT/'RUNTIME_PROTOCOL.json')==expected,'Exact runtime SHA required'
    freeze=read(ROOT/'FREEZE_RECEIPT.json')
    assert freeze['runtime_sha256']==expected and freeze['protocol_sha256']==sha(ROOT/'FROZEN_PROTOCOL.json')
    admission(freeze['pre_freeze_audit'],freeze['pre_freeze_audit_sha256'],'PRE_FREEZE',freeze['protocol_sha256'])
    assert not (ROOT/'UNRESOLVED_OWNED_PROCESSES.json').exists()
    began=time.monotonic();deadline=began+48*3600
    start=dict(start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),start_monotonic=began,
        deadline_monotonic=deadline,global_wall_seconds=48*3600,protocol_sha256=freeze['protocol_sha256'],
        runtime_protocol_sha256=expected,controller_pid=os.getpid(),fits=0)
    with (ROOT/'EXECUTION_STARTED.json').open('x',encoding='utf8') as stream:json.dump(start,stream,indent=2)
    p=protocol()  # Full input/source admission occurs once within the execution deadline.
    assert p['global_wall_seconds']==48*3600
    runtime.RUNTIME=read(ROOT/'RUNTIME_PROTOCOL.json');runtime.RUNTIME_SHA=expected
    runtime.GLOBAL_DEADLINE=deadline
    assert runtime.environment_gate(p)['passed']
    cases=case_map(p);cv=[];final=[];selections=[];stopped=None;attempts=0
    counts=dict(controller_attempts=0,matlab_started=0,worker_native_return_reported=0,strict_eligible=0)
    for c in p['conditions']:
        if c['unsupported']:
            final.append(dict(condition_id=c['id'],dataset=c['dataset'],seed=c['seed'],Q=c['Q'],
                method=METHOD,status='INPUT_UNSUPPORTED',attempted=False,eligible=False,
                finite_partition_retained=False,case_id=None,reason=c['unsupported_reason']))
    def attempt(case):
        nonlocal stopped,attempts
        if time.monotonic()>=deadline:stopped=stopped or 'GLOBAL_EXECUTION_WALL_CAP'
        if stopped:return not_attempted(case,'NOT_ATTEMPTED_AFTER_'+stopped)
        assert attempts<168
        runtime.CURRENT_CASE=case
        r=runtime.run_one(p,case,1);attempts+=1
        counts['controller_attempts']+=1
        counts['matlab_started']+=int(bool(r.get('matlab_started')))
        counts['worker_native_return_reported']+=int(bool(r.get('worker',r.get('unqualified_worker_summary',{})).get('execution_returned')))
        result=outcome(case)
        counts['strict_eligible']+=int(result['eligible'])
        if must_stop(r):stopped=r['status']
        if time.monotonic()>=deadline:stopped=stopped or 'GLOBAL_EXECUTION_WALL_CAP'
        return result
    try:
        for cond in order(p):
            d=paid(cond['query_input'],cond['n']);pairs,answers=d['pairs'][:cond['Q']],d['answers'][:cond['Q']]
            folds=cond['eligible_folds'];scores=np.full((3,len(folds)),np.nan);local=[]
            for fi,fold in enumerate(folds):
                for ci in range(3):
                    case=cases[case_id(cond['dataset'],cond['seed'],cond['Q'],ci,fi)]
                    rec=attempt(case);rec.update(condition_id=cond['id'],candidate=ci,fold=fi)
                    if rec['eligible']:
                        _,z,_=terminal(case)
                        held=np.asarray(fold['validation'],int)
                        scores[ci,fi]=qp.balanced_pair_error(z,pairs[held],answers[held])
                    rec['validation_BER']=float(scores[ci,fi]) if np.isfinite(scores[ci,fi]) else None
                    cv.append(rec);local.append(rec)
                    dump(ROOT/'CV_OUTCOMES.json',cv)
            complete=all(r['attempted'] for r in local)
            if complete:
                index,rule=qp.select_configuration(scores)
                selected=cases[case_id(cond['dataset'],cond['seed'],cond['Q'],index,'final')]
                selection=dict(condition_id=cond['id'],status='SELECTED',scores=serialized_scores(scores),
                    index=index,tau=GRID[index],rule=rule,eligible_folds=len(folds),
                    valid_candidates=int(np.sum(np.all(np.isfinite(scores),axis=1))),final_input_sha256=selected['sha256'])
                immutable(ROOT/(cond['id']+'_selection.json'),selection)
                rec=attempt(selected)
            else:
                selection=dict(condition_id=cond['id'],status='INCOMPLETE_CV_NOT_SELECTED',scores=serialized_scores(scores),
                    index=None,tau=None,rule='not_selected_after_execution_stop',eligible_folds=len(folds))
                immutable(ROOT/(cond['id']+'_selection.json'),selection)
                rec=dict(case_id=None,attempted=False,eligible=False,finite_partition_retained=False,
                    status='NOT_ATTEMPTED_AFTER_'+str(stopped))
            selections.append(selection)
            final.append(dict(rec,condition_id=cond['id'],dataset=cond['dataset'],seed=cond['seed'],Q=cond['Q'],method=METHOD))
            dump(ROOT/'FINAL_OUTCOMES.json',final)
            dump(ROOT/'SELECTIONS.json',selections)
            print(json.dumps(dict(condition_id=cond['id'],status=rec['status'],fit_calls=attempts,global_stop=stopped)),flush=True)
    except BaseException as error:
        # A controller error cannot silently become algorithm success or a resumable run.
        reason='NOT_ATTEMPTED_AFTER_CONTROLLER_FAILURE'
        for cond in order(p):
            for fi in range(2):
                for ci in range(3):
                    if not any((r['condition_id'],r['candidate'],r['fold'])==(cond['id'],ci,fi) for r in cv):
                        c=cases[case_id(cond['dataset'],cond['seed'],cond['Q'],ci,fi)]
                        cv.append(dict(not_attempted(c,reason),condition_id=cond['id'],candidate=ci,fold=fi,validation_BER=None))
            if not any(r['condition_id']==cond['id'] for r in final):
                final.append(dict(condition_id=cond['id'],dataset=cond['dataset'],seed=cond['seed'],Q=cond['Q'],
                    method=METHOD,case_id=None,attempted=False,eligible=False,finite_partition_retained=False,status=reason))
        dump(ROOT/'CV_OUTCOMES.json',cv);dump(ROOT/'FINAL_OUTCOMES.json',final)
        dump(ROOT/'CONTROLLER_FAILURE.json',dict(error=repr(error),controller_attempts=attempts,counts=counts,
            logical_outcomes=len(final),CV_outcomes=len(cv),oracle_read=False,
            warning='Any partial receipt/artifact remains unresolved; global seal is forbidden pending independent technical audit.'))
        raise
    assert len(cv)==144 and len(final)==42 and len(selections)==24 and attempts<=168
    dump(ROOT/'EXECUTION_FINISHED.json',dict(status='FINISHED_WITH_COMPLETE_OUTCOME_LEDGER',
        protocol_sha256=sha(ROOT/'FROZEN_PROTOCOL.json'),runtime_sha256=expected,fit_calls=attempts,
        counts=counts,counting_note='fit_calls means controller attempts, including attempts with no MATLAB start. Native optimization invocation count is not inferred.',
        CV_attempts=sum(r['attempted'] for r in cv),final_attempts=sum(r['attempted'] for r in final),
        logical_outcomes=len(final),global_stop=stopped,elapsed_wall_seconds=time.monotonic()-began,
        end_overshoot_seconds=max(0.,time.monotonic()-deadline),oracle_read=False,scoring_performed=False))
def main():
    a=argparse.ArgumentParser();a.add_argument('--execute-runtime-sha');args=a.parse_args()
    if args.execute_runtime_sha:run(args.execute_runtime_sha)
    else:
        path=ROOT/'RUNTIME_PROTOCOL.json'
        print(json.dumps(dict(status='FROZEN_NOT_EXECUTED' if path.exists() else 'NOT_FROZEN_NO_EXECUTION',
            runtime_sha256=sha(path) if path.exists() else None,already_started=(ROOT/'EXECUTION_STARTED.json').exists()),indent=2))
if __name__=='__main__':main()
