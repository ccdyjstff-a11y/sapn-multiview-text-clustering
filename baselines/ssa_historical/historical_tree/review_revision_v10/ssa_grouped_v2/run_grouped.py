"""Explicit-SHA native entry; no launch or scientific reads by default."""
from pathlib import Path
import argparse
from copy import deepcopy
import datetime
import hashlib
import json
import os
import sys
import time
import uuid
sys.dont_write_bytecode=True
for _thread_name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[_thread_name]='1'
from adapter_policy import ReceiptPolicy, memory_gate
from controller_core import Group, FitResult, ControllerFailure, run

ROOT=Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return h.hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic(path,value,exclusive=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if exclusive and path.exists():raise FileExistsError(str(path))
    pending=path.with_name(path.name+'.pending.'+uuid.uuid4().hex)
    with pending.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    if exclusive:
        # Same-volume hard-link installation is atomic and cannot replace an
        # existing target; fail closed on filesystems without this capability.
        os.link(pending,path)
        pending.unlink()
    else:
        os.replace(pending,path)


def stop_request(identity):
    path=ROOT/'STOP_AFTER_CONDITION.json'
    if not path.exists():return False
    raw=path.read_bytes()
    request=json.loads(raw.decode('utf-8'))
    if type(request.get('schema')) is not int or request['schema']!=1 or request.get('request')!='STOP_AFTER_CONDITION':
        raise ValueError('Invalid stop request schema')
    for key in ['run_id','protocol_sha256','runtime_sha256']:
        if request.get(key)!=identity[key]:raise ValueError('Stop request identity mismatch: '+key)
    return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),request=request)


class NativeAdapter:
    def __init__(self,p,identity,deadline,fc,rt,qp,psutil):
        self.p,self.identity,self.deadline=p,identity,deadline
        self.fc,self.rt,self.qp,self.psutil=fc,rt,qp,psutil
        self.cases={c['case_id']:c for c in p['cases']}
        self.conditions={c['id']:c for c in p['conditions']}
        self.policy=ReceiptPolicy();self.checked_groups=set();self.decisions={};self.commits={}
        self.counts=dict(adapter_calls=0,launcher_dispatches=0,matlab_started=0,terminal_attempt_lifecycles=0)
        self.stop_reason=None

    def __call__(self,request):
        self.counts['adapter_calls']+=1
        condition=self.conditions[request.condition_id]
        if time.monotonic()>=self.deadline:
            self.stop_reason='GLOBAL_EXECUTION_WALL_CAP'
            path=ROOT/'guards'/(request.case_id+'_deadline.json')
            atomic(path,dict(**self.identity,case_id=request.case_id,status=self.stop_reason,
                native_fit_started=False,launcher_dispatched=False,detected_utc=utc()),exclusive=True)
            return FitResult(request.case_id,'GLOBAL_STOP',False,False,True,False,
                reason=self.stop_reason,evidence_ref=dict(kind='PRE_DISPATCH_DEADLINE',path=str(path),sha256=sha(path)))
        group=(request.corpus,request.Q)
        if group not in self.checked_groups:
            last=[-float('inf')]
            def progress(sample):
                if sample['elapsed_seconds']-last[0]>=5:
                    atomic(ROOT/'GROUP_PROGRESS.json',dict(**self.identity,corpus=group[0],Q=group[1],
                        updated_utc=utc(),status='WAITING_FOR_GROUP_MEMORY',**sample))
                    last[0]=sample['elapsed_seconds']
            guard=memory_gate(condition['n'],lambda:self.psutil.virtual_memory().available,
                time.monotonic,time.sleep,self.deadline,progress)
            guard.update(self.identity);guard.update(corpus=group[0],Q=group[1],end_utc=utc())
            path=ROOT/'guards'/f'{group[0]}_Q{group[1]}.json'
            atomic(path,guard,exclusive=True)
            if guard['status']!='READY':
                self.stop_reason=guard['status']
                return FitResult(request.case_id,'GLOBAL_STOP',False,False,True,False,
                    reason=self.stop_reason,evidence_ref=dict(kind='GROUP_GUARD',path=str(path),sha256=sha(path)))
            self.checked_groups.add(group)
        case=self.cases[request.case_id]
        assert (case['candidate'],case['fold'])==(request.candidate,request.fold)
        self.rt.CURRENT_CASE=case
        dispatch=dict(**self.identity,case_id=request.case_id,input_sha256=case['sha256'],
            dispatched_utc=utc(),ordinal=self.counts['launcher_dispatches']+1)
        atomic(ROOT/'dispatches'/(request.case_id+'.json'),dispatch,exclusive=True)
        self.counts['launcher_dispatches']+=1
        receipt=self.rt.run_one(self.p,case,1)
        saved,labels,eligible=self.fc.terminal(case)
        assert saved==receipt
        receipt_path=ROOT/'runs'/case['case_id']/'repeat1/launcher_receipt.json'
        ber=None
        if request.fold!='final' and eligible:
            data=self.fc.paid(condition['query_input'],condition['n'])
            held=condition['eligible_folds'][request.fold]['validation']
            ber=self.qp.balanced_pair_error(labels,data['pairs'][:condition['Q']][held],data['answers'][:condition['Q']][held])
        decision=self.policy.assess(dict(receipt,is_cv=request.fold!='final'),eligible,ber)
        decision.update(self.identity)
        decision.update(case_id=case['case_id'],receipt_path=str(receipt_path),receipt_sha256=sha(receipt_path),
            finite_partition_retained=labels is not None,validation_BER=ber,
            numerical_eligible=eligible,decision_utc=utc())
        self.counts['matlab_started']+=int(decision['matlab_started'])
        self.counts['terminal_attempt_lifecycles']+=int(decision['terminal_attempt_lifecycle'])
        path=ROOT/'decisions'/(case['case_id']+'.json')
        atomic(path,decision,exclusive=True);self.decisions[case['case_id']]=decision
        reference=dict(kind='NATIVE_DECISION',path=str(path),sha256=sha(path))
        # MAT verification, paid BER, JSON writes and hashes are inside the
        # global budget too. Keep the raw receipt/decision; record an override.
        if time.monotonic()>=self.deadline:
            self.stop_reason='GLOBAL_EXECUTION_WALL_CAP'
            cutoff=ROOT/'guards'/(request.case_id+'_postdecision_deadline.json')
            atomic(cutoff,dict(**self.identity,case_id=request.case_id,status=self.stop_reason,
                stage='after_terminal_verification_BER_and_decision_persistence',
                native_decision=reference,detected_utc=utc()),exclusive=True)
            return FitResult(case['case_id'],'GLOBAL_STOP',True,decision['terminal_attempt_lifecycle'],
                receipt.get('cleanup',{}).get('complete') is True,False,reason=self.stop_reason,
                evidence_ref=dict(kind='POST_DECISION_DEADLINE',path=str(cutoff),sha256=sha(cutoff)))
        if decision['disposition']=='GLOBAL_STOP':self.stop_reason=decision['reason']
        return FitResult(case['case_id'],decision['disposition'],True,
            decision['terminal_attempt_lifecycle'],receipt.get('cleanup',{}).get('complete') is True,
            decision['disposition']=='ELIGIBLE',ber,
            irreversible_local_failure=decision['disposition']=='LOCAL_INELIGIBLE',
            reason=decision['reason'],evidence_ref=reference)

    def commit(self,condition):
        if time.monotonic()>=self.deadline:
            self.stop_reason='GLOBAL_EXECUTION_WALL_CAP'
            raise RuntimeError('Global deadline before condition commit; no acknowledgement')
        path=ROOT/'conditions'/(condition['condition_id']+'.json')
        payload=dict(**self.identity,condition=condition,
            persistence_semantics='Atomic complete-condition payload; acknowledgement belongs to controller ledger.')
        atomic(path,payload,exclusive=True)
        digest=sha(path)
        if time.monotonic()>=self.deadline:
            self.stop_reason='GLOBAL_EXECUTION_WALL_CAP'
            raise RuntimeError('Global deadline during condition commit; persisted payload is unacknowledged')
        self.commits[condition['condition_id']]=dict(path=str(path),sha256=digest)
        return True


def logical_ledgers(state,p,adapter):
    cv,final,selections=[],[],[]
    for condition in state['conditions']:
        meta=dict(condition_id=condition['condition_id'],dataset=condition['corpus'],Q=condition['Q'],seed=condition['seed'])
        for record in condition['cv']+[condition['final']]:
            row=deepcopy(record);row.update(meta)
            decision=adapter.decisions.get(row['case_id'],{})
            row.update(eligible=record['status']=='ELIGIBLE',
                finite_partition_retained=decision.get('finite_partition_retained',False),
                receipt_sha256=decision.get('receipt_sha256'),
                validation_BER=decision.get('validation_BER') if record['status']=='ELIGIBLE' and record['fold']!='final' else None,
                condition_committed=condition['committed'])
            (final if record['fold']=='final' else cv).append(row)
        selections.append(dict(**meta,**deepcopy(condition['selection']),condition_committed=condition['committed']))
    for condition in p['conditions']:
        if condition['unsupported']:
            final.append(dict(condition_id=condition['id'],dataset=condition['dataset'],Q=condition['Q'],seed=condition['seed'],
                case_id=None,status='INPUT_UNSUPPORTED',reason=condition['unsupported_reason'],attempted=False,
                eligible=False,finite_partition_retained=False,condition_committed=False))
    assert len(cv)==864 and len(final)==42 and len(selections)==24
    return cv,final,selections


def execute(expected):
    if not __debug__:raise RuntimeError('Assertions must remain enabled for native evidence checks')
    # Dependencies with scientific-array support are loaded only for explicit execution.
    import psutil
    import formal_common as fc
    import run_clock_qualification as rt
    import query_protocol as qp
    from successor_gate import verify as verify_predecessor
    fc.no_oracle_guard()
    assert sha(ROOT/'RUNTIME_PROTOCOL.json')==expected
    freeze=read(ROOT/'FREEZE_RECEIPT.json')
    assert freeze['runtime_sha256']==expected and freeze['preflight']['passed'] is True
    assert freeze['protocol_sha256']==sha(ROOT/'FROZEN_PROTOCOL.json')
    assert not any((ROOT/name).exists() for name in ['EXECUTION_STARTED.json','CONTROLLER_FAILURE.json','UNRESOLVED_OWNED_PROCESSES.json'])
    assert not (ROOT/'runs').exists(), 'No implicit rerun or recovery'
    assert not (ROOT/'STOP_AFTER_CONDITION.json').exists(), 'Create a request bound to the new run identity after startup'
    frozen_runtime=read(ROOT/'RUNTIME_PROTOCOL.json')
    preliminary=read(ROOT/'FROZEN_PROTOCOL.json')
    closure=verify_predecessor(frozen_runtime['previous_batch_root'],preliminary['prior_batch_protocol_sha256'],
        frozen_runtime.get('previous_batch_closure'))
    began=time.monotonic();deadline=began+48*3600
    identity=dict(run_id=uuid.uuid4().hex,protocol_sha256=freeze['protocol_sha256'],runtime_sha256=expected)
    process=psutil.Process()
    atomic(ROOT/'EXECUTION_STARTED.json',dict(**identity,start_utc=utc(),start_monotonic=began,
        deadline_monotonic=deadline,controller_pid=os.getpid(),controller_create_time=process.create_time(),
        controller_executable=sys.executable,controller_argv=sys.argv,controller_cwd=str(Path.cwd()),
        no_prior_results_imported=True,predecessor_closure_check=closure),exclusive=True)
    rt.RUNTIME=read(ROOT/'RUNTIME_PROTOCOL.json');rt.RUNTIME_SHA=expected;rt.GLOBAL_DEADLINE=deadline
    adapter=None
    try:
        p=fc.protocol()
        rt.verify_runtime()
        assert Path(p['matlab_executable']).is_file()
        assert rt.environment_gate(p)['passed'], 'Bound real trace-bridge environment qualification required'
        assert p['batch']=='SSA_GROUPED_V2' and p['prior_batch_results_imported'] is False
        verify_predecessor(frozen_runtime['previous_batch_root'],p['prior_batch_protocol_sha256'],
            frozen_runtime['previous_batch_closure'])
        adapter=NativeAdapter(p,identity,deadline,fc,rt,qp,psutil)
        def after_condition(condition):
            request=stop_request(identity)
            if request:
                atomic(ROOT/'STOP_REQUEST_ACKNOWLEDGED.json',dict(**identity,
                    condition_id=condition['condition_id'],request=request,
                    condition_commit=adapter.commits[condition['condition_id']],acknowledged_utc=utc()),exclusive=True)
                return True
            return False
        groups=[Group(g['corpus'],g['Q'],tuple(g['seeds'])) for g in p['groups']]
        state=run(groups,adapter,adapter.commit,after_condition)
    except ControllerFailure as error:
        state=error.result
        state['scheduler_stop_reason']=(adapter.stop_reason if adapter else None) or state['stop_reason']
        atomic(ROOT/'CONTROLLER_FAILURE.json',dict(**identity,error=repr(error),
            cause=repr(error.__cause__),state=state,counts=adapter.counts if adapter else None),exclusive=True)
        if adapter is not None:
            cv,final,selections=logical_ledgers(state,p,adapter)
            for name,value in [('CV_OUTCOMES.json',cv),('FINAL_OUTCOMES.json',final),('SELECTIONS.json',selections)]:atomic(ROOT/name,value,exclusive=True)
        raise
    except BaseException as error:
        atomic(ROOT/'CONTROLLER_FAILURE.json',dict(**identity,error=repr(error),
            original_exception_type=type(error).__qualname__,no_completion_claim=True,
            counts=adapter.counts if adapter else None),exclusive=True)
        raise
    state['scheduler_stop_reason']=adapter.stop_reason or state['stop_reason']
    state['adapter_counts']=adapter.counts
    state['condition_commits']=adapter.commits
    cv,final,selections=logical_ledgers(state,p,adapter)
    for name,value in [('CV_OUTCOMES.json',cv),('FINAL_OUTCOMES.json',final),('SELECTIONS.json',selections),('CONTROLLER_LEDGER.json',state)]:atomic(ROOT/name,value,exclusive=True)
    atomic(ROOT/'EXECUTION_ENDED.json',dict(**identity,status=state['status'],
        safe_stop_confirmed=state['safe_stop_confirmed'],plan_complete=state['plan_complete'],
        scheduler_stop_reason=state['scheduler_stop_reason'],counts=adapter.counts,
        logical_counts=dict(CV=864,final=42,selections=24),end_utc=utc(),
        no_class_scoring=True,condition_commits=adapter.commits),exclusive=True)
    print(json.dumps(dict(status=state['status'],counts=adapter.counts,score_performed=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--execute-runtime-sha');args=parser.parse_args()
    if args.execute_runtime_sha:execute(args.execute_runtime_sha)
    else:print('READ_ONLY_PREPARATION: no scientific input opened, no MATLAB launch')
