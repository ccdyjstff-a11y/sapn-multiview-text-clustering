"""Standard-library, data-free mocks. No MATLAB, process control or feature I/O."""
import unittest
import json
import io
from types import SimpleNamespace
from unittest.mock import patch,Mock
from adapter_policy import ReceiptPolicy,memory_gate
from controller_core import Group,FitResult,FitRequest,run
from prepare_grouped import group_spec,CORPORA
import run_grouped as native
from successor_gate import validate_result
import successor_gate as closure_gate


def receipt(status='COMPLETE_ZERO_EXIT_VALID_MARKERS',case='synthetic',started=True):
    return dict(case_id=case,status=status,attempt_lifecycle_ended=True,matlab_started=started,
        process_creation_attempted=started,exit_code=0,cleanup=dict(complete=True,no_process_created=not started),
        worker=dict(status='RETURNED_FINITE',execution_returned=True),clock=dict(status=status),is_cv=True)


class Clock:
    def __init__(self):self.t=0
    def now(self):return self.t
    def sleep(self,seconds):self.t+=seconds


class PolicyTests(unittest.TestCase):
    def _closure_source_probe(self,change_at=None,missing_declared_dependency=False):
        hashes=dict(closure_gate.REVIEWED_SOURCE_SHA256)
        declared=dict(hashes)
        if missing_declared_dependency:declared.pop('retirement_evidence.py')
        raw=json.dumps(dict(verifier_source_sha256=declared)).encode()
        spec=dict(verifier_path=str(closure_gate.REVIEWED_DIRECTORY/'closure_verifier.py'),
            verifier_sha256=hashes['closure_verifier.py'],verifier_source_sha256=hashes,
            closure_path=str(closure_gate.REVIEWED_DIRECTORY/'synthetic_no_file.json'),
            closure_sha256=native.hashlib.sha256(raw).hexdigest())
        counts={name:0 for name in hashes}
        def digest(path):
            name=native.Path(path).name
            if name not in hashes:return spec['closure_sha256']
            counts[name]+=1
            if name=='retirement_evidence.py' and counts[name]==change_at:return 'changed'
            return hashes[name]
        function=Mock(return_value=dict(passed=True,fresh_process_observation=True,
            closure_kind='OPERATIONAL_ABORT',prior_protocol_sha256='prior',
            live_previous_batch_processes=0,unverified_previous_batch_processes=0))
        loader=Mock()
        module=SimpleNamespace(verify_closed_for_successor=function)
        with patch.object(closure_gate,'sha',side_effect=digest),\
                patch.object(closure_gate.Path,'read_bytes',return_value=raw),\
                patch.object(closure_gate.importlib.util,'spec_from_file_location',return_value=SimpleNamespace(loader=loader)),\
                patch.object(closure_gate.importlib.util,'module_from_spec',return_value=module):
            if change_at is not None or missing_declared_dependency:
                with self.assertRaises(AssertionError):closure_gate.verify('synthetic_s1','prior',spec)
            else:
                self.assertTrue(closure_gate.verify('synthetic_s1','prior',spec)['passed'])
        return counts,loader,function

    def test_both_closure_sources_checked_before_and_after_call(self):
        counts,loader,function=self._closure_source_probe()
        self.assertEqual(counts,{'closure_verifier.py':2,'retirement_evidence.py':2})
        loader.exec_module.assert_called_once();function.assert_called_once()
        self.assertEqual(set(closure_gate.source_bindings().values()),set(closure_gate.REVIEWED_SOURCE_SHA256.values()))

    def test_changed_closure_dependency_rejected_before_import(self):
        _,loader,function=self._closure_source_probe(change_at=1)
        loader.exec_module.assert_not_called();function.assert_not_called()

    def test_changed_closure_dependency_rejected_after_call(self):
        _,loader,function=self._closure_source_probe(change_at=2)
        loader.exec_module.assert_called_once();function.assert_called_once()

    def test_closure_record_cannot_omit_dependency_binding(self):
        _,loader,function=self._closure_source_probe(missing_declared_dependency=True)
        loader.exec_module.assert_not_called();function.assert_not_called()

    def test_successor_gate_rejects_pid_only_observation_and_unverified_ownership(self):
        good=dict(passed=True,fresh_process_observation=True,closure_kind='OPERATIONAL_ABORT',
            prior_protocol_sha256='p',live_previous_batch_processes=0,unverified_previous_batch_processes=0)
        self.assertEqual(validate_result(good,'p'),good)
        for change in [dict(fresh_process_observation=False),dict(live_previous_batch_processes=1),
                       dict(unverified_previous_batch_processes=1),dict(closure_kind='OBSERVATION_ONLY'),
                       dict(prior_protocol_sha256='other'),dict(live_previous_batch_processes=False)]:
            with self.subTest(change=change):
                with self.assertRaises(AssertionError):validate_result(dict(good,**change),'p')

    def test_explicit_full_group_order_and_denominators(self):
        groups=group_spec({name:{'n':100} for name in CORPORA})
        self.assertEqual([(g['corpus'],g['Q']) for g in groups],[(ds,q) for ds in CORPORA for q in [400,160]])
        self.assertTrue(all(g['seeds']==[23,24,25] for g in groups))
        self.assertEqual(len(groups)*3*18*2,864)
        self.assertEqual(len(groups)*3+18,42)

    def test_numeric_return_failure_is_local_but_exception_is_global(self):
        item=receipt();item['worker']['status']='RETURNED_NONFINITE_OR_INVALID'
        self.assertEqual(ReceiptPolicy().assess(item,False)['disposition'],'LOCAL_INELIGIBLE')
        item['worker']['status']='EXCEPTION'
        self.assertEqual(ReceiptPolicy().assess(item,False)['disposition'],'GLOBAL_STOP')

    def test_active_cap_requires_terminal_cleanup(self):
        item=receipt('WORKER_ACTIVE_TIME_CAP')
        self.assertEqual(ReceiptPolicy().assess(item,False)['disposition'],'LOCAL_INELIGIBLE')
        item['cleanup']['complete']=False
        self.assertEqual(ReceiptPolicy().assess(item,False)['reason'],'OWNED_CLEANUP_INCOMPLETE')

    def test_start_wait_requires_explicit_no_creation_proof(self):
        item=receipt('RESOURCE_START_WAIT_CAP',started=False)
        decision=ReceiptPolicy().assess(item,False)
        self.assertEqual(decision['disposition'],'LOCAL_INELIGIBLE')
        self.assertFalse(decision['matlab_started'])
        self.assertTrue(decision['terminal_attempt_lifecycle'])
        del item['cleanup']
        self.assertEqual(ReceiptPolicy().assess(item,False)['disposition'],'GLOBAL_STOP')
        item=receipt('RESOURCE_START_WAIT_CAP',started=False)
        item['process_creation_attempted']=True
        self.assertEqual(ReceiptPolicy().assess(item,False)['reason'],'ATTEMPT_LIFECYCLE_UNVERIFIED')

    def test_cap_phase_must_match_started_process_and_clock(self):
        self.assertEqual(ReceiptPolicy().assess(receipt('WORKER_ACTIVE_TIME_CAP',started=False),False)['reason'],'CAP_LIFECYCLE_INCONSISTENT')
        item=receipt('RESOURCE_SUSPEND_WAIT_CAP');item['clock']['status']='RUNNING'
        self.assertEqual(ReceiptPolicy().assess(item,False)['reason'],'CAP_LIFECYCLE_INCONSISTENT')
        self.assertEqual(ReceiptPolicy().assess(receipt('RESOURCE_START_WAIT_CAP',started=True),False)['reason'],'CAP_LIFECYCLE_INCONSISTENT')

    def test_consecutive_resource_caps_and_actual_noncap_reset(self):
        policy=ReceiptPolicy()
        first=policy.assess(receipt('RESOURCE_START_WAIT_CAP','c0',False),False)
        self.assertEqual(first['disposition'],'LOCAL_INELIGIBLE')
        # A short-circuited slot makes no assess call and cannot reset the streak.
        second=policy.assess(receipt('RESOURCE_SUSPEND_WAIT_CAP','c1'),False)
        self.assertEqual(second['reason'],'RESOURCE_PERSISTENCE_STOP')
        self.assertEqual(second['previous_resource_cap_case'],'c0')
        policy.assess(receipt(case='c2'),True,.2)
        third=policy.assess(receipt('RESOURCE_SUSPEND_WAIT_CAP','c3'),False)
        self.assertEqual(third['consecutive_resource_caps'],1)
        self.assertEqual(third['disposition'],'LOCAL_INELIGIBLE')

    def test_deadline_startup_and_unknown_states_never_prune(self):
        for status in ['STARTUP_WALL_TIME_CAP','GLOBAL_EXECUTION_WALL_CAP','OWNED_RESUME_FAILED','MISSING_WORKER_SUMMARY_AFTER_RETURN','UNRECOGNIZED']:
            with self.subTest(status=status):
                self.assertEqual(ReceiptPolicy().assess(receipt(status),False)['disposition'],'GLOBAL_STOP')
        item=receipt('WORKER_ACTIVE_TIME_CAP');item['global_deadline_triggered']=True
        self.assertEqual(ReceiptPolicy().assess(item,False)['reason'],'GLOBAL_EXECUTION_WALL_CAP')

    def test_large_group_threshold_and_stability_reset(self):
        small=Clock()
        self.assertEqual(memory_gate(1999,lambda:3*2**30,small.now,small.sleep,10000)['elapsed_seconds'],60)
        clock=Clock()
        result=memory_gate(2000,lambda:3*2**30,clock.now,clock.sleep,10000)
        self.assertEqual(result['status'],'GROUP_MEMORY_WAIT_CAP')
        self.assertEqual(result['elapsed_seconds'],1800)
        clock=Clock()
        result=memory_gate(2000,lambda:(2 if clock.t==30 else 5)*2**30,clock.now,clock.sleep,10000)
        self.assertEqual(result['status'],'READY');self.assertEqual(result['elapsed_seconds'],90.5)

    def test_group_gate_honors_global_deadline(self):
        clock=Clock();result=memory_gate(100,lambda:5*2**30,clock.now,clock.sleep,20)
        self.assertEqual(result['status'],'GLOBAL_EXECUTION_WALL_CAP')
        self.assertEqual(result['elapsed_seconds'],20)
        self.assertFalse(result['native_fit_started'])

    def test_final_second_resource_cap_retains_prior_selection_without_commit(self):
        policy=ReceiptPolicy();calls=[]
        def runner(request):
            calls.append(request)
            cap=request.fold=='final' or (request.candidate,request.fold)==(17,1)
            item=receipt('RESOURCE_SUSPEND_WAIT_CAP' if cap else 'COMPLETE_ZERO_EXIT_VALID_MARKERS',request.case_id)
            item['is_cv']=request.fold!='final'
            decision=policy.assess(item,not cap,.2 if not cap else None)
            return FitResult(request.case_id,decision['disposition'],True,True,True,
                decision['disposition']=='ELIGIBLE',.2 if not cap else None,
                irreversible_local_failure=decision['disposition']=='LOCAL_INELIGIBLE',reason=decision['reason'])
        def never_commit(condition):self.fail('Final resource stop is not a complete condition')
        result=run([Group('synthetic',400,(23,24,25))],runner,never_commit)
        first=result['conditions'][0]
        self.assertEqual(len(calls),37)
        self.assertEqual(first['selection']['status'],'SELECTED')
        self.assertEqual(first['final']['result']['reason'],'RESOURCE_PERSISTENCE_STOP')
        self.assertFalse(first['committed']);self.assertFalse(result['safe_stop_confirmed'])

    def test_stop_request_is_bound_to_exact_identity(self):
        identity=dict(run_id='new',protocol_sha256='p',runtime_sha256='r')
        request=dict(**identity,schema=1,request='STOP_AFTER_CONDITION')
        with patch.object(native.Path,'exists',return_value=True),patch.object(native.Path,'read_bytes',side_effect=lambda:json.dumps(request).encode()):
            self.assertTrue(native.stop_request(identity))
            request['run_id']='old'
            with self.assertRaises(ValueError):native.stop_request(identity)

    def test_condition_commit_does_not_acknowledge_failed_write(self):
        adapter=native.NativeAdapter(dict(cases=[],conditions=[]),{},float('inf'),None,None,None,None)
        with patch.object(native,'atomic',side_effect=OSError('mock write failure')):
            with self.assertRaises(OSError):adapter.commit(dict(condition_id='synthetic'))
        self.assertEqual(adapter.commits,{})

    def test_stop_request_rejects_boolean_schema(self):
        identity=dict(run_id='new',protocol_sha256='p',runtime_sha256='r')
        request=dict(**identity,schema=True,request='STOP_AFTER_CONDITION')
        with patch.object(native.Path,'exists',return_value=True),patch.object(native.Path,'read_bytes',return_value=json.dumps(request).encode()):
            with self.assertRaisesRegex(ValueError,'Invalid stop request schema'):
                native.stop_request(identity)

    def test_deadline_before_or_during_commit_cannot_acknowledge(self):
        for times in [[101],[99,101]]:
            with self.subTest(times=times):
                adapter=native.NativeAdapter(dict(cases=[],conditions=[]),{},100,None,None,None,None)
                with patch.object(native.time,'monotonic',side_effect=times),patch.object(native,'atomic') as writer,patch.object(native,'sha',return_value='hash'):
                    with self.assertRaises(RuntimeError):adapter.commit(dict(condition_id='synthetic'))
                    self.assertEqual(writer.call_count,0 if len(times)==1 else 1)
                self.assertEqual(adapter.commits,{})
                self.assertEqual(adapter.stop_reason,'GLOBAL_EXECUTION_WALL_CAP')

    def test_exclusive_install_collision_never_falls_back_to_replace(self):
        class MemoryFile(io.StringIO):
            def fileno(self):return 123
        with patch.object(native.Path,'mkdir'),patch.object(native.Path,'exists',return_value=False),\
                patch.object(native.Path,'open',return_value=MemoryFile()),patch.object(native.os,'fsync'),\
                patch.object(native.os,'link',side_effect=FileExistsError('synthetic collision')),\
                patch.object(native.os,'replace') as replace,patch.object(native.Path,'unlink') as unlink:
            with self.assertRaises(FileExistsError):native.atomic(native.ROOT/'synthetic.json',{'x':1},exclusive=True)
            replace.assert_not_called();unlink.assert_not_called()

    def test_failed_group_guard_creates_no_launcher_receipt_or_dispatch(self):
        p=dict(cases=[],conditions=[dict(id='synthetic',n=2225)])
        adapter=native.NativeAdapter(p,{},float('inf'),None,None,None,SimpleNamespace())
        request=FitRequest('synthetic','case','synthetic',400,23,0,0)
        guard=dict(status='GROUP_MEMORY_WAIT_CAP',native_fit_started=False,launcher_dispatched=False)
        with patch.object(native,'memory_gate',return_value=guard),patch.object(native,'atomic') as writer,patch.object(native,'sha',return_value='digest'):
            result=adapter(request)
            self.assertEqual(writer.call_count,1)
            self.assertEqual(writer.call_args.args[0].parent.name,'guards')
        self.assertFalse(result.attempted)
        self.assertEqual(adapter.counts['launcher_dispatches'],0)
        self.assertEqual(adapter.counts['matlab_started'],0)

    def test_postdecision_deadline_does_not_change_original_receipt_or_select(self):
        class PaidArray:
            def __getitem__(self,index):return self
        case=dict(case_id='case',candidate=0,fold=0,sha256='input')
        condition=dict(id='synthetic',n=100,Q=400,query_input={},eligible_folds=[dict(validation=[0])])
        item=receipt(case='case')
        fc=SimpleNamespace(terminal=lambda c:(item,[1,2],True),
            paid=lambda card,n:dict(pairs=PaidArray(),answers=PaidArray()))
        rt=SimpleNamespace(run_one=lambda p,c,repetition:item)
        qp=SimpleNamespace(balanced_pair_error=lambda *a:.2)
        adapter=native.NativeAdapter(dict(cases=[case],conditions=[condition]),{},100,fc,rt,qp,None)
        adapter.checked_groups.add(('synthetic',400))
        with patch.object(native.time,'monotonic',side_effect=[0,101]),patch.object(native,'atomic'),patch.object(native,'sha',return_value='digest'):
            result=adapter(FitRequest('synthetic','case','synthetic',400,23,0,0))
        self.assertEqual(result.status,'GLOBAL_STOP')
        self.assertEqual(result.reason,'GLOBAL_EXECUTION_WALL_CAP')
        self.assertEqual(item['status'],'COMPLETE_ZERO_EXIT_VALID_MARKERS')
        self.assertEqual(adapter.decisions['case']['disposition'],'ELIGIBLE')
        self.assertEqual(result.evidence_ref['kind'],'POST_DECISION_DEADLINE')


if __name__=='__main__':unittest.main()
