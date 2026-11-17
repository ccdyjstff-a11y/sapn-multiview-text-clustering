"""Pure/mocked tests only. No MATLAB, dataset, oracle or real child process."""
from pathlib import Path
import contextlib,io,json,sys,tempfile,types,unittest
from unittest.mock import patch
sys.dont_write_bytecode=True
import numpy as np
import run_clock_qualification as rt
import query_protocol as qp
import run_formal as dispatch
from run_formal import must_stop

HERE=Path(__file__).resolve().parent
class FormalTests(unittest.TestCase):
    def test_original_hard_BER(self):
        z=np.array([0,0,1,1]);pairs=np.array([[0,1],[0,2],[2,3],[1,3]])
        self.assertEqual(qp.balanced_pair_error(z,pairs,np.array([1,-1,-1,1])),.5)
    def test_original_tie_invalid_and_fallback(self):
        self.assertEqual(qp.select_configuration([[.1,.2],[.1,.2],[.2,.2]])[0],0)
        self.assertEqual(qp.select_configuration([[np.nan,.1],[.1,.2],[.1,.2]])[0],1)
        self.assertIn('all_validation_fits_failed',qp.select_configuration([[np.nan,0],[0,np.nan],[np.nan,np.nan]])[1])
        self.assertIn('no_eligible_folds',qp.select_configuration(np.empty((3,0)))[1])
    def test_stop_propagation(self):
        for r in [dict(status='GLOBAL_EXECUTION_WALL_CAP',global_deadline_triggered=True),
                  dict(status='OWNED_CLEANUP_INCOMPLETE'),dict(status='LAUNCHER_EXCEPTION'),dict(status='X',interrupted=True)]:
            self.assertTrue(must_stop(r))
        self.assertFalse(must_stop(dict(status='WORKER_ACTIVE_TIME_CAP')))
    def test_shared_monotonic_deadline_not_extended(self):
        with patch.object(rt,'GLOBAL_DEADLINE',10),patch.object(rt.time,'monotonic',return_value=11):
            for stage in ['start_memory_wait','running_or_suspended_poll','before_resume_action','after_resume_action','after_exit_and_marker_observation']:
                record={}
                with self.assertRaises(rt.GlobalDeadline):rt.check_deadline(record,stage)
                self.assertTrue(record['global_deadline_triggered'])
                self.assertEqual(record['global_deadline_stage'],stage)
    def mock_case(self,mode):
        assert 'Documents/Codex' in HERE.as_posix()
        with tempfile.TemporaryDirectory(dir=HERE,prefix='.mock_') as directory:
            root=Path(directory);(root/'FROZEN_PROTOCOL.json').write_text('{}')
            case=dict(case_id='mock',input='unopened.mat',sha256='mock_input',timeout_active_seconds=600)
            p=dict(author_root='unopened_author',matlab_executable='NEVER_EXECUTE',runtime_environment={},
                runtime=dict(memory_wait_limit_seconds=1800,poll_seconds=.5))
            ticks=[0.];verifies=[0];launches=[0]
            def verify():
                verifies[0]+=1
                if mode=='pre_Popen' and verifies[0]==2:ticks[0]=2
            fake=types.SimpleNamespace(pid=999,create_time=lambda:1.,poll=lambda:0,wait=lambda timeout:0)
            def popen(*args,**kwargs):
                launches[0]+=1;ticks[0]=2 if mode in ['after_Popen','cleanup_failure'] else ticks[0]
                return fake
            clean=dict(complete=mode!='cleanup_failure',actions=[],unresolved_original_processes=[])
            memory=lambda:types.SimpleNamespace(available=0 if mode=='memory_wait' else 5*2**30)
            with patch.object(rt,'ROOT',root),patch.object(rt,'CURRENT_CASE',case),patch.object(rt,'GLOBAL_DEADLINE',1),patch.object(rt,'RUNTIME_SHA','mock_runtime'),patch.object(rt,'verify_runtime',side_effect=verify),patch.object(rt.psutil,'virtual_memory',side_effect=memory),patch.object(rt.psutil,'Popen',side_effect=popen),patch.object(rt.time,'monotonic',side_effect=lambda:ticks[0]),patch.object(rt.time,'sleep',side_effect=lambda s:ticks.__setitem__(0,ticks[0]+s)),patch.object(rt,'cleanup',return_value=clean),contextlib.redirect_stdout(io.StringIO()):
                r=rt.run_one(p,case,1)
            self.assertTrue(r['global_deadline_triggered'])
            self.assertNotIn('worker',r)
            self.assertEqual(r['status'],'OWNED_CLEANUP_INCOMPLETE' if mode=='cleanup_failure' else 'GLOBAL_EXECUTION_WALL_CAP')
            self.assertEqual(launches[0],0 if mode in ['memory_wait','pre_Popen'] else 1)
            if mode=='cleanup_failure':self.assertTrue((root/'UNRESOLVED_OWNED_PROCESSES.json').exists())
    def test_memory_wait_hits_global_without_process(self):self.mock_case('memory_wait')
    def test_source_recheck_crosses_deadline_before_Popen(self):self.mock_case('pre_Popen')
    def test_Popen_cold_start_crosses_global(self):self.mock_case('after_Popen')
    def test_global_cleanup_failure_preserves_stop(self):self.mock_case('cleanup_failure')
    def test_current_case_must_match_verified_case(self):
        with patch.object(rt,'CURRENT_CASE',dict(case_id='other')):
            with self.assertRaises(AssertionError):rt.run_one({},dict(case_id='wrong'),1)

    def test_controller_full_or_global_stop_keeps_144_and_42(self):
        for stop in [False,True]:
            with self.subTest(stop=stop), tempfile.TemporaryDirectory(dir=HERE,prefix='.mock_') as directory:
                root=Path(directory);conditions=[];cases=[]
                for ds in dispatch.FIT_DATA+['Reuters','20NG','AGNews']:
                    for seed in [23,24,25]:
                        for Q in [160,400]:
                            cond=dict(id=dispatch.condition_id(ds,seed,Q),dataset=ds,seed=seed,Q=Q,n=4,
                                unsupported=ds not in dispatch.FIT_DATA,unsupported_reason='test',query_input={},
                                eligible_folds=[dict(validation=[0,1]),dict(validation=[0,1])])
                            conditions.append(cond)
                            if not cond['unsupported']:
                                for ci in range(3):
                                    for fold in [0,1,'final']:
                                        cases.append(dict(case_id=dispatch.case_id(ds,seed,Q,ci,fold),sha256='test'))
                p=dict(conditions=conditions,cases=cases,global_wall_seconds=172800)
                (root/'FREEZE_RECEIPT.json').write_text(json.dumps(dict(runtime_sha256='test',protocol_sha256='test',pre_freeze_audit='unused',pre_freeze_audit_sha256='test')))
                (root/'RUNTIME_PROTOCOL.json').write_text('{}')
                called=[]
                def run_one(p,case,repeat):
                    called.append(case['case_id'])
                    return dict(status='GLOBAL_EXECUTION_WALL_CAP' if stop else 'COMPLETE_ZERO_EXIT_VALID_MARKERS',global_deadline_triggered=stop,matlab_started=False)
                def outcome(case):return dict(case_id=case['case_id'],attempted=True,eligible=False,finite_partition_retained=False,status='mock')
                with patch.object(dispatch,'ROOT',root),patch.object(dispatch,'sha',return_value='test'),patch.object(dispatch,'admission'),patch.object(dispatch,'protocol',return_value=p),patch.object(dispatch,'no_oracle_guard'),patch.object(dispatch,'paid',return_value=dict(pairs=np.array([[0,1],[0,2]]),answers=np.array([1,-1]))),patch.object(dispatch,'outcome',side_effect=outcome),patch.object(rt,'run_one',side_effect=run_one),patch.object(rt,'environment_gate',return_value={'passed':True}),contextlib.redirect_stdout(io.StringIO()):
                    dispatch.run('test')
                cv=json.loads((root/'CV_OUTCOMES.json').read_text());final=json.loads((root/'FINAL_OUTCOMES.json').read_text())
                self.assertEqual(len(cv),144);self.assertEqual(len(final),42)
                self.assertEqual(sum(r['status']=='INPUT_UNSUPPORTED' for r in final),18)
                self.assertEqual(len(called),1 if stop else 168)
                self.assertEqual(len(called),len(set(called)))
                self.assertEqual(sum(r['attempted'] for r in final),0 if stop else 24)

if __name__=='__main__':unittest.main(verbosity=2)
