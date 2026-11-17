"""Three representative in-memory cases; no actual export, input or score reads."""
from copy import deepcopy
import unittest
import replay_ssa_compact as replay

np=replay.np


def fixture():
    cid='synthetic_s23_Q4'
    labels=np.array([1,1,2,2,1,1,2,2])
    data=dict(pairs=np.array([[0,1],[0,2],[4,5],[4,6]]),answers=np.array([1,-1,1,-1]))
    meta=dict(Q=4,eligible_folds=[dict(validation=[0,1]),dict(validation=[2,3])])
    rows=[];partitions={}
    for fold in range(2):
        for candidate in range(18):
            name=f'{cid}_c{candidate}_f{fold}'
            rows.append(dict(case_id=name,candidate=candidate,fold=fold,status='ELIGIBLE',attempted=True,
                trigger_case=None,result=dict(case_id=name,status='ELIGIBLE',attempted=True,execution_complete=True,
                    cleanup_complete=True,eligible=True,irreversible_local_failure=False,validation_BER=0.0)))
            partitions[name]=labels
    final_name=cid+'_c0_final';partitions[final_name]=labels
    condition=dict(condition_id=cid,corpus='synthetic',Q=4,seed=23,committed=True,cv=rows,
        selection=replay.L.selected(rows),final=dict(case_id=final_name,status='ELIGIBLE',attempted=True))
    return condition,meta,data,partitions


class ReplayCases(unittest.TestCase):
    def test_complete_condition_paid_BER_default_tie_and_uncommitted_diagnostic(self):
        condition,meta,data,partitions=fixture()
        result=replay.replay_condition_budget(condition,meta,data,partitions)
        self.assertEqual(result['selected_index'],0)
        self.assertEqual(result['training_BER'],0)
        self.assertTrue(result['primary_eligible'])
        condition['committed']=False
        result=replay.replay_condition_budget(condition,meta,data,partitions)
        self.assertFalse(result['primary_eligible'])
        self.assertEqual(result['training_BER_role'],'diagnostic_ineligible_finite')
        partitions[condition['cv'][18]['case_id']]=np.ones(8,dtype=int)
        with self.assertRaisesRegex(ValueError,'Paid validation BER differs'):
            replay.replay_condition_budget(condition,meta,data,partitions)

    def test_local_pruning_keeps_denominator_and_all_invalid_fallback(self):
        condition,meta,data,partitions=fixture()
        def prune(candidate):
            first,later=condition['cv'][candidate],condition['cv'][18+candidate]
            first.update(status='LOCAL_INELIGIBLE')
            first['result'].update(status='LOCAL_INELIGIBLE',eligible=False,irreversible_local_failure=True,validation_BER=None)
            later.update(status='SKIPPED_AFTER_LOCAL_INELIGIBLE',attempted=False,result=None,trigger_case=first['case_id'])
            partitions.pop(first['case_id']);partitions.pop(later['case_id'])
        prune(3);condition['selection']=replay.L.selected(condition['cv'])
        self.assertEqual(condition['selection']['valid_candidates'],17)
        self.assertEqual(len(condition['cv']),36)
        self.assertTrue(replay.replay_condition_budget(condition,meta,data,partitions)['primary_eligible'])
        for candidate in range(18):
            if candidate!=3:prune(candidate)
        condition['selection']=replay.L.selected(condition['cv'])
        result=replay.replay_condition_budget(condition,meta,data,partitions)
        self.assertEqual(result['selected_index'],0)
        self.assertEqual(result['selection_rule'],'all_validation_fits_failed_default_with_failure_flag')
        tampered=deepcopy(condition['cv']);tampered[18]['trigger_case']='another_candidate'
        with self.assertRaises(ValueError):replay.L.selected(tampered)

    def test_persistent_resource_global_stop_stays_unselected(self):
        condition,meta,data,partitions=fixture();condition['committed']=False
        policy=replay.POLICY.ReceiptPolicy()
        decisions=[]
        for ordinal in range(2):
            row=condition['cv'][ordinal]
            receipt=dict(case_id=row['case_id'],status='RESOURCE_START_WAIT_CAP',attempt_lifecycle_ended=True,
                matlab_started=False,process_creation_attempted=False,cleanup=dict(complete=True,no_process_created=True))
            decisions.append(policy.assess(receipt,False))
            row['status']=decisions[-1]['disposition']
            row['result'].update(status=row['status'],eligible=False,irreversible_local_failure=ordinal==0,validation_BER=None)
        self.assertEqual(decisions[1]['reason'],'RESOURCE_PERSISTENCE_STOP')
        for row in condition['cv'][2:]:row.update(status='NOT_ATTEMPTED_AFTER_GLOBAL_STOP',attempted=False,result=None)
        condition['selection']=dict(status='NOT_SELECTED_AFTER_GLOBAL_STOP',index=None,rule=None)
        condition['final']=dict(case_id=None,status='NOT_ATTEMPTED_AFTER_GLOBAL_STOP',attempted=False)
        result=replay.replay_condition_budget(condition,meta,data,{})
        self.assertIsNone(result['selected_index']);self.assertFalse(result['primary_eligible'])
        self.assertIsNone(result['training_BER'])
        with self.assertRaises(ValueError):replay.L.selected(condition['cv'])


if __name__=='__main__':unittest.main()
