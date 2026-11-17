"""Separate all-outcome seal and independently admitted score; never fits/selects anew."""
import argparse,datetime,sys
sys.dont_write_bytecode=True
import numpy as np
from scipy import io
from formal_common import *
import query_protocol as qp

def collect():
    assert not (ROOT/'CONTROLLER_FAILURE.json').exists(),'Unresolved controller failure cannot be sealed/scored'
    p=protocol()  # Recheck all 216 input identities at seal and scoring admission.
    finish=read(ROOT/'EXECUTION_FINISHED.json')
    assert finish['protocol_sha256']==sha(ROOT/'FROZEN_PROTOCOL.json') and finish['logical_outcomes']==42
    cases={c['case_id']:c for c in p['cases']}
    cv=read(ROOT/'CV_OUTCOMES.json');final=read(ROOT/'FINAL_OUTCOMES.json');selections=read(ROOT/'SELECTIONS.json')
    assert len(cv)==144 and len(final)==42 and len(selections)==24
    assert len({(r['condition_id'],r['candidate'],r['fold']) for r in cv})==144
    assert {r['condition_id'] for r in final}=={c['id'] for c in p['conditions']}
    assert len({r['condition_id'] for r in selections})==24
    members={'EXECUTION_STARTED.json','EXECUTION_FINISHED.json','CV_OUTCOMES.json','FINAL_OUTCOMES.json','SELECTIONS.json','FREEZE_RECEIPT.json','FROZEN_PROTOCOL.json','RUNTIME_PROTOCOL.json'}
    attempts=0
    for cond in p['conditions']:
        fr=next(r for r in final if r['condition_id']==cond['id'])
        assert (fr['dataset'],fr['seed'],fr['Q'])==(cond['dataset'],cond['seed'],cond['Q'])
        if cond['unsupported']:
            assert fr['status']=='INPUT_UNSUPPORTED' and not fr['attempted'] and not fr['eligible'] and fr['case_id'] is None
            assert not [r for r in cv if r['condition_id']==cond['id']]
            continue
        d=paid(cond['query_input'],cond['n']);pairs,answers=d['pairs'][:cond['Q']],d['answers'][:cond['Q']]
        folds=qp.eligible_folds(qp.node_folds(d['pool'],pairs,cond['seed']+41000),answers)
        assert read_json_value(folds)==cond['eligible_folds']
        scores=np.full((3,len(folds)),np.nan);local=[]
        for fi,fold in enumerate(folds):
            for ci in range(3):
                r=next(r for r in cv if (r['condition_id'],r['candidate'],r['fold'])==(cond['id'],ci,fi))
                case=cases[case_id(cond['dataset'],cond['seed'],cond['Q'],ci,fi)]
                assert r['case_id']==case['case_id'];local.append(r)
                if r['attempted']:
                    base=outcome(case)
                    assert all(r[k]==v for k,v in base.items())
                    attempts+=1;add_run_members(members,case)
                    if r['eligible']:
                        _,z,_=terminal(case);held=np.asarray(fold['validation'],int)
                        scores[ci,fi]=qp.balanced_pair_error(z,pairs[held],answers[held])
                else:
                    assert not r['eligible'] and r['status'].startswith('NOT_ATTEMPTED_AFTER_')
                    assert not (ROOT/'runs'/case['case_id']).exists()
                expected=float(scores[ci,fi]) if np.isfinite(scores[ci,fi]) else None
                assert r['validation_BER']==expected
        s=next(s for s in selections if s['condition_id']==cond['id'])
        selection_name=cond['id']+'_selection.json';members.add(selection_name)
        assert read(ROOT/selection_name)==s
        assert s['eligible_folds']==len(folds)
        assert s['scores']==[[float(v) if np.isfinite(v) else None for v in row] for row in scores]
        if all(r['attempted'] for r in local):
            index,rule=qp.select_configuration(scores)
            assert s['status']=='SELECTED' and s['index']==index and s['tau']==GRID[index] and s['rule']==rule
            assert s['valid_candidates']==int(np.sum(np.all(np.isfinite(scores),axis=1)))
            selected=cases[case_id(cond['dataset'],cond['seed'],cond['Q'],index,'final')]
            assert s['final_input_sha256']==selected['sha256'] and fr['case_id']==selected['case_id']
        else:
            assert s['status']=='INCOMPLETE_CV_NOT_SELECTED' and s['index'] is None and s['tau'] is None
            assert s['rule']=='not_selected_after_execution_stop'
            assert fr['case_id'] is None and not fr['attempted']
        if fr['attempted']:
            base=outcome(selected);assert all(fr[k]==v for k,v in base.items())
            attempts+=1;add_run_members(members,selected)
        else:assert not fr['eligible'] and fr['status'].startswith('NOT_ATTEMPTED_AFTER_')
    actual_receipts=list((ROOT/'runs').glob('*/repeat1/launcher_receipt.json')) if (ROOT/'runs').exists() else []
    assert len(actual_receipts)==attempts==finish['fit_calls']<=168
    expected_dirs={r['case_id'] for r in cv+final if r['attempted']}
    actual_dirs={d.name for d in (ROOT/'runs').iterdir() if d.is_dir()} if (ROOT/'runs').exists() else set()
    assert actual_dirs==expected_dirs,'Unresolved/unlisted partial or extra run directory'
    assert sum(r['attempted'] for r in cv)<=144 and sum(r['attempted'] for r in final)<=24
    assert not (ROOT/'UNRESOLVED_OWNED_PROCESSES.json').exists(),'Resolve owned-process evidence before sealing'
    return p,final,members
def read_json_value(v):return __import__('json').loads(__import__('json').dumps(v,default=serial))
def add_run_members(members,case):
    out=ROOT/'runs'/case['case_id']/'repeat1';r=read(out/'launcher_receipt.json')
    members.add(str((out/'launcher_receipt.json').relative_to(ROOT)))
    for name in r['output_hashes']:members.add(str((out/name).relative_to(ROOT)))
def seal():
    no_oracle_guard();p,rows,members=collect()
    s=dict(status='SEALED',protocol_sha256=sha(ROOT/'FROZEN_PROTOCOL.json'),runtime_sha256=sha(ROOT/'RUNTIME_PROTOCOL.json'),
        logical_outcomes=42,CV_outcomes=144,no_oracle_read=True,no_metrics_computed=True,
        members_sha256={name:sha(ROOT/name) for name in sorted(members)})
    immutable(ROOT/'GLOBAL_PREDICTIONS_SEALED.json',s)
    print('SEALED',sha(ROOT/'GLOBAL_PREDICTIONS_SEALED.json'),flush=True)
def score(expected,audit_path,audit_sha):
    assert sha(ROOT/'FROZEN_PROTOCOL.json')==expected
    s=read(ROOT/'GLOBAL_PREDICTIONS_SEALED.json')
    assert s['protocol_sha256']==expected
    for name,digest in s['members_sha256'].items():assert sha(ROOT/name)==digest,name
    admission(audit_path,audit_sha,'PRE_SCORE',expected,sha(ROOT/'GLOBAL_PREDICTIONS_SEALED.json'))
    p,rows,members=collect()  # No oracle has been decoded before every boundary passes.
    from metrics_reference import metrics
    cases={c['case_id']:c for c in p['cases']};evaluated=[]
    for ds,card in p['datasets'].items():
        # Unsupported corpora have no native outputs; no need to read their oracle.
        if ds not in FIT_DATA:
            evaluated.extend(dict(r,ACC=None,NMI=None,ARI=None,training_BER=None,unqueried_BER=None) for r in rows if r['dataset']==ds)
            continue
        ref=card['sealed_scoring_input'];assert sha(ref['path'])==ref['sha256']
        y=np.load(ref['path'],allow_pickle=False).ravel();assert len(y)==card['n'] and len(np.unique(y))==card['k']
        for seed in p['seed_map'][ds]:
            d=paid(card['query_inputs'][str(seed)],card['n']);test=d['test']
            ep=qp.sample_query_stream(test,min(10000,len(test)*(len(test)-1)//2),seed+51000);ea=qp.answer_queries(ep,y)
            for r in [r for r in rows if r['dataset']==ds and r['seed']==seed]:
                if r['attempted'] and r['finite_partition_retained']:
                    _,z,_=terminal(cases[r['case_id']]);values=metrics(y[test],z[test])
                    values.update(training_BER=qp.balanced_pair_error(z,d['pairs'][:r['Q']],d['answers'][:r['Q']]),unqueried_BER=qp.balanced_pair_error(z,ep,ea))
                else:values={name:None for name in ['ACC','NMI','ARI','training_BER','unqueried_BER']}
                evaluated.append(dict(r,**values,metric_role='primary_eligible' if r['eligible'] else 'descriptive_or_missing'))
    summary=[]
    for ds in p['datasets']:
        for Q in p['budgets']:
            group=[r for r in evaluated if r['dataset']==ds and r['Q']==Q]
            complete=len(group)==3 and all(r['eligible'] for r in group)
            sgroup=[s for s in read(ROOT/'SELECTIONS.json') if s['condition_id'] in {r['condition_id'] for r in group}]
            item=dict(dataset=ds,Q=Q,n_planned=3,n_eligible=sum(r['eligible'] for r in group),complete=complete,
                fallback_count=sum(s['rule'] in ['no_eligible_folds_default','all_validation_fits_failed_default_with_failure_flag'] for s in sgroup),
                all_CV_failed_default_count=sum(s['rule']=='all_validation_fits_failed_default_with_failure_flag' for s in sgroup),
                selection_denominator=len(sgroup),valid_candidates=[s.get('valid_candidates') for s in sgroup])
            if complete:
                for name in ['ACC','NMI','ARI','training_BER','unqueried_BER']:
                    vals=[r[name] for r in group];item[name]=dict(values=vals,mean=float(np.mean(vals)),sample_sd=float(np.std(vals,ddof=1)))
            summary.append(item)
    assert len(evaluated)==42
    immutable(ROOT/'evaluation.json',evaluated);immutable(ROOT/'summary.json',summary)
    immutable(ROOT/'EVALUATION_MANIFEST.json',dict(status='SCORED_PENDING_POST_SCORE_AUDIT',
        global_seal_sha256=sha(ROOT/'GLOBAL_PREDICTIONS_SEALED.json'),pre_score_audit_sha256=audit_sha,
        evaluation_sha256=sha(ROOT/'evaluation.json'),summary_sha256=sha(ROOT/'summary.json'),
        labels_sha256={ds:card['sealed_scoring_input']['sha256'] for ds,card in p['datasets'].items() if ds in FIT_DATA}))
def main():
    a=argparse.ArgumentParser();a.add_argument('--seal',action='store_true');a.add_argument('--execute-score-sha');a.add_argument('--pre-score-audit');a.add_argument('--audit-sha');args=a.parse_args()
    if args.seal:assert not args.execute_score_sha;seal()
    elif args.execute_score_sha:assert args.pre_score_audit and args.audit_sha;score(args.execute_score_sha,Path(args.pre_score_audit),args.audit_sha)
    else:print('READ_ONLY: no seal or scoring requested')
if __name__=='__main__':main()
