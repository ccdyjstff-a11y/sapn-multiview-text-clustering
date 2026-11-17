"""Prepare finite inputs, then freeze only after independent input/source audit."""
from pathlib import Path
import argparse,datetime,gc,json,shutil,sys,zipfile
sys.dont_write_bytecode=True
import numpy as np
import psutil
from scipy import io,sparse
from formal_common import *

CAT=MAIN/'extension_round_v4/baselines/mvchss_formal_v1/GLOBAL_PROTOCOL.json'
V5=GAP/'native_qualification_clock_v5'
ENV=GAP/'environment_context_check_20261024/normal_exit_v3'
CONFIG=dict(anchor_rate=.7,gama=.1,Iter=30,style=0,IterMax=20,islocal=1,Organize_CL=1,w=1,Theorem2=1,Theorem3=1,Isbreak_Loss=1,Convergence=1e-4)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def preflight():
    assert sha(GAP/'FINITE_NATIVE_COMPARISON_DECISION.json')==DECISION_SHA
    cat=read(CAT)['specification'];sizes={}
    for ds in FIT_DATA:
        sizes[ds]=sum(sum(m.file_size for m in zipfile.ZipFile(path).infolist()) for path in cat['datasets'][ds]['feature_files'])
    disk_bound=54*sum(sizes.values())+2**30
    peak_bound=max(sizes.values())*12+128*2**20
    return dict(status='PREPARATION_ONLY_ESTIMATE',utc=now(),source_npz_uncompressed_bytes=sizes,
        estimated_disk_upper_bytes=disk_bound,estimated_prepare_peak_bytes=peak_bound,
        available_bytes=psutil.virtual_memory().available,free_disk_bytes=shutil.disk_usage(ROOT).free,
        minimum_available_bytes=max(3*2**30,2*2**30+peak_bound),
        minimum_free_disk_bytes=2*disk_bound+2*2**30,
        caveat='Sparse input serialization only; conservative components, not a measured formal MATLAB peak.')
def prepare():
    no_oracle_guard()
    assert not (ROOT/'PROTOCOL_DRAFT.json').exists() and not (ROOT/'inputs').exists(),'Preserve prior prepared inputs'
    pf=preflight();dump(ROOT/'PREPARATION_RESOURCE_PREFLIGHT.json',pf)
    assert pf['available_bytes']>=pf['minimum_available_bytes'] and pf['free_disk_bytes']>=pf['minimum_free_disk_bytes']
    from derive_runtime import derive
    derive()
    import query_protocol as qp
    cat=read(CAT)['specification'];v5=read(V5/'RUNTIME_PROTOCOL.json');science=read(V5/'FROZEN_PROTOCOL.json')
    gate_path=GAP.parent/'content_gate/PCBG_CLOCK_V5_TERMINAL_AUDIT.json'
    assert sha(gate_path)=='5efb5fae37f5827ee4c1d78d493fb1e4d27c1793c7c46ce5162581c9f6da018a'
    envreceipt=read(ENV/'attempt/CHECK_RECEIPT.json')
    assert envreceipt['status']=='PASS' and envreceipt['exit_code']==0 and envreceipt['owned_cleanup']['complete']
    (ROOT/'inputs').mkdir()
    conditions=[];cases=[];feature_signatures={}
    # Unsupported logical records keep original seed identities but never load their views.
    for ds,card in cat['datasets'].items():
        for seed in cat['seeds'][ds]:
            d=paid(card['query_inputs'][str(seed)],card['n'])
            for Q in [160,400]:
                pairs,answers=d['pairs'][:Q],d['answers'][:Q]
                allfolds=qp.node_folds(d['pool'],pairs,seed+41000)
                folds=qp.eligible_folds(allfolds,answers)
                if ds in FIT_DATA:assert len(folds)==2
                conditions.append(dict(id=condition_id(ds,seed,Q),dataset=ds,seed=seed,Q=Q,
                    logical_seed=seed,fit_seed=seed,fold_partition_seed=seed+41000,evaluation_pair_seed=seed+51000,
                    unsupported=ds not in FIT_DATA,unsupported_reason=None if ds in FIT_DATA else 'native_zero_row_normalization_unsupported',
                    query_input=card['query_inputs'][str(seed)],all_folds=allfolds,eligible_folds=folds,
                    n=card['n'],c=card['k']))
    for ds in FIT_DATA:
        assert psutil.virtual_memory().available>=pf['minimum_available_bytes'],'Preparation memory gate'
        card=cat['datasets'][ds];xs=[]
        for path,digest in card['feature_files'].items():
            assert sha(path)==digest
            x=sparse.load_npz(path).tocsc();assert x.shape[0]==card['n']
            assert np.isfinite(x.data).all() and np.all(x.data>=0)
            assert np.all(np.asarray(x.multiply(x).sum(axis=1)).ravel()>0)
            xs.append(x)
        assert len(xs)==card['views']
        cell=np.empty((len(xs),1),object)
        for i,x in enumerate(xs):cell[i,0]=x
        feature_signatures[ds]=dict(shapes=[list(x.shape) for x in xs],nnz=[x.nnz for x in xs])
        for cond in [c for c in conditions if c['dataset']==ds]:
            d=paid(cond['query_input'],card['n']);Q=cond['Q'];pairs,answers=d['pairs'][:Q],d['answers'][:Q]
            for ci,tau in enumerate(GRID):
                m=int(card['n']*tau);k=max(3,min(10,int(tau*.5*card['n']/card['k'])))
                assert m>=k+1 and m>=card['k']+1
                for fold in [0,1,'final']:
                    indices=None if fold=='final' else np.asarray(cond['eligible_folds'][fold]['training'],int)
                    P,N=qp.signed_pairs(pairs,answers,indices)
                    case=case_id(ds,cond['seed'],Q,ci,fold);path=ROOT/'inputs'/(case+'.mat')
                    config=dict(CONFIG,anchor_rate=tau)
                    io.savemat(path,dict(X=cell,c=float(card['k']),ML=(P+1).astype(float),CL=(N+1).astype(float),
                        seed=float(cond['seed']),config={key:float(v) for key,v in config.items()},case_id=case),do_compression=True)
                    assert {name for name,_,_ in io.whosmat(path)}=={'X','c','ML','CL','seed','config','case_id'}
                    cases.append(dict(case_id=case,input=str(path),sha256=sha(path),kind='real',repeats=1,
                        timeout_active_seconds=600,dataset=ds,seed=cond['seed'],logical_seed=cond['seed'],fit_seed=cond['seed'],
                        fold_partition_seed=cond['seed']+41000,evaluation_pair_seed=cond['seed']+51000,Q=Q,candidate=ci,tau=tau,fold=fold,
                        n=card['n'],c=card['k'],positive=len(P),negative=len(N),query_input_sha256=cond['query_input']['sha256']))
        del xs,cell;gc.collect()
        print('PREPARED_INPUTS',ds,len(cases),flush=True)
    assert len(cases)==216 and len(conditions)==42 and sum(c['unsupported'] for c in conditions)==18
    evidence=dict(v5['bound_files'])
    for file in [GAP/'FINITE_NATIVE_COMPARISON_DECISION.json',CAT,V5/'RUNTIME_PROTOCOL.json',V5/'QUALIFICATION_RESULTS.json',gate_path,
        ENV/'CHECK_MANIFEST.json',ENV/'attempt/CHECK_RECEIPT.json',ENV/'attempt/probe.json',
        ENV/'attempt/atomic_smoke/smoke.json',ENV/'attempt/atomic_smoke/started.json',ENV/'attempt/atomic_smoke/finished.json']:
        evidence[str(file)]=sha(file)
    for file,digest in read(ENV/'CHECK_MANIFEST.json')['bound_files'].items():evidence[file]=digest
    # Author source identities are inherited explicitly, not merely via a manifest hash.
    for file,digest in science['source_hashes'].items():evidence[file]=digest
    p=dict(status='DRAFT_INPUTS_PREPARED_NOT_FROZEN',prepared_utc=now(),method=METHOD,
        decision_sha256=DECISION_SHA,grid=GRID,gamma=.1,config=CONFIG,seed_map=cat['seeds'],budgets=[160,400],
        datasets=cat['datasets'],conditions=conditions,cases=cases,feature_metadata=feature_signatures,
        counts=dict(maximum_cv_fits=144,maximum_final_fits=24,maximum_total_fits=168,logical_outcomes=42,unsupported=18,potential_inputs=216),
        author_root=science['author_root'],matlab_executable=science['matlab_executable'],runtime_environment=science['runtime_environment'],
        runtime=dict(science['runtime'],timeout_includes_startup=False),global_wall_seconds=48*3600,source_hashes=sources(),evidence_hashes=evidence,
        full_input_hash_policy='All 216 hashes at execution admission and seal; current MAT plus sources/protocol before/after each fit.',
        execution_order='seed23..25;Q160,400;3Sources,BBC,Cora,CiteSeer;fold0,1;candidate.7,.1,.3;one selected final.',
        randomization='Every candidate/fold/final uses fit_seed=logical seed s and MATLAB rng(s,twister); fold seed=s+41000; evaluation-pair seed=s+51000. No claim of identical random trajectories across algorithms.',
        selection='Original hard-label balanced pair error, all required folds, default-first 1e-12 ties/fallback. No final rescue.',
        score_boundary='All 42 outcomes/144 CV states sealed plus independent PRE_SCORE admission before oracle labels.',
        no_training_or_oracle_read_by_preparation=True)
    dump(ROOT/'PROTOCOL_DRAFT.json',p)
    protocol(ROOT/'PROTOCOL_DRAFT.json')
    print(json.dumps(dict(status=p['status'],draft_sha256=sha(ROOT/'PROTOCOL_DRAFT.json'),input_MATs=len(cases),actual_input_bytes=sum(Path(c['input']).stat().st_size for c in cases)),indent=2))
def freeze(audit_path,audit_sha):
    no_oracle_guard();p=protocol(ROOT/'PROTOCOL_DRAFT.json');draft_sha=sha(ROOT/'PROTOCOL_DRAFT.json')
    admission(audit_path,audit_sha,'PRE_FREEZE',draft_sha)
    assert not (ROOT/'EXECUTION_STARTED.json').exists()
    # Preserve the exact draft bytes; freeze is an external admission, not a rewrite.
    target=ROOT/'FROZEN_PROTOCOL.json'
    assert not target.exists() and not (ROOT/'RUNTIME_PROTOCOL.json').exists()
    target.write_bytes((ROOT/'PROTOCOL_DRAFT.json').read_bytes())
    v5=read(V5/'RUNTIME_PROTOCOL.json')
    r=dict(identity='native_formal_v1_limited_anchor_runtime',original_scientific_protocol_sha256=sha(target),
        original_scientific_protocol=str(target),environment_evidence=dict(receipt=str(ENV/'attempt/CHECK_RECEIPT.json'),
            probe=str(ENV/'attempt/probe.json'),manifest=str(ENV/'CHECK_MANIFEST.json')),
        counts=p['counts'],global_wall_seconds=p['global_wall_seconds'],
        base_qualification_runtime_sha256=sha(V5/'RUNTIME_PROTOCOL.json'),resources=v5['resources'],markers=v5['markers'],
        scope='Formal limited-anchor CV, original author code unchanged. Worker budget600active, new global48h deadline.',
        bound_files={**p['source_hashes'],**p['evidence_hashes'],str(audit_path):audit_sha})
    r['clocks']=dict(v5['clocks'],synthetic_worker_active_seconds=None,real_worker_active_seconds=600)
    r['environment_gate_at_preparation']=dict(passed=True,reasons=[])
    dump(ROOT/'RUNTIME_PROTOCOL.json',r)
    dump(ROOT/'FREEZE_RECEIPT.json',dict(status='FROZEN_NOT_EXECUTED',protocol_sha256=sha(target),runtime_sha256=sha(ROOT/'RUNTIME_PROTOCOL.json'),pre_freeze_audit=str(audit_path),pre_freeze_audit_sha256=audit_sha))
    print(json.dumps(read(ROOT/'FREEZE_RECEIPT.json'),indent=2))
def main():
    a=argparse.ArgumentParser();a.add_argument('--prepare',action='store_true');a.add_argument('--freeze-audit');a.add_argument('--audit-sha');args=a.parse_args()
    if args.prepare:assert not args.freeze_audit;prepare()
    elif args.freeze_audit:assert args.audit_sha;freeze(Path(args.freeze_audit).resolve(),args.audit_sha)
    else:print(json.dumps(preflight(),indent=2))
if __name__=='__main__':main()
