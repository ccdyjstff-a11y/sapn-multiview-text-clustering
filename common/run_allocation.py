"""Standalone allocation CV/finals. Accepts paid relationships, never class IDs."""
from pathlib import Path
import sys,os,argparse,time,traceback,warnings
sys.dont_write_bytecode=True
for name in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[name]='1'
PACKAGE=Path(__file__).resolve().parent
sys.path.insert(0,str(PACKAGE))  # Explicit sibling loading also works under python -I.
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits
from allocation_models import CONFIG,ETAS,fit_allocation
from allocation_core import build_multiview_graph
from init_warm import warm_init
from query_protocol import node_folds,eligible_folds,signed_pairs,balanced_pair_error,select_configuration
from allocation_io import (METHODS,VALUES,LOGICAL_METHODS,read,sha,dump,immutable,array_digest,
    save_state,source_hashes,software,paid_inputs,compact,terminal,seal_outputs,validate_seal)

def fit(out,stem,method,value,X,L,k,H0,Hv0,P,N,seed,protocol):
    identity=dict(method=method,eta=ETAS[method],lambda_pair=value,seed=seed,
        protocol_sha256=sha(out/'PROTOCOL.json'),input_sha256=sha(out/'inputs.npz'),P=array_digest(P),N=array_digest(N))
    immutable(out/(stem+'_identity.json'),identity)
    diag=out/(stem+'_diagnostics.json');failure=out/(stem+'_FAILURE.json')
    if diag.exists() or failure.exists():
        z,q,_=terminal(out,stem,method,value,P,N,protocol);return z,q
    started=time.perf_counter()
    try:
        z,H,q,state=fit_allocation(method,value,X,L,k,H0,Hv0,P,N,seed)
        save_state(out/(stem+'_state.npz'),state)
        temp=out/(stem+'.tmp.npz');np.savez_compressed(temp,labels=z,H=H,P=P,N=N);temp.replace(out/(stem+'.npz'))
        q.update(seconds=time.perf_counter()-started,invariants=q.pop('common_invariants'),
            prediction_sha256=sha(out/(stem+'.npz')),state_sha256=sha(out/(stem+'_state.npz')),
            identity_sha256=sha(out/(stem+'_identity.json')))
        dump(diag,q)
        print('FIT',stem,q['converged'],round(q['seconds'],2),flush=True)
        return z,q
    except Exception:
        q=dict(status='exception',converged=False,kkt=None,seconds=time.perf_counter()-started,
            traceback=traceback.format_exc(),identity_sha256=sha(out/(stem+'_identity.json')))
        dump(failure,q);print('FAILED',stem,flush=True);return None,q

def prepare(data,queries,out,seed,budgets):
    meta=read(data/'metadata.json');n,k,V=[meta[key] for key in ['n','k','views']]
    if not all(isinstance(v,int) for v in [n,k,V]) or not 1<=k<=n or V<1:raise ValueError('Invalid n/k/views metadata')
    if not budgets or budgets!=sorted(set(budgets)) or min(budgets)<0:raise ValueError('Budgets must be sorted unique nonnegative integers')
    inp=paid_inputs(queries,n,budgets)
    X=[sparse.load_npz(data/f'X{v}.npz').tocsr() for v in range(V)]
    if any(x.shape[0]!=n or not np.isfinite(x.data).all() or np.any(x.data<0) or x.multiply(x).sum()<=0 for x in X):raise ValueError('Views must be nonnegative finite aligned matrices with positive energy')
    names=['metadata.json']+[f'X{v}.npz' for v in range(V)]
    for name in ['initialization.npz','MV_graph.npz']:
        if (data/name).exists():names.append(name)
    if (data/'initialization.npz').exists() and meta.get('initialization_seed')!=seed:raise ValueError('Saved initialization_seed must equal requested seed')
    protocol=dict(version='portable_allocation_v1',seed=seed,budgets=budgets,n=n,k=k,views=V,
        dataset=meta.get('dataset','external'),methods=METHODS,logical_methods=LOGICAL_METHODS,
        lambda_grid_default_first=VALUES,eta=ETAS,config=CONFIG,
        prepared_file_sha256={name:sha(data/name) for name in names},paid_query_sha256=sha(queries),
        source_sha256=source_hashes(),software=software(),
        selection='Original same eligible node-disjoint two-fold mean BER; every fold converged; [20,2,200]; 1e-12 default-first ties; no Q/Qf scaling.',
        solver='Unmodified fit_allocation factory: 200/1000/4000 stages, original tolerances and endpoint schedule.',
        boundary='Only nonnegative views, known k, graph, initialization and paid relationship answers enter fitting. No class vector accepted. Separate score command requires every requested logical final and validation artifact sealed.',
        final_logical_records=5*len(budgets),maximum_validation_fits=18*len(budgets),maximum_final_fits=5*len(budgets))
    immutable(out/'PROTOCOL.json',protocol)
    if (out/'GLOBAL_PREDICTIONS_SEALED.json').exists():
        validate_seal(out);return protocol,X,None,None,None,inp
    if (out/'RUN_INPUT_MANIFEST.json').exists():
        manifest=read(out/'RUN_INPUT_MANIFEST.json')
        assert manifest['protocol_sha256']==sha(out/'PROTOCOL.json')
        assert manifest['inputs_sha256']==sha(out/'inputs.npz') and manifest['graph_sha256']==sha(out/'used_graph.npz')
        with np.load(out/'inputs.npz',allow_pickle=False) as z:
            for key,value in inp.items():assert np.array_equal(z[key],value)
            H0=z['H0'].copy();Hv0=[z[f'Hv{v}'].copy() for v in range(V)]
        L=sparse.load_npz(out/'used_graph.npz')
    else:
        assert not (out/'inputs.npz').exists() and not (out/'used_graph.npz').exists(),'Unsealed interrupted preparation; retain and inspect before resuming'
        warnings_seen=[]
        if (data/'initialization.npz').exists():
            with np.load(data/'initialization.npz',allow_pickle=False) as z:
                if {'y','truth','labels','targets','target'}.intersection(z.files):raise ValueError('Initialization must not contain class IDs')
                H0=z['H0'].copy();Hv0=[z[f'Hv{v}'].copy() for v in range(V)]
        else:
            with warnings.catch_warnings(record=True) as ws:
                H0,Hv0=warm_init(X,k,seed,also_hv=True,max_iter=200)
            warnings_seen=[str(w.message) for w in ws]
        if any(h.shape!=(k,n) or not np.isfinite(h).all() or np.any(h<0) for h in [H0,*Hv0]):raise ValueError('Invalid saved warm initialization')
        L=sparse.load_npz(data/'MV_graph.npz') if (data/'MV_graph.npz').exists() else build_multiview_graph(X)
        if L.shape!=(n,n) or not np.isfinite(L.data).all() or np.max(abs((L-L.T).data),initial=0)>1e-12:raise ValueError('Invalid finite symmetric graph Laplacian')
        np.savez_compressed(out/'inputs.npz',**inp,H0=H0,**{f'Hv{v}':h for v,h in enumerate(Hv0)})
        sparse.save_npz(out/'used_graph.npz',L)
        immutable(out/'RUN_INPUT_MANIFEST.json',dict(protocol_sha256=sha(out/'PROTOCOL.json'),
            inputs_sha256=sha(out/'inputs.npz'),graph_sha256=sha(out/'used_graph.npz'),
            initialization='supplied' if (data/'initialization.npz').exists() else 'original warm_init, max_iter=200',
            warnings=warnings_seen,label_vector_stored=False))
    return protocol,X,L,H0,Hv0,inp

def run(prepared_directory,paid_queries,output,seed,budgets):
    data=Path(prepared_directory).resolve();queries=Path(paid_queries).resolve();out=Path(output).resolve()
    if out==data or out==PACKAGE:raise ValueError('Output must be separate from prepared data and source package')
    out.mkdir(parents=True,exist_ok=True)
    p,X,L,H0,Hv0,inp=prepare(data,queries,out,seed,budgets)
    if (out/'GLOBAL_PREDICTIONS_SEALED.json').exists():
        print('ALREADY SEALED',p['final_logical_records'],flush=True);return
    rows=[]
    for Q in budgets:
        qp,qa=inp['pairs'][:Q],inp['answers'][:Q]
        folds=eligible_folds(node_folds(inp['pool'],qp,seed+41000),qa)
        immutable(out/f'Q{Q}_query.json',dict(folds=folds,positive=int(sum(qa==1)),negative=int(sum(qa==-1)),input_sha256=sha(out/'inputs.npz')))
        chosen={}
        for method in METHODS:
            scores=np.full((3,len(folds)),np.inf);cv=[]
            for ci,value in enumerate(VALUES):
                for fi,fold in enumerate(folds):
                    P,N=signed_pairs(qp,qa,fold['training']);stem=f'Q{Q}_{method}_c{ci}_f{fi}'
                    z,q=fit(out,stem,method,value,X,L,p['k'],H0,Hv0,P,N,seed,p)
                    raw=None if z is None else balanced_pair_error(z,qp[fold['validation']],qa[fold['validation']])
                    if q['converged'] and raw is not None:scores[ci,fi]=raw
                    cv.append(dict(candidate=ci,fold=fi,raw_BER=raw,**compact(q,stem)))
            index,rule=select_configuration(scores);chosen[method]=index
            immutable(out/f'Q{Q}_{method}_selection.json',dict(scores=scores,cv=cv,index=index,value=VALUES[index],rule=rule,eligible_folds=len(folds)))
            P,N=signed_pairs(qp,qa);stem=f'Q{Q}_{method}_c{index}_final'
            _,q=fit(out,stem,method,VALUES[index],X,L,p['k'],H0,Hv0,P,N,seed,p)
            rows.append(dict(dataset=p['dataset'],seed=seed,budget=Q,method=method+'-CV',selected_lambda=VALUES[index],
                selection_rule=rule,eligible_folds=len(folds),**compact(q,stem)))
        for method in METHODS[1:]:
            index=chosen['full'];P,N=signed_pairs(qp,qa);stem=f'Q{Q}_{method}_c{index}_final'
            _,q=fit(out,stem,method,VALUES[index],X,L,p['k'],H0,Hv0,P,N,seed,p)
            rows.append(dict(dataset=p['dataset'],seed=seed,budget=Q,method=method+'-at-full-lambda',selected_lambda=VALUES[index],
                selection_rule='Full-selected lambda; no additional selection',eligible_folds=len(folds),
                reused_CV_selected_final=index==chosen[method],**compact(q,stem)))
        dump(out/'runs.json',rows)
    seal_outputs(out);validate_seal(out)
    print('SEALED',len(rows),'requested logical finals; no class vector or evaluation scores read',flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-directory','--data',required=True)
    parser.add_argument('--paid-queries','--queries',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--budgets',required=True,help='Sorted unique comma-separated total query counts, e.g. 160,400')
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.prepared_directory,args.paid_queries,args.output,args.seed,[int(q) for q in args.budgets.split(',')])

if __name__=='__main__':main()
