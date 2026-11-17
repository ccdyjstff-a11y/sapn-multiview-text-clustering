"""Portable original-SAPN/EQ experiment. Training does not accept class labels."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits
from sapn_text_research import fit_sapn_research,warm_init,build_multiview_graph
from query_protocol import node_folds,eligible_folds,signed_pairs,balanced_pair_error,select_configuration

VALUES=[20.,2.,200.]

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):
    def clean(x):
        if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
        if isinstance(x,(tuple,list)):return [clean(v) for v in x]
        if isinstance(x,np.ndarray):return clean(x.tolist())
        if isinstance(x,np.generic):return clean(x.item())
        if isinstance(x,float) and not np.isfinite(x):return None
        return x
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)

def save_state(path,state):
    arrays={};meta={}
    for key,value in state.items():
        if isinstance(value,np.ndarray):arrays[key]=value
        elif key in ('U','Hv'):
            meta[key+'_count']=len(value);arrays.update({key+str(i):v for i,v in enumerate(value)})
        else:meta[key]=value
    arrays['metadata_json']=np.array(json.dumps(meta,ensure_ascii=False))
    temp=path.with_suffix('.tmp.npz');np.savez_compressed(temp,**arrays);temp.replace(path)

def fit(out,stem,X,L,k,H0,Hv0,P,N,seed,value):
    infofile=out/(stem+'_diagnostics.json');artifact=out/(stem+'.npz')
    if infofile.exists():
        q=read(infofile);assert sha(artifact)==q['prediction_sha256'] and sha(out/(stem+'_state.npz'))==q['state_sha256']
        with np.load(artifact) as f:return f['labels'].copy(),q
    print('START',stem,flush=True)
    z,H,q,state=fit_sapn_research(X,P,N,k,variant='original',seed=seed,pair_strength=value,H0=H0,Hv0=Hv0,geometry_graph=L)
    save_state(out/(stem+'_state.npz'),state)
    temp=artifact.with_suffix('.tmp.npz');np.savez_compressed(temp,labels=z,H=H,P=P,N=N);temp.replace(artifact)
    q.update(parameter=value,prediction_sha256=sha(artifact),state_sha256=sha(out/(stem+'_state.npz')))
    dump(infofile,q);print('FIT',stem,q['converged'],q['kkt'],flush=True);return z,q

def run(data,queries,out,seed,budgets):
    data=Path(data).resolve();queries=Path(queries).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    meta=read(data/'metadata.json');n,k,V=meta['n'],meta['k'],meta['views']
    if (data/'initialization.npz').exists() and meta.get('initialization_seed')!=seed:raise ValueError('Saved initialization seed must match requested seed')
    if not budgets or budgets!=sorted(set(budgets)) or min(budgets)<0:raise ValueError('Budgets must be sorted unique nonnegative integers')
    X=[sparse.load_npz(data/f'X{v}.npz').tocsr() for v in range(V)]
    if not 1<=k<=n or not X or any(x.shape[0]!=n or not np.isfinite(x.data).all() or (x.data<0).any() or x.multiply(x).sum()<=0 for x in X):raise ValueError('Invalid nonnegative aligned views')
    with np.load(queries,allow_pickle=False) as f:
        pool=f['pool'].copy();pairs=f['pairs'].copy();answers=f['answers'].copy()
    if pool.ndim!=1 or not np.issubdtype(pool.dtype,np.integer) or len(np.unique(pool))!=len(pool) or (pool<0).any() or (pool>=n).any():raise ValueError('Invalid unique query pool')
    if pairs.ndim!=2 or pairs.shape[1]!=2 or not np.issubdtype(pairs.dtype,np.integer) or not np.isin(pairs,pool).all() or (pairs[:,0]>=pairs[:,1]).any() or len(np.unique(pairs,axis=0))!=len(pairs):raise ValueError('Unique ordered unordered pairs within pool required')
    if answers.shape!=(len(pairs),) or not np.isin(answers,[-1,1]).all() or max(budgets)>len(pairs):raise ValueError('Invalid answers or insufficient paid queries')
    files=[data/'metadata.json',queries]+[data/f'X{v}.npz' for v in range(V)]
    for name in ['initialization.npz','MV_graph.npz','EQ_graph.npz']:
        if (data/name).exists():files.append(data/name)
    script=Path(__file__).resolve();names=['sapn_text_research.py','query_protocol.py',script.name]
    protocol=dict(seed=seed,budgets=budgets,n=n,k=k,views=V,values=VALUES,methods=['SAPN-MV','SAPN-EQ'],
        input_hashes={str(p):sha(p) for p in files},source_hashes={name:sha(script.parent/name) for name in names},
        selection='same node-disjoint query folds, converged candidates only, default-first ties',
        training_labels='No class labels accepted or read. Caller supplies query answers and known k.',
        stages=[200,1000],continuation=4000,scoring='separate command after all requested final predictions sealed')
    if (out/'PROTOCOL.json').exists():assert read(out/'PROTOCOL.json')==protocol,'Resume identity changed'
    else:dump(out/'PROTOCOL.json',protocol)
    EQ=[sparse.hstack([x/np.sqrt(float(x.multiply(x).sum())) for x in X],format='csr')]
    if (data/'initialization.npz').exists():
        with np.load(data/'initialization.npz',allow_pickle=False) as f:H0=f['H0'].copy();Hv0=[f[f'Hv{v}'].copy() for v in range(V)];Heq=f['Heq'].copy()
    else:H0,Hv0=warm_init(X,k,seed,also_hv=True,max_iter=200);Heq=warm_init(EQ,k,seed,max_iter=200)
    graphs={}
    for method,views in [('MV',X),('EQ',EQ)]:
        path=data/(method+'_graph.npz');graphs[method]=sparse.load_npz(path) if path.exists() else build_multiview_graph(views)
    generated=out/'used_initialization.npz'
    if not generated.exists():np.savez_compressed(generated,H0=H0,Heq=Heq,**{f'Hv{v}':a for v,a in enumerate(Hv0)})
    with np.load(generated) as f:
        assert np.array_equal(f['H0'],H0) and np.array_equal(f['Heq'],Heq)
        assert all(np.array_equal(f[f'Hv{v}'],Hv0[v]) for v in range(V))
    for name,L in graphs.items():
        assert L.shape==(n,n)
        path=out/(name+'_graph.npz')
        if path.exists():assert np.max(np.abs((sparse.load_npz(path)-L).data),initial=0)==0
        else:sparse.save_npz(path,L)
    records=read(out/'runs.json') if (out/'runs.json').exists() else []
    for Q in budgets:
        qp,qa=pairs[:Q],answers[:Q];folds=eligible_folds(node_folds(pool,qp,seed+41000),qa)
        dump(out/f'Q{Q}_query_protocol.json',dict(total_queries=Q,positive=int(sum(qa==1)),negative=int(sum(qa==-1)),folds=folds))
        for name,views,h,hv in [('MV',X,H0,Hv0),('EQ',EQ,Heq,[Heq])]:
            method='SAPN-'+name;stem=f'Q{Q}_{method}_final'
            if any(r['stem']==stem for r in records):continue
            scores=np.full((3,len(folds)),np.inf)
            for ci,value in enumerate(VALUES):
                for fi,fold in enumerate(folds):
                    P,N=signed_pairs(qp,qa,fold['training'])
                    z,q=fit(out,f'Q{Q}_{method}_c{ci}_f{fi}',views,graphs[name],k,h,hv,P,N,seed,value)
                    if q['converged']:scores[ci,fi]=balanced_pair_error(z,qp[fold['validation']],qa[fold['validation']])
            index,rule=select_configuration(scores)
            selection=out/f'Q{Q}_{method}_selection.json';dump(selection,dict(scores=scores,index=index,value=VALUES[index],rule=rule))
            P,N=signed_pairs(qp,qa);z,q=fit(out,stem,views,graphs[name],k,h,hv,P,N,seed,VALUES[index])
            records.append(dict(seed=seed,total_queries=Q,method=method,stem=stem,selected_value=VALUES[index],selection_rule=rule,
                eligible_folds=len(folds),converged=q['converged'],kkt=q['kkt'],prediction_sha256=q['prediction_sha256'],state_sha256=q['state_sha256'],selection_sha256=sha(selection)))
            dump(out/'runs.json',records)
    assert len(records)==2*len(budgets) and len({r['stem'] for r in records})==len(records)
    dump(out/'PREDICTIONS_SEALED.json',dict(count=len(records),runs_sha256=sha(out/'runs.json'),protocol_sha256=sha(out/'PROTOCOL.json'),initialization_sha256=sha(generated),graph_hashes={name:sha(out/(name+'_graph.npz')) for name in graphs}))
    print('SEALED',len(records),'final predictions; no class labels read or test scores computed',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--queries',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=23);p.add_argument('--budgets',default='0,160,400');a=p.parse_args()
    with threadpool_limits(limits=1):run(a.data,a.queries,a.output,a.seed,[int(q) for q in a.budgets.split(',')])
