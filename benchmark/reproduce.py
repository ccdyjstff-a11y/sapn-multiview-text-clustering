"""Portable configured experiment: oracle -> fit/select/seal -> separate score.

The train command never opens a class-label file. Source modules are copied
unchanged by build_reproduction.py; all experiment paths are user configuration.
"""
from pathlib import Path
import argparse, hashlib, json, os, sys, time, traceback
import numpy as np
from scipy import sparse, io
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from threadpoolctl import threadpool_limits
from portable_budget_runner import dump, read, sha, save_state
from query_protocol import sample_query_stream, answer_queries, node_folds, eligible_folds, signed_pairs, balanced_pair_error, select_configuration
from sapn_text_research import fit_sapn_research, warm_init, build_multiview_graph

ROOT=Path(__file__).resolve().parent
GRIDS={'SAPN-MV':[20.,2.,200.], 'SAPN-EQ':[20.,2.,200.], 'CMVNMF':[1.,.01,100.], 'MVCHSS':[150.,50.,200.], 'FSSMSC':list(range(9)),
       'full-v5':[20.,2.,200.], 'private-coupled':[20.,2.,200.], 'no-correction':[20.,2.,200.]}
def resolve(base,p):
    p=Path(p);return p.resolve() if p.is_absolute() else (base/p).resolve()
def fresh(p):
    p=Path(p)
    if p.exists() and any(p.iterdir()):raise ValueError('Use a new output directory: '+str(p))
    p.mkdir(parents=True,exist_ok=True);return p
def features(folder):
    m=read(folder/'metadata.json');X=[sparse.load_npz(folder/f'X{i}.npz').tocsr() for i in range(m['views'])]
    assert all(x.shape[0]==m['n'] and np.isfinite(x.data).all() and (x.data>=0).all() and x.multiply(x).sum()>0 for x in X)
    assert 1<m['k']<=m['n']
    return X,m
def metric(y,z):
    _,a=np.unique(y,return_inverse=True);_,b=np.unique(z,return_inverse=True)
    t=np.zeros((a.max()+1,b.max()+1),int);np.add.at(t,(a,b),1);r,c=linear_sum_assignment(-t)
    return dict(ACC=float(t[r,c].sum()/len(y)),NMI=float(normalized_mutual_info_score(y,z)),ARI=float(adjusted_rand_score(y,z)))

def oracle(config):
    p=Path(config).resolve();c=read(p);folder=resolve(p.parent,c['data']);_,m=features(folder)
    labels=resolve(p.parent,c['labels']);y=np.load(labels,allow_pickle=False).ravel();assert len(y)==m['n']
    n=len(y);seed=int(c['seed']);pool=np.sort(np.random.RandomState(seed+21000).permutation(n)[:round(.3*n)])
    pairs=sample_query_stream(pool,int(c['total_queries']),seed+31000)
    output=resolve(p.parent,c['output']);output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise ValueError('Query output already exists')
    np.savez_compressed(output,pool=pool,pairs=pairs,answers=answer_queries(pairs,y))
    dump(output.with_suffix('.oracle.json'),dict(seed=seed,query_sha256=sha(output),labels_sha256=sha(labels),
         pool_sampled_without_labels=True,pair_stream_sampled_without_labels=True,oracle_uses_labels_only_to_answer_sampled_pairs=True))

def fit(method,value,X,L,k,H0,Hv0,P,N,seed,folder,config):
    started=time.perf_counter();state={};q={};z=None;H=None
    try:
        if method in ['full-v5','private-coupled','no-correction']:
            from mechanism_bridge import fit_variant
            z,H,q,state=fit_variant(method,value,X,L,k,H0,Hv0,P,N,seed);ok=bool(q['converged'])
        elif method.startswith('SAPN'):
            z,H,q,state=fit_sapn_research(X,P,N,k,seed=seed,pair_strength=value,H0=H0,Hv0=Hv0,geometry_graph=L)
            ok=bool(q['converged'])
        elif method=='CMVNMF':
            from cmvnmf_accelerated import CMVNMFCoordinate
            model=CMVNMFCoordinate(beta=value,max_iter=3000,identity_links=True,tol=1e-6,tol_kkt=1e-3,seed=seed)
            z,H,a=model.fit_predict(X,P,N,k,H0,Hv0);b=None
            if not a['converged']:z,H,b=model.continue_fit(max_iter=10000)
            last=b or a;q=dict(multiplicative=a,coordinate=b,kkt=last['kkt'],converged=last['converged']);ok=bool(last['converged'])
            state={'U':model.U,'Hv':model.H,'H':H}
        elif method=='MVCHSS':
            import domain_extension as domain
            from mvchss_cached_author import fit_graphs
            graphs,graph_report=domain.mvchss_graphs(X,P,N,p=5,mu=.3,seed=seed)
            trials=[fit_graphs(graphs,k,alpha=50.,beta=value,seed=seed+100000*r,max_iter=1000,profile='author_normalized') for r in range(5)]
            best=int(np.argmin([r.objective_history[-1] for r in trials]));r=trials[best];z=r.labels;H=r.consensus.T
            state=dict(H=H,Hv=r.factors,**{f'restart{j}_history':t.objective_history for j,t in enumerate(trials)})
            q=dict(graph_domain=graph_report,selected_restart=best,restart_objectives=[float(t.objective_history[-1]) for t in trials],
                   converged=None,status='author_normalized_1000_steps_5_restarts',stationarity_certified=False)
            ok=bool(np.isfinite(H).all())
        elif method=='FSSMSC':
            import fssmsc_bridge as f
            f.MATLAB=Path(config['matlab_executable']);common=Path(config['_common'])
            job=f.make_job(folder,'native',common,seed,k,min(500,X[0].shape[0]),P,N,int(value))
            f.batch_jobs(folder,'native',[job]);z,q=f.finish_job(job,seed,P,N,folder)
            ok=bool(q['operational_eligible']);H=np.empty((0,0));state={}
        else:raise ValueError(method)
        assert z is not None and np.asarray(z).shape==(X[0].shape[0],) and np.isfinite(z).all()
    except Exception as error:
        q=dict(status='exception',exception=str(error),traceback=traceback.format_exc());ok=False;z=None
    if z is not None:
        np.savez_compressed(folder/'prediction.npz',labels=z,H=H,P=P,N=N)
        save_state(folder/'state.npz',state)
        q.update(prediction_sha256=sha(folder/'prediction.npz'),state_sha256=sha(folder/'state.npz'))
    q.update(eligible=ok,method=method,value=value,seconds=time.perf_counter()-started)
    dump(folder/'diagnostics.json',q)
    return z,q

def train(config):
    cp=Path(config).resolve();c=read(cp);folder=resolve(cp.parent,c['data']);queries=resolve(cp.parent,c['queries'])
    out=fresh(resolve(cp.parent,c['output']));X,meta=features(folder);n,k=meta['n'],meta['k'];seed=int(c['seed'])
    assert 'labels' not in c, 'Training configuration must not include labels'
    with np.load(queries,allow_pickle=False) as f:pool=f['pool'].copy();pairs=f['pairs'].copy();answers=f['answers'].copy()
    assert pool.ndim==1 and len(np.unique(pool))==len(pool) and ((pool>=0)&(pool<n)).all()
    assert pairs.ndim==2 and pairs.shape[1]==2 and np.isin(pairs,pool).all() and (pairs[:,0]<pairs[:,1]).all()
    assert len(np.unique(pairs,axis=0))==len(pairs) and answers.shape==(len(pairs),) and np.isin(answers,[-1,1]).all()
    budgets=c['budgets'];methods=c['methods'];assert budgets==sorted(set(budgets)) and 0<=min(budgets)<=max(budgets)<=len(pairs)
    assert len(set(methods))==len(methods) and set(methods)<=set(GRIDS)
    if 'FSSMSC' in methods:assert 0 not in budgets,'No author Q0 definition for FSSMSC; run supervised budgets separately'
    names=[p for p in ROOT.glob('*.py')]+list((ROOT/'fssmsc').glob('*.m'))
    if 'FSSMSC' in methods:
        for item in read(ROOT/'THIRD_PARTY_INPUTS.json')['FSSMSC']['required_files']:
            path=ROOT/'literature/FSSMSC/master'/item['path']
            assert path.is_file() and sha(path)==item['sha256'], 'Run acquire_fssmsc.py with the exact author checkout'
            names.append(path)
    source_hashes={str(p.relative_to(ROOT)):sha(p) for p in names}
    input_hashes={f'X{i}.npz':sha(folder/f'X{i}.npz') for i in range(len(X))}
    input_hashes.update(metadata=sha(folder/'metadata.json'),queries=sha(queries))
    for name in ['initialization.npz','MV_graph.npz','EQ_graph.npz']:
        if (folder/name).exists():input_hashes[name]=sha(folder/name)
    protocol=dict(config=c,n=n,k=k,views=len(X),seed=seed,grids={m:GRIDS[m] for m in methods},
                  source_hashes=source_hashes,input_hashes=input_hashes,query_truth_vector_read=False,
                  selection='node-disjoint paid-answer validation; both signs required; default-first ties; no test-label tuning',
                  failed_fits='retained; not scored as zero or counted as competing wins',
                  source_scope='portable reorchestration; main algorithms unchanged; not all historical runs re-trained')
    dump(out/'PROTOCOL.json',protocol);np.savez_compressed(out/'used_queries.npz',pool=pool,pairs=pairs,answers=answers)
    EQ=[sparse.hstack([x/np.sqrt(float(x.multiply(x).sum())) for x in X],format='csr')]
    if (folder/'initialization.npz').exists():
        assert meta.get('initialization_seed')==seed
        with np.load(folder/'initialization.npz') as f:H0=f['H0'];Hv0=[f[f'Hv{i}'] for i in range(len(X))];Heq=f['Heq']
    else:H0,Hv0=warm_init(X,k,seed,also_hv=True,max_iter=200);Heq=warm_init(EQ,k,seed,max_iter=200)
    np.savez_compressed(out/'initialization.npz',H0=H0,Heq=Heq,**{f'Hv{i}':h for i,h in enumerate(Hv0)})
    graphs={name:sparse.load_npz(folder/(name+'_graph.npz')) if (folder/(name+'_graph.npz')).exists() else build_multiview_graph(x) for name,x in [('MV',X),('EQ',EQ)]}
    for name,g in graphs.items():sparse.save_npz(out/(name+'_graph.npz'),g)
    if 'FSSMSC' in methods:
        import fssmsc_bridge as f
        c['matlab_executable']=str(resolve(cp.parent,c['matlab_executable']));f.MATLAB=Path(c['matlab_executable'])
        common=out/'fssmsc_common.mat';cells=np.empty((1,len(X)),object)
        for i,x in enumerate(X):cells[0,i]=x
        inp=out/'fssmsc_features.mat';io.savemat(inp,dict(X=cells,k=k,m=min(500,n),seed=seed),do_compression=True)
        result=f.matlab_call(f"fssmsc_prepare_common('{f.quoted(inp)}','{f.quoted(common)}')",out/'fssmsc_prepare.log')
        assert result['exit_code']==0 and common.exists(),result;c['_common']=str(common)
    records=[]
    for Q in budgets:
        qp,qa=pairs[:Q],answers[:Q];folds=eligible_folds(node_folds(pool,qp,seed+41000),qa)
        dump(out/f'Q{Q}_folds.json',folds)
        for method in methods:
            xx,hh,hv,gg=(EQ,Heq,[Heq],graphs['EQ']) if method=='SAPN-EQ' else (X,H0,Hv0,graphs['MV'])
            values=GRIDS[method];scores=np.full((len(values),len(folds)),np.inf)
            for ci,v in enumerate(values):
                for fi,fold in enumerate(folds):
                    P,N=signed_pairs(qp,qa,fold['training']);d=out/f'Q{Q}_{method}_c{ci}_f{fi}';d.mkdir()
                    z,q=fit(method,v,xx,gg,k,hh,hv,P,N,seed,d,c)
                    if q['eligible']:scores[ci,fi]=balanced_pair_error(z,qp[fold['validation']],qa[fold['validation']])
            selected,rule=select_configuration(scores);selection=out/f'Q{Q}_{method}_selection.json'
            dump(selection,dict(scores=scores,selected_index=selected,value=values[selected],rule=rule,eligible_folds=len(folds)))
            d=out/f'Q{Q}_{method}_final';d.mkdir();P,N=signed_pairs(qp,qa)
            z,q=fit(method,values[selected],xx,gg,k,hh,hv,P,N,seed,d,c)
            records.append(dict(method=method,total_queries=Q,seed=seed,directory=d.name,eligible=q['eligible'],
              selected_value=values[selected],selection_sha256=sha(selection),diagnostics_sha256=sha(d/'diagnostics.json'),
              prediction_sha256=q.get('prediction_sha256'),state_sha256=q.get('state_sha256')))
            dump(out/'runs.json',records);print('FINAL',method,Q,q['eligible'],flush=True)
    files=[p for p in out.rglob('*') if p.is_file()]
    dump(out/'PREDICTIONS_SEALED.json',dict(count=len(records),files={str(p.relative_to(out)):sha(p) for p in files},
         all_requested_final_fits_completed=True,score_computed=False))
    print('SEALED',len(records),flush=True)

def score(config):
    cp=Path(config).resolve();c=read(cp);out=resolve(cp.parent,c['output']);labels=resolve(cp.parent,c['labels'])
    seal=read(out/'PREDICTIONS_SEALED.json');p=read(out/'PROTOCOL.json');records=read(out/'runs.json')
    for name,digest in seal['files'].items():assert sha(out/name)==digest,name
    assert len(records)==seal['count']==len(p['config']['methods'])*len(p['config']['budgets'])
    with np.load(out/'used_queries.npz') as f:pool=f['pool'];pairs=f['pairs']
    test=np.setdiff1d(np.arange(p['n']),pool);assert len(test)>0 and not np.intersect1d(test,pairs).size
    y=np.load(labels,allow_pickle=False).ravel();assert len(y)==p['n']
    results=[]
    for row in records:
        r=dict(row)
        if r['eligible']:
            with np.load(out/r['directory']/'prediction.npz') as f:z=f['labels']
            r.update(metric(y[test],z[test]))
        else:r.update(ACC=None,NMI=None,ARI=None)
        results.append(r)
    dump(out/'evaluation.json',results);dump(out/'EVALUATION_MANIFEST.json',dict(seal_sha256=sha(out/'PREDICTIONS_SEALED.json'),
         label_sha256=sha(labels),evaluation_sha256=sha(out/'evaluation.json'),score_source_sha256=sha(__file__),test_nodes=len(test)))
    print('SCORED',len(results),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['oracle','train','score']);p.add_argument('--config',required=True);a=p.parse_args()
    with threadpool_limits(limits=1):globals()[a.command](a.config)
