"""Exact existing integration functions, relocated; no class vectors accepted."""
from pathlib import Path
import hashlib,json,os,subprocess,time,traceback,warnings
import numpy as np
from scipy import io,sparse
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
import psutil
from portable_budget_runner import read,sha
ROOT=Path(__file__).resolve().parent/'fssmsc'
MATLAB=Path(os.environ.get('MATLAB_EXECUTABLE','matlab'))
GRID=[dict(lambda_M=a,beta=b) for a in [1e4,1e3,1e5] for b in [1e-4,1e-3,1e-5]]
PARAMETERS=dict(maxiter=30,tol=.001,lambda_Z=.2,**{'lambda':100.},lambda0=100.,epsilon=1e-5,epsilon_q=1e-5,mZ=1.)
def serial(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, Path): return str(x)
    raise TypeError(type(x).__name__)

def dump(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=serial), encoding='utf-8')
    os.replace(tmp, path)

def matlab_call(expression, log):
    log = Path(log);log.parent.mkdir(parents=True,exist_ok=True)
    batch = "maxNumCompThreads(1); addpath('"+str(ROOT).replace('\\','/').replace("'","''")+"'); "+expression
    start=time.perf_counter();peak=0
    with log.open('w',encoding='utf-8') as f:
        proc=subprocess.Popen([str(MATLAB),'-batch',batch],stdout=f,stderr=subprocess.STDOUT,
                              creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        while proc.poll() is None:
            try:
                process=psutil.Process(proc.pid);family=[process]+process.children(recursive=True)
                peak=max(peak,sum(p.memory_info().rss for p in family if p.is_running()))
            except (psutil.NoSuchProcess,psutil.AccessDenied): pass
            time.sleep(.5)
    result=dict(exit_code=proc.returncode,wall_seconds=time.perf_counter()-start,peak_family_rss_bytes=peak,log=str(log))
    dump(log.with_suffix('.process.json'),result)
    return result

def quoted(path): return str(path).replace('\\','/').replace("'","''")

def array_hash(x):
    h=hashlib.sha256()
    if sparse.issparse(x):
        x=sparse.csr_matrix(x);x.sort_indices()
        for z in [np.asarray(x.shape,dtype=np.int64),x.indptr.astype(np.int64),x.indices.astype(np.int64),x.data]:h.update(z.tobytes())
    else:
        x=np.asarray(x);h.update(str(x.shape).encode());h.update(str(x.dtype).encode());h.update(np.ascontiguousarray(x).tobytes())
    return h.hexdigest()

def make_job(folder,stem,common,seed,k,m,P,N,ci):
    inp=folder/(stem+'_input.mat');out=folder/(stem+'_state.mat')
    payload=dict(common_path=str(common),P=np.asarray(P,dtype=np.float64)+1,N=np.asarray(N,dtype=np.float64)+1,
        k=k,seed=seed,m=m,para=dict(PARAMETERS,**GRID[ci]),input_id=stem)
    manifest=folder/(stem+'_job.json')
    if inp.exists():
        existing=io.loadmat(inp,simplify_cells=True)
        assert set(existing)-{'__header__','__version__','__globals__'}==set(payload)
        for key,value in payload.items():
            if key in ['P','N']:assert np.array_equal(np.asarray(existing[key]).reshape(-1,2),value.reshape(-1,2)),key
            else:assert existing[key]==value,key
    else:
        assert not out.exists() and not Path(str(out)+'.error.json').exists(), 'Orphan state without its original input'
        io.savemat(inp,payload,do_compression=True)
    record=dict(input=str(inp),output=str(out),stem=stem,candidate=ci,input_sha256=sha(inp),common_sha256=sha(common))
    if manifest.exists():assert read(manifest)==record, 'Immutable job identity changed'
    else:
        assert not out.exists() and not Path(str(out)+'.error.json').exists(), 'Orphan output without prelaunch manifest'
        dump(manifest,record)
    return record

def batch_jobs(folder,name,jobs):
    for j in jobs:
        assert sha(j['input'])==j['input_sha256']
        inp=io.loadmat(j['input'],simplify_cells=True)
        assert sha(inp['common_path'])==j['common_sha256']
    pending=[j for j in jobs if not Path(j['output']).exists() and not Path(j['output']+'.error.json').exists()]
    if not pending:return None
    path=folder/(name+'_jobs.json');dump(path,pending)
    return matlab_call(f"fssmsc_batch('{quoted(path)}')",folder/(name+'.log'))

def finish_job(job,seed,P,N,folder):
    stem=job['stem'];state=Path(job['output']);meta_path=folder/(stem+'_diagnostics.json')
    prediction=folder/(stem+'.npz')
    if meta_path.exists():
        meta=read(meta_path);assert meta['input_sha256']==sha(job['input'])
        if meta.get('state_sha256'):assert sha(state)==meta['state_sha256']
        if meta.get('error_sha256'):assert sha(str(state)+'.error.json')==meta['error_sha256']
        if meta.get('prediction_sha256'):
            assert sha(prediction)==meta['prediction_sha256'] and sha(state)==meta['state_sha256']
            with np.load(prediction) as d:return d['labels'].copy(),meta
        return None,meta
    meta=dict(stem=stem,candidate=job['candidate'],parameters=dict(PARAMETERS,**GRID[job['candidate']]),
              input_sha256=sha(job['input']),operational_eligible=False,test_metrics_computed=False)
    try:
        if not state.exists():raise RuntimeError(read(str(state)+'.error.json') if Path(str(state)+'.error.json').exists() else 'Missing MATLAB output')
        inp=io.loadmat(job['input'],simplify_cells=True)
        common=io.loadmat(inp['common_path'],variable_names=['n','k','m','seed'],simplify_cells=True)
        data=io.loadmat(state,simplify_cells=True);F=np.asarray(data['F'],float)
        k=int(common['k']);n=int(common['n']);m=int(common['m'])
        assert data['input_id']==job['stem'] and data['common_path']==inp['common_path']
        assert data['status']=='embedding_completed' and not bool(data['test_metrics_computed'])
        assert int(data['seed'])==seed==int(inp['seed'])==int(common['seed'])
        assert int(data['k'])==k==int(inp['k']) and int(data['m'])==m==int(inp['m'])
        for key in inp['para']:assert data['para'][key]==inp['para'][key],key
        assert data['para']['C_q']==n and data['para']['k']==k
        def canonical_edges(edges):
            edges=np.sort(np.asarray(edges,dtype=np.int64).reshape(-1,2),axis=1)
            return edges[np.lexsort((edges[:,1],edges[:,0]))]
        for name,edges in [('P',P),('N',N)]:
            assert np.array_equal(canonical_edges(data[name]),canonical_edges(edges)+1)
            assert np.array_equal(canonical_edges(inp[name]),canonical_edges(edges)+1)
        perm=np.asarray(data['permutation'],int).ravel()-1
        assert F.shape==(n,k) and np.array_equal(np.sort(perm),np.arange(n)) and np.isfinite(F).all()
        diag=data['diagnostics'];Z=np.asarray(data['Z']);q=np.asarray(diag['q']);B=np.asarray(data['B']);A=np.asarray(data['A'])
        assert A.shape==(k,m) and B.shape==Z.shape==np.asarray(diag['Lambda']).shape==(m,n) and q.shape==(m,)
        assert all(np.isfinite(x).all() for x in [A,B,Z,q,np.asarray(diag['Lambda'])])
        assert Z.min()>=-1e-12 and Z.max()<=1+1e-12 and q.min()>=1e-5-1e-12 and q.max()<=n+1e-10
        assert int(diag['iterations'])==30 or bool(diag['legacy_stop_reached'])
        history=diag['history'];history=[history] if isinstance(history,dict) else list(history)
        assert len(history)==int(diag['iterations'])
        for entry in history:
            assert bool(entry['finite']) and bool(entry['inner']['finite'])
            assert all(np.isfinite(v) for key,v in entry.items() if key!='inner')
            assert all(np.isfinite(v) for v in entry['inner'].values())
        with warnings.catch_warnings(record=True) as caught,threadpool_limits(limits=1):
            model=KMeans(n_clusters=k,init='k-means++',n_init=10,max_iter=1000,tol=1e-4,algorithm='lloyd',random_state=seed).fit(F)
        labels=np.empty(n,dtype=np.int64);labels[perm]=model.labels_
        embedding=np.empty_like(F);embedding[perm]=F
        np.savez_compressed(prediction,labels=labels,embedding=embedding,P=P,N=N,inertia=model.inertia_,centers=model.cluster_centers_)
        meta.update(state_sha256=sha(state),prediction_sha256=sha(prediction),inertia=float(model.inertia_),
          kmeans_iterations=int(model.n_iter_),realized_clusters=len(np.unique(labels)),kmeans_warnings=[str(w.message) for w in caught],
          operational_eligible=True,degenerate_cluster_count=len(np.unique(labels))<k,status='operational_completed',iterations=int(diag['iterations']),
          native_outer_tolerance=bool(diag['legacy_stop_reached']),native_stop_reason=diag['stop_reason'],
          inner_author_failure_message_count=sum(bool(h['inner']['author_failure_message']) for h in history),
          inner_max_ratio_change=max(float(h['inner']['ratio_change']) for h in history),
          final_diagnostics=history[-1],core_seconds=float(data['timing']['core']),paper_stationarity_claimed=False)
        dump(meta_path,meta);return labels,meta
    except Exception as e:
        meta.update(status='exception_or_invalid_output',exception=str(e),traceback=traceback.format_exc())
        if state.exists():meta['state_sha256']=sha(state)
        if Path(str(state)+'.error.json').exists():meta['error_sha256']=sha(str(state)+'.error.json')
        dump(meta_path,meta);return None,meta
