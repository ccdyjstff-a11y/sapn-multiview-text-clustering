"""Portable native PCBG replay. Author optimizer stays external and unchanged."""
from pathlib import Path
import argparse, hashlib, json, os, subprocess, sys, time
sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
import psutil
from scipy import io, sparse

HERE=Path(__file__).resolve().parent
GRID=[.7,.1,.3]
CONFIG=dict(anchor_rate=.7,gama=.1,Iter=30,style=0,IterMax=20,islocal=1,
    Organize_CL=1,w=1,Theorem2=1,Theorem3=1,Isbreak_Loss=1,Convergence=1e-4)
def read(p): return json.loads(Path(p).read_text('utf8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,d):
    p=Path(p);p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf8')
def require(v,msg):
    if not v: raise ValueError(msg)
def relative(root,name):
    p=(root/name).resolve();require(p.is_relative_to(root),'Path escapes its declared root');return p
def feature_hash(x):
    x=x.astype(np.float64).tocsc();x.sum_duplicates();x.eliminate_zeros();x.sort_indices()
    h=hashlib.sha256()
    for v in [np.array(x.shape,dtype='<i8'),x.indptr.astype('<i8'),x.indices.astype('<i8'),x.data.astype('<f8')]:h.update(v.tobytes())
    return h.hexdigest()
def author_audit(author):
    identity=read(HERE/'IDENTITIES.json')
    for f in identity['author_files']:
        p=relative(author,f['path']);require(p.is_file() and sha(p)==f['sha256'],'Author source differs: '+f['path'])
    require(sha(HERE/'pcbg_formal_worker.m')==identity['worker_sha256'],'Worker changed')
    require(sha(HERE/'pcbg_clock_bridge.m')==identity['bridge_sha256'],'Bridge changed')
    return identity
def sources():return {p.name:sha(p) for p in [Path(__file__),HERE/'IDENTITIES.json',HERE/'pcbg_formal_worker.m',HERE/'pcbg_clock_bridge.m']}
def load_plan(out):
    p=read(out/'PLAN.json');require(p['source_hashes']==sources(),'Portable source identity changed')
    for c in p.get('cases',[]):require(sha(relative(out,c['input']))==c['sha256'],'Prepared MAT changed')
    return p
def folds(pool,pairs,answers,seed):
    halves=np.array_split(np.random.RandomState(seed+41000).permutation(pool),2);result=[]
    for half in halves:
        side=np.isin(pairs,half);tr=np.flatnonzero(~side.any(axis=1));va=np.flatnonzero(side.all(axis=1))
        if set(answers[tr])=={-1,1} and set(answers[va])=={-1,1}:result.append((tr,va))
    return result
def pair_ber(labels,pairs,answers):
    positive=answers==1;negative=answers==-1
    if not positive.any() or not negative.any():return None
    equal=labels[pairs[:,0]]==labels[pairs[:,1]]
    return float(.5*((~equal[positive]).mean()+equal[negative].mean()))

def prepare(a):
    release=Path(a.release_root).resolve();data=Path(a.prepared).resolve();out=Path(a.output).resolve()
    require(not out.exists(),'Use a new output directory')
    require(not out.is_relative_to(release),'Output must be outside release')
    require(a.budget in [160,400] and a.seed in [23,24,25,41,42,43,101,102,103],'Unknown historical budget/seed')
    base=release/'evidence/v10/native_pcbg/partition_export'
    matches=[r for r in read(base/'PAID_STREAMS.json') if r['dataset']==a.dataset and r['seed']==a.seed]
    require(len(matches)==1,'Dataset/seed absent from historical paid streams')
    card=matches[0];qfile=relative(base,card['projection']['file']);require(sha(qfile)==card['projection']['sha256'],'Paid query projection changed')
    identity=read(HERE/'IDENTITIES.json');out.mkdir(parents=True)
    plan=dict(version='portable_native_pcbg_v1',dataset=a.dataset,seed=a.seed,budget=a.budget,
        source_hashes=sources(),historical_protocol_sha256=identity['historical_protocol_sha256'],
        author_commit=identity['author_commit'],query_sha256=sha(qfile),grid=GRID,
        execution_scope='New replay, not historical resource-scheduler equivalence. No manuscript scores are replaced.',cases=[])
    if a.dataset not in identity['dataset_identities']:
        plan.update(status='INPUT_UNSUPPORTED',reason='Historical native preprocessing does not support these zero-row representations.')
        put(out/'PLAN.json',plan);print(plan['status']);return
    expected=identity['dataset_identities'][a.dataset];n=expected['n'];k=expected['k']
    # Only feature arrays are opened. Labels and metadata containing class counts are not loaded.
    xs=[sparse.load_npz(data/f'X{i}.npz').tocsc() for i in range(expected['views'])]
    require([feature_hash(x) for x in xs]==expected['canonical_feature_sha256'],'Prepared feature values/order differ from historical inputs')
    require(all(np.isfinite(x.data).all() and (x.data>=0).all() and (np.asarray(x.multiply(x).sum(axis=1)).ravel()>0).all() for x in xs),'Unsupported feature values/zero rows')
    with np.load(qfile,allow_pickle=False) as z:
        require(set(z.files)=={'pool','test','pairs','answers'},'Unexpected query fields')
        pool,test,pairs,answers=[z[s].copy() for s in ['pool','test','pairs','answers']]
    require(np.array_equal(np.sort(np.r_[pool,test]),np.arange(n)),'Invalid pool/evaluation partition')
    require(np.array_equal(test,np.setdiff1d(np.arange(n),pool)),'Evaluation objects must be pool complement')
    require(pairs.shape==(400,2) and np.isin(pairs,pool).all() and (pairs[:,0]<pairs[:,1]).all() and len(np.unique(pairs,axis=0))==400,'Invalid paid pairs')
    require(answers.shape==(400,) and np.isin(answers,[-1,1]).all(),'Invalid answers')
    pairs,answers=pairs[:a.budget],answers[:a.budget];fs=folds(pool,pairs,answers,a.seed)
    np.savez_compressed(out/'paid_queries.npz',pool=pool,test=test,pairs=pairs,answers=answers)
    cell=np.empty((len(xs),1),object)
    for i,x in enumerate(xs):cell[i,0]=x
    (out/'inputs').mkdir()
    for ci,rate in enumerate(GRID):
        require(int(n*rate)>=max(k+1,max(3,min(10,int(rate*.5*n/k)))+1),'Insufficient anchor dimension')
        for fold in list(range(len(fs)))+['final']:
            ids=np.arange(len(pairs)) if fold=='final' else fs[fold][0]
            pp=pairs[ids][answers[ids]==1];nn=pairs[ids][answers[ids]==-1]
            case=f'{a.dataset}_s{a.seed}_Q{a.budget}_c{ci}_'+('final' if fold=='final' else f'f{fold}')
            dest=out/'inputs'/(case+'.mat')
            io.savemat(dest,dict(X=cell,c=float(k),ML=(pp+1).astype(float),CL=(nn+1).astype(float),
                seed=float(a.seed),config={key:float(v) for key,v in dict(CONFIG,anchor_rate=rate).items()},case_id=case),do_compression=True)
            plan['cases'].append(dict(id=case,input='inputs/'+dest.name,sha256=sha(dest),candidate=ci,fold=fold))
    plan.update(status='PREPARED',n=n,k=k,feature_hashes=[feature_hash(x) for x in xs],
        paid_file_sha256=sha(out/'paid_queries.npz'),folds=[dict(train=t.tolist(),validation=v.tolist()) for t,v in fs])
    put(out/'PLAN.json',plan);print('PREPARED',len(plan['cases']),'MAT inputs')

def mlquote(p):return str(p).replace('\\','/').replace("'","''")
def execute_case(out,plan,case,author,matlab):
    identity=author_audit(author);dest=out/'runs'/case['id'];receipt=dest/'receipt.json'
    if receipt.exists():
        r=read(receipt)
        require(r['plan_sha256']==sha(out/'PLAN.json'),'Receipt plan mismatch')
        for name,digest in r['artifacts'].items():require(sha(relative(dest,name))==digest,'Result changed')
        return r
    require(not dest.exists(),'Partial run exists; preserve it and prepare a separate replay')
    require(psutil.virtual_memory().available>=3*2**30,'Less than 3 GiB available; retry later, without launching MATLAB')
    dest.mkdir(parents=True)
    args=[str(matlab),'-batch',"addpath('"+mlquote(HERE)+"');pcbg_clock_bridge('"+mlquote(relative(out,case['input']))+"','"+mlquote(dest)+"','"+mlquote(author)+"','"+case['id']+"');"]
    environment=os.environ.copy();environment['MATLAB_PREFDIR']=str(dest/'preferences')
    wall=time.monotonic();status='RUNNING';exitcode=None;proc=None
    with (dest/'matlab.log').open('wb') as log:
        try:
            proc=psutil.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=environment,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            started=None;finished=None
            while proc.poll() is None:
                now=time.monotonic()
                if (dest/'worker_started.json').exists() and started is None:started=now
                if (dest/'worker_finished.json').exists() and finished is None:finished=now
                if (started is None and now-wall>300) or (started is not None and finished is None and now-started>600) or (finished is not None and now-finished>60):
                    status='PORTABLE_CLOCK_CAP';break
                time.sleep(.25)
            if status=='RUNNING':exitcode=proc.wait();status='PROCESS_EXITED'
        finally:
            if proc is not None and proc.poll() is None:
                children=proc.children(recursive=True)
                for child in reversed(children):
                    try:child.terminate()
                    except psutil.NoSuchProcess:pass
                proc.terminate();_,alive=psutil.wait_procs(children+[proc],timeout=5)
                for child in alive:
                    try:child.kill()
                    except psutil.NoSuchProcess:pass
    summary=read(dest/'worker_summary.json') if (dest/'worker_summary.json').exists() else {}
    markers=all((dest/name).exists() for name in ['worker_started.json','worker_finished.json'])
    eligible=bool(status=='PROCESS_EXITED' and exitcode==0 and markers and summary.get('benchmark_contract_met'))
    author_audit(author);require(plan['source_hashes']==sources(),'Source changed during execution')
    require(sha(relative(out,case['input']))==case['sha256'],'Input changed during execution')
    r=dict(case_id=case['id'],plan_sha256=sha(out/'PLAN.json'),status=status,exit_code=exitcode,
        eligible=eligible,worker_status=summary.get('status'),finite_partition_retained=bool(summary.get('finite_partition_retained')),
        elapsed_seconds=time.monotonic()-wall,author_commit=identity['author_commit'],
        clock_scope='300 s startup / 600 s worker / 60 s shutdown; wall time with no memory suspension. Original active-time scheduler is not replayed.',
        artifacts={p.name:sha(p) for p in dest.iterdir() if p.is_file()})
    put(receipt,r);return r

def run(a):
    out=Path(a.output).resolve();p=load_plan(out)
    require(p['status']=='PREPARED','Unsupported input cannot fit')
    author=Path(a.author_root).resolve();matlab=Path(a.matlab).resolve()
    require(matlab.is_file(),'Explicit MATLAB executable not found');author_audit(author)
    if a.case:
        cases=[c for c in p['cases'] if c['id']==a.case];require(len(cases)==1,'Unknown case')
        print(json.dumps(execute_case(out,p,cases[0],author,matlab),indent=2));return
    require(not (out/'CONDITION_SEALED.json').exists(),'This condition is already sealed')
    with np.load(out/'paid_queries.npz',allow_pickle=False) as z:pairs=z['pairs'];answers=z['answers']
    require(sha(out/'paid_queries.npz')==p['paid_file_sha256'],'Paid queries changed')
    scores=np.full((3,len(p['folds'])),np.inf)
    for f,fold in enumerate(p['folds']):
        for ci in range(3):
            case=next(c for c in p['cases'] if c['candidate']==ci and c['fold']==f)
            r=execute_case(out,p,case,author,matlab)
            if r['eligible']:
                labels=io.loadmat(out/'runs'/case['id']/'native_outputs.mat',variable_names=['label'])['label'].ravel()
                ids=np.asarray(fold['validation']);scores[ci,f]=pair_ber(labels,pairs[ids],answers[ids])
    means=scores.mean(axis=1) if scores.shape[1] else np.full(3,np.inf)
    tied=np.flatnonzero(abs(means-np.min(means))<=1e-12) if np.isfinite(means).any() else np.array([],int)
    selected=int(0 if 0 in tied or not len(tied) else tied[0])
    rule='minimum_query_BER' if len(tied) else 'no_valid_candidate_default'
    final=next(c for c in p['cases'] if c['candidate']==selected and c['fold']=='final')
    result=execute_case(out,p,final,author,matlab)
    put(out/'SELECTION.json',dict(candidate=selected,anchor_rate=GRID[selected],rule=rule,
        scores=[[float(v) if np.isfinite(v) else None for v in row] for row in scores],eligible_folds=len(p['folds']),final_case=final['id'],eligible=result['eligible']))
    hashes={q.relative_to(out).as_posix():sha(q) for q in out.rglob('*') if q.is_file() and 'preferences' not in q.parts}
    put(out/'CONDITION_SEALED.json',dict(status='SEALED_BEFORE_EXTERNAL_SCORING',files=hashes,scope='One replay condition, not the historical 42-outcome global seal'))
    print('SEALED',final['id'],'eligible=',result['eligible'])

def score(a):
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import normalized_mutual_info_score,adjusted_rand_score
    out=Path(a.output).resolve();p=load_plan(out);seal=read(out/'CONDITION_SEALED.json')
    for name,digest in seal['files'].items():require(sha(relative(out,name))==digest,'Sealed file changed')
    selected=read(out/'SELECTION.json');result={'eligible':selected['eligible'],'ACC':None,'NMI':None,'ARI':None}
    if selected['eligible']:
        y=np.load(a.labels,allow_pickle=False).reshape(-1);require(len(y)==p['n'] and len(np.unique(y))==p['k'],'External labels do not match n/k')
        z=io.loadmat(out/'runs'/selected['final_case']/'native_outputs.mat',variable_names=['label'])['label'].ravel()
        with np.load(out/'paid_queries.npz',allow_pickle=False) as q:
            test=q['test'];require(np.array_equal(np.where(y[q['pairs'][:,0]]==y[q['pairs'][:,1]],1,-1),q['answers']),'External labels disagree with paid answers/order')
        _,yy=np.unique(y[test],return_inverse=True);_,zz=np.unique(z[test],return_inverse=True)
        table=np.zeros((yy.max()+1,zz.max()+1),int);np.add.at(table,(yy,zz),1);i,j=linear_sum_assignment(-table)
        result.update(ACC=float(table[i,j].sum()/len(test)),NMI=float(normalized_mutual_info_score(y[test],z[test])),ARI=float(adjusted_rand_score(y[test],z[test])))
    result.update(labels_sha256=sha(a.labels),seal_sha256=sha(out/'CONDITION_SEALED.json'),not_a_manuscript_result_replacement=True)
    require(not (out/'SCORE.json').exists(),'Score already recorded');put(out/'SCORE.json',result);print(json.dumps(result,indent=2))

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare')
    for name in ['release-root','prepared','dataset','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--budget',type=int,choices=[160,400],required=True)
    p=sub.add_parser('run')
    for name in ['output','author-root','matlab']:p.add_argument('--'+name,required=True)
    p.add_argument('--case',help='Execute only one prepared case; omit for complete within-budget selection and final fit')
    p=sub.add_parser('score');p.add_argument('--output',required=True);p.add_argument('--labels',required=True)
    a=parser.parse_args();globals()[a.command](a)
if __name__=='__main__':main()
