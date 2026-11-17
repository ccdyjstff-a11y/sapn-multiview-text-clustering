"""Portable artifact identities and complete pre-score validation; no oracle I/O."""
from pathlib import Path
import hashlib,json,os,importlib.metadata,itertools
import numpy as np
from query_protocol import node_folds,eligible_folds,signed_pairs,select_configuration,balanced_pair_error

HOME=Path(__file__).resolve().parent
METHODS=['full','consensus-only','views-only']
VALUES=[20.,2.,200.]
LOGICAL_METHODS=[m+'-CV' for m in METHODS]+[m+'-at-full-lambda' for m in METHODS[1:]]
SOURCE_FILES=['allocation_core.py','allocation_models.py','CORE_DERIVATION.json','query_protocol.py',
              'init_warm.py','allocation_io.py','run_allocation.py','score_allocation.py','SOURCE_PROVENANCE.json']

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Path):return str(value)
    raise TypeError(type(value).__name__)
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=serial),encoding='utf-8')
    temp.replace(path)
def immutable(path,value):
    path=Path(path);canonical=json.loads(json.dumps(value,default=serial))
    if path.exists():assert read(path)==canonical,('Frozen artifact changed',str(path))
    else:dump(path,value)
def array_digest(value):
    value=np.ascontiguousarray(value)
    h=hashlib.sha256(str((value.shape,value.dtype.str)).encode());h.update(value.tobytes());return h.hexdigest()
def save_state(path,state):
    arrays={};meta={}
    for key,value in state.items():
        if isinstance(value,np.ndarray):arrays[key]=value
        elif key in ('U','Hv'):
            meta[key+'_count']=len(value);arrays.update({key+str(v):a for v,a in enumerate(value)})
        else:meta[key]=value
    arrays['metadata_json']=np.array(json.dumps(meta,ensure_ascii=False))
    tmp=Path(path).with_suffix('.tmp.npz');np.savez_compressed(tmp,**arrays);tmp.replace(path)
def load_state(path):
    with np.load(path,allow_pickle=False) as f:
        state=json.loads(str(f['metadata_json']))
        for key in ('U','Hv'):state[key]=[f[key+str(v)].copy() for v in range(state.pop(key+'_count'))]
        for key in f.files:
            if key!='metadata_json' and not ((key.startswith('U') and key[1:].isdigit()) or (key.startswith('Hv') and key[2:].isdigit())):state[key]=f[key].copy()
    return state
def source_hashes():return {name:sha(HOME/name) for name in SOURCE_FILES}
def software():return {name:importlib.metadata.version(name) for name in ['numpy','scipy','scikit-learn','threadpoolctl']}
def check_sources(protocol):
    assert source_hashes()==protocol['source_sha256'],'Portable source identity changed'
    assert software()==protocol['software'],'Software identity changed'
    for r in read(HOME/'SOURCE_PROVENANCE.json')['copies']:
        assert sha(HOME/r['file'])==r['sha256'],r['file']
    assert sha(HOME/'allocation_core.py')==read(HOME/'CORE_DERIVATION.json')['derived_sha256']

def paid_inputs(path,n,budgets):
    with np.load(path,allow_pickle=False) as z:
        if {'y','truth','labels','targets','target'}.intersection(z.files):raise ValueError('Paid-query file must not contain a full class vector')
        pool,pairs,answers=[z[key].copy() for key in ['pool','pairs','answers']]
        stored_test=z['test'].copy() if 'test' in z.files else None
    if pool.ndim!=1 or not np.issubdtype(pool.dtype,np.integer) or len(np.unique(pool))!=len(pool) or np.any(pool<0) or np.any(pool>=n):raise ValueError('Invalid unique query pool')
    if pairs.ndim!=2 or pairs.shape[1]!=2 or not np.issubdtype(pairs.dtype,np.integer) or not np.isin(pairs,pool).all() or np.any(pairs[:,0]>=pairs[:,1]) or len(np.unique(pairs,axis=0))!=len(pairs):raise ValueError('Unique ordered unordered pairs entirely within pool required')
    if answers.shape!=(len(pairs),) or not np.isin(answers,[-1,1]).all() or max(budgets)>len(pairs):raise ValueError('Invalid or insufficient paid answers')
    test=np.setdiff1d(np.arange(n),pool)
    if not len(test):raise ValueError('A nonempty held-out object set is required')
    if stored_test is not None and not np.array_equal(test,stored_test):raise ValueError('Stored test indices disagree with pool complement')
    return dict(pool=pool,test=test,pairs=pairs,answers=answers)

def compact(q,stem):
    return dict(stem=stem,**{key:q[key] for key in ['status','converged','kkt','seconds','identity_sha256']},
        **{key:q[key] for key in ['prediction_sha256','state_sha256','invariants','final_a','allocation'] if key in q})

def terminal(out,stem,method,value,P,N,protocol):
    """Validate a saved fit, including failures, before reuse or evaluation."""
    out=Path(out);identpath=out/(stem+'_identity.json');ident=read(identpath)
    expected=dict(method=method,eta={'full':.5,'consensus-only':0.,'views-only':1.}[method],
        lambda_pair=value,seed=protocol['seed'],protocol_sha256=sha(out/'PROTOCOL.json'),
        input_sha256=sha(out/'inputs.npz'),P=array_digest(P),N=array_digest(N))
    assert ident==expected,stem
    diag=out/(stem+'_diagnostics.json');fail=out/(stem+'_FAILURE.json')
    assert diag.exists()!=fail.exists(),('Exactly one terminal outcome required',stem)
    q=read(diag if diag.exists() else fail)
    assert q['status'] in ['ok','exception'] and isinstance(q['converged'],bool)
    assert q['identity_sha256']==sha(identpath)
    files={identpath.name,(diag if diag.exists() else fail).name}
    # Keep any partial artifacts of a failed fit in the seal too.
    files.update(stem+s for s in ['.npz','_state.npz'] if (out/(stem+s)).exists())
    labels=None
    if q['status']=='ok':
        assert diag.exists() and q['method']==method and q['parameter']==value
        assert q['prediction_sha256']==sha(out/(stem+'.npz')) and q['state_sha256']==sha(out/(stem+'_state.npz'))
        assert np.isfinite(q['kkt'])
        if q['converged']:assert q['kkt']<=1e-3
        with np.load(out/(stem+'.npz'),allow_pickle=False) as z:
            labels=z['labels'].copy();H=z['H']
            assert H.shape==(protocol['k'],protocol['n']) and np.isfinite(H).all()
            assert H.min()>=-1e-12 and np.max(abs(H.sum(axis=0)-1))<1e-10
            assert labels.shape==(protocol['n'],) and np.array_equal(labels,H.argmax(axis=0))
            assert np.array_equal(z['P'],P) and np.array_equal(z['N'],N)
    else:assert fail.exists() and not q['converged']
    return labels,q,files

def collect_artifacts(out,protocol,records):
    out=Path(out)
    expected={(Q,m) for Q,m in itertools.product(protocol['budgets'],LOGICAL_METHODS)}
    actual=[(r['budget'],r['method']) for r in records]
    assert len(actual)==len(set(actual))==len(expected) and set(actual)==expected
    assert all(r['seed']==protocol['seed'] for r in records)
    inp=paid_inputs(out/'inputs.npz',protocol['n'],protocol['budgets'])
    manifest=read(out/'RUN_INPUT_MANIFEST.json')
    assert manifest['protocol_sha256']==sha(out/'PROTOCOL.json')
    assert manifest['inputs_sha256']==sha(out/'inputs.npz') and manifest['graph_sha256']==sha(out/'used_graph.npz')
    files={'PROTOCOL.json','inputs.npz','RUN_INPUT_MANIFEST.json','used_graph.npz','runs.json'}
    for Q in protocol['budgets']:
        qp,qa=inp['pairs'][:Q],inp['answers'][:Q]
        folds=eligible_folds(node_folds(inp['pool'],qp,protocol['seed']+41000),qa)
        query_file=f'Q{Q}_query.json';query=read(out/query_file)
        assert query==json.loads(json.dumps(dict(folds=folds,positive=int(sum(qa==1)),negative=int(sum(qa==-1)),input_sha256=sha(out/'inputs.npz')),default=serial))
        files.add(query_file);selected={}
        for method in METHODS:
            name=f'Q{Q}_{method}_selection.json';s=read(out/name);files.add(name)
            scores=np.asarray(s['scores'],float)
            assert scores.shape==(3,len(folds)) and s['eligible_folds']==len(folds)
            index,rule=select_configuration(scores)
            assert s['index']==index and s['value']==VALUES[index] and s['rule']==rule
            assert len(s['cv'])==3*len(folds)
            assert {(c['candidate'],c['fold']) for c in s['cv']}==set(itertools.product(range(3),range(len(folds))))
            for c in s['cv']:
                ci,fi=c['candidate'],c['fold'];stem=f'Q{Q}_{method}_c{ci}_f{fi}'
                assert c['stem']==stem
                P,N=signed_pairs(qp,qa,folds[fi]['training'])
                labels,q,paths=terminal(out,stem,method,VALUES[ci],P,N,protocol);files.update(paths)
                assert all(c[key]==value for key,value in compact(q,stem).items())
                raw=None if labels is None else balanced_pair_error(labels,qp[folds[fi]['validation']],qa[folds[fi]['validation']])
                assert c['raw_BER']==raw
                expected_score=c['raw_BER'] if q['converged'] and c['raw_BER'] is not None else np.inf
                assert scores[ci,fi]==expected_score
            selected[method]=s
        P,N=signed_pairs(qp,qa)
        for logical in LOGICAL_METHODS:
            row=next(r for r in records if r['budget']==Q and r['method']==logical)
            method=logical.removesuffix('-CV').removesuffix('-at-full-lambda')
            s=selected['full' if logical.endswith('-at-full-lambda') else method]
            stem=f'Q{Q}_{method}_c{s["index"]}_final'
            assert row['stem']==stem and row['selected_lambda']==s['value']
            _,q,paths=terminal(out,stem,method,s['value'],P,N,protocol);files.update(paths)
            assert all(row[key]==value for key,value in compact(q,stem).items())
    return files

def seal_outputs(out):
    out=Path(out);p=read(out/'PROTOCOL.json');check_sources(p);records=read(out/'runs.json')
    files=collect_artifacts(out,p,records)
    immutable(out/'GLOBAL_PREDICTIONS_SEALED.json',dict(count=len(records),
        protocol_sha256=sha(out/'PROTOCOL.json'),runs_sha256=sha(out/'runs.json'),
        condition_keys=sorted([[r['budget'],r['method']] for r in records]),
        files={name:sha(out/name) for name in sorted(files)}))
    return records

def validate_seal(out):
    out=Path(out);p=read(out/'PROTOCOL.json');check_sources(p)
    seal=read(out/'GLOBAL_PREDICTIONS_SEALED.json')
    assert seal['protocol_sha256']==sha(out/'PROTOCOL.json') and seal['runs_sha256']==sha(out/'runs.json')
    records=read(out/'runs.json')
    assert seal['count']==len(records)==5*len(p['budgets'])
    assert seal['condition_keys']==sorted([[r['budget'],r['method']] for r in records])
    for name,digest in seal['files'].items():
        assert Path(name).name==name and sha(out/name)==digest,name
    assert set(seal['files'])==collect_artifacts(out,p,records)
    return p,records
