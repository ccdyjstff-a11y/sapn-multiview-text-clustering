"""Finite native PCBG identities and paid-query validation; no oracle loading."""
from pathlib import Path
import hashlib,json,os,sys
sys.dont_write_bytecode=True
import numpy as np
from scipy import io

ROOT=Path(__file__).resolve().parent
GAP=ROOT.parent
MAIN=GAP.parent.parent
GRID=[.7,.1,.3]
FIT_DATA=['3Sources','BBC','Cora','CiteSeer']
METHOD='PCBG-author-limited-anchor-CV'
DECISION_SHA='ac6ef6618ce38eddf2ba45f9bfea27485fc9988590accd6f090421f57c306de4'

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def serial(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    raise TypeError(type(v).__name__)
def dump(p,v):
    p=Path(p);t=p.with_name(p.name+'.tmp')
    t.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=serial,allow_nan=False)+'\n',encoding='utf8');os.replace(t,p)
def immutable(p,v):
    expected=json.loads(json.dumps(v,default=serial,allow_nan=False))
    if Path(p).exists():assert read(p)==expected,str(p)
    else:dump(p,expected)
def no_oracle_guard():
    def guard(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            path=Path(os.fsdecode(args[0]));name=path.name.lower()
            if name in ['y.npy','labels.npy','truth.npy','gnd.npy'] or (name in ['evaluation.json','summary.json'] and path.parent!=ROOT):
                raise PermissionError('Training/sealing cannot read oracle or historical scores: '+str(path))
    sys.addaudithook(guard)
def paid(card,n):
    assert sha(card['path'])==card['sha256']
    with np.load(card['path'],allow_pickle=False) as f:
        assert not set(f.files)&{'y','truth','labels','targets','target','gnd'}
        d={key:f[key].copy() for key in ['pool','test','pairs','answers']}
    pool,pairs,answers=d['pool'],d['pairs'],d['answers']
    assert pool.ndim==1 and np.issubdtype(pool.dtype,np.integer) and len(np.unique(pool))==len(pool)
    assert np.all((pool>=0)&(pool<n)) and np.array_equal(d['test'],np.setdiff1d(np.arange(n),pool))
    assert pairs.shape==(400,2) and np.issubdtype(pairs.dtype,np.integer) and np.isin(pairs,pool).all()
    assert np.all(pairs[:,0]<pairs[:,1]) and len(np.unique(pairs,axis=0))==400
    assert answers.shape==(400,) and np.isin(answers,[-1,1]).all()
    return d
def case_id(ds,seed,Q,ci,fold):return f'{ds}_s{seed}_Q{Q}_c{ci}_'+('final' if fold=='final' else f'f{fold}')
def condition_id(ds,seed,Q):return f'{ds}_s{seed}_Q{Q}'
def sources():
    names=['formal_common.py','prepare_formal.py','derive_runtime.py','run_formal.py','seal_score.py',
        'run_clock_qualification.py','phase_clock.py','owned_processes.py','test_clock.py','test_formal.py',
        'query_protocol.py','metrics_reference.py','pcbg_clock_bridge.m','pcbg_formal_worker.m']
    return {str(ROOT/name):sha(ROOT/name) for name in names}
def protocol(path=None):
    path=Path(path or ROOT/'FROZEN_PROTOCOL.json');p=read(path)
    assert sha(GAP/'FINITE_NATIVE_COMPARISON_DECISION.json')==DECISION_SHA
    assert p['source_hashes']==sources()
    for file,digest in p['evidence_hashes'].items():assert sha(file)==digest,file
    for ds,card in p['datasets'].items():
        for record in card['query_inputs'].values():assert sha(record['path'])==record['sha256']
        if ds in FIT_DATA:
            for file,digest in card['feature_files'].items():assert sha(file)==digest,file
    for c in p['cases']:assert sha(c['input'])==c['sha256'],c['case_id']
    assert len(p['cases'])==216 and len(p['conditions'])==42
    assert p['grid']==GRID and p['counts']==dict(maximum_cv_fits=144,maximum_final_fits=24,maximum_total_fits=168,logical_outcomes=42,unsupported=18,potential_inputs=216)
    return p
def terminal(case):
    out=ROOT/'runs'/case['case_id']/'repeat1'
    r=read(out/'launcher_receipt.json')
    assert r['case_id']==case['case_id'] and r['repeat']==1 and r['input_sha256']==case['sha256']
    assert r['protocol_sha256']==sha(ROOT/'FROZEN_PROTOCOL.json')
    assert r['runtime_protocol_sha256']==sha(ROOT/'RUNTIME_PROTOCOL.json')
    for name,digest in r['output_hashes'].items():assert sha(out/name)==digest,name
    eligible=bool(r['status']=='COMPLETE_ZERO_EXIT_VALID_MARKERS' and r.get('exit_code')==0 and
        r.get('cleanup',{}).get('complete') and r.get('worker',{}).get('benchmark_contract_met'))
    worker=r.get('worker',r.get('unqualified_worker_summary',{}))
    finite=bool(worker.get('finite_partition_retained'))
    z=None
    if finite:
        z=np.asarray(io.loadmat(out/'native_outputs.mat',variable_names=['label'])['label']).ravel()
        assert z.shape==(case['n'],) and np.isfinite(z).all() and np.all(z==np.floor(z)) and np.all(z>=1)
    return r,z,eligible
def outcome(case):
    r,z,eligible=terminal(case)
    return dict(case_id=case['case_id'],attempted=True,eligible=eligible,status=r['status'],
        worker_status=r.get('worker',r.get('unqualified_worker_summary',{})).get('status'),
        finite_partition_retained=z is not None,receipt_sha256=sha(ROOT/'runs'/case['case_id']/'repeat1/launcher_receipt.json'))
def not_attempted(case,reason):return dict(case_id=case['case_id'],attempted=False,eligible=False,status=reason,finite_partition_retained=False)
def admission(path,digest,kind,p_sha,seal_sha=None):
    assert sha(path)==digest,'Admission identity changed'
    a=read(path)
    assert a['passed'] is True and a['kind']==kind and a['protocol_sha256']==p_sha
    if seal_sha is not None:assert a['global_seal_sha256']==seal_sha
    return a
