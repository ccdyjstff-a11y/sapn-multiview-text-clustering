"""Relocated, isolated-process comparison of original and English common solvers.

Synthetic inputs test numerical equivalence, not scientific performance.
The temporary historical Python copies are removed when the test finishes.
"""
from pathlib import Path
import argparse, hashlib, json, os, shutil, subprocess, sys, tempfile
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import numpy as np
from scipy import sparse

ROOT=Path(__file__).resolve().parents[1]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    if a.output.exists():raise ValueError('Use a new report file')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    report=dict(scope='Synthetic relocated execution only; no corpus performance claims',commands=[],comparisons=[])
    with tempfile.TemporaryDirectory(prefix='.qa_runtime_',dir=ROOT) as tmp:
        t=Path(tmp); hist=t/'historical';current=t/'english';hist.mkdir();current.mkdir()
        for f in (ROOT/'common').glob('*'):
            if f.is_file():shutil.copyfile(f,current/f.name)
        for f in (ROOT/'provenance/original_sources/common').glob('*.py.txt'):
            shutil.copyfile(f,hist/f.name[:-4])
        for name in ['CORE_DERIVATION.json','SOURCE_PROVENANCE.json']:
            shutil.copyfile(ROOT/'provenance/historical_common'/name,hist/name)
        data=t/'prepared';data.mkdir();seed=17;n=60;k=3;rng=np.random.RandomState(926)
        y=np.arange(n)%k
        for v,dim in enumerate([12,15]):
            centers=rng.gamma(2.,1.,(k,dim));x=np.maximum(centers[y]+rng.normal(0,.15,(n,dim)),0)
            sparse.save_npz(data/f'X{v}.npz',sparse.csr_matrix(x))
        (data/'metadata.json').write_text(json.dumps(dict(dataset='Synthetic',n=n,k=k,views=2)),encoding='utf-8')
        np.save(t/'labels.npy',y)
        pool=np.sort(np.random.RandomState(seed+21000).permutation(n)[:round(.3*n)])
        pairs=np.array([(int(i),int(j)) for i in pool for j in pool if i<j],dtype=int)
        pairs=pairs[np.random.RandomState(seed+31000).permutation(len(pairs))][:100]
        answers=np.where(y[pairs[:,0]]==y[pairs[:,1]],1,-1)
        np.savez_compressed(t/'queries.npz',pool=pool,test=np.setdiff1d(np.arange(n),pool),pairs=pairs,answers=answers)
        def execute(cmd,name):
            run=subprocess.run(cmd,cwd=t,env=dict(os.environ,PYTHONPATH=''),capture_output=True,text=True,encoding='utf-8',errors='replace')
            report['commands'].append(dict(name=name,returncode=run.returncode,stdout=run.stdout,stderr=run.stderr,isolated=True))
            if run.returncode:raise RuntimeError(name+' failed\n'+run.stderr)
        for label,source in [('historical',hist),('english',current)]:
            execute([sys.executable,'-B','-I',str(source/'run_allocation.py'),'--prepared-directory',str(data),
              '--paid-queries',str(t/'queries.npz'),'--seed',str(seed),'--budgets','100','--output',str(t/(label+'_run'))],label+'_train')
        for label,source in [('historical',hist),('english',current)]:
            execute([sys.executable,'-B','-I',str(source/'score_allocation.py'),'--output',str(t/(label+'_run')),'--labels',str(t/'labels.npy')],label+'_score')
        before=t/'historical_run';after=t/'english_run';arrays=0;json_checks=0
        for f in sorted(before.glob('*.npz')):
            g=after/f.name
            with np.load(f,allow_pickle=False) as x,np.load(g,allow_pickle=False) as y2:
                assert set(x.files)==set(y2.files),f.name
                for key in x.files:
                    assert np.array_equal(x[key],y2[key]),(f.name,key)
                    arrays+=1
        def cleaned(obj):
            if isinstance(obj,dict):return {k:cleaned(v) for k,v in obj.items() if k not in ['seconds','prediction_sha256','state_sha256','identity_sha256']}
            if isinstance(obj,list):return [cleaned(x) for x in obj]
            return obj
        for f in sorted(before.glob('*_selection.json'))+sorted(before.glob('*_diagnostics.json'))+[before/'evaluation.json']:
            assert cleaned(json.loads(f.read_text(encoding='utf-8')))==cleaned(json.loads((after/f.name).read_text(encoding='utf-8'))),f.name
            json_checks+=1
        rows=json.loads((after/'evaluation.json').read_text(encoding='utf-8'))
        assert {r['allocation']['eta'] for r in rows}=={0.,.5,1.}
        report.update(status='PASS',arrays_bitwise_equal=arrays,numerical_json_equal=json_checks,
            logical_finals=len(rows),etas=[0,.5,1],eligible_finals=sum(bool(r['converged']) for r in rows),
            changed_fields='Source/protocol/identity hashes and elapsed seconds are expected to differ; arrays and numerical diagnostics do not.',
            original_core_sha256=sha(hist/'allocation_core.py'),english_core_sha256=sha(current/'allocation_core.py'))
    a.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='commands'}))

if __name__=='__main__':main()
