"""Verify and rescore the delivered primary predictions using external labels.

Usage: python -I rescore.py --labels-map labels_map.json --output audit.json
The JSON maps each manifest dataset to an .npy path, or to {"path":...,"key":"y"}
for a MAT file. Relative paths resolve against labels_map.json. No fitting occurs.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
from scipy.io import loadmat
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

ROOT=Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf8'))
def partition_hash(y):
    mapping={}; z=[]
    for label in np.asarray(y).ravel().tolist():
        if label not in mapping: mapping[label]=len(mapping)
        z.append(mapping[label])
    return hashlib.sha256(np.asarray(z,dtype='<i8').tobytes()).hexdigest()

def run(labels_map,output):
    labels_map=Path(labels_map).resolve(); label_paths=read(labels_map)
    audit=read(ROOT/'EXPORT_AUDIT.json')
    if sha(ROOT/'MANIFEST.json') != audit['manifest_sha256']: raise ValueError('Manifest hash mismatch')
    manifest=read(ROOT/'MANIFEST.json'); results=[]; cache={}
    if len(manifest['records']) != 210: raise ValueError('Expected 210 primary records')
    for record in manifest['records']:
        ds=record['dataset']; n=record['n']; seed=record['seed']
        if ds not in cache:
            entry=label_paths[ds]; spec={'path':entry} if isinstance(entry,str) else entry
            p=Path(spec['path']); p=p if p.is_absolute() else labels_map.parent/p
            y=loadmat(p,variable_names=[spec.get('key','y')])[spec.get('key','y')].ravel() if p.suffix.lower()=='.mat' else np.load(p,allow_pickle=False).ravel()
            if len(y)!=n or partition_hash(y)!=record['partition_sha256']:
                raise ValueError(f'Wrong label partition or document order: {ds}')
            cache[ds]=y
        y=cache[ds]; p=(ROOT/record['file']).resolve()
        if ROOT not in p.parents or sha(p)!=record['sha256']: raise ValueError(f'Prediction path/hash mismatch: {ds}')
        with np.load(p,allow_pickle=False) as f:
            z=f['labels']; pool=f['query_pool']; test=f['evaluation_indices']
        expected_pool=np.sort(np.random.RandomState(seed+21000).permutation(n)[:round(.3*n)])
        expected_test=np.setdiff1d(np.arange(n),expected_pool)
        if not np.array_equal(pool,expected_pool) or not np.array_equal(test,expected_test):
            raise ValueError(f'Evaluation split mismatch: {ds}/{seed}')
        if z.shape!=(n,) or not np.issubdtype(z.dtype,np.integer): raise ValueError('Invalid predicted labels')
        yt=y[test]; zt=z[test]
        _,a=np.unique(yt,return_inverse=True); _,b=np.unique(zt,return_inverse=True)
        t=np.zeros((a.max()+1,b.max()+1),dtype=np.int64); np.add.at(t,(a,b),1)
        r,c=linear_sum_assignment(-t)
        got=dict(ACC=float(t[r,c].sum()/len(yt)),NMI=float(normalized_mutual_info_score(yt,zt,average_method='arithmetic')),ARI=float(adjusted_rand_score(yt,zt)))
        error=max(abs(got[k]-record['metrics'][k]) for k in got)
        if error>=1e-12: raise ValueError(f'Metric mismatch: {ds}/{record["method"]}/{seed}/{record["Q"]}')
        results.append(dict(dataset=ds,method=record['method'],Q=record['Q'],seed=seed,eligible=record['eligible'],status=record['status'],metrics=got,max_abs_error=error))
    grouped=[]
    for ds,method,q in sorted({(r['dataset'],r['method'],r['Q']) for r in results}):
        cell=[r for r in results if (r['dataset'],r['method'],r['Q'])==(ds,method,q)]
        valid=[r for r in cell if r['eligible']]
        item=dict(dataset=ds,method=method,Q=q,planned=3,eligible=len(valid),complete=len(valid)==3)
        if len(valid)==3:
            item['mean_percent']={k:float(np.mean([r['metrics'][k]*100 for r in valid])) for k in ['ACC','NMI','ARI']}
            item['sample_sd_percent']={k:float(np.std([r['metrics'][k]*100 for r in valid],ddof=1)) for k in ['ACC','NMI','ARI']}
        grouped.append(item)
    report=dict(status='PASS',scope='Historical prediction rescoring, not optimizer rerunning',predictions=len(results),metric_comparisons=3*len(results),
        max_abs_error=max(r['max_abs_error'] for r in results),manifest_sha256=sha(ROOT/'MANIFEST.json'),scorer_sha256=sha(Path(__file__)),records=results,table_cells=grouped)
    Path(output).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print({k:v for k,v in report.items() if k not in ['records','table_cells']})

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--labels-map',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.labels_map,a.output)
