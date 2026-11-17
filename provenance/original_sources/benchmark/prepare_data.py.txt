"""Explicit JSON source configuration; no machine-specific drive is required."""
from pathlib import Path
import argparse, io, json, tarfile, zipfile
import numpy as np
from scipy import io as sio, sparse
from portable_budget_runner import read,dump,sha
from reproduce import resolve,fresh
from verify_original_sources import minmax

def main(config):
    cp=Path(config).resolve();c=read(cp);kind=c['format'];src=resolve(cp.parent,c['source']);out=fresh(resolve(cp.parent,c['output']))
    hashes={};names=None;ids=None
    if kind=='prepared_directory':
        m=read(src/'metadata.json');X=[sparse.load_npz(src/f'X{i}.npz') for i in range(m['views'])];y=np.load(src/'y.npy',allow_pickle=False)
        for p in [src/'metadata.json',src/'y.npy']+[src/f'X{i}.npz' for i in range(len(X))]:hashes[p.name]=sha(p)
        names=m.get('view_meaning');ids=m.get('sample_ids')
    elif kind=='3sources_original_zip':
        assert sha(src)==c['expected_sha256'];hashes['archive']=sha(src);z=zipfile.ZipFile(src)
        def raw(suffix):
            names=[n for n in z.namelist() if n==suffix or n.endswith('/'+suffix)];assert len(names)==1;return z.read(names[0])
        names=['bbc','guardian','reuters'];docs=[raw('3sources_'+v+'.docs').decode().split() for v in names]
        label={}
        for i,line in enumerate(raw('3sources.disjoint.clist').decode().splitlines()):
            _,values=line.split(':');label.update({v.strip():i for v in values.split(',') if v.strip()})
        ids=sorted(set(docs[0]).intersection(*map(set,docs[1:])));ids=[d for d in ids if d in label]
        X=[]
        for name,documents in zip(names,docs):
            m=sio.mmread(io.BytesIO(raw('3sources_'+name+'.mtx'))).tocsr();index={d:i for i,d in enumerate(documents)}
            X.append(m[:,[index[d] for d in ids]].T.tocsr())
        y=np.array([label[d] for d in ids])
    elif kind=='reuters_grimal_tar':
        assert sha(src)==c['expected_sha256'];hashes['archive']=sha(src);t=tarfile.open(src,'r:gz');sample=int(c.get('sample',0));assert 0<=sample<6
        def raw(suffix):
            names=[n for n in t.getnames() if n==suffix or n.endswith('/'+suffix)];assert len(names)==1;return t.extractfile(names[0]).read()
        names=['EN','FR','GR','IT','SP'];X=[sparse.csr_matrix(minmax(sio.mmread(io.BytesIO(raw(f'reutersEN_{sample}_{v}.mtx'))))) for v in names]
        rows=[raw(f'reutersEN_{sample}_{v}.maprow.txt') for v in names];assert len(set(rows))==1
        ids=rows[0].decode().splitlines();y=np.loadtxt(io.BytesIO(raw('reutersEN_act.txt')),dtype=int)
    elif kind=='synthetic_smoke':
        rng=np.random.RandomState(926);n=60;k=3;y=np.arange(n)%k;X=[]
        for dim in [12,15]:
            centers=rng.gamma(2.,1.,(k,dim));X.append(sparse.csr_matrix(np.maximum(centers[y]+rng.normal(0,.15,(n,dim)),0)))
        names=['synthetic_a','synthetic_b'];hashes['generator']=sha(__file__)
    else:raise ValueError('Unsupported format '+kind)
    _,y=np.unique(np.asarray(y).ravel(),return_inverse=True)
    for i,x in enumerate(X):
        x=sparse.csr_matrix(x,dtype=float);assert x.shape[0]==len(y) and np.isfinite(x.data).all() and (x.data>=0).all()
        x.sum_duplicates();x.eliminate_zeros();x.sort_indices();sparse.save_npz(out/f'X{i}.npz',x)
    np.save(out/'labels_FOR_ORACLE_AND_SEALED_SCORING_ONLY.npy',y,allow_pickle=False)
    dump(out/'metadata.json',dict(dataset=c['dataset'],n=len(y),k=len(np.unique(y)),views=len(X),dimensions=[x.shape[1] for x in X],
        preparation_kind=kind,view_meaning=names,sample_ids=ids,source_hashes=hashes,configuration=c,preparation_source_sha256=sha(__file__),
        training_labels='separate oracle and scoring file; never read by train'))
    dump(out/'INPUT_SHA256.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})
    print('PREPARED',c['dataset'],len(y),len(X),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();main(a.config)
