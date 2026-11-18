"""Portable integrity and external-feature helpers; no fitting or class-label reads."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
from scipy import sparse, io


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def verify_resource(root):
    if not __debug__:
        raise RuntimeError('Run without -O/-OO: integrity assertions are required')
    root=Path(root).resolve()
    expected=(root/'MANIFEST.sha256').read_text(encoding='ascii').split()[0]
    assert sha256(root/'MANIFEST.json')==expected,'Resource manifest identity differs'
    manifest=read_json(root/'MANIFEST.json')
    assert manifest['schema']=='sapn-minimal-diagnostics-v1'
    for name,digest in manifest['files'].items():
        target=(root/name).resolve()
        assert root in target.parents,'Manifest path escapes the resource'
        assert sha256(target)==digest,'Resource input changed: '+name
    return manifest


def canonical_csr_hash(matrix):
    """Digest matrix values and row/feature order independently of NPZ encoding."""
    x=matrix.astype(np.float64).tocsr(copy=True)
    x.sum_duplicates();x.sort_indices();x.eliminate_zeros()
    h=hashlib.sha256(b'sapn-csr-f64-v1\0')
    for a in [np.asarray(x.shape,dtype='<i8'),np.asarray(x.indptr,dtype='<i8'),
              np.asarray(x.indices,dtype='<i8'),np.asarray(x.data,dtype='<f8')]:
        h.update(a.tobytes(order='C'))
    return h.hexdigest()


def reuters_views(path):
    # Request X explicitly; do not deserialize a class vector from the MAT file.
    raw=io.loadmat(path,variable_names=['X'])['X'].ravel()
    views=[]
    for x in raw:
        if x.shape[0]!=1200:x=x.T
        x=x.toarray() if sparse.issparse(x) else np.asarray(x,dtype=float)
        lo=x.min(0);span=x.max(0)-lo
        transformed=(x-lo)/np.where(span>1e-12,span,1)
        transformed[:,span<=1e-12]=0
        views.append(sparse.csr_matrix(transformed))
    return views


def load_features(root,feature_map,dataset):
    """Validate external numerical identity, then return normalized d-by-n views."""
    if isinstance(feature_map,(str,Path)):feature_map=read_json(feature_map)
    specification=feature_map.get('corpora',feature_map)[dataset]
    requirements=read_json(Path(root)/'FEATURE_REQUIREMENTS.json')['corpora'][dataset]
    if isinstance(specification,dict) and 'reuters_mat' in specification:
        assert dataset=='Reuters'
        path=Path(specification['reuters_mat'])
        assert sha256(path)==requirements['historical_files'][0]['sha256'],'Reuters MAT identity mismatch'
        views=reuters_views(path)
    else:
        paths=specification['views'] if isinstance(specification,dict) else specification
        assert len(paths)==len(requirements['views'])
        views=[sparse.load_npz(Path(p)).tocsr() for p in paths]
    assert len(views)==len(requirements['views'])
    output=[]
    for x,expected in zip(views,requirements['views']):
        assert list(x.shape)==expected['shape'],'External view shape/order mismatch'
        assert canonical_csr_hash(x)==expected['canonical_csr_sha256'],'External matrix values/order differ'
        assert np.isfinite(x.data).all() and (x.data>=0).all()
        n=x.shape[0];norm2=float(x.multiply(x).sum());assert norm2>0
        output.append((x*np.sqrt(n/norm2)).T.tocsr())
    return output
