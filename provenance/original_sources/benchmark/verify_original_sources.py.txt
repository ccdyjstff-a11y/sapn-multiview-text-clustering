"""Compare local training representations with original publishers' archives.

Reads archives directly, never extracts arbitrary archive paths, changes no
training input and copies no corpus into the distributable reproduction code.
"""
from pathlib import Path
import argparse, hashlib, io, json, tarfile, zipfile, sys, importlib.util
import numpy as np
from scipy import io as sio, sparse

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrsha(x):
    x = np.ascontiguousarray(x, dtype=np.float64)
    return hashlib.sha256(str(x.shape).encode()+x.tobytes()).hexdigest()
def minmax(x):
    x = x.toarray() if sparse.issparse(x) else np.asarray(x, float)
    lo = x.min(0); span = x.max(0)-lo
    z = (x-lo)/np.where(span>1e-12,span,1)
    z[:,span<=1e-12] = 0
    return z

def audit_3sources(archive, local, prepared=None):
    z = zipfile.ZipFile(archive)
    def content(suffix):
        names = [n for n in z.namelist() if n.endswith('/'+suffix) or n == suffix]
        assert len(names)==1, suffix
        return z.read(names[0])
    names = ['bbc','guardian','reuters']
    views=[]; ids=[]
    for name in names:
        views.append(sio.mmread(io.BytesIO(content('3sources_'+name+'.mtx'))).tocsr())
        ids.append(content('3sources_'+name+'.docs').decode().split())
    mapping={}; category_names=[]
    for c,line in enumerate(content('3sources.disjoint.clist').decode().splitlines()):
        label,items=line.split(':'); category_names.append(label.strip())
        mapping.update({item.strip():c for item in items.split(',') if item.strip()})
    common=sorted(set(ids[0]).intersection(*map(set,ids[1:])))
    common=[doc for doc in common if doc in mapping]
    original=[mat[:,[{doc:i for i,doc in enumerate(docids)}[doc] for doc in common]].T.toarray()
              for mat,docids in zip(views,ids)]
    y=np.array([mapping[d] for d in common])
    with np.load(local,allow_pickle=False) as d:
        tests=[dict(view=i,source=names[i],shape=list(x.shape),equal=np.array_equal(x,d[f'X{i}']),
                    max_abs_error=float(np.max(np.abs(x-d[f'X{i}']))),array_sha256=arrsha(x)) for i,x in enumerate(original)]
        labels_equal=np.array_equal(y,d['y'])
    prepared_report=None
    if prepared:
        prepared=Path(prepared)
        prepared_report=dict(feature_files=[dict(name=f'X{i}.npz',sha256=sha(prepared/f'X{i}.npz'),
             equal_original=np.array_equal(sparse.load_npz(prepared/f'X{i}.npz').toarray(),x)) for i,x in enumerate(original)],
             labels_file_sha256=sha(prepared/'y.npy'),labels_equal_original=np.array_equal(np.load(prepared/'y.npy'),y),
             metadata_sha256=sha(prepared/'metadata.json'))
        assert all(r['equal_original'] for r in prepared_report['feature_files']) and prepared_report['labels_equal_original']
    return dict(status='EXACT_MATCH' if all(t['equal'] for t in tests) and labels_equal else 'MISMATCH',
        source_url='http://mlg.ucd.ie/files/datasets/3sources.zip', source_page='http://mlg.ucd.ie/datasets/3sources.html',
        archive_sha256=sha(archive),local_sha256=sha(local),views=tests,labels_equal=labels_equal,
        row_order='lexicographically sorted intersection of story IDs, retain disjoint labels',
        prepared_training_inputs=prepared_report,story_ids=common,categories=category_names,story_ids_sha256=hashlib.sha256('\n'.join(common).encode()).hexdigest(),
        scope='Exact reconstruction of three aligned numerical matrices and label vector; does not independently re-annotate original stories.')

def audit_reuters(archive,local,experiment=None):
    d=sio.loadmat(local);y=np.asarray(d['y']).ravel();X=[a.toarray() if sparse.issparse(a) else np.asarray(a) for a in d['X'].ravel()]
    X=[a if a.shape[0]==len(y) else a.T for a in X]
    transform=minmax; actual_transform=None
    if experiment:
        experiment=Path(experiment).resolve();sys.path.insert(0,str(experiment.parent))
        spec=importlib.util.spec_from_file_location('actual_experiment',experiment);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        transform=module.minmax
        actual_transform=dict(source_sha256=sha(experiment),function='experiment.minmax',
                             matches_independent_transform=all(np.array_equal(transform(x).toarray(),minmax(x)) for x in X))
        assert actual_transform['matches_independent_transform']
    raw_hash=[arrsha(x) for x in X];scaled_hash=[arrsha(transform(x).toarray() if sparse.issparse(transform(x)) else transform(x)) for x in X]
    tar=tarfile.open(archive,'r:gz')
    def content(suffix):
        members=[n for n in tar.getnames() if n.endswith('/'+suffix) or n==suffix];assert len(members)==1,suffix
        return tar.extractfile(members[0]).read()
    yo=np.loadtxt(io.BytesIO(content('reutersEN_act.txt')),dtype=int).ravel()
    labels_equiv=np.array_equal(np.unique(y,return_inverse=True)[1],np.unique(yo,return_inverse=True)[1])
    candidates=[]
    for sample in range(6):
        for language in ['EN','FR','GR','IT','SP']:
            stem=f'reutersEN_{sample}_{language}'
            raw=sio.mmread(io.BytesIO(content(stem+'.mtx'))).toarray()
            rh=arrsha(raw);sh=arrsha(minmax(raw))
            candidates.append(dict(sample=sample,language=language,
                raw_equal_local_views=[i for i,h in enumerate(raw_hash) if h==rh],
                prepared_equal_local_views=[i for i,h in enumerate(scaled_hash) if h==sh],
                raw_sha256=rh,prepared_sha256=sh,maprow_sha256=hashlib.sha256(content(stem+'.maprow.txt')).hexdigest()))
    matches=[c for c in candidates if c['raw_equal_local_views'] or c['prepared_equal_local_views']]
    covered=sorted(set(i for c in matches for i in c['prepared_equal_local_views']))
    return dict(status='PREPARED_EXACT_MATCH' if covered==list(range(5)) and labels_equiv else 'NO_COMPLETE_MATCH',
        source_url='https://lig-membres.imag.fr/grimal/data/ReutersEN.tar.gz',
        source_readme='https://lig-membres.imag.fr/grimal/data/ReutersEN-readme.txt',
        archive_sha256=sha(archive),local_sha256=sha(local),label_partition_equal=labels_equiv,actual_training_transform=actual_transform,
        matches=matches,all_candidates=candidates,
        scope='Compared all six published samples and five named languages, both raw and actual featurewise min-max transforms. No inference from shape alone.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--three-archive',type=Path,required=True);p.add_argument('--three-local',type=Path,required=True)
    p.add_argument('--reuters-archive',type=Path,required=True);p.add_argument('--reuters-local',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prepared-three',type=Path);p.add_argument('--actual-experiment',type=Path);a=p.parse_args()
    report=dict(three_sources=audit_3sources(a.three_archive,a.three_local,a.prepared_three),reuters=audit_reuters(a.reuters_archive,a.reuters_local,a.actual_experiment),auditor_sha256=sha(__file__))
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v['status'] for k,v in report.items() if isinstance(v,dict)}))
if __name__=='__main__': main()
