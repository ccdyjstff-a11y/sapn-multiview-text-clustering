"""Portable oracle, training and sealed scoring for declared common-candidate studies."""
from pathlib import Path
import argparse, hashlib, json, os, subprocess, sys

ROOT=Path(__file__).resolve().parents[1]
PRESETS={
    'historical':dict(corpora=['3Sources','BBC','Reuters','Cora','CiteSeer'],seeds=list(range(504,514)),budgets=[160,400]),
    'allocation-extension':dict(corpora=['20NG','AGNews'],seeds=list(range(1101,1106)),budgets=[400]),
}

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def immutable(p,value):
    if p.exists():
        if read(p)!=value:raise ValueError('Frozen study record changed: '+str(p))
    else:
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
def mapped(mapping,key,base):
    q=Path(mapping[key]);return q.resolve() if q.is_absolute() else (base/q).resolve()
def integers(value):
    answer=[]
    for item in value.split(','):
        if ':' in item:
            start,stop=map(int,item.split(':'));answer.extend(range(start,stop+1))
        else:answer.append(int(item))
    if not answer or answer!=sorted(set(answer)) or min(answer)<0:raise ValueError('Sorted unique nonnegative integers required')
    return answer
def settings(a):
    override=dict(corpora=a.corpora.split(',') if a.corpora else None,
        seeds=integers(a.seeds) if a.seeds else None,budgets=integers(a.budgets) if a.budgets else None)
    if a.profile=='custom':
        if not all(override.values()):raise ValueError('Custom studies require --corpora, --seeds and --budgets')
        config=override
    else:
        config=PRESETS[a.profile]
        if any(v is not None and v!=config[k] for k,v in override.items()):
            raise ValueError('Overrides differ from the named protocol; use --profile custom')
    if len(set(config['corpora']))!=len(config['corpora']):raise ValueError('Duplicate corpus')
    if any(not d or any(c in d for c in '/\\:') or d in ['.','..'] for d in config['corpora']):raise ValueError('Corpus names must be safe directory names')
    return config
def check_data(a,config):
    data=read(a.data_map)
    if set(data)!=set(config['corpora']):raise ValueError('Data-map keys must equal the declared corpus set')
    folders={d:mapped(data,d,a.data_map.resolve().parent) for d in config['corpora']}
    identities={}
    for d,folder in folders.items():
        meta=read(folder/'metadata.json')
        names=['metadata.json']+[f'X{v}.npz' for v in range(meta['views'])]
        names.extend(n for n in ['MV_graph.npz','initialization.npz'] if (folder/n).exists())
        if (folder/'initialization.npz').exists() and len(config['seeds'])>1:
            raise ValueError('One saved initialization cannot represent multiple requested seeds')
        identities[d]={n:digest(folder/n) for n in names}
    return folders,identities
def oracle(a,config,folders,queries):
    """Separate full-label boundary: answer only the pre-sampled unique pairs."""
    import numpy as np
    sys.path.insert(0,str(ROOT/'common'))
    from query_protocol import sample_query_stream,answer_queries
    labels=read(a.labels_map)
    if set(labels)!=set(config['corpora']):raise ValueError('Label-map keys must equal the declared corpus set')
    targets=[queries/d/f'seed{s}.npz' for d in config['corpora'] for s in config['seeds']]
    if any(p.exists() or p.with_suffix('.oracle.json').exists() for p in targets):raise ValueError('Oracle requires fresh query paths')
    for d in config['corpora']:
        meta=read(folders[d]/'metadata.json');labelpath=mapped(labels,d,a.labels_map.resolve().parent)
        y=np.load(labelpath,allow_pickle=False).reshape(-1);n=meta['n']
        if len(y)!=n or len(np.unique(y))!=meta['k']:raise ValueError('Label count or cluster count differs from prepared metadata')
        for seed in config['seeds']:
            pool=np.sort(np.random.RandomState(seed+21000).permutation(n)[:round(.3*n)])
            pairs=sample_query_stream(pool,max(config['budgets']),seed+31000)
            target=queries/d/f'seed{seed}.npz';target.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(target,pool=pool,pairs=pairs,answers=answer_queries(pairs,y),test=np.setdiff1d(np.arange(n),pool))
            immutable(target.with_suffix('.oracle.json'),dict(dataset=d,seed=seed,query_sha256=digest(target),
                labels_sha256=digest(labelpath),query_protocol_sha256=digest(ROOT/'common/query_protocol.py'),
                sampling='30% label-blind pool seed+21000; unique uniform unordered stream seed+31000',
                labels_used_only_to_answer_sampled_pairs=True))
    print('ORACLE',len(targets),'paid streams; no fit or evaluation performed')
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['oracle','plan','train','score'])
    p.add_argument('--profile',choices=[*PRESETS,'custom'],default='historical')
    p.add_argument('--corpora',help='Comma-separated exact keys; required for custom studies')
    p.add_argument('--seeds',help='Sorted comma-separated seeds or inclusive start:stop range')
    p.add_argument('--budgets',help='Sorted comma-separated query budgets')
    p.add_argument('--data-map',required=True,type=Path,help='JSON mapping corpus keys to prepared directories')
    p.add_argument('--labels-map',type=Path,help='Allowed only for oracle and score')
    p.add_argument('--queries-root',type=Path,help='Paid streams: <root>/<corpus>/seed<seed>.npz; required outside the historical preset')
    p.add_argument('--output',required=True,type=Path,help='Study output directory; not a prepared directory')
    a=p.parse_args();config=settings(a);out=a.output.resolve()
    if a.phase in ['train','plan'] and a.labels_map is not None:raise ValueError('Training and planning do not accept labels')
    if a.phase in ['oracle','score'] and a.labels_map is None:raise ValueError('--labels-map is required for oracle and score')
    if a.queries_root is None and (a.profile!='historical' or a.phase=='oracle'):
        raise ValueError('Specify --queries-root for new streams or nonhistorical studies')
    queries=a.queries_root.resolve() if a.queries_root else ROOT/'evidence/v9/allocation_repeat/paid_queries'
    folders,identities=check_data(a,config)
    if out in folders.values() or out==ROOT:raise ValueError('Use a separate study output directory')
    if a.phase=='oracle':oracle(a,config,folders,queries);return
    blocks=[(d,s,out/d/f'seed{s}') for d in config['corpora'] for s in config['seeds']]
    query_hashes={f'{d}/seed{s}.npz':digest(queries/d/f'seed{s}.npz') for d,s,_ in blocks}
    sources={f.name:digest(f) for f in sorted((ROOT/'common').iterdir()) if f.suffix in ['.py','.json']}
    plan=dict(profile=a.profile,**config,expected_blocks=len(blocks),expected_logical_finals=len(blocks)*5*len(config['budgets']),
        input_sha256=identities,paid_query_sha256=query_hashes,common_source_sha256=sources,
        launcher_sha256=digest(Path(__file__)),eta=[0.,.5,1.],lambda_grid_default_first=[20.,2.,200.],
        selection='Original common candidates, eligible node-disjoint two-fold mean BER and default-first ties',
        gate='Every declared block must seal before any evaluation; incomplete blocks require inspection, not replacement',
        compute='Sequential workers; one BLAS thread; extension starts only with >=3 GiB available RAM',
        boundary='Prepared views and paid pair answers enter fitting. Oracle and post-seal labels are separate commands.')
    protocol=out/'STUDY_PROTOCOL.json';seal=out/'STUDY_PREDICTIONS_SEALED.json'
    if a.phase=='plan':print(json.dumps(plan,indent=2));return
    if a.phase=='train':
        if seal.exists():raise ValueError('Study already sealed; use score or a fresh output')
        immutable(protocol,plan)
        sys.path.insert(0,str(ROOT/'common'))
        from allocation_io import validate_seal
        env=dict(os.environ)
        for name in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[name]='1'
        for d,seed,block in blocks:
            if (block/'GLOBAL_PREDICTIONS_SEALED.json').exists():validate_seal(block);continue
            if block.exists() and any(block.iterdir()):raise ValueError('Unsealed partial block requires explicit inspection: '+str(block))
            if a.profile=='allocation-extension':
                import psutil
                if psutil.virtual_memory().available<3*1024**3:raise RuntimeError('Resource gate: less than 3 GiB available; no new block started')
            subprocess.run([sys.executable,'-B','-I',str(ROOT/'common/run_allocation.py'),
                '--prepared-directory',str(folders[d]),'--paid-queries',str(queries/d/f'seed{seed}.npz'),
                '--seed',str(seed),'--budgets',','.join(map(str,config['budgets'])),'--output',str(block)],check=True,env=env)
        records=[]
        for d,seed,block in blocks:
            _,rows=validate_seal(block)
            if len(rows)!=5*len(config['budgets']):raise ValueError('Incomplete logical finals')
            records.append(dict(dataset=d,seed=seed,block=block.relative_to(out).as_posix(),
                logical_finals=len(rows),seal_sha256=digest(block/'GLOBAL_PREDICTIONS_SEALED.json')))
        immutable(seal,dict(protocol_sha256=digest(protocol),expected_blocks=len(blocks),blocks=records,evaluation_read=False))
        print('STUDY_SEALED',len(records),'blocks',plan['expected_logical_finals'],'logical finals')
    else:
        if read(protocol)!=plan:raise ValueError('Requested scoring plan/source/input differs from frozen training plan')
        frozen=read(seal)
        if frozen['protocol_sha256']!=digest(protocol):raise ValueError('Study protocol changed')
        expected=[(d,s,b.relative_to(out).as_posix()) for d,s,b in blocks]
        if [(r['dataset'],r['seed'],r['block']) for r in frozen['blocks']]!=expected:raise ValueError('Incomplete or changed block roster')
        sys.path.insert(0,str(ROOT/'common'))
        from allocation_io import validate_seal
        for _,_,block in blocks:validate_seal(block)
        for r in frozen['blocks']:
            if digest(out/r['block']/'GLOBAL_PREDICTIONS_SEALED.json')!=r['seal_sha256']:raise ValueError('Block seal changed')
        labels=read(a.labels_map)
        if set(labels)!=set(config['corpora']):raise ValueError('Label-map keys must equal the declared corpus set')
        rows=[]
        for d,_,block in blocks:
            subprocess.run([sys.executable,'-B','-I',str(ROOT/'common/score_allocation.py'),
                '--output',str(block),'--labels',str(mapped(labels,d,a.labels_map.resolve().parent))],check=True)
            rows.extend(read(block/'evaluation.json'))
        if len(rows)!=plan['expected_logical_finals']:raise ValueError('Incomplete evaluation; no reduced complete-case summary written')
        immutable(out/'STUDY_EVALUATION.json',rows)
        print('SCORED',len(rows),'logical finals; existing selections retained')

if __name__=='__main__':main()
