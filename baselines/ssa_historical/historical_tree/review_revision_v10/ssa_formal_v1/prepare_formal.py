"""Prepare an immutable, budget-only SSA comparison; never invoke MATLAB or class truth."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, itertools, json, sys
sys.dont_write_bytecode = True
import numpy as np
from scipy import io

ROOT = Path(__file__).resolve().parent
MAIN = ROOT.parent.parent
OLD = MAIN/'review_revision_v9/baseline_gap/native_formal_v1'
BRIDGE = ROOT.parent/'ssa_trace_bridge'
DESIGN = ROOT.parent/'ssa_formal_design/PROTOCOL_RECOMMENDATION.json'
OLD_SHA = '4a334d1319a17b3d8a266962267c6e19cd2d7ecd9bfff2a78c0afea2b608b117'
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20), b''): h.update(b)
    return h.hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def exclusive(p, obj):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
def copy_exact(name):
    with (ROOT/name).open('xb') as f: f.write((OLD/name).read_bytes())
def source_hashes():
    files = list(ROOT.glob('*.py'))+list(ROOT.glob('*.m'))+list((ROOT/'private_generated').glob('*.m'))
    return {str(p.resolve()):sha(p) for p in sorted(files)}
def prepare():
    assert __debug__, 'Do not disable assertions'
    assert sha(OLD/'FROZEN_PROTOCOL.json') == OLD_SHA
    old = read(OLD/'FROZEN_PROTOCOL.json')
    assert not (ROOT/'FROZEN_PROTOCOL.json').exists()
    assert (ROOT/'WORKER_PREPARATION.json').is_file()
    for name in ['phase_clock.py', 'owned_processes.py', 'query_protocol.py']: copy_exact(name)
    rt = (OLD/'run_clock_qualification.py').read_text(encoding='utf-8')
    rt = rt.replace('pcbg_clock_bridge(', 'ssa_clock_bridge(')
    needle = "assert sha(CURRENT_CASE['input']) == CURRENT_CASE['sha256'], CURRENT_CASE['input']"
    assert rt.count(needle) == 1
    rt = rt.replace(needle, needle+"\n        assert sha(CURRENT_CASE['source_input']) == CURRENT_CASE['source_input_sha256'], CURRENT_CASE['source_input']")
    start = rt.index('def environment_gate(p):')
    end = rt.index('\ndef remember(', start)
    rt = rt[:start]+'''def environment_gate(p):
    e = RUNTIME['bridge_evidence']
    result, receipt = read(e['result']), read(e['receipt'])
    passed = (sha(e['result']) == e['result_sha256'] and sha(e['receipt']) == e['receipt_sha256']
        and result.get('passed') is True and result.get('class_truth_loaded') is False
        and result.get('ACC_NMI_ARI_computed') is False and receipt.get('exit_code') == 0
        and receipt.get('status') == 'RETURNED')
    return dict(passed=passed, scope='Actual completed trace bridge in the same MATLAB environment')

''' + rt[end:]
    rt = rt.replace('Original scientific protocol, author code and MATLAB worker are not changed.',
                    'SSA finite-profile worker and exact new inputs are bound by this protocol.')
    with (ROOT/'run_clock_qualification.py').open('x', encoding='utf-8', newline='\n') as f: f.write(rt)
    import query_protocol as qp
    from formal_common import GRID, METHOD, paid, no_oracle_guard, case_id
    no_oracle_guard()
    expected = [dict(beta=b,lam=l,kap=k) for b,l,k in itertools.product([.1,.2,.3],[.2,.3],[.8,.9,.95])]
    default = dict(beta=.1,lam=.2,kap=.95)
    expected.remove(default); expected.insert(0,default)
    assert GRID == expected and len(GRID) == 18
    result = BRIDGE/'bridge_run/TRACE_BRIDGE_RESULT.json'
    receipt = BRIDGE/'bridge_run/EXECUTION_RECEIPT.json'
    assert read(result)['passed'] and read(receipt)['exit_code'] == 0
    inputs = ROOT/'inputs'; inputs.mkdir()
    old_cases = {c['case_id']:c for c in old['cases']}
    conditions = old['conditions']; cases = []; sources = {}; query_sources = {}
    for cond in conditions:
        query_sources[cond['query_input']['path']] = cond['query_input']['sha256']
        d = paid(cond['query_input'],cond['n'])
        pairs, answers = d['pairs'][:cond['Q']], d['answers'][:cond['Q']]
        all_folds = qp.node_folds(d['pool'], pairs, cond['fold_partition_seed'])
        folds = qp.eligible_folds(all_folds, answers)
        serial_folds = lambda fs: [{k:v.tolist() for k,v in f.items()} for f in fs]
        assert serial_folds(folds) == cond['eligible_folds'] and serial_folds(all_folds) == cond['all_folds']
        if cond['unsupported']:
            assert cond['unsupported_reason'] == 'native_zero_row_normalization_unsupported'
            continue
        assert len(folds) == 2
        for fold in [0,1,'final']:
            source = old_cases[case_id(cond['dataset'],cond['seed'],cond['Q'],0,fold)]
            sp = Path(source['input']); assert sha(sp) == source['sha256']
            assert set(n for n,_,_ in io.whosmat(sp)) == {'X','c','ML','CL','seed','config','case_id'}
            m = io.loadmat(sp, variable_names=['c','ML','CL','seed'])
            assert int(m['c'].item()) == cond['c'] and int(m['seed'].item()) == cond['fit_seed']
            ix = np.arange(cond['Q']) if fold == 'final' else np.asarray(folds[fold]['training'],int)
            ml, cl = qp.signed_pairs(pairs[ix],answers[ix])
            assert np.array_equal(m['ML'], ml+1) and np.array_equal(m['CL'], cl+1)
            sources[str(sp)] = source['sha256']
            for ci, config in enumerate(GRID):
                cid = case_id(cond['dataset'],cond['seed'],cond['Q'],ci,fold)
                target = inputs/(cid+'.mat')
                io.savemat(target, dict(source_input=str(sp),source_sha256=source['sha256'],config=config,case_id=cid), do_compression=False)
                req = io.loadmat(target, simplify_cells=True)
                assert req['source_input'] == str(sp) and req['source_sha256'] == source['sha256'] and req['config'] == config and req['case_id'] == cid
                c = {k:v for k,v in source.items() if k not in ['tau','input','sha256','case_id','candidate']}
                c.update(case_id=cid,input=str(target.resolve()),sha256=sha(target),source_input=str(sp),
                         source_input_sha256=source['sha256'],candidate=ci,config=config)
                cases.append(c)
    assert len(cases)==1296 and len(sources)==72 and len(conditions)==42
    evidence = [OLD/'FROZEN_PROTOCOL.json', DESIGN, ROOT/'WORKER_PREPARATION.json',
                BRIDGE/'PREPARATION.json', result, receipt]
    p = dict(status='PREPARED_PENDING_ROOT_FREEZE',prepared_utc=datetime.now(timezone.utc).isoformat(),
        method=METHOD,grid=GRID,conditions=conditions,cases=cases,base_inputs=sources,query_sources=query_sources,
        datasets=old['datasets'],
        source_hashes=source_hashes(),evidence_hashes={str(x.resolve()):sha(x) for x in evidence},
        counts=dict(maximum_cv_fits=864,maximum_final_fits=24,maximum_total_fits=888,logical_outcomes=42,unsupported=18,potential_inputs=1296),
        author_root=str((ROOT/'private_generated').resolve()),matlab_executable=old['matlab_executable'],
        runtime_environment=old['runtime_environment'],runtime=old['runtime'],global_wall_seconds=48*3600,
        execution_order='seed23,24,25; Q160,400; 3Sources,BBC,Cora,CiteSeer; fold then candidate; selected final',
        randomization=old['randomization'],selection=old['selection'],
        preprocessing='Original features, per-object L2 normalization inside source profile; known k; zero-row corpora unsupported without imputation or object deletion.',
        budget_comparability='Same query stream, folds and BER selector. Eighteen SSA candidates is a disclosed limited grid, not the same search size as other methods or exhaustive tuning.',
        adaptation='Anonymous-pair function adapter, missing LM definition and pairwise CL indexing repairs; scalar trace evaluated elementwise; otherwise source updates unchanged.',
        numerical_eligibility='Finite correctly shaped model states; nonnegative primal states, positive consensus row mass, exactly known k nonempty hard clusters. Hard orthogonality/simplex residuals diagnostic only. cap500 allowed but not evidence of convergence.',
        artifact_scope='Compact memberships, labels, full objective/diagnostic trajectories and live-state eligibility witness. Dense fitted states are checked in memory but not retained; not a full-state replay package.',
        score_boundary='All predictions and outcomes sealed before any new full-class scoring. Failed, unsupported and unattempted cases cannot count as SAPN wins.',
        resource_scope='One hidden worker, thread1, 600 seconds active per fit, 300 startup/60 shutdown. Total48h may yield incomplete CV; never select from a partial grid.',
        no_class_truth_or_accuracy_read=True)
    exclusive(ROOT/'FROZEN_PROTOCOL.json',p)
    runtime = dict(original_scientific_protocol_sha256=sha(ROOT/'FROZEN_PROTOCOL.json'),
        bound_files=p['source_hashes'],bridge_evidence=dict(result=str(result.resolve()),result_sha256=sha(result),
        receipt=str(receipt.resolve()),receipt_sha256=sha(receipt)),global_wall_seconds=p['global_wall_seconds'])
    exclusive(ROOT/'RUNTIME_PROTOCOL.json',runtime)
    print(json.dumps(dict(status='PREPARED_NOT_EXECUTED',protocol_sha256=sha(ROOT/'FROZEN_PROTOCOL.json'),runtime_sha256=sha(ROOT/'RUNTIME_PROTOCOL.json'),counts=p['counts']),indent=2))

if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare', action='store_true');args=parser.parse_args()
    if args.prepare: prepare()
    else: print('Read-only: pass --prepare once after worker and trace bridge are complete.')
