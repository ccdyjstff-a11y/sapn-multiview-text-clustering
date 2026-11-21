"""Inspect or generate 45 SSA carriers and five PCBG inputs; never launch MATLAB."""
import argparse
from pathlib import Path
import shutil
import numpy as np
from scipy import io, sparse
from completion_common import (HERE, require, read, sha, immutable, module, load_spec,
                               own_sources, case_id, plain_folds, relative, identity)

FIELDS = {'X', 'c', 'ML', 'CL', 'seed', 'config', 'case_id'}

def extended_launcher_bytes(original):
    """Exactly the two reviewed clock changes; no optimizer or worker changes."""
    text = original.decode('utf8')
    for old, new in [('now-started>600', 'now-started>1800'),
                     ('300 s startup / 600 s worker / 60 s shutdown',
                      '300 s startup / 1800 s worker / 60 s shutdown')]:
        require(text.count(old) == 1, 'Unexpected public clock expression')
        text = text.replace(old, new)
    return text.encode('utf8')

def paid(release, c, query):
    base = release / 'evidence/v10/native_pcbg/partition_export'
    rows = read(base / 'PAID_STREAMS.json')
    cards = [r for r in rows if r['dataset'] == c['dataset'] and r['seed'] == c['seed']]
    require(len(cards) == 1, 'Paid stream not uniquely identified')
    card = cards[0]['projection']; source = relative(base, card['file'])
    require(sha(source) == card['sha256'], 'Paid projection hash changed')
    with np.load(source, allow_pickle=False) as z:
        require(set(z.files) == {'pool', 'test', 'pairs', 'answers'}, 'Paid projection schema')
        a = {k: z[k].copy() for k in z.files}
    n = c['n']
    require(np.array_equal(np.sort(np.r_[a['pool'], a['test']]), np.arange(n)), 'Pool/test object order')
    require(np.array_equal(a['test'], np.setdiff1d(np.arange(n), a['pool'])), 'Test complement')
    require(a['pairs'].shape == (400, 2) and a['answers'].shape == (400,), 'Paid full-stream shape')
    require(np.isin(a['answers'], [-1, 1]).all() and np.isin(a['pairs'], a['pool']).all(), 'Paid domain')
    require((a['pairs'][:, 0] < a['pairs'][:, 1]).all() and
            len(np.unique(a['pairs'], axis=0)) == 400, 'Paid pair uniqueness/order')
    a['pairs'] = a['pairs'][:c['Q']]; a['answers'] = a['answers'][:c['Q']]
    folds = query.node_folds(a['pool'], a['pairs'], c['fold_partition_seed'])
    require(plain_folds(folds) == c['all_folds'], 'Frozen all-fold identities differ')
    require(plain_folds(query.eligible_folds(folds, a['answers'])) == c['eligible_folds'], 'Eligible folds differ')
    return a, card['sha256']

def feature_arrays(prepared, expected, native):
    xs = [sparse.load_npz(prepared / ('X%d.npz' % i)).tocsc() for i in range(expected['views'])]
    require([native.feature_hash(x) for x in xs] == expected['canonical_feature_sha256'],
            'Feature values/object order differ from original input')
    require(all(x.shape[0] == expected['n'] and np.isfinite(x.data).all() and
                (x.data >= 0).all() and (np.asarray(x.multiply(x).sum(axis=1)).ravel() > 0).all()
                for x in xs), 'Unsupported feature values')
    return xs

def semantic(path, native):
    require({name for name, _, _ in io.whosmat(path)} == FIELDS, 'MAT must be label-free seven-field carrier')
    d = io.loadmat(path, variable_names=sorted(FIELDS), simplify_cells=True)
    xs = list(d['X']) if isinstance(d['X'], np.ndarray) else [d['X']]
    require(all(np.isfinite(d[k]) and float(d[k]) == int(d[k]) for k in ('c', 'seed')), 'Noninteger c/seed carrier')
    return {'X': [native.feature_hash(sparse.csc_matrix(x)) for x in xs],
            'c': int(d['c']), 'seed': int(d['seed']), 'config': d['config'], 'case_id': str(d['case_id']),
            'ML': np.asarray(d['ML']).reshape(-1, 2).tolist(),
            'CL': np.asarray(d['CL']).reshape(-1, 2).tolist()}

def compare_reference(output, reference_map, native):
    refs = read(reference_map)
    report = []
    manifest = read(output / 'INPUT_MANIFEST.json')
    all_cases = [(method, x) for method in ('SSA', 'PCBG') for x in read(output / method / 'PLAN.json')['cases']]
    require(set(refs) == {m + '/' + c['id'] for m, c in all_cases}, 'Reference map needs exact 50 carriers')
    for method, case in all_cases:
        key = method + '/' + case['id']; ref = refs[key]
        original = Path(ref['path'])
        require(sha(original) == ref['sha256'], 'Frozen reference MAT hash differs')
        generated = output / method / case['input']
        require(sha(generated) == case['sha256'], 'Generated input changed')
        left, right = semantic(original, native), semantic(generated, native)
        require(left == right, 'Numerical input mismatch: ' + key)
        report.append({'case': key, 'original_container_sha256': ref['sha256'],
                       'new_container_sha256': case['sha256'], 'semantic_sha256': identity(right),
                       'equal_fields': sorted(FIELDS)})
    result = {'status': 'PASS_50_INPUTS_SEMANTICALLY_IDENTICAL', 'cases': report,
              'input_manifest_sha256': sha(output / 'INPUT_MANIFEST.json'),
              'new_container_hashes_are_not_historical_identity': True,
              'full_class_labels_loaded': False, 'new_native_outputs_or_test_scores_read': False,
              'matlab_started': False}
    immutable(output / 'FROZEN_INPUT_SEMANTIC_QA.json', result)
    return result

def prepare(release, prepared_map, output, reference_map=None):
    release, output = Path(release).resolve(), Path(output).resolve()
    spec = load_spec(release)
    require(not output.exists(), 'Use a fresh private work directory')
    require(not output.is_relative_to(release) and not output.is_relative_to(HERE),
            'Raw feature payloads must stay outside release and distributable tool directory')
    native = module(release / 'baselines/portable/pcbg/pcbg_portable.py', 'prep_native')
    query = module(release / 'common/query_protocol.py', 'prep_query')
    mapping_path = Path(prepared_map).resolve(); mapping = read(mapping_path)
    require(set(mapping) == {'BBC', 'Cora', 'CiteSeer'}, 'Prepared map needs exactly BBC/Cora/CiteSeer')
    expected = read(release / 'baselines/portable/pcbg/IDENTITIES.json')['dataset_identities']
    xs = {ds: feature_arrays((mapping_path.parent / mapping[ds]).resolve(), expected[ds], native) for ds in mapping}
    output.mkdir(parents=True)
    runtime = output / 'pcbg_runtime'; runtime.mkdir()
    base = release / 'baselines/portable/pcbg'
    (runtime / 'pcbg_portable.py').write_bytes(extended_launcher_bytes((base / 'pcbg_portable.py').read_bytes()))
    for name in ('IDENTITIES.json', 'pcbg_clock_bridge.m', 'pcbg_formal_worker.m'):
        shutil.copyfile(base / name, runtime / name)
    extended = module(runtime / 'pcbg_portable.py', 'prepared_extended_native')
    require(extended.sources()['pcbg_portable.py'] == 'a8f9e47c7d67de5a5b48d42ce2f5488b8e168302c7e18ef960c1c0de504d4d70',
            'Extended launcher differs from original reviewed extension')
    for method in ('SSA', 'PCBG'):
        root = output / method; (root / 'inputs').mkdir(parents=True); (root / 'paid').mkdir()
        plan = {'schema': 'portable-completion-inputs-v1', 'method': method,
                'spec_sha256': sha(HERE / 'PORTABLE_COMPLETION_SPEC.json'),
                'source_hashes': extended.sources() if method == 'PCBG' else own_sources(),
                'cases': [], 'paid': {}, 'feature_hashes': {}, 'class_labels_loaded': False}
        for c in spec['methods'][method]['conditions']:
            arrays, original_paid_hash = paid(release, c, query)
            paid_path = root / 'paid' / (c['id'] + '.npz')
            np.savez_compressed(paid_path, **arrays)
            plan['paid'][c['id']] = {'file': 'paid/' + paid_path.name, 'sha256': sha(paid_path),
                                      'original_projection_sha256': original_paid_hash}
            plan['feature_hashes'][c['dataset']] = expected[c['dataset']]['canonical_feature_sha256']
            folds = (0, 1, 'final') if method == 'SSA' else [x['fold'] for x in spec['methods'][method]['new_cases']
                                                           if x['case_id'].startswith(c['id'] + '_')]
            for fold in folds:
                ids = np.arange(c['Q']) if fold == 'final' else np.asarray(c['eligible_folds'][fold]['training'], int)
                pp, aa = arrays['pairs'][ids], arrays['answers'][ids]
                cell = np.empty((len(xs[c['dataset']]), 1), object)
                for i, x in enumerate(xs[c['dataset']]): cell[i, 0] = x
                cid = case_id(c, 0, fold); path = root / 'inputs' / (cid + '.mat')
                # SSA ignores this legacy PCBG carrier config and c0 case_id; request carries its effective config.
                io.savemat(path, dict(X=cell, c=float(c['c']), ML=(pp[aa == 1] + 1).astype(float),
                                     CL=(pp[aa == -1] + 1).astype(float), seed=float(c['fit_seed']),
                                     config={k: float(v) for k, v in spec['methods']['PCBG']['native_config'].items()},
                                     case_id=cid), do_compression=True)
                plan['cases'].append({'id': cid, 'condition_id': c['id'], 'candidate': 0, 'fold': fold,
                                      'input': 'inputs/' + path.name, 'sha256': sha(path),
                                      'semantic_sha256': identity(semantic(path, native))})
        require(len(plan['cases']) == (45 if method == 'SSA' else 5), 'Carrier count')
        immutable(root / 'PLAN.json', plan)
    files = {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob('*')) if p.is_file()}
    immutable(output / 'INPUT_MANIFEST.json', {'schema': 'portable-completion-private-input-manifest-v1',
              'status': 'PREPARED_WITHOUT_OPTIMIZATION', 'spec_sha256': sha(HERE / 'PORTABLE_COMPLETION_SPEC.json'),
              'own_sources': own_sources(), 'files': files, 'semantic_not_container_replay': True,
              'contains_external_features_do_not_distribute': True, 'class_labels_loaded': False,
              'matlab_started': False})
    if reference_map:
        compare_reference(output, reference_map, native)
    return {'status': 'PREPARED_WITHOUT_OPTIMIZATION', 'SSA_carriers': 45, 'PCBG_carriers': 5,
            'input_manifest_sha256': sha(output / 'INPUT_MANIFEST.json')}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release-root', required=True)
    p.add_argument('--write-inputs', action='store_true', help='Explicitly write private MAT payloads; never fit')
    p.add_argument('--prepared-map'); p.add_argument('--output'); p.add_argument('--verify-frozen-reference-map')
    a = p.parse_args()
    if not a.write_inputs:
        spec = load_spec(Path(a.release_root).resolve())
        print('VALID_METADATA_ONLY: 15 SSA conditions / 45 carriers / <=555 fits; 4 PCBG conditions / 5 new fits')
        return
    require(a.prepared_map and a.output, '--write-inputs needs --prepared-map and --output')
    print(prepare(a.release_root, a.prepared_map, a.output, a.verify_frozen_reference_map))

if __name__ == '__main__':
    main()
