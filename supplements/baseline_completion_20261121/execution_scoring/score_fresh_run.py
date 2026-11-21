"""Score an independently replayed, completely sealed method. Never fit or replace historical scores."""
import argparse
import hashlib
import importlib.util
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
RUNNER_MANIFEST_SHA256 = '47989992a4f98fd581fa7e7342a5814eb83829aa325d6d74a8e0de22a809675d'
MATH_SHA256 = 'e8e97cc332e9e6ca76a08de9125029ee62461b5b2d3d6f0482b53fdc081aa695'
PUBLIC_PROTOCOL = 'evidence/v10/native_pcbg/partition_export/evidence/FROZEN_PROTOCOL.json'
PUBLIC_BINDINGS = {PUBLIC_PROTOCOL: '4a334d1319a17b3d8a266962267c6e19cd2d7ecd9bfff2a78c0afea2b608b117',
                   'primary_predictions/MANIFEST.json': '81ee5eca0f4f5c67238d560bb2a8eddf43d6a36ed5d10d4d5ad4609b7448f9cd'}

def need(value, message):
    if not value: raise ValueError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1048576), b''): h.update(data)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))

need(sha(HERE / 'baseline_math.py') == MATH_SHA256, 'Reviewed five-metric implementation changed')
_module = importlib.util.spec_from_file_location('fresh_reviewed_math', HERE / 'baseline_math.py')
math = importlib.util.module_from_spec(_module); _module.loader.exec_module(math)
import numpy as np
from scipy import io

def verify_runner_files(root):
    root = Path(root).resolve()
    need(sha(root / 'TOOLS_SHA256.json') == RUNNER_MANIFEST_SHA256, 'Reviewed runner manifest differs')
    need((root / 'TOOLS_SHA256.sha256').read_text().strip() == RUNNER_MANIFEST_SHA256, 'Runner manifest sidecar differs')
    manifest = read(root / 'TOOLS_SHA256.json')
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    need(actual == set(manifest['files']) | {'TOOLS_SHA256.json', 'TOOLS_SHA256.sha256'}, 'Runner whitelist differs')
    for name, digest in manifest['files'].items():
        need(sha(math.child(root, name)) == digest, 'Reviewed runner file differs: ' + name)

def public_requirements(release):
    release = Path(release).resolve()
    for name, digest in PUBLIC_BINDINGS.items():
        need(sha(math.child(release, name)) == digest, 'Trusted public class-metadata source differs: ' + name)
    original = read(release / PUBLIC_PROTOCOL)
    primary = read(release / 'primary_predictions/MANIFEST.json')
    result = {}
    for ds in ('BBC', 'Cora', 'CiteSeer'):
        identities = {r['partition_sha256'] for r in primary['records'] if r['dataset'] == ds}
        need(len(identities) == 1, 'Public ordered class identity is not unique')
        d = original['datasets'][ds]
        result[ds] = {'n': d['n'], 'k': d['k'],
                      'original_label_file_sha256': d['sealed_scoring_input']['sha256'],
                      'ordered_partition_sha256': identities.pop()}
    carried = read(HERE / 'CLASS_IDENTITIES.json')
    need(carried['provenance'] == PUBLIC_BINDINGS and carried['corpora'] == result,
         'Companion class identity projection differs from trusted public sources')
    return result

def reviewed_verify(args):
    """Only the fixed reviewed *verify* CLI is admitted; no run command is constructed."""
    runner, work = Path(args.runner_root).resolve(), Path(args.work).resolve()
    method = args.method; verify_runner_files(runner)
    seal_path = work / method / 'ledger/GLOBAL_PREDICTIONS_SEALED.json'
    need(seal_path.is_file(), 'Complete method seal missing; external categories are not admitted')
    need(isinstance(args.expected_seal_sha256, str) and len(args.expected_seal_sha256) == 64
         and sha(seal_path) == args.expected_seal_sha256, 'Explicit expected fresh-run seal hash differs')
    seal = read(seal_path)
    need(seal['status'] == 'NEW_PORTABLE_PREDICTIONS_SEALED_BEFORE_EXTERNAL_SCORING'
         and seal['method'] == method and seal['full_class_labels_loaded'] is False,
         'Not a complete pre-scoring fresh-run seal')
    spec = read(runner / 'PORTABLE_COMPLETION_SPEC.json')['methods'][method]
    need([x['condition_id'] for x in seal['conditions']] == [c['id'] for c in spec['conditions']]
         and len(seal['conditions']) == (15 if method == 'SSA' else 4), 'Incomplete fresh-run denominator')
    binding_path = work / method / 'RUN_BINDING.json'
    need(sha(binding_path) == seal['run_binding'], 'Fresh run binding differs')
    binding = read(binding_path)
    cmd = [sys.executable, '-B', str(runner / 'completion_portable.py'), 'verify', '--method', method,
           '--release-root', str(Path(args.release_root).resolve()), '--work', str(work),
           '--matlab', binding['matlab']]
    cmd += (['--ssa-runtime', binding['ssa_runtime']] if method == 'SSA' else ['--pcbg-author', binding['pcbg_author']])
    finished = subprocess.run(cmd, capture_output=True, text=True)
    need(finished.returncode == 0, 'Reviewed whole-method verification failed before class access:\n' + finished.stdout + finished.stderr)
    need(sha(seal_path) == args.expected_seal_sha256, 'Seal changed during full verification')
    verify_runner_files(runner)
    return {'seal_sha256': args.expected_seal_sha256, 'run_binding_sha256': seal['run_binding'],
            'conditions': spec['conditions'], 'verification_stdout': finished.stdout.strip()}

def snapshot(args, verified):
    """Rebind final metadata and paid inputs; no category array or final partition decoded here."""
    work = Path(args.work).resolve(); method = args.method; root = work / method
    seal_path = root / 'ledger/GLOBAL_PREDICTIONS_SEALED.json'; seal = read(seal_path)
    need(sha(seal_path) == verified['seal_sha256'], 'Verified seal changed')
    files = {}; states = []
    def bound(path, digest=None):
        actual = sha(path)
        need(digest is None or actual == digest, 'Sealed artifact differs: ' + str(path))
        files[str(path)] = actual
        return actual
    bound(seal_path, verified['seal_sha256']); bound(root / 'RUN_BINDING.json', verified['run_binding_sha256'])
    binding = read(root / 'RUN_BINDING.json')
    manifest_path = work / 'INPUT_MANIFEST.json'
    bound(manifest_path, binding['input_manifest_sha256']); manifest = read(manifest_path)
    bound(root / 'PLAN.json', manifest['files'][method + '/PLAN.json']); plan = read(root / 'PLAN.json')
    for c in verified['conditions']:
        commit_path = root / 'ledger/conditions' / (c['id'] + '.json')
        commit_hash = next(x['sha256'] for x in seal['conditions'] if x['condition_id'] == c['id'])
        bound(commit_path, commit_hash); record = read(commit_path)
        selection_path = root / 'ledger/selections' / (c['id'] + '.json')
        bound(selection_path, record['selection_sha256']); selection = read(selection_path)
        if method == 'SSA':
            final_path = root / 'ledger/fits' / (record['final_case'] + '.json')
            bound(final_path, record['final_decision_sha256']); final = read(final_path)
            candidate, fallback = selection['selected_candidate'], selection['fallback']
            receipt_path = root / 'runs' / final['case_id'] / 'EXECUTION.json'
            bound(receipt_path, final['execution_sha256']); receipt = read(receipt_path); artifacts = receipt['files']
            variable = 'pred'
        else:
            final = record['final']; candidate, fallback = selection['candidate'], selection['default_fallback']
            receipt_path = root / 'runs' / final['case_id'] / 'receipt.json'
            bound(receipt_path, final['receipt_sha256']); receipt = read(receipt_path); artifacts = receipt['artifacts']
            variable = 'label'
        prediction = None
        if final['occupied_clusters'] is not None:
            prediction = root / 'runs' / final['case_id'] / 'native_outputs.mat'
            need('native_outputs.mat' in artifacts, 'Finite final partition is not receipt-bound')
            bound(prediction, artifacts['native_outputs.mat'])
        need(not final['eligible'] or prediction is not None, 'Eligible final partition missing')
        card = plan['paid'][c['id']]; paid_path = math.child(root, card['file']); bound(paid_path, card['sha256'])
        with np.load(paid_path, allow_pickle=False) as z:
            need(set(z.files) == {'pool', 'test', 'pairs', 'answers'}, 'Paid packet schema differs')
            a = {key: math.integer_vector(z[key]) for key in z.files}; a['pairs'] = a['pairs'].reshape(-1, 2)
        validate_paid(c, a)
        states.append({'condition': c, 'final': final, 'candidate': candidate, 'fallback': fallback,
                       'prediction': prediction, 'variable': variable, 'arrays': a})
    need(len(states) == (15 if method == 'SSA' else 4), 'Expected complete method final outcomes')
    return states, files

def validate_paid(c, a):
    n, q = c['n'], c['Q']; pool, test, pairs, answers = [a[k] for k in ('pool', 'test', 'pairs', 'answers')]
    expected = np.sort(np.random.RandomState(c['seed'] + 21000).permutation(n)[:round(.3 * n)])
    need(np.array_equal(pool, expected) and np.array_equal(test, np.setdiff1d(np.arange(n), pool)),
         'Frozen paid pool/test complement or order differs')
    need(pairs.shape == (q, 2) and answers.shape == (q,) and np.isin(pairs, pool).all()
         and np.isin(answers, [-1, 1]).all() and (pairs[:, 0] < pairs[:, 1]).all()
         and len(np.unique(pairs, axis=0)) == q, 'Paid pair domain/budget/uniqueness differs')

def admit_labels(mapping_path, requirements, required, forbidden_roots):
    mapping_path = Path(mapping_path).resolve(); mapping = read(mapping_path)
    mapping = mapping.get('corpora', mapping); files = {}; labels = {}
    # Byte hashes are checked for all required files before decoding any category array.
    for ds in sorted(required):
        item = mapping[ds]
        need(isinstance(item, dict) and isinstance(item.get('path'), str)
             and isinstance(item.get('sha256'), str), 'Label map needs explicit path and byte SHA256')
        path = (mapping_path.parent / item['path']).resolve()
        need(path.suffix.lower() == '.npy' and not any(path.is_relative_to(r) for r in forbidden_roots),
             'External category NPY must be outside runner, work, release and scoring tools')
        need(sha(path) == item['sha256'], 'External label bytes differ from declared input')
        files[ds] = (path, item['sha256'])
    for ds, (path, digest) in files.items():
        req = requirements[ds]; y = math.integer_vector(np.load(path, allow_pickle=False), req['n'])
        need(len(np.unique(y)) == req['k'] and math.partition_hash(y) == req['ordered_partition_sha256'],
             'External class partition/object order differs from independently pinned public identity: ' + ds)
        labels[ds] = y
    return labels, files

def score_states(method, states, labels):
    rows = []; contexts = {}
    for s in states:
        c, a = s['condition'], s['arrays']; y = labels[c['dataset']]; pairs = a['pairs']
        need(np.array_equal(np.where(y[pairs[:, 0]] == y[pairs[:, 1]], 1, -1), a['answers']),
             'Paid answers contradict supplied object order')
        key = c['dataset'], c['seed']
        if key in contexts:
            old = contexts[key]
            need(np.array_equal(old[0], a['test']) and old[1] == c['evaluation_pair_seed'],
                 'Cross-budget evaluation context differs')
        else:
            ep = math.sample_pairs(a['test'], c['evaluation_pair_seed'])
            need(not np.intersect1d(ep, a['pool']).size, 'Evaluation pairs overlap query pool')
            need(np.isin(ep, a['test']).all() and (ep[:, 0] < ep[:, 1]).all()
                 and len(np.unique(ep, axis=0)) == len(ep), 'Evaluation pair construction differs')
            ea = np.where(y[ep[:, 0]] == y[ep[:, 1]], 1, -1)
            contexts[key] = (a['test'], c['evaluation_pair_seed'], ep, ea)
    # Every paid answer/context has passed before the first final partition is decoded for scoring.
    for s in states:
        c, final, a = s['condition'], s['final'], s['arrays']; y = labels[c['dataset']]
        ep, ea = contexts[c['dataset'], c['seed']][2:]
        row = {'method': method, 'condition_id': c['id'], 'dataset': c['dataset'], 'seed': c['seed'],
               'Q': c['Q'], 'n': c['n'], 'k': c['c'], 'selected_candidate': s['candidate'], 'fallback': s['fallback'],
               'final_case_id': final['case_id'], 'eligible': bool(final['eligible']),
               'partition_present': s['prediction'] is not None, 'occupied_clusters': final['occupied_clusters'],
               'execution_status': final['execution_status'], 'worker_status': final['worker_status'],
               'timed_out': final.get('timed_out'), 'evaluation_objects': len(a['test']),
               'evaluation_pairs': len(ep), 'evaluation_pair_seed': c['evaluation_pair_seed'],
               'metrics_scope': 'strict_eligible_final' if final['eligible'] else 'ineligible_metrics_not_reported'}
        row.update({m: None for m in math.METRICS})
        if final['eligible']:
            pred = math.integer_vector(io.loadmat(s['prediction'], variable_names=[s['variable']])[s['variable']], c['n'])
            need(len(np.unique(pred)) == c['c'] == final['occupied_clusters'], 'Eligible final occupied-k mismatch')
            row.update(math.metrics(y[a['test']], pred[a['test']]))
            row['training_BER'] = math.ber(pred, a['pairs'], a['answers'])
            row['unqueried_BER'] = math.ber(pred, ep, ea)
        rows.append(row)
    return rows

def run(args):
    output = Path(args.output).resolve()
    roots = [Path(x).resolve() for x in (args.runner_root, args.release_root, args.work)] + [HERE]
    need(not output.exists() and not any(output.is_relative_to(r) for r in roots),
         'Score output must be a fresh directory outside immutable inputs/run/release/tools')
    own_sources = {name: sha(HERE / name) for name in ('score_fresh_run.py', 'baseline_math.py', 'CLASS_IDENTITIES.json')}
    verified = reviewed_verify(args)
    requirements = public_requirements(args.release_root)
    states, bound_files = snapshot(args, verified)
    required = {s['condition']['dataset'] for s in states}
    for s in states:
        c = s['condition']; need(c['n'] == requirements[c['dataset']]['n'] and c['c'] == requirements[c['dataset']]['k'],
                                 'Run dimensions differ from trusted category identity')
    labels, label_files = admit_labels(args.labels_map, requirements, required, roots)
    rows = score_states(args.method, states, labels)
    need(len(rows) == len(verified['conditions']) and len({r['condition_id'] for r in rows}) == len(rows),
         'All planned fresh-run outcomes must remain in the score report')
    # Verify the whole method again, then recheck every consumed final/paid/category byte before output.
    again = reviewed_verify(args)
    need(again == verified, 'Whole-method verification changed while scoring')
    need(public_requirements(args.release_root) == requirements, 'Trusted class requirements changed during scoring')
    for name, digest in own_sources.items(): need(sha(HERE / name) == digest, 'Scoring source changed during calculation')
    for name, digest in bound_files.items(): need(sha(name) == digest, 'Sealed scoring input changed during calculation')
    for path, digest in label_files.values(): need(sha(path) == digest, 'External labels changed during calculation')
    result = {'status': 'FRESH_RUN_SCORED_AFTER_COMPLETE_METHOD_VERIFICATION', 'method': args.method,
              'n_planned': len(rows), 'n_eligible': sum(r['eligible'] for r in rows),
              'n_ineligible': sum(not r['eligible'] for r in rows),
              'fresh_global_seal_sha256': verified['seal_sha256'], 'run_binding_sha256': verified['run_binding_sha256'],
              'reviewed_runner_manifest_sha256': RUNNER_MANIFEST_SHA256, 'metric_source_sha256': MATH_SHA256,
              'scorer_sha256': sha(__file__), 'scoring_sources': own_sources, 'class_identity_public_sources': PUBLIC_BINDINGS,
              'external_label_file_sha256': {ds: h for ds, (_, h) in label_files.items()},
              'label_serialization_matches_original': {ds: h == requirements[ds]['original_label_file_sha256']
                                                       for ds, (_, h) in label_files.items()},
              'external_ordered_class_identity': {ds: requirements[ds]['ordered_partition_sha256'] for ds in sorted(required)},
              'versions': {p: version(p) for p in ('numpy', 'scipy', 'scikit-learn')},
              'clustering_metrics_on_test_objects_only': True, 'evaluation_pairs_exclude_query_pool': True,
              'full_method_verified_before_class_decode': True, 'fitting_invoked': False,
              'historical_84_rows_replaced': False, 'historical_metric_equality_required': False,
              'ineligible_metrics_imputed': False, 'group_means_or_inference_computed': False}
    output.mkdir(parents=True, exist_ok=False)
    math.write_json(output / 'FRESH_RUN_ROWS.json', rows); math.write_csv(output / 'fresh_run_metrics.csv', rows)
    math.write_json(output / 'SCORING_RECEIPT.json', result)
    math.write_json(output / 'OUTPUT_SHA256.json', {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    return result

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['check', 'score'], nargs='?', default='check')
    for key in ('runner-root', 'release-root', 'work', 'expected-seal-sha256'): p.add_argument('--' + key, required=True)
    p.add_argument('--method', required=True, choices=['SSA', 'PCBG'])
    p.add_argument('--labels-map'); p.add_argument('--output'); args = p.parse_args()
    if args.action == 'check':
        result = reviewed_verify(args); public_requirements(args.release_root)
        print(json.dumps({'status': 'VERIFIED_NO_CLASS_ARRAYS_OPENED', 'method': args.method,
                          'seal_sha256': result['seal_sha256'], 'conditions': len(result['conditions'])}, indent=2))
    else:
        need(args.labels_map and args.output, 'score needs explicit --labels-map and --output')
        print(json.dumps(run(args), indent=2))

if __name__ == '__main__': main()
