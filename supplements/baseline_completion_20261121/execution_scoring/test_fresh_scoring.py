"""Temporary synthetic fresh seals and category vectors only; no actual experiment result access."""
import argparse
import copy
import itertools
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
from scipy import io
import score_fresh_run as scorer

RUNNER = None

def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf8')

class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
    def tearDown(self): self.temp.cleanup()

    def fixture(self, method='SSA'):
        work = self.root / 'fresh'; release = self.root / 'public'; release.mkdir(exist_ok=True)
        original = scorer.read(RUNNER / 'PORTABLE_COMPLETION_SPEC.json')['methods'][method]['conditions']
        conditions = []; requirements = {}; labels = {}; mapping = {}
        external = self.root / 'external'; external.mkdir(exist_ok=True)
        for ds in sorted({c['dataset'] for c in original}):
            y = np.arange(12, dtype=np.int64) % 3; labels[ds] = y
            file = external / (ds + '.npy'); np.save(file, y)
            requirements[ds] = {'n': 12, 'k': 3, 'original_label_file_sha256': scorer.sha(file),
                                'ordered_partition_sha256': scorer.math.partition_hash(y)}
            mapping[ds] = {'path': str(file), 'sha256': scorer.sha(file)}
        mapping_file = external / 'labels_map.json'; put(mapping_file, mapping)
        root = work / method; root.mkdir(parents=True)
        binding_path = root / 'RUN_BINDING.json'; put(binding_path, {'method': method, 'synthetic': True})
        plan = {'paid': {}}; commitments = []
        for index, old in enumerate(original):
            c = dict(old); c.update(id='synthetic_' + old['id'], n=12, c=3, Q=6 if old['Q'] == 400 else 5)
            conditions.append(c); ds = c['dataset']; y = labels[ds]
            pool = np.sort(np.random.RandomState(c['seed'] + 21000).permutation(12)[:4])
            test = np.setdiff1d(np.arange(12), pool)
            allpairs = np.array(list(itertools.combinations(pool, 2)))
            signs = np.where(y[allpairs[:, 0]] == y[allpairs[:, 1]], 1, -1)
            pos, neg = np.flatnonzero(signs == 1)[0], np.flatnonzero(signs == -1)[0]
            order = [pos, neg] + [i for i in range(6) if i not in (pos, neg)]
            pairs, answers = allpairs[order][:c['Q']], signs[order][:c['Q']]
            paid_path = root / 'paid' / (c['id'] + '.npz'); paid_path.parent.mkdir(exist_ok=True)
            np.savez_compressed(paid_path, pool=pool, test=test, pairs=pairs, answers=answers)
            plan['paid'][c['id']] = {'file': 'paid/' + paid_path.name, 'sha256': scorer.sha(paid_path)}
            cid = c['id'] + '_c0_final'; dest = root / 'runs' / cid; dest.mkdir(parents=True)
            eligible = index > 1; present = index != 0
            artifacts = {}
            if present:
                pred = y + 1 if eligible else np.ones(12, dtype=int)
                io.savemat(dest / 'native_outputs.mat', {'pred' if method == 'SSA' else 'label': pred})
                artifacts['native_outputs.mat'] = scorer.sha(dest / 'native_outputs.mat')
            receipt_path = dest / ('EXECUTION.json' if method == 'SSA' else 'receipt.json')
            put(receipt_path, {'files' if method == 'SSA' else 'artifacts': artifacts})
            final = {'case_id': cid, 'eligible': eligible, 'occupied_clusters': 3 if eligible else (1 if present else None),
                     'execution_status': 'EXECUTED' if eligible else 'FAILED_SYNTHETIC',
                     'worker_status': 'ELIGIBLE' if eligible else ('WRONG_K' if present else 'TIMEOUT'),
                     'timed_out': not present,
                     'execution_sha256' if method == 'SSA' else 'receipt_sha256': scorer.sha(receipt_path)}
            selection = ({'selected_candidate': 0, 'fallback': False} if method == 'SSA'
                         else {'candidate': 0, 'default_fallback': False})
            selection_path = root / 'ledger/selections' / (c['id'] + '.json'); put(selection_path, selection)
            commit = {'condition_id': c['id'], 'selection_sha256': scorer.sha(selection_path)}
            if method == 'SSA':
                final_path = root / 'ledger/fits' / (cid + '.json'); put(final_path, final)
                commit.update(final_case=cid, final_decision_sha256=scorer.sha(final_path))
            else: commit['final'] = final
            commit_path = root / 'ledger/conditions' / (c['id'] + '.json'); put(commit_path, commit)
            commitments.append({'condition_id': c['id'], 'sha256': scorer.sha(commit_path)})
        put(root / 'PLAN.json', plan)
        manifest_path = work / 'INPUT_MANIFEST.json'
        put(manifest_path, {'files': {method + '/PLAN.json': scorer.sha(root / 'PLAN.json')}})
        put(binding_path, {'method': method, 'synthetic': True, 'input_manifest_sha256': scorer.sha(manifest_path)})
        seal = {'status': 'NEW_PORTABLE_PREDICTIONS_SEALED_BEFORE_EXTERNAL_SCORING', 'method': method,
                'full_class_labels_loaded': False, 'conditions': commitments, 'run_binding': scorer.sha(binding_path)}
        seal_path = root / 'ledger/GLOBAL_PREDICTIONS_SEALED.json'; put(seal_path, seal)
        bound = {str(p): scorer.sha(p) for p in work.rglob('*') if p.is_file()}
        args = types.SimpleNamespace(runner_root=str(RUNNER), release_root=str(release), work=str(work), method=method,
                    expected_seal_sha256=scorer.sha(seal_path), labels_map=str(mapping_file), output=str(self.root / 'score'))
        verified = {'seal_sha256': scorer.sha(seal_path), 'run_binding_sha256': scorer.sha(binding_path),
                    'conditions': conditions, 'verification_stdout': 'SYNTHETIC_VERIFIER_ONLY'}
        calls = []
        def fake_verify(a):
            calls.append(True)
            scorer.need(seal_path.is_file() and scorer.sha(seal_path) == a.expected_seal_sha256, 'Synthetic seal absent/changed')
            for p, digest in bound.items(): scorer.need(scorer.sha(p) == digest, 'Synthetic sealed artifact changed')
            return verified
        return args, verified, requirements, fake_verify, labels, calls

    def test_all_15_ssa_rows_and_missing_not_zero(self):
        a, v, req, verify, labels, calls = self.fixture('SSA')
        with patch.object(scorer, 'reviewed_verify', side_effect=verify), patch.object(scorer, 'public_requirements', return_value=req):
            result = scorer.run(a)
        self.assertEqual(len(calls), 2); self.assertEqual(result['n_planned'], 15); self.assertEqual(result['n_ineligible'], 2)
        rows = scorer.read(Path(a.output) / 'FRESH_RUN_ROWS.json')
        self.assertEqual(len(rows), 15)
        for row in rows[:2]: self.assertTrue(all(row[m] is None for m in scorer.math.METRICS))
        self.assertFalse(rows[0]['partition_present']); self.assertTrue(rows[1]['partition_present'])
        for row in rows[2:]:
            self.assertEqual([row[m] for m in scorer.math.METRICS], [1., 1., 1., 0., 0.])
        self.assertFalse(result['historical_metric_equality_required']); self.assertFalse(result['group_means_or_inference_computed'])

    def test_all_4_pcbg_rows_no_three_seed_aggregate(self):
        a, v, req, verify, labels, calls = self.fixture('PCBG')
        with patch.object(scorer, 'reviewed_verify', side_effect=verify), patch.object(scorer, 'public_requirements', return_value=req):
            result = scorer.run(a)
        self.assertEqual(result['n_planned'], 4)
        self.assertEqual(set(p.name for p in Path(a.output).iterdir()),
                         {'FRESH_RUN_ROWS.json', 'fresh_run_metrics.csv', 'SCORING_RECEIPT.json', 'OUTPUT_SHA256.json'})

    def test_missing_seal_rejected_before_class_decode(self):
        a, v, req, verify, labels, calls = self.fixture()
        (Path(a.work) / a.method / 'ledger/GLOBAL_PREDICTIONS_SEALED.json').unlink()
        with patch.object(scorer, 'reviewed_verify', side_effect=verify), patch.object(scorer.np, 'load', side_effect=AssertionError('array opened')):
            with self.assertRaisesRegex(ValueError, 'seal absent'): scorer.run(a)
        self.assertFalse(Path(a.output).exists())

    def test_tampered_final_rejected_before_class_decode(self):
        a, v, req, verify, labels, calls = self.fixture()
        path = next((Path(a.work) / a.method / 'runs').glob('*/native_outputs.mat')); path.write_bytes(b'tampered')
        with patch.object(scorer, 'reviewed_verify', side_effect=verify), patch.object(scorer.np, 'load', side_effect=AssertionError('array opened')):
            with self.assertRaisesRegex(ValueError, 'artifact changed'): scorer.run(a)
        self.assertFalse(Path(a.output).exists())

    def test_real_gate_no_seal_does_not_call_subprocess_or_labels(self):
        args = types.SimpleNamespace(runner_root=str(RUNNER), work=str(self.root), method='SSA', expected_seal_sha256='0' * 64)
        with patch.object(scorer.subprocess, 'run', side_effect=AssertionError('called')), patch.object(scorer.np, 'load', side_effect=AssertionError('decoded')):
            with self.assertRaisesRegex(ValueError, 'seal missing'): scorer.reviewed_verify(args)

    def test_wrong_expected_hash_does_not_call_subprocess_or_labels(self):
        a, v, req, verify, labels, calls = self.fixture(); a.expected_seal_sha256 = '0' * 64
        with patch.object(scorer.subprocess, 'run', side_effect=AssertionError('called')), patch.object(scorer.np, 'load', side_effect=AssertionError('decoded')):
            with self.assertRaisesRegex(ValueError, 'seal hash differs'): scorer.reviewed_verify(a)

    def test_self_declared_hash_cannot_change_class_object_order(self):
        a, v, req, verify, labels, calls = self.fixture()
        mapping = scorer.read(a.labels_map); item = mapping['BBC']; path = Path(item['path'])
        y = np.load(path); y[[0, 1]] = y[[1, 0]]; np.save(path, y)
        item['sha256'] = scorer.sha(path); put(Path(a.labels_map), mapping)
        with patch.object(scorer, 'reviewed_verify', side_effect=verify), patch.object(scorer, 'public_requirements', return_value=req):
            with self.assertRaisesRegex(ValueError, 'independently pinned public identity'): scorer.run(a)
        self.assertFalse(Path(a.output).exists())

    def test_equivalent_class_numbering_requires_same_ordered_partition(self):
        a, v, req, verify, labels, calls = self.fixture()
        mapping = scorer.read(a.labels_map)
        for item in mapping.values():
            path = Path(item['path']); np.save(path, np.load(path) * 7 + 11); item['sha256'] = scorer.sha(path)
        put(Path(a.labels_map), mapping)
        with patch.object(scorer, 'reviewed_verify', side_effect=verify), patch.object(scorer, 'public_requirements', return_value=req):
            result = scorer.run(a)
        self.assertFalse(any(result['label_serialization_matches_original'].values()))

    def test_paid_answer_mismatch_rejected_before_scoring_prediction(self):
        a, v, req, verify, labels, calls = self.fixture(); states, files = scorer.snapshot(a, v)
        states[0]['arrays']['answers'][0] *= -1
        with patch.object(scorer.io, 'loadmat', side_effect=AssertionError('prediction decoded')):
            with self.assertRaisesRegex(ValueError, 'Paid answers'): scorer.score_states('SSA', states, labels)

    def test_pool_excluded_metrics_and_pair_sampling(self):
        a, v, req, verify, labels, calls = self.fixture(); states, files = scorer.snapshot(a, v)
        s = next(s for s in states if s['final']['eligible']); base = scorer.score_states('SSA', [s], labels)[0]
        c = s['condition']; pred = labels[c['dataset']].copy() + 1; pred[s['arrays']['pool']] = 1
        io.savemat(s['prediction'], {'pred': pred})
        changed = scorer.score_states('SSA', [s], labels)[0]
        for m in ('ACC', 'NMI', 'ARI', 'unqueried_BER'): self.assertEqual(changed[m], base[m])
        self.assertEqual(changed['training_BER'], .5)
        pairs = scorer.math.sample_pairs(s['arrays']['test'], c['evaluation_pair_seed'])
        self.assertEqual(len(pairs), 28); self.assertFalse(np.intersect1d(pairs, s['arrays']['pool']).size)
        self.assertTrue(np.array_equal(pairs, scorer.math.sample_pairs(s['arrays']['test'], c['evaluation_pair_seed'])))

    def test_wrong_test_complement_and_cross_budget_context(self):
        a, v, req, verify, labels, calls = self.fixture(); states, files = scorer.snapshot(a, v)
        s = states[0]; broken = {k: x.copy() for k, x in s['arrays'].items()}; broken['test'][0] = broken['pool'][0]
        with self.assertRaisesRegex(ValueError, 'test complement'): scorer.validate_paid(s['condition'], broken)
        first = next(s for s in states if s['condition']['dataset'] == 'Cora')
        second = copy.deepcopy(first); second['condition']['evaluation_pair_seed'] += 1
        with self.assertRaisesRegex(ValueError, 'Cross-budget'): scorer.score_states('SSA', [first, second], labels)

    def test_five_metrics_independent_small_example(self):
        y = np.array([0, 0, 1, 1]); pred = np.array([0, 1, 0, 1])
        self.assertEqual(scorer.math.metrics(y, pred), {'ACC': .5, 'NMI': 0., 'ARI': -.5})
        paid = np.array([[0, 1], [0, 2]])
        self.assertEqual(scorer.math.ber(pred, paid, np.array([1, -1])), 1.)
        ep = scorer.math.sample_pairs(np.arange(4), 123); ea = np.where(y[ep[:, 0]] == y[ep[:, 1]], 1, -1)
        self.assertEqual(scorer.math.ber(pred, ep, ea), .75)

    def test_second_verification_failure_leaves_no_score_files(self):
        a, v, req, verify, labels, calls = self.fixture()
        def fail_second(args):
            if calls: raise ValueError('Second verification rejected')
            return verify(args)
        with patch.object(scorer, 'reviewed_verify', side_effect=fail_second), patch.object(scorer, 'public_requirements', return_value=req):
            with self.assertRaisesRegex(ValueError, 'Second verification'): scorer.run(a)
        self.assertFalse(Path(a.output).exists())

    def test_snapshot_plan_directly_bound_to_run_manifest(self):
        a, v, req, verify, labels, calls = self.fixture()
        path = Path(a.work) / a.method / 'PLAN.json'
        value = scorer.read(path); value['extra_changed_field'] = True; put(path, value)
        with patch.object(scorer.np, 'load', side_effect=AssertionError('array decoded')):
            with self.assertRaisesRegex(ValueError, 'Sealed artifact differs'): scorer.snapshot(a, v)

    def test_real_gate_constructs_only_reviewed_verify_command(self):
        method = 'PCBG'; work = self.root / 'fresh'; root = work / method
        spec = scorer.read(RUNNER / 'PORTABLE_COMPLETION_SPEC.json')['methods'][method]
        binding_path = root / 'RUN_BINDING.json'
        put(binding_path, {'matlab': 'C:/synthetic/matlab.exe', 'pcbg_author': 'C:/synthetic/author'})
        seal_path = root / 'ledger/GLOBAL_PREDICTIONS_SEALED.json'
        put(seal_path, {'status': 'NEW_PORTABLE_PREDICTIONS_SEALED_BEFORE_EXTERNAL_SCORING', 'method': method,
            'full_class_labels_loaded': False, 'conditions': [{'condition_id': c['id']} for c in spec['conditions']],
            'run_binding': scorer.sha(binding_path)})
        args = types.SimpleNamespace(runner_root=str(RUNNER), work=str(work), method=method,
                release_root=str(self.root / 'synthetic-public'), expected_seal_sha256=scorer.sha(seal_path))
        calls = []
        def fake(cmd, **kw):
            calls.append(cmd)
            self.assertEqual(cmd[3], 'verify'); self.assertNotIn('run', cmd)
            return types.SimpleNamespace(returncode=0, stdout='SYNTHETIC_VERIFY_SUCCESS', stderr='')
        with patch.object(scorer.subprocess, 'run', side_effect=fake), patch.object(scorer.np, 'load', side_effect=AssertionError('decoded')):
            checked = scorer.reviewed_verify(args)
        self.assertEqual(len(calls), 1); self.assertEqual(len(checked['conditions']), 4)

    def test_label_change_during_scoring_rejected_without_output(self):
        a, v, req, verify, labels, calls = self.fixture()
        original = scorer.score_states
        def changing(method, states, loaded):
            rows = original(method, states, loaded)
            path = Path(scorer.read(a.labels_map)['BBC']['path']); path.write_bytes(path.read_bytes() + b'changed')
            return rows
        with patch.object(scorer, 'reviewed_verify', side_effect=verify), patch.object(scorer, 'public_requirements', return_value=req), \
             patch.object(scorer, 'score_states', side_effect=changing):
            with self.assertRaisesRegex(ValueError, 'labels changed during'): scorer.run(a)
        self.assertFalse(Path(a.output).exists())

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--runner-root', required=True); args, rest = p.parse_known_args()
    RUNNER = Path(args.runner_root).resolve()
    unittest.main(argv=[__file__] + rest, verbosity=2)
