"""Synthetic metadata only. Temporary complete-looking fixtures are never published."""
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch
import copy
import json
import tempfile
import unittest

import replay_exact_k as app

CORE, PROTOCOL = app.load_core()


def inputs_fixture():
    rows, records = [], []
    for method, spec in PROTOCOL['methods'].items():
        for index, c in enumerate(spec['conditions']):
            supported = c['supported']; wrong = supported and index == 0
            count = c['c'] - 1 if wrong else c['c']
            row = dict(method=method, condition_id=c['id'], dataset=c['dataset'], seed=c['seed'],
                       Q=c['Q'], n=c['n'], extension_condition=c['extension_condition'],
                       occupied_clusters=count if supported else None, eligible=supported and not wrong,
                       SAPN_eligible=True, outcome_category='finite_wrong_k' if wrong else 'strict_eligible' if supported else 'input_unsupported')
            for metric in CORE.METRICS:
                row['baseline_' + metric] = .6 + .1 * ((c['seed'] - 23) % 3) if supported else None
                row['SAPN_' + metric] = .5 if supported else None
            if supported and index == 2:
                row['baseline_unqueried_BER'] = None
            record = dict(schema=CORE.PROJECTION_SCHEMA, method=method, condition_id=c['id'], evidence=[])
            if not supported:
                record['unavailable_reason'] = 'outside_predefined_admission_categories'
            else:
                case = c['id'] + '_c0_final'
                w = dict(case_id=case, status='RETURNED_FINITE_CONTRACT_FAILED' if wrong else 'RETURNED_FINITE',
                         execution_returned=True, finite_partition_retained=True, numeric_outputs_finite=True,
                         output_shapes_valid=True, n=c['n'], c=c['c'], strict_c=not wrong,
                         benchmark_contract_met=not wrong, sample_partition_count=count)
                if method == 'SSA':
                    w.update(views=3, state_checks=dict.fromkeys(CORE.STATE_FIELDS, True), initial_CL_nonzeros=[0, 0, 0],
                             final_CL_max_abs=0, full_class_truth_used=False, clustering_scores_computed=False)
                else:
                    w.update(graph_component_count=count, minimum_P=0., minimum_alpha=0., alpha_simplex_residual=0.,
                             native_label_partition_matches_graph=True, initializer_repeated_B_exact=True)
                if c['extension_condition']:
                    if method == 'SSA':
                        rt = dict(schema='ssa_portable', status='EXECUTED', matlab_exit_code=0, timed_out=False,
                                  failure=None, input_and_runtime_hashes_unchanged=True, class_labels_loaded=False,
                                  worker_status=w['status'])
                    else:
                        rt = dict(schema='pcbg_extension', status='PROCESS_EXITED', exit_code=0, worker_status=w['status'],
                                  finite_partition_retained=True, started_marker_bound=True, finished_marker_bound=True)
                else:
                    rt = dict(schema='legacy_launcher', status='COMPLETE_ZERO_EXIT_VALID_MARKERS', exit_code=0, cleanup_complete=True)
                record.update(case_id=case, worker=w, runtime=rt,
                              evidence=[{'key': 'public/synthetic/' + case + '.json', 'sha256': 'a'*64}])
            rows.append(row); records.append(record)
    metadata = {(r['method'], r['condition_id']): r for r in records}
    runs, groups = CORE.aggregate(PROTOCOL, rows, metadata)
    receipt = dict(status='SECONDARY_ANALYSIS_COMPLETE_FOR_ROOT_REVIEW',
        finished_utc=(datetime.fromisoformat(PROTOCOL['frozen_utc']) + timedelta(seconds=1)).isoformat(),
        protocol_sha256=app.PROTOCOL_SHA, tools_manifest_sha256=app.ANALYSIS_TOOLS_SHA,
        pipeline_complete_sha256='b'*64, preparation_complete_sha256='c'*64,
        seed_rows_sha256={'PCBG': 'd'*64, 'SSA': 'e'*64}, planned_slots_per_method=42,
        supported_slots_per_method=24, group_rows=16, original_two_group_overlap_rows=6,
        original_two_group_overlap_exclusions=[], raw_class_arrays_read=False, native_arrays_read=False,
        new_fits=0, CV_or_primary_modified=False, manuscript_modified=False,
        interpretation='Fixed selected outputs; relax only native exact-k after verifying all other clauses. '
                       'Not evidence about retuning CV, all stopping rules, or unsupported corpora.')
    return {'QUALIFICATION_METADATA.json': dict(schema=CORE.PROJECTION_SCHEMA, protocol_sha256=app.PROTOCOL_SHA, records=records),
            'SEED_SENSITIVITY.json': runs, 'GROUP_SENSITIVITY.json': groups, 'ANALYSIS_RECEIPT.json': receipt}


def write_bundle(root, data):
    root.mkdir()
    for name, value in data.items():
        app.write_json(root / name, value)
    hashes = {name: app.sha(root / name) if name in data else 'f'*64 for name in app.SOURCE_OUTPUTS}
    app.write_json(root / 'OUTPUT_SHA256.json', hashes)
    return app.sha(root / 'OUTPUT_SHA256.json')


class PublicReplayTests(unittest.TestCase):
    def test_standalone_replay_all_slots_and_independent_statistics(self):
        data = inputs_fixture()
        with tempfile.TemporaryDirectory(prefix='synthetic_exact_k_') as temp:
            root = Path(temp); bundle = root / 'data'; output = root / 'replay'
            digest = write_bundle(bundle, data)
            status = app.replay(CORE, PROTOCOL, bundle, digest, output)
            self.assertEqual(status, 'PUBLIC_PROJECTION_AND_AGGREGATION_REPLAY_MATCHED')
            groups = app.read_json(output / 'RECOMPUTED_GROUPS.json')
            self.assertEqual(len(groups), 16)
            self.assertAlmostEqual(groups[0]['ACC_mean'], .2); self.assertAlmostEqual(groups[0]['ACC_sample_sd'], .1)
            self.assertIsNone(groups[0]['unqueried_BER_mean']); self.assertEqual(groups[0]['unqueried_BER_pairs'], 2)
            counts = app.read_json(output / 'COVERAGE.json')
            for method in ('PCBG', 'SSA'):
                self.assertEqual((counts[method]['planned_slots'], counts[method]['supported_slots']), (42, 24))
                self.assertEqual(counts[method]['wrong_k_non_k_qualified_runs'], 1)
            with self.assertRaises(ValueError): app.replay(CORE, PROTOCOL, bundle, digest, output)

    def test_missing_duplicate_or_revised_derived_rows_refuse(self):
        for mutation in ('missing', 'duplicate', 'delta', 'null_to_zero', 'boolean_to_integer', 'integer_to_float'):
            data = inputs_fixture(); rows = data['SEED_SENSITIVITY.json']
            if mutation == 'missing': rows.pop()
            elif mutation == 'duplicate': rows[-1] = copy.deepcopy(rows[0])
            elif mutation == 'delta': rows[0]['paired_delta_ACC'] += .01
            elif mutation == 'null_to_zero': rows[-1]['paired_delta_ACC'] = 0
            elif mutation == 'boolean_to_integer': rows[0]['paired_qualified'] = 1
            else: rows[0]['requested_k'] = float(rows[0]['requested_k'])
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): app.recompute(CORE, PROTOCOL, data)

    def test_non_k_failure_cannot_be_hidden_under_wrong_k_category(self):
        data = inputs_fixture(); record = data['QUALIFICATION_METADATA.json']['records'][0]
        self.assertEqual(record['method'], 'PCBG')
        record['worker']['minimum_P'] = -1e-6
        with self.assertRaises(ValueError): app.recompute(CORE, PROTOCOL, data)
        # An honestly published exclusion remains an exclusion and preserves the group denominator.
        meta = {(r['method'], r['condition_id']): r for r in data['QUALIFICATION_METADATA.json']['records']}
        runs, groups = CORE.aggregate(PROTOCOL, data['SEED_SENSITIVITY.json'], meta)
        data['SEED_SENSITIVITY.json'], data['GROUP_SENSITIVITY.json'] = runs, groups
        data['ANALYSIS_RECEIPT.json']['original_two_group_overlap_exclusions'] = [
            dict(condition_id=r['condition_id'], reasons=r['exclusion_reasons']) for r in runs
            if r['overlaps_original_two_group_PCBG_scope'] and not r['paired_qualified']]
        _, result, counts = app.recompute(CORE, PROTOCOL, data)
        self.assertEqual(result[0]['qualified_pairs'], 2); self.assertIsNone(result[0]['ACC_mean'])
        self.assertEqual(counts['PCBG']['planned_slots'], 42)

    def test_projection_privacy_nested_containers_and_unknown_fields_refuse(self):
        for mutation in ('private_key', 'nested_list', 'absolute_evidence', 'traversal', 'unknown_field', 'unknown_schema', 'no_evidence'):
            data = inputs_fixture(); record = data['QUALIFICATION_METADATA.json']['records'][0]
            if mutation == 'private_key': record['worker']['minimum_P'] = {'Z:/private/user/model': 1}
            elif mutation == 'nested_list': record['worker']['numeric_outputs_finite'] = [{'secret': ['Z:/private/array']}]
            elif mutation == 'absolute_evidence': record['evidence'][0]['key'] = 'Z:/private/worker.json'
            elif mutation == 'traversal': record['evidence'][0]['key'] = 'public/../worker.json'
            elif mutation == 'unknown_field': record['runtime']['unknown'] = {'secret': 1}
            elif mutation == 'unknown_schema': record['schema'] = 'future'
            else: record['evidence'] = []
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): app.recompute(CORE, PROTOCOL, data)

    def test_manifest_integrity_flat_whitelist_and_hash_first(self):
        with tempfile.TemporaryDirectory(prefix='synthetic_exact_k_') as temp:
            root = Path(temp) / 'data'; digest = write_bundle(root, inputs_fixture())
            with patch.object(app, 'read_json', side_effect=AssertionError('Payload opened before manifest admission')):
                with self.assertRaises(ValueError): app.read_inputs(root, '0'*64)
            (root / 'raw.npy').write_bytes(b'not a real array')
            with self.assertRaises(ValueError): app.read_inputs(root, digest)
            (root / 'raw.npy').unlink()
            (root / 'SEED_SENSITIVITY.json').write_text('[]')
            with self.assertRaises(ValueError): app.read_inputs(root, digest)

    def test_json_nonfinite_and_duplicate_fields_refuse(self):
        with tempfile.TemporaryDirectory(prefix='synthetic_exact_k_') as temp:
            path = Path(temp) / 'input.json'
            for content in ('{"a":NaN}', '{"a":Infinity}', '{"a":1,"a":2}'):
                path.write_text(content)
                with self.subTest(content=content), self.assertRaises(ValueError): app.read_json(path)


if __name__ == '__main__':
    unittest.main(verbosity=2)
