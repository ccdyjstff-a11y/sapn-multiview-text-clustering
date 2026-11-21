"""Public metadata/aggregation replay only; default checks tools without reading data."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import re
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
CORE_SHA = 'f63fe05614e9e4b304223e26a3b01ea3a709f570bb1fa81fb3a0a50c1837a1db'
PROTOCOL_SHA = 'd940ffa84917ca6cd7349dc300c17a793e3f187db6fec75a6c756a27c9e1fef9'
ANALYSIS_TOOLS_SHA = '539ff939c1bfcb30681111191ddaccfab5c21bd28c07f6aebc937ec4ac1fc7e7'
DATA_JSON = {'QUALIFICATION_METADATA.json', 'SEED_SENSITIVITY.json',
             'GROUP_SENSITIVITY.json', 'ANALYSIS_RECEIPT.json'}
SOURCE_OUTPUTS = DATA_JSON | {'seed_sensitivity.csv', 'group_sensitivity.csv',
                             'exact_k_sensitivity_en.tex', 'exact_k_sensitivity_zh.tex'}
DATA_FILES = DATA_JSON | {'OUTPUT_SHA256.json'}
UNAVAILABLE = {'outside_predefined_admission_categories', 'not_bound_in_pre_truth_receipt',
               'invalid_json_metadata', 'unknown_json_metadata_schema', 'worker_summary_missing',
               'missing_or_unknown_route_metadata_schema'}
RUNTIME_FIELDS = {
    'legacy_launcher': {'schema', 'status', 'exit_code', 'cleanup_complete'},
    'ssa_portable': {'schema', 'status', 'matlab_exit_code', 'timed_out', 'failure',
                     'input_and_runtime_hashes_unchanged', 'class_labels_loaded', 'worker_status'},
    'pcbg_extension': {'schema', 'status', 'exit_code', 'worker_status', 'finite_partition_retained',
                       'started_marker_bound', 'finished_marker_bound'},
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def is_hash(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, 'Duplicate JSON key')
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError('Nonfinite JSON constant: ' + value)


def read_json(path):
    need(Path(path).stat().st_size <= 8 * 1024**2, 'Unexpectedly large metadata file')
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),
                      object_pairs_hook=unique_object, parse_constant=reject_constant)


def safe_key(key):
    need(isinstance(key, str) and re.fullmatch(r'[A-Za-z0-9_./-]+', key)
         and not key.startswith('/') and all(p not in ('', '.', '..') for p in key.split('/')),
         'Unsafe relative evidence key')
    return key


def equal(actual, expected, where='value'):
    """Check every key/slot, preserving None and booleans; tolerate only roundoff."""
    if actual is None or expected is None or type(actual) is bool or type(expected) is bool:
        need(type(actual) is type(expected) and actual == expected, 'Null/boolean mismatch: ' + where)
    elif isinstance(actual, dict) and isinstance(expected, dict):
        need(set(actual) == set(expected), 'Field whitelist mismatch: ' + where)
        for key in actual:
            equal(actual[key], expected[key], where + '.' + key)
    elif isinstance(actual, list) and isinstance(expected, list):
        need(len(actual) == len(expected), 'List length mismatch: ' + where)
        for index, (a, b) in enumerate(zip(actual, expected)):
            equal(a, b, where + '[' + str(index) + ']')
    elif type(actual) is int or type(expected) is int:
        need(type(actual) is type(expected) and actual == expected, 'Integer/count mismatch: ' + where)
    elif type(actual) in (int, float) and type(expected) in (int, float):
        need(math.isfinite(actual) and math.isfinite(expected)
             and abs(actual - expected) <= 1e-12, 'Numeric mismatch: ' + where)
    else:
        need(type(actual) is type(expected) and actual == expected, 'Value mismatch: ' + where)


def load_core():
    need(sha(HERE / 'non_k_core.py') == CORE_SHA and sha(HERE / 'PROTOCOL.json') == PROTOCOL_SHA,
         'Frozen author core/protocol changed')
    spec = importlib.util.spec_from_file_location('public_frozen_non_k_core', HERE / 'non_k_core.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module, read_json(HERE / 'PROTOCOL.json')


def verify_tools():
    path = HERE / 'TOOLS_SHA256.json'
    need(sha(path) == (HERE / 'TOOLS_SHA256.sha256').read_text().strip(), 'Public tool manifest changed')
    manifest = read_json(path)
    need(manifest['schema'] == 'sapn-exact-k-public-tools-v1', 'Unknown public tool schema')
    for name, expected in manifest['files'].items():
        safe_key(name)
        target = (HERE / name).resolve()
        need(target.is_relative_to(HERE) and not (HERE / name).is_symlink()
             and is_hash(expected) and sha(target) == expected, 'Public tool member changed')
    return load_core()


def validate_projection(core, protocol, projection):
    need(isinstance(projection, dict) and set(projection) == {'schema', 'protocol_sha256', 'records'}
         and projection['schema'] == core.PROJECTION_SCHEMA and projection['protocol_sha256'] == PROTOCOL_SHA,
         'Wrong qualification projection schema/protocol')
    records = projection['records']; need(isinstance(records, list) and len(records) == 84, 'Projection needs all 84 slots')
    conditions = {(m, c['id']): c for m, spec in protocol['methods'].items() for c in spec['conditions']}
    indexed = {}
    for record in records:
        need(isinstance(record, dict), 'Projection record is not an object')
        base = {'schema', 'method', 'condition_id', 'evidence'}
        branch = {'unavailable_reason'} if 'unavailable_reason' in record else {'case_id', 'worker', 'runtime'}
        need(set(record) == base | branch and record['schema'] == core.PROJECTION_SCHEMA,
             'Unknown projection record fields/schema')
        key = (record['method'], record['condition_id'])
        need(key in conditions and key not in indexed, 'Duplicate/unknown projection identity')
        need(isinstance(record['evidence'], list), 'Evidence must be an array')
        seen = set()
        for ref in record['evidence']:
            need(isinstance(ref, dict) and set(ref) == {'key', 'sha256'} and is_hash(ref['sha256']), 'Unknown evidence descriptor')
            name = safe_key(ref['key'])
            need(name.startswith(('public/', 'revision/')) and name.endswith('.json') and name not in seen,
                 'Evidence key must identify one relative JSON record')
            seen.add(name)
        if 'unavailable_reason' in record:
            need(record['unavailable_reason'] in UNAVAILABLE, 'Unknown metadata-unavailability reason')
        else:
            need(bool(record['evidence']), 'Worker projection has no source evidence descriptor')
            need(isinstance(record['case_id'], str) and
                 re.fullmatch(re.escape(record['condition_id']) + r'_c[0-9]+_final', record['case_id']),
                 'Unsafe or non-final case identity')
            w, rt = record['worker'], record['runtime']
            need(isinstance(w, dict) and isinstance(rt, dict), 'Worker/runtime projection must be objects')
            # Equality rejects unknown keys and malformed nested containers; no cleaning is performed here.
            equal(w, core.project_worker(record['method'], w), 'worker_public_whitelist')
            kind = rt.get('schema')
            need(kind in RUNTIME_FIELDS and set(rt) <= RUNTIME_FIELDS[kind], 'Unknown runtime projection schema/fields')
            expected_kind = ('ssa_portable' if record['method'] == 'SSA' else 'pcbg_extension') if conditions[key]['extension_condition'] else 'legacy_launcher'
            need(kind == expected_kind, 'Runtime origin differs from the frozen condition scope')
            equal(rt, core.safe_projection(rt), 'runtime_public_whitelist')
        indexed[key] = record
    need(set(indexed) == set(conditions), 'Lost projection denominator')
    return indexed


def independent_groups(core, protocol, runs):
    groups = []
    for method in ('PCBG', 'SSA'):
        for dataset in protocol['supported_corpora']:
            for budget in protocol['budgets']:
                rs = [r for r in runs if (r['method'], r['dataset'], r['Q']) == (method, dataset, budget)]
                need(len(rs) == 3, 'Group denominator changed')
                row = dict(method=method, dataset=dataset, Q=budget, planned_pairs=3,
                           primary_eligible_pairs=sum(r['eligible'] and r['SAPN_eligible'] for r in rs),
                           non_k_qualified_runs=sum(r['non_k_qualified'] for r in rs),
                           qualified_pairs=sum(r['paired_qualified'] for r in rs))
                for metric in core.METRICS:
                    values = [r['paired_delta_' + metric] for r in rs if r['paired_delta_' + metric] is not None]
                    n = len(values)
                    mean = math.fsum(values) / 3 if n == 3 else None
                    sd = math.sqrt(math.fsum((x - mean)**2 for x in values) / 2) if n == 3 else None
                    row.update({metric + '_pairs': n, metric + '_mean': mean, metric + '_sample_sd': sd})
                groups.append(row)
    return groups


def coverage(core, protocol, runs, groups):
    result = {}
    for method in ('PCBG', 'SSA'):
        rs = [r for r in runs if r['method'] == method]
        gs = [g for g in groups if g['method'] == method]
        result[method] = dict(planned_slots=len(rs), supported_slots=sum(r['dataset'] in protocol['supported_corpora'] for r in rs),
            supported_groups=len(gs), original_categories=dict(sorted(Counter(r['outcome_category'] for r in rs).items())),
            primary_eligible_runs=sum(r['eligible'] for r in rs),
            primary_paired_runs=sum(r['eligible'] and r['SAPN_eligible'] for r in rs),
            non_k_qualified_runs=sum(r['non_k_qualified'] for r in rs),
            secondary_paired_runs=sum(r['paired_qualified'] for r in rs),
            wrong_k_non_k_qualified_runs=sum(r['outcome_category'] == 'finite_wrong_k' and r['non_k_qualified'] for r in rs),
            complete_groups_by_metric={metric: sum(g[metric + '_pairs'] == 3 for g in gs) for metric in core.METRICS},
            exclusion_reasons=dict(sorted(Counter(reason for r in rs for reason in r['exclusion_reasons']).items())))
        need(result[method]['planned_slots'] == 42 and result[method]['supported_slots'] == 24
             and result[method]['supported_groups'] == 8, 'Coverage scope changed')
    return result


def validate_receipt(receipt, protocol, runs):
    fixed = dict(status='SECONDARY_ANALYSIS_COMPLETE_FOR_ROOT_REVIEW', protocol_sha256=PROTOCOL_SHA,
        tools_manifest_sha256=ANALYSIS_TOOLS_SHA, planned_slots_per_method=42, supported_slots_per_method=24,
        group_rows=16, original_two_group_overlap_rows=6, raw_class_arrays_read=False, native_arrays_read=False,
        new_fits=0, CV_or_primary_modified=False, manuscript_modified=False,
        interpretation='Fixed selected outputs; relax only native exact-k after verifying all other clauses. '
                       'Not evidence about retuning CV, all stopping rules, or unsupported corpora.')
    extra = {'finished_utc', 'pipeline_complete_sha256', 'preparation_complete_sha256',
             'seed_rows_sha256', 'original_two_group_overlap_exclusions'}
    need(isinstance(receipt, dict) and set(receipt) == set(fixed) | extra, 'Unknown analysis receipt fields')
    for key, value in fixed.items():
        need(type(receipt[key]) is type(value) and receipt[key] == value, 'Analysis receipt differs: ' + key)
    finished = datetime.fromisoformat(receipt['finished_utc'])
    need(finished.tzinfo is not None and finished >= datetime.fromisoformat(protocol['frozen_utc']), 'Analysis predates protocol freeze')
    need(is_hash(receipt['pipeline_complete_sha256']) and is_hash(receipt['preparation_complete_sha256']), 'Invalid lineage hash')
    need(isinstance(receipt['seed_rows_sha256'], dict) and set(receipt['seed_rows_sha256']) == {'PCBG', 'SSA'}
         and all(is_hash(h) for h in receipt['seed_rows_sha256'].values()), 'Invalid original seed-table hash mapping')
    overlap = [r for r in runs if r['overlaps_original_two_group_PCBG_scope']]
    need(len(overlap) == 6, 'Historical overlap changed')
    equal(receipt['original_two_group_overlap_exclusions'],
          [dict(condition_id=r['condition_id'], reasons=r['exclusion_reasons']) for r in overlap if not r['paired_qualified']],
          'historical_overlap_exclusions')


def recompute(core, protocol, inputs):
    metadata = validate_projection(core, protocol, inputs['QUALIFICATION_METADATA.json'])
    source = inputs['SEED_SENSITIVITY.json']
    runs, expected_groups = core.aggregate(protocol, source, metadata)
    # The published seed table contains derived flags/deltas too: recompute and check all of them.
    equal(source, runs, 'published_84_seed_rows')
    groups = independent_groups(core, protocol, runs)
    equal(groups, expected_groups, 'independent_vs_frozen_group_arithmetic')
    equal(groups, inputs['GROUP_SENSITIVITY.json'], 'published_16_groups')
    validate_receipt(inputs['ANALYSIS_RECEIPT.json'], protocol, runs)
    return runs, groups, coverage(core, protocol, runs, groups)


def read_inputs(root, expected_manifest):
    root = Path(root).resolve()
    need(is_hash(expected_manifest), 'An explicit published output-manifest SHA256 is required')
    need(root.is_dir() and {p.name for p in root.iterdir()} == DATA_FILES, 'Input directory must contain exactly the five planned JSON files')
    for name in DATA_FILES:
        path = root / name
        need(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root), 'Input file escapes resource root')
    need(sha(root / 'OUTPUT_SHA256.json') == expected_manifest, 'Published source output manifest differs')
    manifest = read_json(root / 'OUTPUT_SHA256.json')
    need(isinstance(manifest, dict) and set(manifest) == SOURCE_OUTPUTS and all(is_hash(h) for h in manifest.values()),
         'Original output manifest whitelist differs')
    snapshots = {'OUTPUT_SHA256.json': expected_manifest}
    inputs = {}
    for name in sorted(DATA_JSON):
        need(sha(root / name) == manifest[name], 'Published input changed: ' + name)
        inputs[name] = read_json(root / name); snapshots[name] = manifest[name]
    return inputs, snapshots


def write_json(path, data):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write('\n')


def replay(core, protocol, input_root, expected_manifest, output):
    root, output = Path(input_root).resolve(), Path(output).resolve()
    need(not output.exists() and not output.is_relative_to(root) and not output.is_relative_to(HERE),
         'Replay output must be new and outside both inputs and tools')
    inputs, snapshots = read_inputs(root, expected_manifest)
    runs, groups, counts = recompute(core, protocol, inputs)
    for name, expected in snapshots.items():
        need(sha(root / name) == expected, 'Input changed during replay')
    output.mkdir(parents=False, exist_ok=False)
    for name, data in (('RECOMPUTED_SEED_SENSITIVITY.json', runs), ('RECOMPUTED_GROUPS.json', groups), ('COVERAGE.json', counts)):
        write_json(output / name, data)
    with (output / 'recomputed_groups.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(groups[0])); writer.writeheader(); writer.writerows(groups)
    for name, expected in snapshots.items():
        need(sha(root / name) == expected, 'Input changed while writing replay outputs')
    receipt = dict(status='PUBLIC_PROJECTION_AND_AGGREGATION_REPLAY_MATCHED', finished_utc=datetime.now(timezone.utc).isoformat(),
        source_output_manifest_sha256=expected_manifest, input_file_sha256=snapshots,
        frozen_protocol_sha256=PROTOCOL_SHA, frozen_core_sha256=CORE_SHA, replay_source_sha256=sha(Path(__file__)),
        paired_slots_checked=84, groups_checked=16, comparison_absolute_tolerance=1e-12,
        class_arrays_read=False, native_arrays_read=False, fits_run=0, CV_repeated=False,
        original_worker_execution_reverified=False, projection_lineage_reverified_from_original_workers=False,
        interpretation='Verified the published minimal projection and its fixed-selection aggregate arithmetic; '
                       'the source manifest SHA must come from the trusted publication. Original workers and CV are outside this replay.')
    write_json(output / 'REPLAY_RECEIPT.json', receipt)
    write_json(output / 'OUTPUT_SHA256.json', {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    return receipt['status']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', action='store_true')
    parser.add_argument('--input-root', type=Path)
    parser.add_argument('--expect-source-output-manifest-sha256')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    core, protocol = verify_tools()
    if not args.replay:
        result = dict(status='FROZEN_PUBLIC_TOOL_PREFLIGHT_PASS', result_data_read=False, files_written=False)
    else:
        need(args.input_root is not None and args.output is not None, '--replay requires --input-root and --output')
        result = dict(status=replay(core, protocol, args.input_root, args.expect_source_output_manifest_sha256, args.output))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
