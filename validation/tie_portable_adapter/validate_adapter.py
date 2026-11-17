"""Validate the adapter by planning and replay only; never invoke a real fit.

Requires completed historical blocks and the already completed frozen sensitivity
study. Hard links are read only in this harness; no optimizer is called. Retain
the small report and logs, not the temporary linked scientific states, for release.
"""
from pathlib import Path
import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import uuid

sys.dont_write_bytecode = True


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def equal_table(new, reference, keys):
    first, second = rows(new), rows(reference)
    lookup = {tuple(r[k] for k in keys): r for r in second}
    assert len(first) == len(second) == len(lookup)
    fields = set(first[0]) & set(second[0]) - {'source_block'}
    comparisons = 0
    for row in first:
        old = lookup[tuple(row[k] for k in keys)]
        for field in fields:
            assert row[field] == old[field], (new.name, row, field, row[field], old[field])
            comparisons += 1
    return dict(rows=len(first), common_fields=sorted(fields), equal_cells=comparisons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ['adapter', 'reference', 'common-dir', 'historical-root', 'data-map', 'labels-map', 'output']:
        parser.add_argument('--' + arg, type=Path, required=True)
    a = parser.parse_args()
    for key, value in vars(a).items():
        setattr(a, key, value.resolve())
    a.output.mkdir(parents=True, exist_ok=True)
    logs = a.output / 'logs'
    logs.mkdir(exist_ok=True)
    report = dict(new_fits_run=0, optimizer_called=False, train_executed=False,
                  train_dry_run_executed=False, score_replay_executed=False)

    def run(label, args, expected=0):
        result = subprocess.run([sys.executable, '-B', '-I', *map(str, args)], capture_output=True, text=True)
        (logs / (label + '.txt')).write_text(result.stdout + result.stderr, encoding='utf-8')
        assert result.returncode == expected, (label, result.returncode, result.stdout[-1500:], result.stderr[-1500:])
        print(label, 'PASS', flush=True)

    def command(script, phase, output, common=None, data_map=None):
        return [script, phase, '--common-dir', common or a.common_dir, '--historical-root', a.historical_root,
                '--data-map', data_map or a.data_map, '--output', output]

    ast.parse(a.adapter.read_text(encoding='utf-8'))
    report['ast_parse'] = True
    for phase in ['', 'plan', 'train', 'score']:
        run('help_' + (phase or 'root'), [a.adapter, *([phase] if phase else []), '--help'])
    run('reject_plan_labels', command(a.adapter, 'plan', a.output / 'parser_probe') + ['--labels-map', a.labels_map], 2)
    run('reject_train_labels', command(a.adapter, 'train', a.output / 'parser_probe') + ['--labels-map', a.labels_map], 2)
    original = a.output / 'original_plan'
    if not (original / 'PLAN_FREEZE.json').exists():
        run('original_plan', command(a.adapter, 'plan', original))
    else:
        print('original_plan: using existing completed adapter plan; train dry-run will recheck its hashes', flush=True)
    p = read(original / 'PLAN.json')
    assert p['summary']['missing_unique_fits'] == 172
    assert p['summary']['verified_reuse_unique_fits'] == 386
    assert p['summary']['unique_final_requirements'] == 558
    report['plan_summary'] = p['summary']
    report['choices_comparison'] = equal_table(original / 'selection_policy_index_900.csv',
        a.reference / 'selection_policy_index_900.csv', ['dataset', 'seed', 'budget', 'allocation', 'policy'])
    report['targets_comparison'] = equal_table(original / 'logical_final_targets_1500.csv',
        a.reference / 'logical_final_targets_1500.csv', ['dataset', 'seed', 'budget', 'policy', 'logical_method'])
    assert {r['fit_id'] for r in rows(original / 'missing_fits.csv')} == {r['fit_id'] for r in rows(a.reference / 'missing_fits.csv')}
    assert {r['fit_id'] for r in rows(original / 'verified_reuse_fits.csv')} == {r['fit_id'] for r in rows(a.reference / 'verified_reuse_fits.csv')}
    report['exact_missing_and_reuse_id_sets'] = True

    relocated = a.output / ('relocated_' + uuid.uuid4().hex[:12])
    (relocated / 'tools').mkdir(parents=True)
    (relocated / 'common').mkdir()
    shutil.copy2(a.adapter, relocated / 'tools' / a.adapter.name)
    for source in a.common_dir.iterdir():
        if source.is_file() and source.suffix in ['.py', '.json']:
            shutil.copy2(source, relocated / 'common' / source.name)
    data = {d: os.path.relpath((a.data_map.parent / value).resolve(), relocated)
            for d, value in read(a.data_map).items()}
    relocated_map = relocated / 'data_map.json'
    relocated_map.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    run('relocated_plan', command(relocated / 'tools' / a.adapter.name, 'plan', relocated / 'plan',
                                 relocated / 'common', relocated_map))
    report['relocated_relative_directory'] = relocated.relative_to(a.output).as_posix()
    report['relocated_byte_equal_plan_artifacts'] = []
    for name, digest in read(original / 'PLAN_FREEZE.json').items():
        assert sha(relocated / 'plan' / name) == digest
        report['relocated_byte_equal_plan_artifacts'].append(name)
    run('train_dry_run', command(a.adapter, 'train', original) + ['--dry-run'])
    report['train_dry_run_executed'] = True
    run('reject_score_without_seal', command(a.adapter, 'score', original) + ['--labels-map', a.labels_map], 1)

    # Link only already completed, hash-sealed states. Do not call train or fit.
    prior_seal = read(a.reference / 'ALL_FINALS_SEALED.json')
    linked = {}
    for row in p['unique']:
        if row['status'] != 'missing_fit':
            continue
        rel_block = Path('fits') / row['source_block']
        source = a.reference / rel_block
        target = original / rel_block
        target.mkdir(parents=True, exist_ok=True)
        files = [source / name for name in ['PROTOCOL.json', 'inputs.npz', 'used_graph.npz']]
        files += list(source.glob(row['expected_stem'] + '*'))
        assert len(files) >= 5
        for source_file in files:
            relative = source_file.relative_to(a.reference)
            expected = prior_seal['files'].get(str(relative), prior_seal['files'].get(relative.as_posix()))
            assert expected and sha(source_file) == expected
            target_file = target / source_file.name
            if not target_file.exists():
                os.link(source_file, target_file)
            assert sha(target_file) == expected
            linked[relative.as_posix()] = expected
    report['linked_existing_files'] = len(linked)
    report['linked_existing_finals'] = 172

    # Use the adapter's normal validation gate to construct a replay-only seal.
    spec = importlib.util.spec_from_file_location('portable_tie_replay', a.adapter)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    options = module.arguments(list(map(str, command(a.adapter, 'score', original)[1:])) + ['--labels-map', str(a.labels_map)])
    context = module.Context(options)
    frozen = module.load_plan(context)
    records = module.final_records(context, frozen)
    module.freeze(original / 'ALL_FINALS_SEALED.json', dict(plan_freeze_sha256=sha(original / 'PLAN_FREEZE.json'),
        logical_records=1500, unique_final_records=558, planned_new_fits=172, reused_unique=386,
        records=records, test_scores_read=False, validation_replay_only=True,
        terminal_states='Hard links to existing frozen results; train and optimizer never invoked'))
    module.score(context)
    # The audit hook confines writes to original_plan for the rest of this process.
    report['score_replay_executed'] = True
    metrics = ['ACC', 'NMI', 'ARI', 'BER_train', 'BER_unqueried']
    old = {tuple(r[k] for k in ['policy', 'dataset', 'seed', 'budget', 'logical_method']): r
           for r in rows(a.reference / 'evaluation' / 'logical_metrics_1500.csv')}
    errors = []
    for row in rows(original / 'evaluation' / 'logical_metrics_1500.csv'):
        reference = old[tuple(row[k] for k in ['policy', 'dataset', 'seed', 'budget', 'logical_method'])]
        for metric in metrics:
            errors.append(abs(float(row[metric]) - float(reference[metric])))
        assert row['eligible'] == reference['eligible']
    assert len(errors) == 7500 and max(errors) <= 1e-12
    report['logical_metric_comparisons'] = len(errors)
    report['logical_metrics_max_abs_error'] = max(errors)
    report['paired_comparison'] = equal_table(original / 'evaluation' / 'paired_five_metrics_6000.csv',
        a.reference / 'evaluation' / 'paired_five_metrics_6000.csv', ['policy', 'dataset', 'seed', 'budget', 'comparator', 'metric'])
    report['summary_comparison'] = equal_table(original / 'evaluation' / 'paired_summary_600.csv',
        a.reference / 'evaluation' / 'paired_summary_600.csv', ['policy', 'dataset', 'budget', 'comparator', 'metric'])
    for relative, digest in linked.items():
        assert sha(a.reference / relative) == digest
    report['all_linked_sources_unchanged_after_replay'] = True
    report['adapter_sha256'] = sha(a.adapter)
    module.freeze(original / 'VALIDATION_REPORT.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
