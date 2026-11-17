"""Verify compact tie-sensitivity metrics from published predictions and external labels.

This command performs no fitting or selection and does not need optimizer states.
It requires the complete accompanying release, including its original paid queries.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import statistics
import sys

sys.dont_write_bytecode = True
for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def table(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-root', type=Path, required=True)
    parser.add_argument('--labels-map', type=Path, required=True,
                        help='JSON mapping five corpus names to local ordered .npy truth vectors; paths relative to JSON')
    parser.add_argument('--output', type=Path, required=True, help='New verification JSON outside the release')
    args = parser.parse_args()
    release, label_config, output = args.release_root.resolve(), args.labels_map.resolve(), args.output.resolve()
    require(not output.is_relative_to(release), 'Verification output must be outside the frozen release')
    evidence = release / 'evidence/tie_sensitivity'
    for row in read(evidence / 'PUBLIC_MANIFEST.json')['files']:
        path = (evidence / row['file']).resolve()
        require(path.is_relative_to(evidence) and sha(path) == row['sha256'], 'Compact manifest mismatch: ' + row['file'])
    config = read(label_config)
    datasets = ['3Sources', 'BBC', 'Reuters', 'Cora', 'CiteSeer']
    require(set(config) == set(datasets), 'Exactly the five original dataset keys are required')
    sys.path.insert(0, str(release / 'common'))
    import numpy as np
    import query_protocol as query
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    require(Path(query.__file__).resolve().parent == release / 'common', 'Unexpected query module')
    metrics = ['ACC', 'NMI', 'ARI', 'BER_train', 'BER_unqueried']
    source = table(evidence / 'reporting/unique_fit_metrics_558.csv')
    require(len(source) == 558, 'Unexpected unique-final roster')
    label_hashes, calculated, maximum = {}, {}, {name: 0.0 for name in metrics}
    for dataset in datasets:
        label_path = (label_config.parent / config[dataset]).resolve()
        truth = np.load(label_path, allow_pickle=False).reshape(-1)
        label_hashes[dataset] = sha(label_path)
        for seed in range(504, 514):
            block = release / f'evidence/v9/allocation_repeat/paid_queries/{dataset}/seed{seed}.npz'
            with np.load(block, allow_pickle=False) as z:
                test, pool = z['test'].copy(), z['pool'].copy()
                pairs, answers = z['pairs'].copy(), z['answers'].copy()
            require(np.array_equal(np.sort(np.concatenate([pool, test])), np.arange(len(truth))), 'Pool/test partition changed')
            require(np.array_equal(query.answer_queries(pairs, truth), answers), 'Class labels disagree with paid relationships')
            ep = query.sample_query_stream(test, min(10000, len(test) * (len(test) - 1) // 2), seed + 51000)
            ea = query.answer_queries(ep, truth)
            for row in source:
                if row['dataset'] != dataset or int(row['seed']) != seed:
                    continue
                prediction = (release / row['prediction_path']).resolve()
                require(prediction.is_relative_to(release) and sha(prediction) == row['prediction_sha256'], 'Prediction hash mismatch')
                require(row['eligible'] == 'True' and row['status'] == 'ok', 'This compact author-run replay expects all 558 eligible fits')
                with np.load(prediction, allow_pickle=False) as z:
                    pred = z['labels'].copy()
                require(pred.shape == truth.shape, 'Prediction/truth length mismatch')
                _, left = np.unique(truth[test], return_inverse=True)
                _, right = np.unique(pred[test], return_inverse=True)
                confusion = np.zeros((left.max() + 1, right.max() + 1), dtype=np.int64)
                np.add.at(confusion, (left, right), 1)
                i, j = linear_sum_assignment(-confusion)
                budget = int(row['budget'])
                values = dict(ACC=float(confusion[i, j].sum() / len(test)),
                    NMI=float(normalized_mutual_info_score(truth[test], pred[test], average_method='arithmetic')),
                    ARI=float(adjusted_rand_score(truth[test], pred[test])),
                    BER_train=query.balanced_pair_error(pred, pairs[:budget], answers[:budget]),
                    BER_unqueried=query.balanced_pair_error(pred, ep, ea))
                for metric, value in values.items():
                    require(value is not None, 'Undefined BER is not present in the preserved author run')
                    error = abs(value - float(row[metric]))
                    maximum[metric] = max(maximum[metric], error)
                    require(error <= 1e-12, 'Metric differs: ' + row['fit_id'] + '/' + metric)
                calculated[row['fit_id']] = values
    logical = table(evidence / 'reporting/logical_metrics_1500.csv')
    require(len(logical) == 1500 and len(calculated) == 558, 'Incomplete replay')
    for row in logical:
        for metric in metrics:
            require(abs(float(row[metric]) - calculated[row['fit_id']][metric]) <= 1e-12, 'Logical metric differs')
    groups, point_error = {}, 0.0
    points = table(evidence / 'reporting/paired_five_metrics_6000.csv')
    for row in points:
        metric = row['metric']
        value = 100 * (calculated[row['full_fit_id']][metric] - calculated[row['control_fit_id']][metric])
        error = abs(value - float(row['Full_minus_control_pp']))
        point_error = max(point_error, error)
        require(error <= 1e-10, 'Paired difference differs')
        key = tuple(row[name] for name in ['policy', 'dataset', 'budget', 'comparator', 'metric'])
        groups.setdefault(key, []).append(value)
    summaries = table(evidence / 'reporting/paired_summary_600.csv')
    summary_error = 0.0
    for row in summaries:
        key = tuple(row[name] for name in ['policy', 'dataset', 'budget', 'comparator', 'metric'])
        values = groups[key]
        require(len(values) == 10 and row['complete'] == 'True', 'Incomplete paired block')
        expected = dict(mean_pp=statistics.mean(values), sample_SD_pp=statistics.stdev(values),
                        median_pp=statistics.median(values), minimum_pp=min(values), maximum_pp=max(values))
        for name, value in expected.items():
            error = abs(value - float(row[name]))
            summary_error = max(summary_error, error)
            require(error <= 1e-10, 'Summary differs: ' + str(key))
        signed = [-v for v in values] if row['metric'].startswith('BER') else values
        counts = [sum(v > 1e-12 for v in signed), sum(abs(v) <= 1e-12 for v in signed), sum(v < -1e-12 for v in signed)]
        require(counts == [int(row[name]) for name in ['wins', 'ties', 'losses']], 'W/T/L differs')
    require(len(points) == 6000 and len(summaries) == 600, 'Unexpected paired roster')
    report = dict(status='PASS', scope='Compact prediction replay; no selection, fitting, or optimizer-state use',
        script_sha256=sha(Path(__file__)), public_manifest_sha256=sha(evidence / 'PUBLIC_MANIFEST.json'),
        labels_sha256=label_hashes, query_protocol_sha256=sha(release / 'common/query_protocol.py'),
        unique_predictions=558, logical_records=1500, paired_points=6000, summary_groups=600,
        metric_max_absolute_errors=maximum, paired_max_absolute_error_pp=point_error,
        summary_max_absolute_error_pp=summary_error, all_WTL_counts_identical=True, new_fits_run=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
