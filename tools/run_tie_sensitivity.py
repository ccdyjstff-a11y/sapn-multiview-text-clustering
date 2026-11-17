"""Portable, frozen tie-policy planning, missing final fitting and sealed scoring.

The supplied common package is imported unchanged. Historical blocks must have
been produced by that exact package and software environment. No raw class
vector is accepted by plan or train. All output references are root-relative.
"""
from pathlib import Path
import argparse
import collections
import csv
import hashlib
import io
import json
import math
import os
import statistics
import sys

sys.dont_write_bytecode = True
for _name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_name] = '1'

DATA = ['3Sources', 'BBC', 'Reuters', 'Cora', 'CiteSeer']
SEEDS = list(range(504, 514))
BUDGETS = [160, 400]
METHODS = ['full', 'consensus-only', 'views-only']
LOGICAL = [m + '-CV' for m in METHODS] + [m + '-at-full-lambda' for m in METHODS[1:]]
POLICIES = ['primary_default', 'sensitivity_min', 'sensitivity_max']
VALUES = [20., 2., 200.]
METRICS = ['ACC', 'NMI', 'ARI', 'BER_train', 'BER_unqueried']
RULES = dict(version='tie_sensitivity_portable_v1', datasets=DATA, seeds=SEEDS,
             budgets=BUDGETS, methods=METHODS, policies=POLICIES, grid=VALUES,
             tie_tolerance=1e-12, primary='Original default-first selection',
             sensitivity='Minimum/maximum numeric lambda in original best mean BER tie',
             fallback='No eligible finite candidate: preserve original default and failure flag for all policies',
             finals='Complete Q-query finals only; independent CV and each policy Full-lambda controls',
             interpretation='Posthoc sensitivity; primary preserved; no independent confirmation or p-values',
             evaluation='Pool complement; up to 10000 pairs sampled at seed+51000 after global seal',
             summary='Full-minus-control percentage points; all ten eligible pairs required; BER lower is better')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def freeze_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == payload, 'Existing immutable artifact differs: ' + str(path))
    else:
        with path.open('xb') as stream:
            stream.write(payload)


def freeze(path, value):
    freeze_bytes(path, (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8'))


def table(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ['fit_id']
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    freeze_bytes(path, stream.getvalue().encode('utf-8-sig'))


def safe(root, name):
    path = (root / name).resolve()
    require(path.is_relative_to(root), 'Reference escapes its declared root: ' + name)
    return path


def mapping(path):
    values = read(path)
    require(isinstance(values, dict) and set(values) == set(DATA), 'Map keys must be exactly ' + ', '.join(DATA))
    require(all(isinstance(v, str) for v in values.values()), 'Map values must be paths')
    return {d: (path.parent / values[d]).resolve() for d in DATA}


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='phase', required=True)
    for name in ['plan', 'train', 'score']:
        sub = subs.add_parser(name)
        sub.add_argument('--common-dir', type=Path, required=True,
                         help='Exact unchanged common package that created the historical blocks')
        sub.add_argument('--historical-root', type=Path, required=True,
                         help='Directory containing <dataset>/seed504 through seed513')
        sub.add_argument('--data-map', type=Path, required=True,
                         help='JSON mapping the five corpus keys to prepared directories; relative to this JSON')
        sub.add_argument('--output', type=Path, required=True, help='Separate immutable sensitivity output')
        if name == 'score':
            sub.add_argument('--labels-map', type=Path, required=True,
                             help='External ordered class vectors (.npy); relative to this JSON')
        if name == 'train':
            sub.add_argument('--dry-run', action='store_true', help='Verify frozen inputs only; do not fit or seal')
    args = parser.parse_args(argv)
    if sys.flags.optimize:
        parser.error('Do not use -O: the unchanged common package uses assertions for integrity checks')
    for key in ['common_dir', 'historical_root', 'data_map', 'output']:
        setattr(args, key, getattr(args, key).resolve())
    if hasattr(args, 'labels_map'):
        args.labels_map = args.labels_map.resolve()
    return args


class Context:
    def __init__(self, args):
        self.a = args
        self.data = mapping(args.data_map)
        protected = [args.common_dir, args.historical_root, *self.data.values()]
        for root in protected:
            require(not args.output.is_relative_to(root) and not root.is_relative_to(args.output),
                    'Output must be separate from common, historical blocks and prepared data')
        self.allow_labels = False
        sys.addaudithook(self.guard)
        sys.path.insert(0, str(args.common_dir))
        import numpy as np
        import allocation_io as aio
        import query_protocol as query
        require(Path(aio.__file__).resolve().parent == args.common_dir, 'Wrong common package imported')
        self.np, self.aio, self.query = np, aio, query

    def guard(self, event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else 0
        writing = ((isinstance(mode, str) and any(c in mode for c in 'wax+')) or
                   (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)))
        if writing and not path.is_relative_to(self.a.output):
            raise PermissionError('Writes are confined to --output: ' + str(path))
        banned = {'evaluation.json', 'labels.npy', 'y.npy', 'truth.npy', 'all_logical_evaluation.json',
                  'all500_logical.csv', 'postseal_evaluation_pairs.npz', 'EVALUATION_MANIFEST.json'}
        if not self.allow_labels and (path.name in banned or 'postseal_evaluation' in path.parts):
            raise PermissionError('No test outcomes before complete final seal validation: ' + str(path))

    def block(self, dataset, seed):
        return self.a.historical_root / dataset / f'seed{seed}'

    def input(self, block, protocol):
        return self.aio.paid_inputs(block / 'inputs.npz', protocol['n'], BUDGETS)

    def terminal(self, block, row):
        protocol = read(block / 'PROTOCOL.json')
        inp = self.input(block, protocol)
        q = row['budget']
        P, N = self.query.signed_pairs(inp['pairs'][:q], inp['answers'][:q])
        require(self.aio.array_digest(P) == row['P_sha256'] and self.aio.array_digest(N) == row['N_sha256'],
                'Complete-query identity changed: ' + row['fit_id'])
        for name, field in [('PROTOCOL.json', 'source_protocol_sha256'),
                            ('inputs.npz', 'inputs_sha256'), ('used_graph.npz', 'graph_sha256')]:
            require(sha(block / name) == row[field], 'Copied numerical identity changed: ' + str(block / name))
        return self.aio.terminal(block, row['expected_stem'], row['allocation'], row['lambda_pair'], P, N, protocol)


def plan(ctx):
    a, np, aio, query = ctx.a, ctx.np, ctx.aio, ctx.query
    choices, candidates, targets, unique, blocks = [], [], [], {}, []
    sources = aio.source_hashes()
    prepared = {}
    verified_cv = 0
    for ds in DATA:
        for seed in SEEDS:
            block = ctx.block(ds, seed)
            protocol, _ = aio.validate_seal(block)
            require(protocol['dataset'] == ds and protocol['seed'] == seed and protocol['budgets'] == BUDGETS,
                    'Historical block roster/protocol differs: ' + str(block))
            require(protocol['methods'] == METHODS and protocol['lambda_grid_default_first'] == VALUES,
                    'Historical method or candidate roster differs')
            for name, digest in protocol['prepared_file_sha256'].items():
                require(Path(name).name == name and sha(ctx.data[ds] / name) == digest, 'Prepared input changed: ' + name)
            require(ds not in prepared or prepared[ds] == protocol['prepared_file_sha256'], 'Prepared inputs vary across seeds')
            prepared[ds] = protocol['prepared_file_sha256']
            seal = read(block / 'GLOBAL_PREDICTIONS_SEALED.json')
            block_rel = f'{ds}/seed{seed}'
            blocks.append(dict(block=block_rel, seal_sha256=sha(block / 'GLOBAL_PREDICTIONS_SEALED.json'),
                               files=seal['files']))
            inp = ctx.input(block, protocol)
            for q in BUDGETS:
                selected = {}
                for method in METHODS:
                    name = f'Q{q}_{method}_selection.json'
                    selection = read(block / name)
                    scores = np.asarray(selection['scores'], float)
                    default, rule = query.select_configuration(scores)
                    require(default == selection['index'] and rule == selection['rule'], 'Primary selection changed')
                    means = np.full(3, np.inf) if scores.shape[1] == 0 else np.where(
                        np.all(np.isfinite(scores), axis=1), np.mean(scores, axis=1), np.inf)
                    tied = np.flatnonzero(np.isfinite(means) & (abs(means - min(means)) <= 1e-12)).tolist() if np.isfinite(means).any() else []
                    selected[method] = dict(primary_default=default,
                        sensitivity_min=min(tied, key=lambda i: VALUES[i]) if tied else default,
                        sensitivity_max=max(tied, key=lambda i: VALUES[i]) if tied else default)
                    verified_cv += len(selection['cv'])
                    for ci in range(3):
                        candidates.append(dict(dataset=ds, seed=seed, budget=q, allocation=method, candidate=ci,
                            lambda_pair=VALUES[ci], mean_validation_BER=float(means[ci]) if math.isfinite(means[ci]) else None,
                            best_tie=ci in tied, eligible_folds=selection['eligible_folds'], source_selection_sha256=sha(block / name)))
                    for policy in POLICIES:
                        ci = selected[method][policy]
                        choices.append(dict(dataset=ds, seed=seed, budget=q, allocation=method, policy=policy,
                            selected_candidate=ci, selected_lambda=VALUES[ci], primary_lambda=VALUES[default],
                            changed_from_primary=ci != default, eligible_folds=selection['eligible_folds'],
                            tied_candidates='|'.join(map(str, tied)), tied_lambdas='|'.join(str(VALUES[i]) for i in tied),
                            tie_count=len(tied), mean_validation_BER=float(means[ci]) if math.isfinite(means[ci]) else None,
                            fallback_without_eligible_candidate=not bool(tied), source_selection_sha256=sha(block / name)))
                P, N = query.signed_pairs(inp['pairs'][:q], inp['answers'][:q])
                for policy in POLICIES:
                    logical = [(m, selected[m][policy], m + '-CV') for m in METHODS]
                    logical += [(m, selected['full'][policy], m + '-at-full-lambda') for m in METHODS[1:]]
                    for method, ci, logical_method in logical:
                        stem = f'Q{q}_{method}_c{ci}_final'
                        fit_id = f'{block_rel}/{stem}'
                        row = dict(dataset=ds, seed=seed, budget=q, allocation=method, candidate=ci,
                            lambda_pair=VALUES[ci], fit_id=fit_id, source_block=block_rel, expected_stem=stem,
                            inputs_sha256=sha(block / 'inputs.npz'), graph_sha256=sha(block / 'used_graph.npz'),
                            source_protocol_sha256=sha(block / 'PROTOCOL.json'), P_sha256=aio.array_digest(P),
                            N_sha256=aio.array_digest(N), query_count=len(P) + len(N))
                        unique[fit_id] = row
                        targets.append(dict(policy=policy, logical_method=logical_method, **row))
            print('VERIFIED_BLOCK', block_rel, flush=True)
    reused, missing = [], []
    for fit_id, row in sorted(unique.items()):
        block = safe(a.historical_root, row['source_block'])
        stem = row['expected_stem']
        existing = list(block.glob(stem + '*'))
        if existing:
            _, diag, files = ctx.terminal(block, row)
            seal_files = next(b['files'] for b in blocks if b['block'] == row['source_block'])
            require(all(name in seal_files and sha(block / name) == seal_files[name] for name in files),
                    'Existing final is outside the historical seal: ' + fit_id)
            row.update(status='verified_reuse', terminal_status=diag['status'], converged=diag['converged'],
                       files={name: sha(block / name) for name in sorted(files)})
            reused.append(row)
        else:
            row.update(status='missing_fit')
            missing.append(row)
    for row in targets:
        row['status'] = unique[row['fit_id']]['status']
    require(len(choices) == 900 and len(targets) == 1500, 'Incomplete planned logical roster')
    summary = dict(choices=len(choices), logical_finals=len(targets), unique_final_requirements=len(unique),
                   verified_reuse_unique_fits=len(reused), missing_unique_fits=len(missing),
                   cv_predictions_replayed=verified_cv, historical_blocks=len(blocks),
                   new_fits_run=0, test_scores_read=False, posthoc=True, primary_preserved=True)
    document = dict(rules=RULES, adapter_sha256=sha(Path(__file__)), common_source_sha256=sources,
                    software=aio.software(), prepared_file_sha256=prepared, blocks=blocks,
                    choices=choices, targets=targets, unique=list(unique.values()), summary=summary)
    freeze(a.output / 'PLAN.json', document)
    freeze(a.output / 'PLAN_SUMMARY.json', summary)
    table(a.output / 'selection_policy_index_900.csv', choices)
    table(a.output / 'candidate_cv_index_900.csv', candidates)
    table(a.output / 'logical_final_targets_1500.csv', targets)
    table(a.output / 'verified_reuse_fits.csv', [{k: v for k, v in r.items() if k != 'files'} for r in reused])
    table(a.output / 'missing_fits.csv', missing)
    names = ['PLAN.json', 'PLAN_SUMMARY.json', 'selection_policy_index_900.csv', 'candidate_cv_index_900.csv',
             'logical_final_targets_1500.csv', 'verified_reuse_fits.csv', 'missing_fits.csv']
    freeze(a.output / 'PLAN_FREEZE.json', {name: sha(a.output / name) for name in names})
    print(json.dumps(summary, indent=2))


def load_plan(ctx):
    a = ctx.a
    for name, digest in read(a.output / 'PLAN_FREEZE.json').items():
        require(sha(safe(a.output, name)) == digest, 'Frozen plan changed: ' + name)
    p = read(a.output / 'PLAN.json')
    require(p['rules'] == RULES and p['adapter_sha256'] == sha(Path(__file__)), 'Adapter or declared rules changed')
    require(p['common_source_sha256'] == ctx.aio.source_hashes() and p['software'] == ctx.aio.software(),
            'Exact common source and software are required')
    for ds, files in p['prepared_file_sha256'].items():
        for name, digest in files.items():
            require(sha(safe(ctx.data[ds], name)) == digest, 'Prepared data changed')
    for b in p['blocks']:
        block = safe(a.historical_root, b['block'])
        require(sha(block / 'GLOBAL_PREDICTIONS_SEALED.json') == b['seal_sha256'], 'Historical seal changed')
        for name, digest in b['files'].items():
            require(sha(safe(block, name)) == digest, 'Historical sealed member changed: ' + name)
    return p


def final_records(ctx, p):
    records = []
    for row in p['unique']:
        origin = 'historical' if row['status'] == 'verified_reuse' else 'output'
        root = ctx.a.historical_root if origin == 'historical' else ctx.a.output
        rel = row['source_block'] if origin == 'historical' else 'fits/' + row['source_block']
        block = safe(root, rel)
        _, diag, files = ctx.terminal(block, row)
        files.update(['PROTOCOL.json', 'inputs.npz', 'used_graph.npz'])
        records.append(dict(fit_id=row['fit_id'], origin=origin, block=rel,
                            status=diag['status'], converged=diag['converged'],
                            files={name: sha(block / name) for name in sorted(files)}))
    return records


def train(ctx):
    a, np = ctx.a, ctx.np
    p = load_plan(ctx)
    missing = [r for r in p['unique'] if r['status'] == 'missing_fit']
    if a.dry_run:
        print(json.dumps(dict(phase='train_dry_run', inputs_verified=True, planned_missing_fits=len(missing),
                              new_fits_run=0, seal_written=False, test_scores_read=False), indent=2))
        return
    require(not (a.output / 'ALL_FINALS_SEALED.json').exists(), 'Finals already sealed; use score')
    from scipy import sparse
    from threadpoolctl import threadpool_limits
    import psutil
    from run_allocation import fit
    lock = a.output / 'TRAIN.lock'
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(descriptor, str(os.getpid()).encode('ascii'))
        groups = collections.defaultdict(list)
        for row in missing:
            groups[(row['dataset'], row['seed'])].append(row)
        for (ds, seed), rows in groups.items():
            require(psutil.virtual_memory().available >= 3 * 1024**3,
                    'Less than 3 GiB RAM available; no new block started; completed artifacts retained')
            source = ctx.block(ds, seed)
            out = a.output / 'fits' / ds / f'seed{seed}'
            out.mkdir(parents=True, exist_ok=True)
            for name in ['PROTOCOL.json', 'inputs.npz', 'used_graph.npz']:
                freeze_bytes(out / name, (source / name).read_bytes())
            protocol = read(out / 'PROTOCOL.json')
            ctx.aio.check_sources(protocol)
            X = [sparse.load_npz(ctx.data[ds] / f'X{v}.npz').tocsr() for v in range(protocol['views'])]
            L = sparse.load_npz(out / 'used_graph.npz')
            with np.load(out / 'inputs.npz', allow_pickle=False) as z:
                H0 = z['H0'].copy()
                Hv0 = [z[f'Hv{v}'].copy() for v in range(protocol['views'])]
                pairs, answers = z['pairs'].copy(), z['answers'].copy()
            for row in rows:
                stem, q = row['expected_stem'], row['budget']
                if list(out.glob(stem + '*')):
                    ctx.terminal(out, row)
                else:
                    P, N = ctx.query.signed_pairs(pairs[:q], answers[:q])
                    with threadpool_limits(limits=1):
                        fit(out, stem, row['allocation'], row['lambda_pair'], X, L,
                            protocol['k'], H0, Hv0, P, N, seed, protocol)
                    ctx.terminal(out, row)
            del X, L, H0, Hv0, pairs, answers
        load_plan(ctx)
        records = final_records(ctx, p)
        freeze(a.output / 'ALL_FINALS_SEALED.json', dict(plan_freeze_sha256=sha(a.output / 'PLAN_FREEZE.json'),
            logical_records=len(p['targets']), unique_final_records=len(records), planned_new_fits=len(missing),
            reused_unique=len(records) - len(missing), records=records, test_scores_read=False))
        print('ALL_FINALS_SEALED', len(records), 'unique finals;', len(p['targets']), 'logical targets')
    finally:
        os.close(descriptor)
        lock.unlink()


def summarize(values, eligible, metric):
    complete = len(values) == 10 and all(eligible) and all(v is not None and math.isfinite(v) for v in values)
    row = dict(eligible_pairs=sum(eligible), planned_pairs=10, complete=complete,
               mean_pp=None, sample_SD_pp=None, median_pp=None, minimum_pp=None, maximum_pp=None,
               wins=None, ties=None, losses=None)
    if complete:
        signed = [-v for v in values] if metric.startswith('BER') else values
        row.update(mean_pp=statistics.mean(values), sample_SD_pp=statistics.stdev(values),
                   median_pp=statistics.median(values), minimum_pp=min(values), maximum_pp=max(values),
                   wins=sum(v > 1e-12 for v in signed), ties=sum(abs(v) <= 1e-12 for v in signed),
                   losses=sum(v < -1e-12 for v in signed))
    return row


def score(ctx):
    a, np = ctx.a, ctx.np
    p = load_plan(ctx)
    seal = read(a.output / 'ALL_FINALS_SEALED.json')
    require(seal['plan_freeze_sha256'] == sha(a.output / 'PLAN_FREEZE.json') and not seal['test_scores_read'],
            'Global final seal is not bound to the frozen plan')
    records = final_records(ctx, p)
    require(seal['records'] == records and seal['logical_records'] == 1500 and
            seal['unique_final_records'] == len(p['unique']), 'Final roster or terminal artifact changed')
    out = a.output / 'evaluation'
    freeze(out / 'PRE_SCORE_GATE.json', dict(final_seal_sha256=sha(a.output / 'ALL_FINALS_SEALED.json'),
           plan_freeze_sha256=sha(a.output / 'PLAN_FREEZE.json'), all_unique_finals_checked=len(records),
           scorer_sha256=sha(Path(__file__)), choices_frozen_before_labels=True))
    ctx.allow_labels = True
    labels_map = mapping(a.labels_map)
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    lookup_records = {r['fit_id']: r for r in records}
    byfit, sources = {}, {}
    for ds in DATA:
        y = np.load(labels_map[ds], allow_pickle=False).reshape(-1)
        sources[ds] = dict(labels_sha256=sha(labels_map[ds]), blocks={})
        for seed in SEEDS:
            block = ctx.block(ds, seed)
            protocol = read(block / 'PROTOCOL.json')
            require(len(y) == protocol['n'] and len(np.unique(y)) == protocol['k'], 'Class vector count or known k differs')
            inp = ctx.input(block, protocol)
            require(np.array_equal(ctx.query.answer_queries(inp['pairs'], y), inp['answers']), 'Labels disagree with paid relationships')
            test = inp['test']
            ep = ctx.query.sample_query_stream(test, min(10000, len(test) * (len(test) - 1) // 2), seed + 51000)
            ea = ctx.query.answer_queries(ep, y)
            oldpairs = block / 'postseal_evaluation_pairs.npz'
            oldmanifest = block / 'EVALUATION_MANIFEST.json'
            if oldmanifest.exists():
                old = read(oldmanifest)
                require(old['labels_sha256'] == sha(labels_map[ds]) and oldpairs.exists() and
                        old['evaluation_pairs_sha256'] == sha(oldpairs), 'Historical evaluation source differs')
            if oldpairs.exists():
                with np.load(oldpairs, allow_pickle=False) as z:
                    require(np.array_equal(z['pairs'], ep) and np.array_equal(z['answers'], ea), 'Historical pool-excluded pairs differ')
            pairs_path = out / 'pairs' / ds / f'seed{seed}.npz'
            if pairs_path.exists():
                with np.load(pairs_path, allow_pickle=False) as z:
                    require(np.array_equal(z['pairs'], ep) and np.array_equal(z['answers'], ea), 'Saved evaluation pairs differ')
            else:
                pairs_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(pairs_path, pairs=ep, answers=ea)
            sources[ds]['blocks'][f'seed{seed}'] = dict(evaluation_pairs_sha256=sha(pairs_path), test_count=len(test), pair_count=len(ep))
            for row in p['unique']:
                if row['dataset'] != ds or row['seed'] != seed:
                    continue
                record = lookup_records[row['fit_id']]
                root = a.historical_root if record['origin'] == 'historical' else a.output
                terminal_block = safe(root, record['block'])
                values = {metric: None for metric in METRICS}
                if record['status'] == 'ok':
                    with np.load(terminal_block / (row['expected_stem'] + '.npz'), allow_pickle=False) as z:
                        pred = z['labels'].copy()
                    yt, zt = y[test], pred[test]
                    _, first = np.unique(yt, return_inverse=True)
                    _, second = np.unique(zt, return_inverse=True)
                    matrix = np.zeros((first.max() + 1, second.max() + 1), int)
                    np.add.at(matrix, (first, second), 1)
                    i, j = linear_sum_assignment(-matrix)
                    q = row['budget']
                    values = dict(ACC=float(matrix[i, j].sum() / len(test)),
                        NMI=float(normalized_mutual_info_score(yt, zt, average_method='arithmetic')),
                        ARI=float(adjusted_rand_score(yt, zt)),
                        BER_train=ctx.query.balanced_pair_error(pred, inp['pairs'][:q], inp['answers'][:q]),
                        BER_unqueried=ctx.query.balanced_pair_error(pred, ep, ea))
                byfit[row['fit_id']] = dict(fit_id=row['fit_id'], dataset=ds, seed=seed, budget=row['budget'],
                    allocation=row['allocation'], lambda_pair=row['lambda_pair'], origin=record['origin'],
                    status=record['status'], eligible=record['status'] == 'ok' and record['converged'], **values)
    logical = [dict(policy=r['policy'], logical_method=r['logical_method'], **byfit[r['fit_id']]) for r in p['targets']]
    lookup = {(r['policy'], r['dataset'], r['seed'], r['budget'], r['logical_method']): r for r in logical}
    points, summaries = [], []
    for policy in POLICIES:
        for ds in DATA:
            for q in BUDGETS:
                for control in LOGICAL[1:]:
                    for metric in METRICS:
                        values, eligible = [], []
                        for seed in SEEDS:
                            full = lookup[policy, ds, seed, q, 'full-CV']
                            other = lookup[policy, ds, seed, q, control]
                            value = None if full[metric] is None or other[metric] is None else 100 * (full[metric] - other[metric])
                            ok = full['eligible'] and other['eligible'] and value is not None
                            values.append(value)
                            eligible.append(ok)
                            points.append(dict(policy=policy, dataset=ds, seed=seed, budget=q, comparator=control,
                                metric=metric, Full_minus_control_pp=value, eligible=ok,
                                full_fit_id=full['fit_id'], control_fit_id=other['fit_id']))
                        summaries.append(dict(policy=policy, dataset=ds, budget=q, comparator=control, metric=metric,
                            positive_difference_better=not metric.startswith('BER'), **summarize(values, eligible, metric)))
    require(len(logical) == 1500 and len(points) == 6000 and len(summaries) == 600, 'Incomplete scoring roster')
    table(out / 'unique_fit_metrics.csv', list(byfit.values()))
    table(out / 'logical_metrics_1500.csv', logical)
    table(out / 'paired_five_metrics_6000.csv', points)
    table(out / 'paired_summary_600.csv', summaries)
    freeze(out / 'SCORING_AUDIT.json', dict(source_hashes=sources, unique_fits=len(byfit), logical_records=len(logical),
        paired_values=len(points), summary_groups=len(summaries), eligible_unique=sum(r['eligible'] for r in byfit.values()),
        complete_summary_groups=sum(r['complete'] for r in summaries), no_choice_changed_after_scoring=True,
        original_results_preserved=True, posthoc=True, independent_confirmation=False, p_values_computed=False))
    freeze(out / 'EVALUATION_FREEZE.json', {f.relative_to(out).as_posix(): sha(f) for f in sorted(out.rglob('*'))
           if f.is_file() and f.name != 'EVALUATION_FREEZE.json'})
    print('SCORED', len(logical), 'logical records;', len(points), 'paired values;', len(summaries), 'summary groups')


def main(argv=None):
    args = arguments(argv)
    ctx = Context(args)
    {'plan': plan, 'train': train, 'score': score}[args.phase](ctx)


if __name__ == '__main__':
    main()
