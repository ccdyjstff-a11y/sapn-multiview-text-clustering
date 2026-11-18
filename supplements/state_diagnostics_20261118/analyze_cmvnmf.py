"""Replay all 42 CMVNMF checkpoint and 252 scale diagnostics without fitting.

Only exported U/H factors, paid relations and separately supplied feature views
are read. No category file, test accuracy, third-party implementation or MATLAB
is needed. Input verification and feature loading are supplied by resource_common.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path

if not __debug__:
    raise RuntimeError('Run without -O/-OO; resource_common integrity assertions must remain enabled')

for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import numpy as np
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits
from resource_common import verify_resource, load_features, read_json, write_csv

HERE = Path(__file__).resolve().parent
CORPORA = ('3Sources', 'BBC', 'Reuters', 'Cora', 'CiteSeer', '20NG', 'AGNews')
SCALES = (2., 10., 100.)


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                                    allow_nan=False).encode('utf-8')).hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False); f.write('\n')


def integer_array(value):
    value = np.asarray(value)
    need(np.isrealobj(value) and np.isfinite(value).all() and np.equal(value, np.floor(value)).all(),
         'Expected finite integer object IDs')
    return value.astype(np.int64)


def load_state(root, record):
    path = (root / record['state_file']).resolve()
    need(path.is_relative_to(root), 'State path escapes sealed resource')
    v, n, k = record['views'], record['n'], record['k']
    need(v >= 2 and k >= 2 and n > k, 'Invalid factor dimensions')
    need(bool(record['mu_aliases_final']) == (not bool(record['continued'])), 'MU alias/continuation metadata disagree')
    required = {f'{stage}_{name}{i}' for stage in (['final', 'MU'] if record['continued'] else ['final'])
                for name in ('U', 'Hv') for i in range(v)} | {'P', 'N', 'pool', 'test'}
    with np.load(path, allow_pickle=False) as data:
        need(set(data.files) == required, 'State schema differs from the exported factor whitelist')
        final_u = [data[f'final_U{i}'].copy() for i in range(v)]
        final_h = [data[f'final_Hv{i}'].copy() for i in range(v)]
        mu_u = [data[f'MU_U{i}'].copy() for i in range(v)] if record['continued'] else final_u
        mu_h = [data[f'MU_Hv{i}'].copy() for i in range(v)] if record['continued'] else final_h
        P, N = [integer_array(data[name]).reshape(-1, 2) for name in ('P', 'N')]
        pool, test = [integer_array(data[name]).ravel() for name in ('pool', 'test')]
    for us, hs in ((final_u, final_h), (mu_u, mu_h)):
        for u, h in zip(us, hs):
            need(u.ndim == 2 and u.shape[1] == k and h.shape == (k, n), 'Factor shape mismatch')
            need(np.isfinite(u).all() and np.isfinite(h).all() and (u >= 0).all() and (h >= 0).all(),
                 'Saved factors must be finite and nonnegative')
    need(len(P) > 0 and len(N) > 0 and len(P) + len(N) == record['budget'], 'Paid-relation count differs from budget')
    need(len(np.unique(pool)) == len(pool) and len(np.unique(test)) == len(test)
         and np.array_equal(np.sort(np.r_[pool, test]), np.arange(n)), 'Pool/test must partition all objects')
    need(np.isin(P, pool).all() and np.isin(N, pool).all(), 'Paid relation outside original query pool')
    return final_u, final_h, mu_u, mu_h, P, N


def output(hs):
    # Preserve the original finite-precision floor exactly. It is relevant to
    # the retained tiny-factor / near-tie scaling exception.
    return np.mean([h / np.maximum(h.sum(0, keepdims=True), 1e-30) for h in hs], axis=0)


def penalty(hs, P, N, beta):
    total = 0.
    for a in range(len(hs)):
        for b in range(a):
            A, B = hs[a], hs[b]
            total += np.square(A - B).sum()
            total += np.square(A[:, P[:, 0]] - B[:, P[:, 1]]).sum() + np.square(A[:, P[:, 1]] - B[:, P[:, 0]]).sum()
            total += 2 * ((A[:, N[:, 0]] * B[:, N[:, 1]]).sum() + (A[:, N[:, 1]] * B[:, N[:, 0]]).sum())
    return float(beta * total)


def norms(arrays):
    return float(np.sqrt(sum(np.square(a).sum() for a in arrays)))


def reconstruction(xs, us, hs):
    return float(sum(float(x.multiply(x).sum()) + np.sum((u.T @ u) * (h @ h.T))
                     - 2 * np.sum(u * np.asarray(x @ h.T)) for x, u, h in zip(xs, us, hs)))


def ber(labels, P, N):
    return float(.5 * (np.mean(labels[P[:, 0]] != labels[P[:, 1]]) + np.mean(labels[N[:, 0]] == labels[N[:, 1]])))


def partition_changes(a, b, k):
    table = np.zeros((k, k), int)
    np.add.at(table, (a, b), 1)
    i, j = linear_sum_assignment(-table)
    return int(len(a) - table[i, j].sum())


def validate_inventory(records):
    keys = [(r['dataset'], r['seed'], r['budget']) for r in records]
    expected = {(ds, seed, q) for ds in CORPORA
                for seed in ((41, 42, 43) if ds == '20NG' else (101, 102, 103) if ds == 'AGNews' else (23, 24, 25))
                for q in (160, 400)}
    need(len(keys) == len(set(keys)) == 42 and set(keys) == expected, 'Full fixed 42-condition inventory required; no filtering')
    for r in records:
        need(np.isfinite(r['beta']) and r['beta'] >= 0, 'Invalid frozen relation coefficient')
        need(all(np.isfinite(r[name]) for name in ('MU_J', 'final_J', 'MU_residual', 'final_residual')), 'Invalid checkpoint metadata')
    return sorted(records, key=lambda r: (CORPORA.index(r['dataset']), r['seed'], r['budget']))


def replay(root, feature_map_path, destination):
    root, feature_map_path, destination = root.resolve(), feature_map_path.resolve(), destination.resolve()
    need(not destination.is_relative_to(root), 'Write replay outputs outside the sealed resource')
    need(not destination.exists(), 'Replay output must be a new directory')
    # Full manifest verification precedes state/feature access.
    manifest = verify_resource(root)
    manifest_hash = canonical_sha(manifest)
    feature_map_hash = sha(feature_map_path)
    feature_map = read_json(feature_map_path)
    records = validate_inventory(read_json(root / 'CMVNMF_RUNS.json')['records'])
    rows, scalerows = [], []
    current_ds, xs = None, None
    for r in records:
        ds, seed, q = r['dataset'], r['seed'], r['budget']
        if ds != current_ds:
            # Release the previous dataset before loading the next one.
            xs = None
            xs = load_features(root, feature_map, ds)
            current_ds = ds
        U, H, Umu, Hmu, P, N = load_state(root, r)
        n, k, beta = r['n'], r['k'], r['beta']
        need(len(xs) == len(H) == r['views'], 'Feature view order/count mismatch')
        for x, u in zip(xs, U):
            need(x.shape == (u.shape[0], n), 'Feature/factor shape mismatch')
            need(abs(float(x.multiply(x).sum()) - n) <= 1e-10 * max(1, n), 'Feature normalization differs from original per-view squared norm n')
        final_h = output(H); labels = final_h.argmax(0)
        muz = output(Hmu).argmax(0)
        muJ, endJ = float(r['MU_J']), float(r['final_J'])
        muP, endP = penalty(Hmu, P, N, beta), penalty(H, P, N, beta)
        muR, endR = reconstruction(xs, Umu, Hmu), reconstruction(xs, U, H)
        muJerr, endJerr = abs(muR + muP - muJ), abs(endR + endP - endJ)
        need(max(muJerr, endJerr) < 1e-8 * max(1, abs(muJ), abs(endJ)), 'Independently recomputed objective differs from frozen checkpoint: ' + str((ds, seed, q)))
        need(min(muR, endR) > -1e-8, 'Negative reconstruction residual beyond roundoff')
        normU, normUmu, normH, normHmu = norms(U), norms(Umu), norms(H), norms(Hmu)
        need(normUmu > 0 and normHmu > 0, 'Undefined factor norm ratio')
        changes = partition_changes(muz, labels, k)
        mu_ber, final_ber = ber(muz, P, N), ber(labels, P, N)
        row = dict(dataset=ds, seed=seed, budget=q, beta=beta, n=n, k=k, views=len(H), continued=bool(r['continued']),
                   MU_iterations=r['MU_iterations'], CD_iterations=r['CD_iterations'],
                   MU_J=muJ, final_J=endJ, MU_relation=muP, final_relation=endP,
                   MU_reconstruction_recomputed=muR, final_reconstruction_recomputed=endR,
                   MU_objective_recompute_error=muJerr, final_objective_recompute_error=endJerr,
                   objective_change=endJ-muJ, relation_change=endP-muP, reconstruction_change=endR-muR,
                   MU_U_norm=normUmu, final_U_norm=normU, MU_H_norm=normHmu, final_H_norm=normH,
                   U_norm_ratio=normU/normUmu, H_norm_ratio=normH/normHmu,
                   changed_objects_raw=int(np.sum(muz != labels)), changed_objects_permutation_matched=changes,
                   changed_fraction_permutation_matched=changes/n,
                   MU_paid_BER=mu_ber, final_paid_BER=final_ber, paid_BER_change=final_ber-mu_ber,
                   MU_residual=r['MU_residual'], final_residual=r['final_residual'])
        rows.append(row)
        for stage, us, hs, R, J, rel in [('MU', Umu, Hmu, muR, muJ, muP), ('final', U, H, endR, endJ, endP)]:
            base = output(hs); zbase = base.argmax(0)
            columns = np.unique(np.linspace(0, n-1, 17, dtype=int))
            for scale in SCALES:
                uu, hh = [u*scale for u in us], [h/scale for h in hs]
                transformed = output(hh)
                rel2, expected = penalty(hh, P, N, beta), rel/(scale*scale)
                product_error = max(float(np.max(abs((uc @ hc[:, columns]) - (u @ h[:, columns]))))
                                    for u, h, uc, hc in zip(us, hs, uu, hh))
                normdiff = float(np.max(abs(transformed-base)))
                changed = transformed.argmax(0) != zbase
                change = int(np.sum(changed))
                pe = abs(rel2-expected)
                Rscaled = reconstruction(xs, uu, hh)
                Jscaled, predicted = Rscaled+rel2, R+expected
                ordered = np.sort(base, axis=0)
                margin = float(np.max((ordered[-1]-ordered[-2])[changed], initial=0))
                need(normdiff < 1e-12 and pe < 1e-10*max(1, rel), 'Scale membership/penalty identity failed')
                # Do not require zero hard-partition changes: all three changes
                # from the known tiny-factor MU checkpoint must remain visible.
                need(margin < 1e-12, 'Scaling changed a non-negligible original argmax margin')
                need(product_error < 1e-10 and abs(Rscaled-R) < 1e-8*max(1, abs(R)), 'Scale reconstruction/product identity failed')
                scalerows.append(dict(dataset=ds, seed=seed, budget=q, stage=stage, scale=scale,
                    original_J=J, scaled_J=Jscaled, original_relation=rel, scaled_relation=rel2,
                    relation_scaling_abs_error=pe, objective_identity_abs_error=abs(Jscaled-predicted),
                    product_spot_max_abs_error=product_error, normalized_membership_max_abs_error=normdiff,
                    partition_changes=change, changed_max_original_argmax_margin=margin,
                    original_floor_active_view_columns=sum(int(np.sum(h.sum(0) <= 1e-30)) for h in hs),
                    scaled_floor_active_view_columns=sum(int(np.sum(h.sum(0) <= 1e-30)) for h in hh)))
        print(json.dumps(dict(event='CHECKPOINT_REPLAYED', dataset=ds, seed=seed, budget=q)), flush=True)
    need(len(rows) == 42 and len(scalerows) == 252, 'Incomplete replay')
    summary = []
    for ds in CORPORA:
        for q in (160, 400):
            group = [r for r in rows if r['dataset'] == ds and r['budget'] == q]
            need(len(group) == 3, 'Incomplete summary group')
            item = dict(dataset=ds, budget=q, count=3, continued=sum(r['continued'] for r in group),
                        partitions_changed=sum(r['changed_objects_permutation_matched'] > 0 for r in group))
            for name in ('objective_change', 'relation_change', 'reconstruction_change', 'U_norm_ratio', 'H_norm_ratio',
                         'changed_fraction_permutation_matched', 'paid_BER_change', 'final_relation'):
                values = np.asarray([r[name] for r in group])
                item.update({name+'_mean':float(values.mean()), name+'_min':float(values.min()), name+'_max':float(values.max())})
            summary.append(item)
    qa = dict(status='PASS', finals=42, continued=sum(r['continued'] for r in rows), transform_checks=252,
              partition_changes_under_scaling=sum(r['partition_changes'] for r in scalerows),
              partitions_changed_by_continuation=sum(r['changed_objects_permutation_matched'] > 0 for r in rows),
              max_relation_scaling_error=max(r['relation_scaling_abs_error'] for r in scalerows),
              max_objective_identity_error=max(r['objective_identity_abs_error'] for r in scalerows),
              max_product_spot_error=max(r['product_spot_max_abs_error'] for r in scalerows),
              max_membership_error=max(r['normalized_membership_max_abs_error'] for r in scalerows),
              no_new_fit=True, no_external_labels_or_accuracy=True, reconstruction_from_logged_objective=False)
    for key in ('U_norm_ratio', 'H_norm_ratio', 'changed_fraction_permutation_matched', 'paid_BER_change',
                'objective_change', 'relation_change', 'reconstruction_change'):
        qa[key+'_range'] = [min(r[key] for r in rows), max(r[key] for r in rows)]
    qa.update(independent_feature_residuals_recomputed=True,
              max_MU_objective_recompute_error=max(r['MU_objective_recompute_error'] for r in rows),
              max_final_objective_recompute_error=max(r['final_objective_recompute_error'] for r in rows),
              final_partition_changes_under_scaling=sum(r['partition_changes'] for r in scalerows if r['stage'] == 'final'))
    need(canonical_sha(verify_resource(root)) == manifest_hash, 'Resource changed during replay')
    need(sha(feature_map_path) == feature_map_hash, 'External feature-map file changed during replay')
    destination.mkdir(parents=True, exist_ok=False)
    write_csv(destination/'cmvnmf_checkpoints_42.csv', rows)
    write_csv(destination/'cmvnmf_scale_transforms_252.csv', scalerows)
    write_csv(destination/'cmvnmf_summary_14.csv', summary)
    write_json(destination/'CMVNMF_DIAGNOSTIC_QA.json', qa)
    write_json(destination/'CMVNMF_REPLAY_RECEIPT.json', dict(status='REPLAYED_WITHOUT_FITTING',
        verified_resource_manifest_canonical_sha256=manifest_hash, run_inventory_sha256=sha(root/'CMVNMF_RUNS.json'),
        feature_map_sha256=feature_map_hash, script_sha256=sha(Path(__file__)),
        records=42, transforms=252, summary_groups=14, new_fits=0, external_class_arrays_read=False,
        external_accuracy_selection=False, third_party_model_code_imported=False,
        source='All declared MU and final checkpoints; no excluded condition or scale.',
        product_check_scope='All feature rows and 17 uniformly spaced object columns; full reconstruction residuals independently recomputed using sparse feature matrices.',
        scale_qualification='Retain every changed argmax and its original margin, including the tiny-factor MU exception; scaling invariance and continuation are distinct diagnostics.',
        output_sha256={p.name:sha(p) for p in destination.iterdir() if p.is_file()}))
    print(json.dumps(qa, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feature-map', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=HERE)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        replay(args.root, args.feature_map, args.output)


if __name__ == '__main__':
    main()
