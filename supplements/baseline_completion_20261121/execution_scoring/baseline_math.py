"""Portable mathematical definitions copied from the frozen author scorer; no model imports."""
import hashlib,json,csv
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score,adjusted_rand_score

def need(value, message):
    if not value:
        raise RuntimeError(message)



def integer_vector(value, length=None):
    a = np.asarray(value).ravel()
    need(np.isrealobj(a) and np.isfinite(a).all() and np.equal(a, np.floor(a)).all(), 'Invalid integer vector')
    need(length is None or len(a) == length, 'Integer vector length mismatch')
    return a.astype(np.int64)



def ber(labels, pairs, answers):
    pos, neg = answers == 1, answers == -1
    if not pos.any() or not neg.any():
        return None
    same = labels[pairs[:, 0]] == labels[pairs[:, 1]]
    return .5 * (np.count_nonzero(~same & pos) / int(pos.sum()) + np.count_nonzero(same & neg) / int(neg.sum()))



def sample_pairs(nodes, seed):
    total = min(10000, len(nodes) * (len(nodes) - 1) // 2)
    rng = np.random.RandomState(seed); seen = set(); pairs = []
    while len(pairs) < total:
        pair = tuple(sorted(int(x) for x in rng.choice(nodes, 2, replace=False)))
        if pair not in seen:
            seen.add(pair); pairs.append(pair)
    return np.asarray(pairs, dtype=int).reshape(-1, 2)


def metrics(y, z):
    need(len(y) == len(z) and len(y) > 0, 'Empty/mismatched metric vectors')
    _, a = np.unique(y, return_inverse=True); _, b = np.unique(z, return_inverse=True)
    table = np.zeros((a.max() + 1, b.max() + 1), dtype=np.int64)
    np.add.at(table, (a, b), 1)
    rows, cols = linear_sum_assignment(-table)
    return dict(ACC=float(table[rows, cols].sum() / len(y)),
                NMI=float(normalized_mutual_info_score(y, z, average_method='arithmetic')),
                ARI=float(adjusted_rand_score(y, z)))



METRICS = ('ACC', 'NMI', 'ARI', 'training_BER', 'unqueried_BER')
CORPORA = ('3Sources', 'BBC', 'Reuters', 'Cora', 'CiteSeer', '20NG', 'AGNews')
CATEGORIES = ('strict_eligible', 'input_unsupported', 'runtime_failure', 'finite_wrong_k',
              'not_attempted', 'finite_other_contract', 'invalid_partition')


def require(value, message):
    if not value:
        raise ValueError(message)


def aggregate(rows, method):
    require(method in ('SSA', 'PCBG'), 'Unknown supplementary method')
    require(len(rows) == 42 and len({r['condition_id'] for r in rows}) == 42, 'Exactly 42 unique logical outcomes required')
    require({r['dataset'] for r in rows} == set(CORPORA), 'Seven fixed corpora required')
    require(all(r['method'] == method and r['outcome_category'] in CATEGORIES and r['before_category'] in CATEGORIES for r in rows), 'Invalid method/category')
    require(all(r['eligible'] == (r['outcome_category'] == 'strict_eligible') for r in rows), 'Eligibility/category disagreement')
    expected_extension = 15 if method == 'SSA' else 4
    require(sum(r['extension_condition'] for r in rows) == expected_extension, 'Extension denominator changed')
    groups = []
    for ds in CORPORA:
        for q in (160, 400):
            group = sorted([r for r in rows if r['dataset'] == ds and r['Q'] == q], key=lambda r: r['seed'])
            require(len(group) == 3 and len({r['seed'] for r in group}) == 3, 'Every group needs all three planned seeds')
            complete = all(r['eligible'] and r['SAPN_eligible'] for r in group)
            if complete:
                require(all(r.get(prefix + '_' + m) is not None and np.isfinite(r[prefix + '_' + m])
                            for r in group for prefix in ('baseline', 'SAPN', 'delta') for m in METRICS), 'Complete group has missing metrics')
            item = dict(method=method, dataset=ds, Q=q, n_planned=3,
                        n_extension=sum(r['extension_condition'] for r in group),
                        n_eligible=sum(r['eligible'] for r in group),
                        n_SAPN_eligible=sum(r['SAPN_eligible'] for r in group),
                        n_paired_eligible=sum(r['eligible'] and r['SAPN_eligible'] for r in group),
                        paired_group_complete=complete,
                        fallback_count=sum(r['fallback'] is True for r in group),
                        selection_denominator=sum(r['fallback'] is not None for r in group))
            for cat in CATEGORIES:
                item['before_' + cat] = sum(r['before_category'] == cat for r in group)
                item['after_' + cat] = sum(r['outcome_category'] == cat for r in group)
            require(sum(item['after_' + cat] for cat in CATEGORIES) == 3, 'Coverage categories do not partition denominator')
            for m in METRICS:
                for prefix in ('baseline', 'SAPN', 'delta'):
                    name = prefix + '_' + m
                    if complete:
                        values = [r[name] for r in group]
                        if prefix == 'delta':
                            require(all(abs(r[name] - (r['baseline_' + m] - r['SAPN_' + m])) <= 1e-15 for r in group), 'Paired delta sign/value error')
                        item[name + '_mean'] = float(np.mean(values))
                        item[name + '_sample_sd'] = float(np.std(values, ddof=1))
                    else:
                        item[name + '_mean'] = item[name + '_sample_sd'] = None
            groups.append(item)
    coverage = dict(method=method, n_planned=42, n_extension_planned=expected_extension,
                    before={cat: sum(r['before_category'] == cat for r in rows) for cat in CATEGORIES},
                    after={cat: sum(r['outcome_category'] == cat for r in rows) for cat in CATEGORIES},
                    extension_after={cat: sum(r['extension_condition'] and r['outcome_category'] == cat for r in rows) for cat in CATEGORIES},
                    complete_three_seed_groups=sum(g['paired_group_complete'] for g in groups), n_planned_groups=14,
                    original_primary_overwritten=False, available_case_group_means_computed=False,
                    failed_or_unsupported_counted_as_win=False, significance_tests_performed=False)
    return groups, coverage



def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()

def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write_json(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')

def write_csv(path,rows):
    with Path(path).open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def partition_hash(labels):
    mapping={};codes=[]
    for x in np.asarray(labels).reshape(-1).tolist():
        if x not in mapping:mapping[x]=len(mapping)
        codes.append(mapping[x])
    return hashlib.sha256(np.asarray(codes,dtype='<i8').tobytes()).hexdigest()

def child(root,name):
    root=Path(root).resolve();path=(root/name).resolve()
    need(path.is_relative_to(root) and path!=root,'Resource path escapes root')
    return path
