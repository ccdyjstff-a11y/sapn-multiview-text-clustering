"""Own compact-array and paid-query primitives; no model imports.

Only relative, manifest-bound packet members are opened. Full-state numerical
eligibility remains an audited input: this is neither a solver rerun nor a
recalculation of class/test metrics. No action without an explicit --root.
"""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import sys
import zipfile

sys.dont_write_bytecode = True
import numpy as np

EXPECTED_SCOPE = dict(CV=864, final=42, paid_streams=21)
SEEDS = {name: [23, 24, 25] for name in ('3Sources', 'BBC', 'Reuters', 'Cora', 'CiteSeer')}
SEEDS.update({'20NG': [41, 42, 43], 'AGNews': [101, 102, 103]})
SUPPORTED = {'3Sources', 'BBC', 'Cora', 'CiteSeer'}
DEFAULT = dict(beta=.1, lam=.2, kap=.95)
GRID = [DEFAULT] + [dict(beta=b, lam=l, kap=k)
    for b in [.1, .2, .3] for l in [.2, .3] for k in [.8, .9, .95]
    if (b, l, k) != (.1, .2, .95)]


def insist(ok, why):
    if not ok:
        raise ValueError(why)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def member(base, name):
    insist(isinstance(name, str) and name, 'Empty/non-string member')
    normalized = name.replace('\\', '/')
    parts = PurePosixPath(normalized)
    insist(not parts.is_absolute() and ':' not in normalized
           and '..' not in parts.parts, 'Nonrelative/escaping member')
    path = (Path(base) / normalized).resolve()
    insist(path.is_relative_to(Path(base).resolve()), 'Escaping member')
    return path


def identity(array):
    header = json.dumps(dict(dtype=array.dtype.str, shape=list(array.shape)),
                        sort_keys=True, separators=(',', ':')).encode('ascii')
    return dict(dtype=array.dtype.str, shape=list(array.shape),
        sha256=hashlib.sha256(header + b'\n' + array.tobytes(order='C')).hexdigest())


def npz_members(path, allowed):
    with zipfile.ZipFile(path) as packet:
        insist(sorted(packet.namelist()) == sorted(allowed), 'Unexpected NPZ member/schema')
        raw = {name: packet.read(name) for name in allowed}
    arrays = {name[:-4]: np.load(io.BytesIO(payload), allow_pickle=False)
              for name, payload in raw.items()}
    return raw, arrays


def same_score(actual, expected):
    if actual is None or expected is None:
        return actual is None and expected is None
    return (isinstance(actual, (int, float)) and not isinstance(actual, bool)
            and math.isfinite(actual) and abs(actual - expected) <= 1e-12)


def folds_rebuilt(pool, pairs, seed, answers):
    """Reconstruct node-disjoint folds independently of the training module."""
    halves = np.array_split(np.random.RandomState(seed).permutation(pool), 2)
    folds = []
    for nodes in halves:
        left, right = np.isin(pairs[:, 0], nodes), np.isin(pairs[:, 1], nodes)
        folds.append(dict(training=np.flatnonzero(~left & ~right).tolist(),
            validation=np.flatnonzero(left & right).tolist(),
            unused_crossing=np.flatnonzero(left ^ right).tolist(),
            validation_nodes=sorted(nodes.tolist())))
    eligible = [fold for fold in folds if all(
        set(answers[fold[key]].tolist()) == {-1, 1} for key in ('training', 'validation'))]
    return folds, eligible


def hard_ber(partition, pairs, answers):
    """Hard pair BER; undefined when a sign is absent (not an imputed zero)."""
    partition, pairs, answers = np.asarray(partition), np.asarray(pairs), np.asarray(answers)
    insist(pairs.ndim == 2 and pairs.shape[1] == 2
           and answers.shape == (len(pairs),) and np.isin(answers, [-1, 1]).all(), 'BER pair schema')
    if not np.any(answers == 1) or not np.any(answers == -1):
        return None
    same = partition[pairs[:, 0]] == partition[pairs[:, 1]]
    positive, negative = answers == 1, answers == -1
    return float(.5 * np.mean(~same[positive]) + .5 * np.mean(same[negative]))


def receipt_eligible(receipt, partition, k):
    return bool(receipt['status'] == 'COMPLETE_ZERO_EXIT_VALID_MARKERS'
        and receipt.get('exit_code') == 0 and receipt.get('cleanup', {}).get('complete')
        and receipt.get('worker', {}).get('benchmark_contract_met')
        and partition is not None and len(np.unique(partition)) == k)


