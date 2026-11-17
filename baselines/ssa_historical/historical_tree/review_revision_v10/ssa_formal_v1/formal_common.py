"""SSA finite-comparison identities and paid-query checks; never loads class truth."""
from pathlib import Path
import hashlib
import json
import os
import sys

sys.dont_write_bytecode = True
import numpy as np
from scipy import io

ROOT = Path(__file__).resolve().parent
MAIN = ROOT.parent.parent
DEFAULT = dict(beta=.1, lam=.2, kap=.95)
GRID = [DEFAULT] + [dict(beta=b, lam=l, kap=k)
    for b in [.1, .2, .3] for l in [.2, .3] for k in [.8, .9, .95]
    if (b, l, k) != (.1, .2, .95)]
FIT_DATA = ['3Sources', 'BBC', 'Cora', 'CiteSeer']
METHOD = 'SSA-SNMF-two-repair-trace-limited-CV'
COUNTS = dict(maximum_cv_fits=864, maximum_final_fits=24,
    maximum_total_fits=888, logical_outcomes=42, unsupported=18,
    potential_inputs=1296)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def dump(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
        default=serial, allow_nan=False) + '\n', encoding='utf8')
    os.replace(temp, path)


def immutable(path, value):
    expected = json.loads(json.dumps(value, default=serial, allow_nan=False))
    if Path(path).exists():
        assert read(path) == expected, str(path)
    else:
        dump(path, expected)


def no_oracle_guard():
    """Guard obvious truth/score files; feature and paid-answer schemas are checked too."""
    def guard(event, args):
        if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0]))
            name = path.name.lower()
            truth_names = {'y.npy', 'labels.npy', 'truth.npy', 'gnd.npy',
                'y.mat', 'labels.mat', 'truth.mat', 'gnd.mat'}
            if name in truth_names or (name in {'evaluation.json', 'summary.json'}
                    and path.parent != ROOT):
                raise PermissionError('Training/sealing cannot read oracle or historical scores: ' + str(path))
    sys.addaudithook(guard)


def paid(card, n):
    assert sha(card['path']) == card['sha256']
    with np.load(card['path'], allow_pickle=False) as packet:
        assert not set(packet.files) & {'y', 'truth', 'labels', 'targets', 'target', 'gnd'}
        data = {key: packet[key].copy() for key in ['pool', 'test', 'pairs', 'answers']}
    pool, pairs, answers = data['pool'], data['pairs'], data['answers']
    assert pool.ndim == 1 and np.issubdtype(pool.dtype, np.integer)
    assert len(np.unique(pool)) == len(pool) and np.all((pool >= 0) & (pool < n))
    assert np.array_equal(data['test'], np.setdiff1d(np.arange(n), pool))
    assert pairs.shape == (400, 2) and np.issubdtype(pairs.dtype, np.integer)
    assert np.isin(pairs, pool).all() and np.all(pairs[:, 0] < pairs[:, 1])
    assert len(np.unique(pairs, axis=0)) == 400
    assert answers.shape == (400,) and np.isin(answers, [-1, 1]).all()
    return data


def case_id(ds, seed, Q, ci, fold):
    return f'{ds}_s{seed}_Q{Q}_c{ci}_' + ('final' if fold == 'final' else f'f{fold}')


def condition_id(ds, seed, Q):
    return f'{ds}_s{seed}_Q{Q}'


def sources():
    paths = set(ROOT.glob('*.py')) | set(ROOT.glob('*.m'))
    paths |= set((ROOT / 'private_generated').glob('*.m'))
    return {str(path): sha(path) for path in sorted(paths, key=lambda p: str(p))}


def protocol(path=None):
    """Check the complete frozen identity, including all requests and shared inputs."""
    p = read(path or ROOT / 'FROZEN_PROTOCOL.json')
    assert p['method'] == METHOD and p['grid'] == GRID and p['counts'] == COUNTS
    assert p['global_wall_seconds'] == 48 * 3600
    assert p['source_hashes'] == sources(), 'Executable source set or identity changed'
    for file, digest in p['evidence_hashes'].items():
        assert sha(file) == digest, file
    for ds, card in p['datasets'].items():
        for record in card['query_inputs'].values():
            paid(record, card['n'])
        if ds in FIT_DATA:
            for file, digest in card['feature_files'].items():
                assert sha(file) == digest, file
    conditions = {c['id']: c for c in p['conditions']}
    assert len(conditions) == len(p['conditions']) == COUNTS['logical_outcomes']
    assert sum(bool(c['unsupported']) for c in conditions.values()) == COUNTS['unsupported']
    wanted, source_inputs, source_groups = set(), {}, {}
    for cond in conditions.values():
        assert cond['id'] == condition_id(cond['dataset'], cond['seed'], cond['Q'])
        assert cond['unsupported'] == (cond['dataset'] not in FIT_DATA)
        assert cond['Q'] in [160, 400]
        assert cond['query_input'] == p['datasets'][cond['dataset']]['query_inputs'][str(cond['seed'])]
        if not cond['unsupported']:
            for ci in range(len(GRID)):
                for fold in [*range(len(cond['eligible_folds'])), 'final']:
                    wanted.add(case_id(cond['dataset'], cond['seed'], cond['Q'], ci, fold))
    actual = set()
    for case in p['cases']:
        identity = case['case_id']
        assert identity not in actual
        actual.add(identity)
        cond = conditions[condition_id(case['dataset'], case['seed'], case['Q'])]
        assert not cond['unsupported']
        ci, fold = case['candidate'], case['fold']
        assert isinstance(ci, int) and 0 <= ci < len(GRID)
        assert case['config'] == GRID[ci]
        assert identity == case_id(case['dataset'], case['seed'], case['Q'], ci, fold)
        assert case['n'] == cond['n'] and case['c'] == cond['c']
        assert case['fit_seed'] == case['logical_seed'] == case['seed']
        assert case['timeout_active_seconds'] == 600
        assert sha(case['input']) == case['sha256'], identity
        assert {name for name, _, _ in io.whosmat(case['input'])} == {
            'source_input', 'source_sha256', 'config', 'case_id'}
        source, digest = case['source_input'], case['source_input_sha256']
        if source in source_inputs:
            assert source_inputs[source] == digest
        else:
            assert sha(source) == digest, source
            assert {name for name, _, _ in io.whosmat(source)} == {
                'X', 'c', 'ML', 'CL', 'seed', 'config', 'case_id'}
            source_inputs[source] = digest
        group = (cond['id'], str(fold))
        pair = (source, digest)
        assert source_groups.setdefault(group, pair) == pair
    assert actual == wanted, 'Missing, duplicated or unexpected request case'
    assert len(actual) == COUNTS['potential_inputs']
    assert len(source_inputs) == len(source_groups) == 72
    return p


def terminal(case):
    out = ROOT / 'runs' / case['case_id'] / 'repeat1'
    receipt = read(out / 'launcher_receipt.json')
    assert receipt['case_id'] == case['case_id'] and receipt['repeat'] == 1
    assert receipt['input_sha256'] == case['sha256']
    assert receipt['protocol_sha256'] == sha(ROOT / 'FROZEN_PROTOCOL.json')
    assert receipt['runtime_protocol_sha256'] == sha(ROOT / 'RUNTIME_PROTOCOL.json')
    for name, digest in receipt['output_hashes'].items():
        assert sha(out / name) == digest, name
    worker = receipt.get('worker', receipt.get('unqualified_worker_summary', {}))
    z = None
    if worker.get('finite_partition_retained'):
        assert 'native_outputs.mat' in receipt['output_hashes']
        z = np.asarray(io.loadmat(out / 'native_outputs.mat', variable_names=['label'])['label']).ravel()
        assert z.shape == (case['n'],) and np.isfinite(z).all()
        assert np.all(z == np.floor(z)) and np.all(z >= 1)
    eligible = bool(receipt['status'] == 'COMPLETE_ZERO_EXIT_VALID_MARKERS'
        and receipt.get('exit_code') == 0 and receipt.get('cleanup', {}).get('complete')
        and receipt.get('worker', {}).get('benchmark_contract_met')
        and z is not None and len(np.unique(z)) == case['c'])
    # A valid cap-500 result is eligible; convergence is not inferred here.
    return receipt, z, eligible


def outcome(case):
    receipt, z, eligible = terminal(case)
    worker = receipt.get('worker', receipt.get('unqualified_worker_summary', {}))
    return dict(case_id=case['case_id'], attempted=True, eligible=eligible,
        status=receipt['status'], worker_status=worker.get('status'),
        finite_partition_retained=z is not None,
        receipt_sha256=sha(ROOT / 'runs' / case['case_id'] / 'repeat1' / 'launcher_receipt.json'))


def not_attempted(case, reason):
    return dict(case_id=case['case_id'], attempted=False, eligible=False,
        status=reason, finite_partition_retained=False)
