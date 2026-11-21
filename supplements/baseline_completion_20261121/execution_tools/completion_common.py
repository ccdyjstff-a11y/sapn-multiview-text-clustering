"""Own portable orchestration helpers. No optimizer, category data, or test scoring."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import time
from contextlib import contextmanager
import numpy as np

HERE = Path(__file__).resolve().parent
CODE = ('completion_common.py', 'prepare_inputs.py', 'completion_portable.py', 'native_backend.py',
        'PORTABLE_COMPLETION_SPEC.json', 'SSA_RUNTIME_IDENTITY.json')

def require(ok, message):
    if not ok:
        raise ValueError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':')).encode('utf8')

def identity(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def immutable(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(read(path) == value, 'Immutable record differs: ' + str(path))
        return
    with path.open('x', encoding='utf8', newline='\n') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')

def relative(root, name):
    root = Path(root).resolve()
    p = (root / name).resolve()
    require(p.is_relative_to(root), 'Path escapes root: ' + str(name))
    return p

def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

def own_sources():
    return {name: sha(HERE / name) for name in CODE}

def load_spec(release):
    spec = read(HERE / 'PORTABLE_COMPLETION_SPEC.json')
    require(sha(HERE / 'PORTABLE_COMPLETION_SPEC.json') ==
            (HERE / 'PORTABLE_COMPLETION_SPEC.sha256').read_text().strip(), 'Spec hash changed')
    require(spec['edition'] == 'major_revision_20261118', 'Wrong edition')
    for name, digest in spec['public_sources'].items():
        require(sha(relative(release, name)) == digest, 'Public release differs: ' + name)
    require(len(spec['methods']['SSA']['conditions']) == 15 and
            len(spec['methods']['SSA']['grid']) == 18, 'SSA scope changed')
    require([c['id'] for c in spec['methods']['PCBG']['conditions']] ==
            ['BBC_s24_Q400', 'BBC_s25_Q400', 'CiteSeer_s23_Q400', 'CiteSeer_s25_Q400'],
            'PCBG condition order changed')
    require(len(spec['methods']['PCBG']['new_cases']) == 5, 'PCBG scope changed')
    return spec

def case_id(condition, candidate, fold):
    return condition['id'] + '_c' + str(candidate) + '_' + ('final' if fold == 'final' else 'f' + str(fold))

def plain_folds(folds):
    return [{k: np.asarray(v).tolist() for k, v in x.items()} for x in folds]

def paid_ber(labels, pairs, answers):
    positive, negative = answers == 1, answers == -1
    require(positive.any() and negative.any(), 'Paid BER requires both signs')
    same = labels[pairs[:, 0]] == labels[pairs[:, 1]]
    return float(.5 * (np.count_nonzero(~same & positive) / np.count_nonzero(positive)
                      + np.count_nonzero(same & negative) / np.count_nonzero(negative)))

def select(scores):
    """Global minimum then default-first 1e-12 tie set; incomplete candidates fail."""
    a = np.asarray([[np.nan if x is None else x for x in row] for row in scores], float)
    require(a.ndim == 2 and a.shape[1] == 2, 'Two folds required')
    means = [float(row.mean()) if np.isfinite(row).all() else None for row in a]
    available = [x for x in means if x is not None]
    if not available:
        return 0, means, True
    best = min(available)
    chosen = next(i for i, x in enumerate(means) if x is not None and abs(x - best) <= 1e-12)
    return chosen, means, False

def check_stop(work):
    require(not (Path(work) / 'STOP').exists(), 'STOP requested; no new fit dispatched')

def before_dispatch(work, available_gib=3, wait_seconds=1800, *, clock=time.monotonic,
                    sleep=time.sleep, psutil_module=None):
    if psutil_module is None:
        import psutil as psutil_module
    def no_matlab():
        active = []
        for proc in psutil_module.process_iter(['pid', 'name']):
            try:
                if (proc.info['name'] or '').lower() in ('matlab', 'matlab.exe'):
                    active.append(proc.info['pid'])
            except (psutil_module.NoSuchProcess, psutil_module.AccessDenied):
                continue
        require(not active, 'Existing MATLAB prevents dispatch: ' + str(active))
    check_stop(work)
    no_matlab()
    started = clock()
    while psutil_module.virtual_memory().available < available_gib * 2**30:
        check_stop(work)
        require(clock() - started < wait_seconds, 'Memory wait cap reached; no fit started')
        sleep(5)
    check_stop(work)
    no_matlab()

@contextmanager
def controller_lock(work):
    """OS lock, released after crashes; the file is retained for provenance."""
    path = Path(work) / 'controller.lock'
    f = path.open('a+b')
    try:
        if f.tell() == 0:
            f.write(b'0'); f.flush()
        f.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        f.close()
