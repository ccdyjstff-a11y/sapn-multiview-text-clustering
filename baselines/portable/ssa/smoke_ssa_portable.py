"""Execute preparation/input/refusal checks. Never start MATLAB or score classes."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import numpy as np
from scipy import io, sparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('release-root', 'author-root', 'work-root'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    release, author, work = [Path(x).resolve() for x in (args.release_root, args.author_root, args.work_root)]
    if work.exists() or work == release or release in work.parents:
        raise ValueError('A new work root outside the release is required')
    work.mkdir(parents=True)
    script = Path(__file__).with_name('ssa_portable.py')
    checks = []

    def call(name, words, expected=0, message=None):
        result = subprocess.run([sys.executable, '-B', str(script), *map(str, words)],
                                capture_output=True, text=True, encoding='utf8')
        passed = (result.returncode == 0) if expected == 0 else (result.returncode != 0)
        if message: passed = passed and message in result.stderr
        checks.append({'name': name, 'exit_code': result.returncode, 'passed': passed})
        if not passed: raise AssertionError(name + '\n' + result.stdout + result.stderr)

    call('private_preparation', ['prepare', '--release-root', release, '--author-root', author, '--output', work / 'runtime'])
    call('refuse_public_runtime', ['prepare', '--release-root', release, '--author-root', author, '--output', release / '__invalid_runtime'], 1, 'outside the release')
    data = work / 'data'; data.mkdir()
    (data / 'metadata.json').write_text(json.dumps({'n': 24, 'k': 3, 'views': 2}), encoding='utf8')
    rng = np.random.RandomState(42)
    for v in range(2): sparse.save_npz(data / f'X{v}.npz', sparse.csr_matrix(.1 + rng.rand(24, 4 + v)))
    pool = np.arange(8)
    pairs = np.array([[i, j] for i in pool for j in pool if i < j], dtype=np.int64)
    answers = np.where(np.arange(len(pairs)) % 2, 1, -1)
    packet = dict(pool=pool, test=np.arange(8, 24), pairs=pairs, answers=answers)
    np.savez(data / 'paid.npz', **packet)
    common = ['make-input', '--release-root', release, '--prepared-directory', data,
              '--fit-seed', '17', '--query-count', '12', '--case-id', 'synthetic']
    call('valid_double_input', common + ['--paid-queries', data / 'paid.npz', '--output', work / 'source.mat'])
    typed = {key: kind for key, _, kind in io.whosmat(work / 'source.mat')}
    assert all(typed[x] == 'double' for x in ('c', 'seed', 'ML', 'CL'))
    checks.append({'name': 'historical_mat_numeric_types', 'passed': True})
    np.savez(data / 'bad_labels.npz', **packet, labels=np.zeros(24))
    call('refuse_class_packet', common + ['--paid-queries', data / 'bad_labels.npz', '--output', work / 'bad.mat'], 1, 'paid-query fields')
    call('refuse_public_input', common + ['--paid-queries', data / 'paid.npz', '--output', release / '__invalid_input.mat'], 1, 'outside the release')
    call('refuse_input_overwrite', common + ['--paid-queries', data / 'paid.npz', '--output', work / 'source.mat'], 1, 'already exists')
    zero = sparse.load_npz(data / 'X0.npz').tolil(); zero[0, :] = 0
    sparse.save_npz(data / 'X0.npz', zero.tocsr())
    call('refuse_zero_row', common + ['--paid-queries', data / 'paid.npz', '--output', work / 'zero.mat'], 1, 'zero object rows')
    fit = ['fit', '--runtime', work / 'runtime', '--source-mat', work / 'source.mat', '--case-id', 'synthetic',
           '--matlab', work / 'not_a_matlab_executable', '--plan-only']
    call('plan_without_matlab', fit + ['--output', work / 'plan'])
    call('refuse_public_fit', fit + ['--output', release / '__invalid_fit'], 1, 'outside the release')
    call('refuse_invalid_parameter', fit + ['--beta', '5', '--output', work / 'invalid_beta'], 1, 'declared grid')
    call('refuse_invalid_time_limit', fit + ['--timeout-seconds', '0', '--output', work / 'invalid_timeout'], 1, 'Finite timeout')
    runtime_manifest = json.loads((work / 'runtime/PORTABLE_PREPARATION.json').read_text('utf8'))
    victim = work / 'runtime' / next(iter(runtime_manifest['runtime_files']))
    original_victim = victim.read_bytes()
    victim.write_bytes(original_victim + b'\n% Intentional test mutation\n')
    call('refuse_runtime_mutation', fit + ['--output', work / 'mutated'], 1, 'Runtime file mismatch')
    victim.write_bytes(original_victim)
    altered = dict(runtime_manifest); altered['worker'] = '../../outside_worker.m'
    manifest_path = work / 'runtime/PORTABLE_PREPARATION.json'
    original_manifest = manifest_path.read_bytes()
    manifest_path.write_text(json.dumps(altered), encoding='utf8')
    call('refuse_runtime_entry_escape', fit + ['--output', work / 'escape'], 1, 'entry escapes')
    altered['worker'] = 'unbound/ssa_formal_worker.m'
    manifest_path.write_text(json.dumps(altered), encoding='utf8')
    call('refuse_unbound_worker', fit + ['--output', work / 'unbound'], 1, 'not hash-bound')
    manifest_path.write_bytes(original_manifest)
    result = {'status': 'PASS', 'scope': 'Preparation, typed input and refusal checks; no MATLAB fit or class scoring',
              'checks': checks, 'script_sha256': hashlib.sha256(script.read_bytes()).hexdigest()}
    (work / 'SMOKE_QA.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
    print(json.dumps({'status': 'PASS', 'checks': len(checks), 'report': str(work / 'SMOKE_QA.json')}))


if __name__ == '__main__':
    main()
