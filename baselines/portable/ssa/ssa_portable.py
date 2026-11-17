"""Relocatable SSA preparation and single-fit entry, without class-label inputs.

Author code is acquired separately. This entry reconstructs the unchanged
historical patch-generator layout in a NEW private workspace, verifies source
identities, and runs the unchanged formal MATLAB worker with explicit paths.
It does not replace the historical full-study scheduling/selection protocol.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open('xb') as stream:
        stream.write(src.read_bytes())


def prepare(args):
    release = Path(args.release_root).resolve()
    source = release / 'baselines/ssa_historical'
    author = Path(args.author_root).resolve()
    output = Path(args.output).resolve()
    need(output != release and release not in output.parents, 'Private output must be outside the release')
    need(output != author and author not in output.parents, 'Private output must be outside the author-source directory')
    need(not output.exists(), 'Output must be a new private directory')
    identity = read(source / 'EXTERNAL_AUTHOR_IDENTITY.json')
    need(identity['commit'] == '4d37f15621ee7bd9d412ad224408658ad243699c', 'Wrong author commit')
    for row in identity['files']:
        need(sha(author / row['path']) == row['sha256'], 'Author hash mismatch: ' + row['path'])
    own = {row['packaged_path']: row['sha256'] for row in read(source / 'OWN_SOURCE_MANIFEST.json')['files']}
    relative = [
        'review_revision_v10/ssa_minimal_qualification/prepare_profile.py',
        'review_revision_v10/ssa_minimal_qualification/run_ssa_synthetic_qualification.m',
        'review_revision_v10/ssa_formal_v1/prepare_worker.py',
        'review_revision_v10/ssa_formal_v1/ssa_formal_worker.m',
        'review_revision_v9/baseline_gap/native_formal_v1/pcbg_clock_bridge.m',
        'review_revision_v6/literature_followup/SSA_download_manifest.json',
    ]
    download_name = relative[-1]
    download = read(source / 'historical_tree' / download_name)
    expected_author = {row['path']: row['sha256'] for row in download['files']}
    need(download['commit'] == identity['commit'] and all(expected_author[row['path']] == row['sha256'] for row in identity['files']), 'Download identity record differs')
    own['historical_tree/' + download_name] = sha(source / 'historical_tree' / download_name)
    for name in relative:
        key = 'historical_tree/' + name
        need(key in own and sha(source / key) == own[key], 'Own source hash mismatch: ' + key)
    output.mkdir(parents=True)
    tree = output / 'private_tree'
    for name in relative:
        copy(source / 'historical_tree' / name, tree / name)
    dest = tree / 'review_revision_v6/literature_followup/author_source_private_analysis'
    for row in identity['files']:
        copy(author / row['path'], dest / row['path'])
    minimal = tree / 'review_revision_v10/ssa_minimal_qualification'
    formal = tree / 'review_revision_v10/ssa_formal_v1'
    for script in (minimal / 'prepare_profile.py', formal / 'prepare_worker.py'):
        completed = subprocess.run([sys.executable, '-I', '-B', str(script)], capture_output=True, text=True, encoding='utf8')
        need(completed.returncode == 0, completed.stdout + completed.stderr)
    bound = [formal / 'ssa_formal_worker.m', *(formal / 'private_generated').glob('*.m')]
    receipt = {
        'status': 'PREPARED_NOT_EXECUTED',
        'release_root': str(release),
        'author_commit': identity['commit'],
        'author_sources': {row['path']: row['sha256'] for row in identity['files']},
        'own_preparers': {name: own['historical_tree/' + name] for name in relative},
        'worker': (formal / 'ssa_formal_worker.m').relative_to(output).as_posix(),
        'profile_root': (formal / 'private_generated').relative_to(output).as_posix(),
        'runtime_files': {p.relative_to(output).as_posix(): sha(p) for p in bound},
        'algorithm_changed': False,
        'historical_controller_or_runtime_receipt_required': False,
        'third_party_derivatives_private': True,
    }
    write(output / 'PORTABLE_PREPARATION.json', receipt)
    return {'status': receipt['status'], 'runtime_files': len(bound), 'output': str(output)}


def make_input(args):
    import numpy as np
    from scipy import io, sparse
    directory = Path(args.prepared_directory).resolve()
    release = Path(args.release_root).resolve()
    output = Path(args.output).resolve()
    need(output != release and release not in output.parents, 'Input output must be outside the release')
    meta = read(directory / 'metadata.json')
    n, k, views = (int(meta[x]) for x in ('n', 'k', 'views'))
    need(n >= 11 and 2 <= k < n and views >= 2, 'SSA requires n>=11, 2<=k<n and >=2 views')
    packet = Path(args.paid_queries).resolve()
    with np.load(packet, allow_pickle=False) as z:
        need(set(z.files) == {'pool', 'test', 'pairs', 'answers'}, 'Expected only paid-query fields')
        q = {name: z[name].copy() for name in z.files}
    pool, pairs, answers = q['pool'], q['pairs'], q['answers']
    need(pool.ndim == 1 and np.issubdtype(pool.dtype, np.integer) and len(np.unique(pool)) == len(pool)
         and np.all((pool >= 0) & (pool < n)), 'Invalid query pool')
    need(np.array_equal(q['test'], np.setdiff1d(np.arange(n), pool)), 'Evaluation complement mismatch')
    need(pairs.ndim == 2 and pairs.shape[1] == 2 and np.issubdtype(pairs.dtype, np.integer)
         and np.isin(pairs, pool).all() and (pairs[:, 0] < pairs[:, 1]).all()
         and len(np.unique(pairs, axis=0)) == len(pairs), 'Invalid unique ordered paid pairs')
    need(answers.shape == (len(pairs),) and np.isin(answers, [-1, 1]).all(), 'Invalid paid signs')
    need(0 < args.query_count <= len(pairs) and args.fit_seed >= 0, 'Invalid query budget or fit seed')
    pairs, answers = pairs[:args.query_count], answers[:args.query_count]
    ix = np.arange(len(pairs))
    if args.fold != 'final':
        need(args.fold_seed is not None, 'Validation input needs the declared node-partition seed')
        held = np.array_split(np.random.RandomState(args.fold_seed).permutation(pool), 2)[int(args.fold)]
        mask = np.isin(pairs, held)
        ix = np.flatnonzero((~mask).all(axis=1))
        validation = np.flatnonzero(mask.all(axis=1))
        need(all(set(answers[x].tolist()) == {-1, 1} for x in (ix, validation)), 'Fold lacks both signs; do not fit it')
    selected, signs = pairs[ix], answers[ix]
    matrix = np.empty((views, 1), dtype=object)
    hashes = {'metadata.json': sha(directory / 'metadata.json')}
    for v in range(views):
        path = directory / ('X' + str(v) + '.npz')
        x = sparse.load_npz(path).astype(np.float64)
        need(x.shape[0] == n and x.shape[1] >= 2 and np.isfinite(x.data).all() and (x.data >= 0).all(), 'Invalid view')
        need(np.all(np.asarray(x.multiply(x).sum(axis=1)).ravel() > 0), 'Native SSA does not support zero object rows')
        matrix[v, 0] = x.tocsc()
        hashes[path.name] = sha(path)
    need(not output.exists() and not output.with_suffix('.provenance.json').exists(), 'Input output already exists')
    output.parent.mkdir(parents=True, exist_ok=True)
    io.savemat(output, {'X': matrix, 'c': float(k), 'ML': (selected[signs == 1] + 1).astype(np.float64),
        'CL': (selected[signs == -1] + 1).astype(np.float64), 'seed': float(args.fit_seed),
        'config': {'beta': .1, 'lam': .2, 'kap': .95}, 'case_id': args.case_id})
    write(output.with_suffix('.provenance.json'), {'status': 'PREPARED_NOT_FITTED', 'source_sha256': sha(output),
        'prepared_feature_hashes': hashes, 'paid_query_sha256': sha(packet), 'query_count': args.query_count,
        'training_indices': ix.tolist(), 'fold': args.fold, 'fold_seed': args.fold_seed,
        'fit_seed': args.fit_seed, 'class_labels_loaded': False, 'feature_normalization_added': False})
    return {'status': 'PREPARED_NOT_FITTED', 'input': str(output), 'training_pairs': len(ix)}


def runtime_check(root):
    root = root.resolve()
    manifest = read(root / 'PORTABLE_PREPARATION.json')
    worker = (root / manifest['worker']).resolve()
    profile = (root / manifest['profile_root']).resolve()
    need(root in worker.parents and root in profile.parents, 'Runtime entry escapes its root')
    need(worker.name == 'ssa_formal_worker.m' and worker.relative_to(root).as_posix() in manifest['runtime_files'], 'Worker is not hash-bound')
    required = ['ssa_formal_profile.m', 'Create_KNNgraph.m', 'NormalizationK.m', 'normalization.m',
                'Update_V_v.m', 'Update_W_v.m', 'Update_Teta.m', 'Cost_function.m']
    need(all((profile / name).relative_to(root).as_posix() in manifest['runtime_files'] for name in required), 'Profile closure is incomplete')
    for name, digest in manifest['runtime_files'].items():
        p = (root / name).resolve()
        need(root in p.parents and sha(p) == digest, 'Runtime file mismatch: ' + name)
    return manifest


def mquote(path):
    return "'" + str(path).replace('\\', '/').replace("'", "''") + "'"


def fit(args):
    from scipy import io
    root, source = Path(args.runtime).resolve(), Path(args.source_mat).resolve()
    manifest = runtime_check(root)
    need(set(name for name, _, _ in io.whosmat(source)) == {'X','c','ML','CL','seed','config','case_id'}, 'Input MAT schema is not label-free worker schema')
    need(args.beta in [.1, .2, .3] and args.lam in [.2, .3] and args.kap in [.8, .9, .95], 'Outside declared grid')
    output = Path(args.output).resolve()
    release = Path(manifest['release_root']).resolve()
    need(output != release and release not in output.parents, 'Fit output must be outside the release')
    need(1 <= args.timeout_seconds <= 86400, 'Finite timeout must be between 1 and 86400 seconds')
    need(not output.exists(), 'Fit output must be new')
    output.mkdir(parents=True)
    config = {'beta': args.beta, 'lam': args.lam, 'kap': args.kap}
    request = output / 'request.mat'
    io.savemat(request, {'source_input': str(source), 'source_sha256': sha(source), 'config': config, 'case_id': args.case_id})
    worker = root / manifest['worker']
    launch = output / 'launch_ssa.m'
    launch.write_text('addpath(' + mquote(worker.parent) + ');\nssa_formal_worker(' + mquote(request) + ',' + mquote(output) + ',' + mquote(root / manifest['profile_root']) + ');\n', encoding='utf8')
    write(output / 'JOB.json', {'status': 'PREPARED_NOT_EXECUTED', 'runtime': str(root), 'runtime_manifest_sha256': sha(root / 'PORTABLE_PREPARATION.json'),
        'source': str(source), 'source_sha256': sha(source), 'request_sha256': sha(request), 'launch_sha256': sha(launch),
        'config': config, 'case_id': args.case_id, 'matlab': str(Path(args.matlab).resolve()),
        'timeout_seconds': args.timeout_seconds, 'class_labels_loaded': False})
    if args.plan_only:
        return {'status': 'PREPARED_NOT_EXECUTED', 'job_dir': str(output)}
    return execute(argparse.Namespace(job_dir=str(output)))


def execute(args):
    import psutil
    output = Path(args.job_dir).resolve()
    job = read(output / 'JOB.json')
    root = Path(job['runtime'])
    need(not (output / 'EXECUTION.json').exists() and not (output / 'worker_summary.json').exists(), 'Do not overwrite an executed job')
    need(sha(root / 'PORTABLE_PREPARATION.json') == job['runtime_manifest_sha256'], 'Runtime manifest changed')
    runtime = runtime_check(root)
    release = Path(runtime['release_root']).resolve()
    need(output != release and release not in output.parents, 'Execution output must be outside the release')
    source, request, launch = Path(job['source']), output / 'request.mat', output / 'launch_ssa.m'
    for path, key in [(source, 'source_sha256'), (request, 'request_sha256'), (launch, 'launch_sha256')]:
        need(sha(path) == job[key], 'Bound input changed')
    need(Path(job['matlab']).is_file(), 'MATLAB executable missing')
    env = os.environ.copy()
    prefs = output / 'matlab_preferences'
    prefs.mkdir()
    env['MATLAB_PREFDIR'] = str(prefs)
    flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
    timeout = job.get('timeout_seconds', 600)
    need(isinstance(timeout, (int, float)) and 1 <= timeout <= 86400, 'Invalid execution timeout')
    process = None; exit_code = None; failure = None; timed_out = False; cleanup = []
    write(output / 'EXECUTION_STARTED.json', {'status': 'STARTED', 'time_unix': time.time(), 'timeout_seconds': timeout})
    with (output / 'matlab_stdout.log').open('xb') as log:
        try:
            if os.name == 'nt':
                with Path(job['matlab']).open('rb') as executable:
                    need(executable.read(2) == b'MZ', 'Configured MATLAB executable lacks a Windows executable header')
            process = subprocess.Popen([job['matlab'], '-batch', 'run(' + mquote(launch) + ')'], cwd=output,
                env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
            owner = psutil.Process(process.pid)
            owner_created = owner.create_time()
            try:
                exit_code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                # Only the task-created PID and its current descendants are eligible.
                # Each PID's creation time is rechecked before termination to exclude reuse.
                same_owner = psutil.Process(process.pid)
                need(same_owner.create_time() == owner_created, 'Task process identity changed')
                owned = [(p.pid, p.create_time()) for p in same_owner.children(recursive=True)]
                owned.append((same_owner.pid, owner_created))
                for pid, created in reversed(owned):
                    try:
                        child = psutil.Process(pid)
                        if child.create_time() == created and created >= owner_created:
                            child.terminate(); cleanup.append({'pid': pid, 'create_time': created, 'action': 'terminate_owned'})
                    except psutil.NoSuchProcess:
                        continue
                    except psutil.AccessDenied:
                        cleanup.append({'pid': pid, 'create_time': created, 'action': 'permission_denied'})
                try: exit_code = process.wait(timeout=10)
                except subprocess.TimeoutExpired: failure = 'Task process remains after bounded cleanup'
        except Exception as exc:
            failure = type(exc).__name__ + ': ' + str(exc)
    hashes_unchanged = False
    try:
        hashes_unchanged = (sha(source) == job['source_sha256'] and sha(request) == job['request_sha256']
            and sha(launch) == job['launch_sha256']
            and sha(root / 'PORTABLE_PREPARATION.json') == job['runtime_manifest_sha256'])
        runtime_check(root)
    except Exception as exc:
        failure = (failure or '') + ' Identity check: ' + str(exc)
        hashes_unchanged = False
    try:
        summary = read(output / 'worker_summary.json') if (output / 'worker_summary.json').is_file() else {}
    except (ValueError, OSError) as exc:
        summary = {}
        failure = (failure or '') + ' Unreadable worker receipt: ' + str(exc)
    result = {'status': 'EXECUTED' if exit_code == 0 and summary.get('execution_returned') is True and hashes_unchanged and not failure and not timed_out else 'EXECUTION_FAILED',
        'matlab_exit_code': exit_code, 'worker_status': summary.get('status'), 'timed_out': timed_out,
        'failure': failure, 'owned_process_cleanup': cleanup,
        'benchmark_contract_met': summary.get('benchmark_contract_met', False),
        'input_and_runtime_hashes_unchanged': hashes_unchanged, 'class_labels_loaded': False,
        'scope': 'One fit only. Worker eligibility is preserved; execution success is not convergence or benchmark performance.',
        'files': {p.name: sha(p) for p in (output / 'worker_summary.json', output / 'native_outputs.mat') if p.is_file()}}
    write(output / 'EXECUTION.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('prepare')
    for key in ('release-root', 'author-root', 'output'): p.add_argument('--' + key, required=True)
    p = sub.add_parser('make-input')
    for key in ('release-root', 'prepared-directory', 'paid-queries', 'output', 'case-id'): p.add_argument('--' + key, required=True)
    p.add_argument('--fit-seed', type=int, required=True); p.add_argument('--query-count', type=int, required=True)
    p.add_argument('--fold', choices=['final', '0', '1'], default='final'); p.add_argument('--fold-seed', type=int)
    p = sub.add_parser('fit')
    for key in ('runtime', 'source-mat', 'output', 'matlab', 'case-id'): p.add_argument('--' + key, required=True)
    p.add_argument('--beta', type=float, default=.1); p.add_argument('--lam', type=float, default=.2); p.add_argument('--kap', type=float, default=.95)
    p.add_argument('--timeout-seconds', type=float, default=600)
    p.add_argument('--plan-only', action='store_true')
    p = sub.add_parser('execute'); p.add_argument('--job-dir', required=True)
    args = parser.parse_args()
    result = {'prepare': prepare, 'make-input': make_input, 'fit': fit, 'execute': execute}[args.action](args)
    print(json.dumps(result, indent=2))
    if result.get('status') == 'EXECUTION_FAILED': raise SystemExit(2)


if __name__ == '__main__':
    main()
