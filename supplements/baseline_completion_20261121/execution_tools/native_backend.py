"""Byte-bound public single-fit adapters; category arrays are never accepted."""
import argparse
from pathlib import Path
import numpy as np
from scipy import io
from completion_common import (HERE, require, read, sha, immutable, relative, module,
                               load_spec, own_sources, case_id, paid_ber, before_dispatch, identity)
from prepare_inputs import paid, semantic

class Context:
    def __init__(self, release, work):
        self.release, self.work = Path(release).resolve(), Path(work).resolve()
        require(not self.work.is_relative_to(self.release) and not self.work.is_relative_to(HERE),
                'Private execution work cannot be in release/tool directory')
        self.spec = load_spec(self.release)
        self.manifest = read(self.work / 'INPUT_MANIFEST.json')
        require(self.manifest['own_sources'] == own_sources(), 'Own source changed since input preparation')
        require(self.manifest['spec_sha256'] == sha(HERE / 'PORTABLE_COMPLETION_SPEC.json'), 'Prepared spec differs')
        for name, digest in self.manifest['files'].items():
            require(sha(relative(self.work, name)) == digest, 'Private prepared artifact changed: ' + name)
        self.plans = {m: read(self.work / m / 'PLAN.json') for m in ('SSA', 'PCBG')}
        require(sha(self.work / 'pcbg_runtime/pcbg_portable.py') ==
                'a8f9e47c7d67de5a5b48d42ce2f5488b8e168302c7e18ef960c1c0de504d4d70',
                'PCBG launcher is not the reviewed 1800-second extension')
        for name in ('IDENTITIES.json', 'pcbg_clock_bridge.m', 'pcbg_formal_worker.m'):
            require(sha(self.work / 'pcbg_runtime' / name) ==
                    self.spec['public_sources']['baselines/portable/pcbg/' + name],
                    'PCBG private support differs from pinned public content: ' + name)
        self.native = module(self.work / 'pcbg_runtime/pcbg_portable.py', 'portable_completion_pcbg')
        require(self.plans['PCBG']['source_hashes'] == self.native.sources(), 'Extended launcher source differs')
        self.ssa = module(self.release / 'baselines/portable/ssa/ssa_portable.py', 'portable_completion_ssa')
        self.arrays = {}
        query = module(self.release / 'common/query_protocol.py', 'completion_context_query')
        feature_ids = read(self.release / 'baselines/portable/pcbg/IDENTITIES.json')['dataset_identities']
        for method in ('SSA', 'PCBG'):
            conditions = self.spec['methods'][method]['conditions']
            require(set(self.plans[method]['paid']) == {c['id'] for c in conditions}, 'Paid condition set changed')
            expected_cases = ({case_id(c, 0, f) for c in conditions for f in (0, 1, 'final')}
                              if method == 'SSA' else {x['case_id'] for x in self.spec['methods']['PCBG']['new_cases']})
            cases = self.plans[method]['cases']
            require(len(cases) == len(expected_cases) and {x['id'] for x in cases} == expected_cases,
                    'Prepared carrier case set changed')
            for cid, card in self.plans[method]['paid'].items():
                c = next(c for c in conditions if c['id'] == cid)
                path = relative(self.work / method, card['file'])
                require(sha(path) == card['sha256'], 'Paid prepared data changed')
                with np.load(path, allow_pickle=False) as z:
                    require(set(z.files) == {'pool', 'test', 'pairs', 'answers'}, 'Paid schema changed')
                    self.arrays[method, cid] = {k: z[k].copy() for k in z.files}
                original, original_hash = paid(self.release, c, query)
                require(card['original_projection_sha256'] == original_hash and
                        all(np.array_equal(original[k], self.arrays[method, cid][k]) for k in original),
                        'Prepared paid values differ from pinned public projection')
                for case in [x for x in cases if x['condition_id'] == cid]:
                    require(case['candidate'] == 0 and case['id'] == case_id(c, 0, case['fold']), 'Carrier identity differs')
                    source = relative(self.work / method, case['input'])
                    require(sha(source) == case['sha256'], 'Carrier container changed')
                    value = semantic(source, self.native)
                    require(identity(value) == case['semantic_sha256'], 'Carrier semantic record changed')
                    fold = case['fold']
                    ids = np.arange(c['Q']) if fold == 'final' else np.asarray(c['eligible_folds'][fold]['training'], int)
                    pp, aa = original['pairs'][ids], original['answers'][ids]
                    expected = {'X': feature_ids[c['dataset']]['canonical_feature_sha256'],
                                'c': c['c'], 'seed': c['fit_seed'], 'case_id': case['id'],
                                'config': self.spec['methods']['PCBG']['native_config'],
                                'ML': (pp[aa == 1] + 1).tolist(), 'CL': (pp[aa == -1] + 1).tolist()}
                    require(value == expected, 'Carrier values differ from trusted scientific inputs: ' + case['id'])
            require(all(x['condition_id'] in {c['id'] for c in conditions} for x in cases), 'Unknown carrier condition')

    def case(self, method, condition, fold):
        found = [x for x in self.plans[method]['cases']
                 if x['condition_id'] == condition['id'] and x['fold'] == fold]
        require(len(found) == 1, 'Input carrier absent or ambiguous')
        return found[0]

    def old_pcbg(self):
        base = self.release / 'evidence/v10/native_pcbg/partition_export'
        rows = read(base / 'evidence/CV_OUTCOMES.json')
        items = {x['outcome']['case_id']: x for x in read(base / 'CV_PARTITIONS.json')}
        conditions = {c['id']: c for c in self.spec['methods']['PCBG']['conditions']}
        result = {}
        capped = []
        for row in rows:
            if row['condition_id'] not in conditions: continue
            c = conditions[row['condition_id']]; cid = row['case_id']; item = items[cid]
            require(item['outcome'] == row, 'Public CV outcome/export disagreement')
            if row['status'] == 'WORKER_ACTIVE_TIME_CAP':
                capped.append(cid); continue
            score = None
            if row['eligible']:
                path = relative(base, item['partition']['file'])
                require(sha(path) == item['partition']['sha256'], 'Retained CV partition changed')
                with np.load(path, allow_pickle=False) as z:
                    require(set(z.files) == {'partition'}, 'Old CV partition schema')
                    labels = z['partition'].ravel()
                validate_partition(labels, c, True)
                idx = np.asarray(c['eligible_folds'][row['fold']]['validation'], int)
                a = self.arrays['PCBG', c['id']]
                score = paid_ber(labels, a['pairs'][idx], a['answers'][idx])
                require(abs(score - row['validation_BER']) <= 1e-15, 'Old paid BER differs')
            result[cid] = {'case_id': cid, 'eligible': bool(row['eligible']), 'validation_BER': score,
                           'origin': 'original_completed_cv', 'original_status': row['status'],
                           'original_worker_status': row.get('worker_status'),
                           'original_receipt_sha256': row['receipt_sha256']}
        require(len(result) == 23 and capped == ['BBC_s24_Q400_c0_f0'], 'PCBG retained/replaced CV scope changed')
        return result

def validate_partition(pred, c, eligible):
    require(pred.shape == (c['n'],) and np.isfinite(pred).all() and (pred == np.floor(pred)).all(),
            'Retained native partition invalid')
    if eligible:
        require(len(np.unique(pred)) == c['c'], 'Eligible native partition has wrong k')

def validate_ssa_runtime(launcher, runtime):
    receipt = launcher.runtime_check(runtime)
    trusted = read(HERE / 'SSA_RUNTIME_IDENTITY.json')
    for key in ('author_commit', 'author_sources', 'own_preparers', 'worker', 'profile_root', 'runtime_files'):
        require(receipt.get(key) == trusted[key], 'SSA runtime is not the reviewed content: ' + key)
    return receipt

class NativeBackend:
    def __init__(self, context, method, matlab, ssa_runtime=None, pcbg_author=None, readonly=False):
        self.ctx, self.method, self.readonly = context, method, readonly
        self.matlab = str(Path(matlab).resolve())
        require(Path(self.matlab).is_file(), 'Explicit MATLAB executable missing')
        self.runtime = Path(ssa_runtime).resolve() if ssa_runtime else None
        self.author = Path(pcbg_author).resolve() if pcbg_author else None
        binding = {'method': method, 'input_manifest_sha256': sha(context.work / 'INPUT_MANIFEST.json'),
                   'own_sources': own_sources(), 'matlab': self.matlab,
                   'matlab_executable_sha256': sha(self.matlab), 'class_labels_loaded': False}
        if method == 'SSA':
            require(self.runtime is not None, 'SSA needs explicitly prepared runtime')
            receipt = validate_ssa_runtime(context.ssa, self.runtime)
            require(receipt['author_commit'] == '4d37f15621ee7bd9d412ad224408658ad243699c', 'SSA author version')
            # The portable prepare receipt must bind the same verified public preparation scripts.
            public_own = read(context.release / 'baselines/ssa_historical/OWN_SOURCE_MANIFEST.json')
            known = {r['packaged_path']: r['sha256'] for r in public_own['files']}
            for name, digest in receipt['own_preparers'].items():
                key = 'historical_tree/' + name
                if key in known: require(digest == known[key], 'SSA own preparer differs')
                require(sha(context.release / 'baselines/ssa_historical' / key) == digest, 'SSA public preparation differs')
            binding.update(ssa_runtime=str(self.runtime),
                           runtime_manifest_sha256=sha(self.runtime / 'PORTABLE_PREPARATION.json'))
        else:
            require(self.author is not None, 'PCBG needs explicit external author source')
            audited = context.native.author_audit(self.author)
            binding.update(pcbg_author=str(self.author), author_commit=audited['author_commit'],
                           author_files=audited['author_files'], native_sources=context.native.sources())
        self.binding = binding
        path = context.work / method / 'RUN_BINDING.json'
        if readonly: require(path.is_file() and read(path) == binding, 'Run binding absent or changed')
        else: immutable(path, binding)

    def fit(self, c, candidate, fold):
        return self.ssa_fit(c, candidate, fold) if self.method == 'SSA' else self.pcbg_fit(c, candidate, fold)

    def ssa_fit(self, c, candidate, fold):
        ctx = self.ctx; config = ctx.spec['methods']['SSA']['grid'][candidate]
        cid = case_id(c, candidate, fold); carrier = ctx.case('SSA', c, fold)
        source = relative(ctx.work / 'SSA', carrier['input'])
        require(sha(source) == carrier['sha256'], 'SSA source changed')
        validate_ssa_runtime(ctx.ssa, self.runtime)
        dest = ctx.work / 'SSA/runs' / cid
        if (ctx.work / 'SSA/ledger/fits' / (cid + '.json')).exists():
            require((dest / 'EXECUTION.json').is_file(), 'Recorded SSA attempt missing; never refit a recorded case')
        cap = ctx.spec['methods']['SSA']['policy']['per_fit_whole_process_cap_seconds']
        require(cap == 600, 'SSA cap must be whole-process 600 s')
        if not dest.exists():
            require(not self.readonly, 'Unexecuted SSA case: ' + cid)
            before_dispatch(ctx.work)
            ctx.ssa.fit(argparse.Namespace(runtime=str(self.runtime), source_mat=str(source), output=str(dest),
                matlab=self.matlab, case_id=cid, timeout_seconds=cap, plan_only=False, **config))
        job = read(dest / 'JOB.json')
        require(job['case_id'] == cid and job['config'] == config, 'SSA existing job science differs')
        require(job['source_sha256'] == carrier['sha256'] == sha(job['source']), 'SSA existing source differs')
        require(Path(job['source']).resolve() == source and Path(job['runtime']).resolve() == self.runtime,
                'SSA source/runtime path differs from current run binding')
        require(job['runtime_manifest_sha256'] == self.binding['runtime_manifest_sha256'] ==
                sha(self.runtime / 'PORTABLE_PREPARATION.json'), 'SSA runtime receipt changed')
        require(job['timeout_seconds'] == cap and job['matlab'] == self.matlab, 'SSA existing launch policy differs')
        for name, key in [('request.mat', 'request_sha256'), ('launch_ssa.m', 'launch_sha256')]:
            require(sha(dest / name) == job[key], 'SSA bound request/launch changed')
        require({n for n, _, _ in io.whosmat(dest / 'request.mat')} ==
                {'source_input', 'source_sha256', 'config', 'case_id'}, 'SSA request schema differs')
        request = io.loadmat(dest / 'request.mat', simplify_cells=True,
                             variable_names=['source_input', 'source_sha256', 'config', 'case_id'])
        require(Path(request['source_input']).resolve() == source and
                request['source_sha256'] == carrier['sha256'] and request['config'] == config and
                request['case_id'] == cid, 'SSA effective request does not match bound candidate')
        require((dest / 'EXECUTION.json').is_file(), 'Incomplete SSA attempt preserved; explicit repair required')
        execution = read(dest / 'EXECUTION.json')
        summary = read(dest / 'worker_summary.json') if (dest / 'worker_summary.json').is_file() else {}
        require(summary or execution.get('timed_out'),
                'SSA preworker infrastructure failure unresolved; preserve attempt, do not prune or retry')
        require(execution.get('input_and_runtime_hashes_unchanged') is True, 'SSA post-fit identity failed')
        eligible = bool(execution['status'] == 'EXECUTED' and execution.get('benchmark_contract_met') is True
                        and summary.get('benchmark_contract_met') is True and summary.get('execution_returned') is True)
        if summary:
            require(summary.get('case_id') == cid and summary.get('source_input_sha256') == carrier['sha256'],
                    'SSA worker identity differs')
        if eligible:
            require(summary.get('n') == c['n'] and summary.get('c') == c['c'] and
                    summary.get('native_options') == config, 'SSA worker effective config/dimensions differ')
            require({'worker_summary.json', 'native_outputs.mat'} <= set(execution['files']), 'SSA bound outputs missing')
        for name, digest in execution.get('files', {}).items():
            require(sha(relative(dest, name)) == digest, 'SSA execution artifact changed')
        score = None; occupied = None
        if summary.get('finite_partition_retained'):
            require('native_outputs.mat' in execution.get('files', {}), 'Retained partition must be hash-bound')
            d = io.loadmat(dest / 'native_outputs.mat', variable_names=['pred', 'V_star'])
            pred = np.asarray(d['pred']).ravel(); validate_partition(pred, c, eligible)
            occupied = len(np.unique(pred))
            if eligible:
                v = np.asarray(d['V_star'])
                require(v.shape == (c['n'], c['c']) and np.isfinite(v).all() and
                        (v >= -1e-10).all() and (v.sum(axis=1) > 0).all(), 'SSA membership domain differs')
                require(np.array_equal(np.argmax(v, axis=1) + 1, pred), 'SSA argmax decode differs')
                if fold != 'final':
                    a = ctx.arrays['SSA', c['id']]; ids = np.asarray(c['eligible_folds'][fold]['validation'], int)
                    score = paid_ber(pred, a['pairs'][ids], a['answers'][ids])
        require(not eligible or occupied is not None, 'Eligible SSA retained partition absent')
        require(not eligible or fold == 'final' or score is not None, 'SSA eligible CV BER absent')
        return {'case_id': cid, 'condition_id': c['id'], 'candidate': candidate, 'fold': fold,
                'attempted': True, 'eligible': eligible, 'validation_BER': score,
                'classification': 'eligible' if eligible else 'terminal_ineligible',
                'occupied_clusters': occupied, 'execution_status': execution['status'],
                'worker_status': summary.get('status'), 'timed_out': bool(execution.get('timed_out')),
                'config': config, 'source_sha256': carrier['sha256'], 'job_sha256': sha(dest / 'JOB.json'),
                'execution_sha256': sha(dest / 'EXECUTION.json'), 'full_class_labels_loaded': False}

    def pcbg_fit(self, c, candidate, fold):
        require(candidate == 0, 'Frozen five-fit PCBG plan cannot expand silently')
        ctx = self.ctx; case = ctx.case('PCBG', c, fold); root = ctx.work / 'PCBG'
        dest = root / 'runs' / case['id']; plan = ctx.native.load_plan(root)
        committed = root / 'ledger' / ('conditions' if fold == 'final' else 'selections') / (c['id'] + '.json')
        if committed.exists():
            require((dest / 'receipt.json').is_file(), 'Recorded PCBG attempt missing; never refit a recorded case')
        require(sha(relative(root, case['input'])) == case['sha256'], 'PCBG source changed')
        ctx.native.author_audit(self.author)
        if not (dest / 'receipt.json').is_file():
            require(not self.readonly, 'Unexecuted PCBG case: ' + case['id'])
            require(not dest.exists(), 'Partial PCBG attempt preserved; no automatic retry')
            before_dispatch(ctx.work)
        r = (read(dest / 'receipt.json') if self.readonly else
             ctx.native.execute_case(root, plan, case, self.author, Path(self.matlab)))
        require(r['case_id'] == case['id'] and r['plan_sha256'] == sha(root / 'PLAN.json'), 'PCBG receipt identity')
        require(r['author_commit'] == self.binding['author_commit'], 'PCBG author receipt differs')
        for name, digest in r['artifacts'].items():
            require(sha(relative(dest, name)) == digest, 'PCBG artifact changed')
        score = None; occupied = None
        if r['finite_partition_retained']:
            require('native_outputs.mat' in r['artifacts'], 'PCBG retained partition not bound')
            labels = io.loadmat(dest / 'native_outputs.mat', variable_names=['label'])['label'].ravel()
            validate_partition(labels, c, r['eligible']); occupied = len(np.unique(labels))
            if r['eligible'] and fold != 'final':
                a = ctx.arrays['PCBG', c['id']]; ids = np.asarray(c['eligible_folds'][fold]['validation'], int)
                score = paid_ber(labels, a['pairs'][ids], a['answers'][ids])
        if r['eligible']:
            require({'worker_summary.json', 'native_outputs.mat', 'worker_started.json', 'worker_finished.json'}
                    <= set(r['artifacts']), 'PCBG eligible artifacts absent')
            summary = read(dest / 'worker_summary.json')
            require(r['status'] == 'PROCESS_EXITED' and r['exit_code'] == 0 and
                    summary.get('case_id') == case['id'] and summary.get('benchmark_contract_met') is True,
                    'PCBG eligible contract differs')
            require(occupied is not None and (fold == 'final' or score is not None), 'Eligible PCBG partition absent')
        return {'case_id': case['id'], 'eligible': bool(r['eligible']), 'validation_BER': score,
                'origin': 'new_portable_execution_extension', 'execution_status': r['status'],
                'worker_status': r['worker_status'], 'occupied_clusters': occupied,
                'receipt_sha256': sha(dest / 'receipt.json')}
