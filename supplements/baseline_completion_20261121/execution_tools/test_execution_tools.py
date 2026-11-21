"""Isolated synthetic checks. No MATLAB, real new predictions, or category files."""
import argparse
from pathlib import Path
import tempfile
import types
import unittest
import subprocess
import sys
from unittest.mock import patch
import numpy as np
from scipy import io
import completion_common as common
import completion_portable as controller
import native_backend as native
from prepare_inputs import extended_launcher_bytes

RELEASE = None

class FakeBackend:
    def __init__(self, outcome=None):
        self.outcome = outcome or (lambda c, ci, f: (True, .5))
        self.calls = []; self.results = {}
    def fit(self, c, ci, fold):
        cid = common.case_id(c, ci, fold)
        if cid not in self.results:
            self.calls.append(cid)
            eligible, score = self.outcome(c, ci, fold)
            self.results[cid] = {'case_id': cid, 'eligible': eligible,
                                'validation_BER': score if eligible and fold != 'final' else None}
        return self.results[cid]

class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
    def tearDown(self): self.temp.cleanup()
    def spec(self, method): return common.read(common.HERE / 'PORTABLE_COMPLETION_SPEC.json')['methods'][method]

    def test_global_minimum_nontransitive_tie(self):
        scores = [[.1 + 1.5e-12] * 2, [.1 + .9e-12] * 2, [.1] * 2]
        self.assertEqual(common.select(scores)[0], 1)
        self.assertEqual(common.select([[.1] * 2, [.1 - .5e-12] * 2])[0], 0)
        self.assertEqual(common.select([[None, .0], [.3, .3]])[0], 1)
        self.assertTrue(common.select([[None, .0], [None, None]])[2])

    def test_ssa_complete_555_and_resume_without_new_launch(self):
        backend = FakeBackend(); ledger = controller.Ledger(self.root)
        result = controller.run_ssa(self.spec('SSA'), backend, ledger, 'binding', make_seal=False)
        self.assertEqual(len(backend.calls), 555)
        self.assertEqual(len(result['conditions']), 15)
        self.assertFalse((self.root / 'GLOBAL_PREDICTIONS_SEALED.json').exists())
        ledger.readonly = True
        self.assertEqual(controller.run_ssa(self.spec('SSA'), backend, ledger, 'binding', make_seal=False), result)
        self.assertEqual(len(backend.calls), 555)

    def test_terminal_pruning_only_and_fallback_final(self):
        spec = dict(self.spec('SSA')); spec['conditions'] = spec['conditions'][:1]
        backend = FakeBackend(lambda c, i, f: (False, None))
        controller.run_ssa(spec, backend, controller.Ledger(self.root), 'binding')
        self.assertEqual(len(backend.calls), 19)
        self.assertTrue(backend.calls[-1].endswith('_c0_final'))
        skips = [p for p in (self.root / 'fits').glob('*.json') if common.read(p).get('attempted') is False]
        self.assertEqual(len(skips), 18)

    def test_bad_ber_is_not_pruned(self):
        spec = dict(self.spec('SSA')); spec['conditions'] = spec['conditions'][:1]
        backend = FakeBackend(lambda c, i, f: (True, 1.0 if i else .0))
        controller.run_ssa(spec, backend, controller.Ledger(self.root), 'binding')
        self.assertEqual(len(backend.calls), 37)

    def test_unresolved_failure_and_stop_leave_no_seal(self):
        class Broken:
            def fit(self, *args): raise ValueError('preworker infrastructure failure')
        with self.assertRaisesRegex(ValueError, 'infrastructure'):
            controller.run_ssa(self.spec('SSA'), Broken(), controller.Ledger(self.root), 'binding')
        self.assertFalse((self.root / 'GLOBAL_PREDICTIONS_SEALED.json').exists())
        (self.root / 'STOP').touch()
        b = FakeBackend()
        with self.assertRaisesRegex(ValueError, 'STOP'):
            controller.run_ssa(self.spec('SSA'), b, controller.Ledger(self.root), 'binding',
                               lambda: common.check_stop(self.root))
        self.assertEqual(b.calls, [])

    def old(self):
        result = {}
        for c in self.spec('PCBG')['conditions']:
            for i in range(3):
                for f in (0, 1):
                    cid = common.case_id(c, i, f)
                    if cid != 'BBC_s24_Q400_c0_f0':
                        result[cid] = {'case_id': cid, 'eligible': i == 0,
                                       'validation_BER': .4 if i == 0 else None,
                                       'original_status': 'FINITE_WRONG_K' if i else 'ELIGIBLE'}
        return result

    def test_pcbg_23_reused_5_fits_order_and_wrongk(self):
        b = FakeBackend(lambda c, i, f: (True, .4))
        result = controller.run_pcbg(self.spec('PCBG'), b, self.old(), controller.Ledger(self.root), 'binding')
        self.assertEqual(b.calls, ['BBC_s24_Q400_c0_f0', 'BBC_s24_Q400_c0_final',
                                  'BBC_s25_Q400_c0_final', 'CiteSeer_s23_Q400_c0_final',
                                  'CiteSeer_s25_Q400_c0_final'])
        self.assertEqual(len(result['conditions']), 4)
        s = common.read(self.root / 'selections/BBC_s24_Q400.json')
        self.assertEqual(s['mean_query_BER'], [.4, None, None])

    def test_pcbg_selection_change_stops_scope_expansion(self):
        old = self.old()
        for key, value in old.items():
            if '_c1_' in key: value.update(eligible=True, validation_BER=.0)
        b = FakeBackend(lambda c, i, f: (True, .4))
        with self.assertRaisesRegex(ValueError, 'frozen five-fit'):
            controller.run_pcbg(self.spec('PCBG'), b, old, controller.Ledger(self.root), 'binding')
        self.assertEqual(b.calls, ['BBC_s24_Q400_c0_f0'])
        self.assertFalse((self.root / 'GLOBAL_PREDICTIONS_SEALED.json').exists())

    def test_changed_ledger_refused(self):
        p = self.root / 'decision.json'; common.immutable(p, {'x': 1})
        with self.assertRaisesRegex(ValueError, 'Immutable'):
            common.immutable(p, {'x': 2})
        with self.assertRaisesRegex(ValueError, 'escapes'):
            common.relative(self.root, '../elsewhere')

    def test_memory_wait_1800_stop_and_foreign_matlab(self):
        t = [0.]; ps = types.SimpleNamespace(process_iter=lambda fields: [],
            virtual_memory=lambda: types.SimpleNamespace(available=0), NoSuchProcess=RuntimeError, AccessDenied=RuntimeError)
        with self.assertRaisesRegex(ValueError, 'Memory wait cap'):
            common.before_dispatch(self.root, psutil_module=ps, clock=lambda: t[0],
                                   sleep=lambda seconds: t.__setitem__(0, t[0] + seconds))
        self.assertEqual(t[0], 1800)
        ps.process_iter = lambda fields: [types.SimpleNamespace(info={'name': 'MATLAB.exe', 'pid': 123})]
        with self.assertRaisesRegex(ValueError, 'Existing MATLAB'):
            common.before_dispatch(self.root, psutil_module=ps)
        ps.process_iter = lambda fields: []
        t[0] = 0
        def stopped(seconds): t[0] += seconds; (self.root / 'STOP').touch()
        with self.assertRaisesRegex(ValueError, 'STOP'):
            common.before_dispatch(self.root, psutil_module=ps, clock=lambda: t[0], sleep=stopped)
        self.assertEqual(t[0], 5)

    def test_exclusive_lock(self):
        with common.controller_lock(self.root):
            with self.assertRaises(OSError):
                with common.controller_lock(self.root): pass

    def test_untrusted_runtime_self_hash_is_not_trusted(self):
        receipt = common.read(common.HERE / 'SSA_RUNTIME_IDENTITY.json')
        receipt['runtime_files'] = dict(receipt['runtime_files'])
        receipt['runtime_files'][next(iter(receipt['runtime_files']))] = '0' * 64
        fake = types.SimpleNamespace(runtime_check=lambda root: receipt)
        with self.assertRaisesRegex(ValueError, 'reviewed content: runtime_files'):
            native.validate_ssa_runtime(fake, self.root)
        receipt = common.read(common.HERE / 'SSA_RUNTIME_IDENTITY.json'); receipt['own_preparers'] = {}
        with self.assertRaisesRegex(ValueError, 'own_preparers'):
            native.validate_ssa_runtime(types.SimpleNamespace(runtime_check=lambda root: receipt), self.root)

    def ssa_fixture(self, mode='eligible'):
        c = {'id': 'tiny_s23_Q4', 'n': 4, 'c': 2, 'eligible_folds': [{'validation': [0, 1]}] * 2}
        cfg = {'beta': .1, 'lam': .2, 'kap': .95}; cid = common.case_id(c, 0, 0)
        root = self.root; (root / 'SSA/inputs').mkdir(parents=True)
        source = root / 'SSA/inputs/carrier.mat'; source.write_bytes(b'fake numeric source')
        runtime = root / 'runtime'; runtime.mkdir(); (runtime / 'PORTABLE_PREPARATION.json').write_text('{}')
        carrier = {'input': 'inputs/carrier.mat', 'sha256': common.sha(source)}
        dest = root / 'SSA/runs' / cid; dest.mkdir(parents=True)
        io.savemat(dest / 'request.mat', {'source_input': str(source), 'source_sha256': carrier['sha256'],
                                        'config': cfg, 'case_id': cid})
        (dest / 'launch_ssa.m').write_text('synthetic fixture only')
        job = {'case_id': cid, 'config': cfg, 'source': str(source), 'source_sha256': carrier['sha256'],
               'runtime': str(runtime), 'runtime_manifest_sha256': common.sha(runtime / 'PORTABLE_PREPARATION.json'),
               'timeout_seconds': 600, 'matlab': 'fixture', 'request_sha256': common.sha(dest / 'request.mat'),
               'launch_sha256': common.sha(dest / 'launch_ssa.m')}
        common.immutable(dest / 'JOB.json', job)
        summary = {'case_id': cid, 'source_input_sha256': carrier['sha256'], 'n': 4, 'c': 2,
                   'native_options': cfg, 'benchmark_contract_met': mode == 'eligible', 'execution_returned': True,
                   'finite_partition_retained': True, 'status': mode}
        if mode not in ('preworker', 'timeout', 'partial'):
            common.immutable(dest / 'worker_summary.json', summary)
            v = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], float)
            pred = np.argmax(v, axis=1) + 1 if mode == 'eligible' else np.ones(4)
            io.savemat(dest / 'native_outputs.mat', {'pred': pred, 'V_star': v})
        if mode != 'partial':
            common.immutable(dest / 'EXECUTION.json', {'status': 'EXECUTED' if mode == 'eligible' else 'EXECUTION_FAILED',
                'input_and_runtime_hashes_unchanged': True, 'benchmark_contract_met': mode == 'eligible',
                'timed_out': mode == 'timeout', 'files': {p.name: common.sha(p) for p in dest.iterdir()
                                                        if p.name in ('worker_summary.json', 'native_outputs.mat')}})
        ctx = types.SimpleNamespace(work=root, ssa=None, spec={'methods': {'SSA': {'grid': [cfg],
                'policy': {'per_fit_whole_process_cap_seconds': 600}}}}, case=lambda *args: carrier,
                arrays={('SSA', c['id']): {'pairs': np.array([[0, 1], [0, 2]]), 'answers': np.array([1, -1])}})
        backend = native.NativeBackend.__new__(native.NativeBackend)
        backend.ctx = ctx; backend.runtime = runtime; backend.binding = job
        backend.matlab = 'fixture'; backend.readonly = True; backend.method = 'SSA'
        return backend, c, dest

    def test_ssa_resume_real_receipt_binding_and_ber(self):
        b, c, dest = self.ssa_fixture()
        with patch.object(native, 'validate_ssa_runtime', return_value={}):
            r = b.ssa_fit(c, 0, 0); self.assertTrue(r['eligible']); self.assertEqual(r['validation_BER'], 0.)
            (dest / 'launch_ssa.m').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'request/launch changed'): b.ssa_fit(c, 0, 0)

    def test_ssa_preworker_never_terminal(self):
        b, c, dest = self.ssa_fixture('preworker')
        with patch.object(native, 'validate_ssa_runtime', return_value={}):
            with self.assertRaisesRegex(ValueError, 'infrastructure failure unresolved'): b.ssa_fit(c, 0, 0)

    def test_ssa_timeout_terminal(self):
        b, c, dest = self.ssa_fixture('timeout')
        with patch.object(native, 'validate_ssa_runtime', return_value={}):
            r = b.ssa_fit(c, 0, 0)
        self.assertFalse(r['eligible']); self.assertTrue(r['timed_out'])

    def test_ssa_partial_never_automatically_retried(self):
        b, c, dest = self.ssa_fixture('partial')
        with patch.object(native, 'validate_ssa_runtime', return_value={}):
            with self.assertRaisesRegex(ValueError, 'Incomplete SSA attempt'): b.ssa_fit(c, 0, 0)

    def test_ssa_wrongk_retained_but_not_score(self):
        b, c, dest = self.ssa_fixture('wrongk')
        with patch.object(native, 'validate_ssa_runtime', return_value={}):
            r = b.ssa_fit(c, 0, 0)
        self.assertFalse(r['eligible']); self.assertEqual(r['occupied_clusters'], 1)
        self.assertIsNone(r['validation_BER'])

    def test_pcbg_exact_clock_patch_1800_and_fake_process(self):
        source = RELEASE / 'baselines/portable/pcbg/pcbg_portable.py'
        target = self.root / 'pcbg_portable.py'; target.write_bytes(extended_launcher_bytes(source.read_bytes()))
        self.assertEqual(common.sha(target), 'a8f9e47c7d67de5a5b48d42ce2f5488b8e168302c7e18ef960c1c0de504d4d70')
        tool = common.module(target, 'fake_pcbg_clock')
        tool.author_audit = lambda root: {'author_commit': 'fixture'}
        tool.sources = lambda: {'fixture': 'fixed'}
        inp = self.root / 'input.mat'; inp.write_bytes(b'fixture')
        common.immutable(self.root / 'PLAN.json', {'source_hashes': tool.sources()})
        t = [0.]; killed = []
        class Process:
            def __init__(self, *a, **k):
                d = self.root if hasattr(self, 'root') else None
                (out / 'runs/case/worker_started.json').write_text('{}')
            def poll(self): return None
            def children(self, recursive): return []
            def terminate(self): killed.append(True)
            def kill(self): killed.append(True)
        out = self.root
        tool.psutil = types.SimpleNamespace(Popen=Process, NoSuchProcess=RuntimeError,
            virtual_memory=lambda: types.SimpleNamespace(available=4 * 2**30), wait_procs=lambda ps, timeout: (ps, []))
        tool.time = types.SimpleNamespace(monotonic=lambda: t[0], sleep=lambda s: t.__setitem__(0, t[0] + s))
        case = {'id': 'case', 'input': 'input.mat', 'sha256': common.sha(inp)}
        r = tool.execute_case(out, {'source_hashes': tool.sources()}, case, self.root, self.root / 'fake.exe')
        self.assertEqual(r['status'], 'PORTABLE_CLOCK_CAP')
        self.assertGreater(t[0], 1800); self.assertLess(t[0], 1801)
        self.assertEqual(killed, [True]); self.assertFalse(r['eligible'])

    def test_public_ssa_whole_process_600_with_fake_launcher(self):
        tool = common.module(RELEASE / 'baselines/portable/ssa/ssa_portable.py', 'fake_ssa_clock')
        out = self.root / 'run'; out.mkdir(); runtime = self.root / 'runtime'; runtime.mkdir()
        common.immutable(runtime / 'PORTABLE_PREPARATION.json', {'release_root': str(RELEASE)})
        source = self.root / 'source.mat'; source.write_bytes(b'fixture')
        for name in ('request.mat', 'launch_ssa.m'): (out / name).write_bytes(b'fixture')
        exe = self.root / 'fake.exe'; exe.write_bytes(b'MZ synthetic fixture, never launched')
        common.immutable(out / 'JOB.json', {'runtime': str(runtime), 'runtime_manifest_sha256': common.sha(runtime / 'PORTABLE_PREPARATION.json'),
            'source': str(source), 'source_sha256': common.sha(source), 'request_sha256': common.sha(out / 'request.mat'),
            'launch_sha256': common.sha(out / 'launch_ssa.m'), 'matlab': str(exe), 'timeout_seconds': 600})
        tool.runtime_check = lambda path: {'release_root': str(RELEASE)}
        waits = []; cleanup = []
        class Process:
            pid = 123
            def wait(self, timeout):
                waits.append(timeout)
                if len(waits) == 1: raise subprocess.TimeoutExpired('fake', timeout)
                return 1
        class Owner:
            pid = 123
            def __init__(self, pid): self.pid = pid
            def create_time(self): return 10.
            def children(self, recursive): return []
            def terminate(self): cleanup.append(self.pid)
        tool.subprocess = types.SimpleNamespace(Popen=lambda *args, **kw: Process(), STDOUT=-2,
            TimeoutExpired=subprocess.TimeoutExpired)
        fake_psutil = types.SimpleNamespace(Process=Owner, NoSuchProcess=RuntimeError, AccessDenied=PermissionError)
        with patch.dict(sys.modules, psutil=fake_psutil):
            result = tool.execute(types.SimpleNamespace(job_dir=str(out)))
        self.assertEqual(waits, [600, 10]); self.assertEqual(cleanup, [123])
        self.assertTrue(result['timed_out']); self.assertFalse(result['benchmark_contract_met'])

    def test_pcbg_readonly_never_calls_launcher(self):
        c = {'id': 'tiny', 'n': 4, 'c': 2}; cid = 'tiny_c0_final'; root = self.root / 'PCBG'
        dest = root / 'runs' / cid; dest.mkdir(parents=True)
        inp = root / 'input.mat'; inp.write_bytes(b'fixture'); common.immutable(root / 'PLAN.json', {'x': 1})
        case = {'id': cid, 'input': 'input.mat', 'sha256': common.sha(inp)}
        receipt = {'case_id': cid, 'plan_sha256': common.sha(root / 'PLAN.json'), 'author_commit': 'fixture',
                   'eligible': False, 'finite_partition_retained': False, 'artifacts': {},
                   'status': 'PORTABLE_CLOCK_CAP', 'worker_status': None}
        common.immutable(dest / 'receipt.json', receipt)
        def forbidden(*args): raise AssertionError('readonly invoked launcher')
        ctx = types.SimpleNamespace(work=self.root, case=lambda *args: case,
                native=types.SimpleNamespace(load_plan=lambda out: {'x': 1}, author_audit=lambda root: {},
                                             execute_case=forbidden))
        backend = native.NativeBackend.__new__(native.NativeBackend)
        backend.ctx = ctx; backend.readonly = True; backend.binding = {'author_commit': 'fixture'}
        backend.author = self.root; backend.matlab = 'fixture'
        self.assertFalse(backend.pcbg_fit(c, 0, 'final')['eligible'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--release-root', required=True)
    args, rest = parser.parse_known_args(); RELEASE = Path(args.release_root).resolve()
    unittest.main(argv=[__file__] + rest, verbosity=2)
