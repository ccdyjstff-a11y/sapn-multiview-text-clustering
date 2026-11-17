"""Deterministic simulations only: never invokes a real process or MATLAB."""
import unittest
from unittest.mock import patch
from phase_clock import PhaseClock
from owned_processes import act_owned, identity_state
import run_clock_qualification as launcher

def started(run_id='sim'):
    return dict(schema=1, run_id=run_id, phase='started', sequence=1)

def finished(wall=1, outcome='returned', run_id='sim'):
    return dict(schema=1, run_id=run_id, phase='finished', sequence=2,
                worker_call_wall_seconds=wall, outcome=outcome)

class FakeProcess:
    def __init__(self, pid=1, created=10, denied=False):
        self.pid, self.created, self.denied = pid, created, denied
        self.live, self.actions = True, []
    def is_running(self): return self.live
    def create_time(self):
        if self.denied: raise PermissionError('simulated denied identity')
        return self.created
    def kill(self): self.actions.append('kill'); self.live = False
    def suspend(self): self.actions.append('suspend')
    def resume(self): self.actions.append('resume')

class ClockSimulations(unittest.TestCase):
    def test_function_pass_does_not_upgrade_resource_abort_environment(self):
        receipt = dict(status='RESOURCE_RUNTIME_ABORT_BELOW_2_GIB', exit_code=15,
                       manifest_sha256='manifest', runtime_environment={}, dataset_loaded=False, model_invoked=False)
        probe = dict(status='PASS', dataset_loaded=False, model_invoked=False)
        result = launcher.assess_environment(receipt, probe, 'probehash', 'manifest', {})
        self.assertFalse(result['passed'])
        self.assertEqual(len(result['reasons']), 3)

    def test_long_cold_start_does_not_spend_worker_budget(self):
        c = PhaseClock('sim', 60)
        for tick in range(1, 400): c.observe(tick / 2)
        c.observe(200, started())
        c.observe(201, started(), finished(1))
        result = c.observe(202, started(), finished(1), True, 0)
        self.assertEqual(result, 'COMPLETE_ZERO_EXIT_VALID_MARKERS')
        self.assertAlmostEqual(c.active['worker'], 1.5)
        self.assertEqual(c.startup_wall_upper, 200)

    def test_worker_cap_remains_sixty(self):
        c = PhaseClock('sim', 60)
        c.observe(.5)
        c.observe(1, started())
        self.assertEqual(c.observe(60.5, started()), 'WORKER_ACTIVE_TIME_CAP')

    def test_real_worker_cap_remains_six_hundred(self):
        c = PhaseClock('sim', 600)
        c.observe(.5); c.observe(1, started())
        self.assertIsNone(c.observe(100, started()))
        self.assertEqual(c.observe(600.5, started()), 'WORKER_ACTIVE_TIME_CAP')

    def test_memory_suspend_excludes_only_verified_paused_time(self):
        c = PhaseClock('sim', 60)
        c.observe(.5); c.observe(1, started()); c.observe(10, started())
        c.set_paused(10, True)
        c.observe(200, started())
        self.assertAlmostEqual(c.active['worker'], 9.5)
        c.set_paused(200, False)
        c.observe(210, started(), finished(209))
        self.assertEqual(c.observe(211, started(), finished(209), True, 0), 'COMPLETE_ZERO_EXIT_VALID_MARKERS')
        self.assertAlmostEqual(c.active['worker'], 19.5)
        self.assertAlmostEqual(c.wall['worker'], 209.5)

    def test_suspend_action_time_is_charged_before_pause(self):
        c = PhaseClock('sim', 60)
        c.observe(.5); c.observe(1, started()); c.observe(5, started())
        c.observe(7, started())  # Two seconds spent freezing owned descendants.
        c.set_paused(7, True); c.observe(100, started())
        self.assertAlmostEqual(c.active['worker'], 6.5)

    def test_continuous_memory_wait_is_bounded(self):
        c = PhaseClock('sim', 60)
        c.observe(.5); c.observe(1, started()); c.set_paused(1, True)
        self.assertEqual(c.observe(1801, started()), 'RESOURCE_SUSPEND_WAIT_CAP')

    def test_startup_deadline_is_conservative_at_marker_boundary(self):
        c = PhaseClock('sim', 60)
        c.observe(299.9)
        self.assertEqual(c.observe(300.2, started()), 'STARTUP_WALL_TIME_CAP')

    def test_missing_startup_marker_hits_startup_cap(self):
        c = PhaseClock('sim', 60)
        self.assertEqual(c.observe(300), 'STARTUP_WALL_TIME_CAP')

    def test_missing_marker_after_zero_exit_is_not_complete(self):
        c = PhaseClock('sim', 60)
        self.assertEqual(c.observe(1, exited=True, exit_code=0), 'MISSING_PHASE_MARKER_ON_EXIT')

    def test_out_of_order_marker_is_invalid(self):
        c = PhaseClock('sim', 60)
        self.assertEqual(c.observe(1, finished=finished()), 'INVALID_PHASE_MARKER')

    def test_invalid_nonce_schema_and_sequence_are_rejected(self):
        for bad in [started('other'), dict(started(), schema=True), dict(started(), sequence=2)]:
            with self.subTest(bad=bad):
                self.assertEqual(PhaseClock('sim', 60).observe(1, bad), 'INVALID_PHASE_MARKER')

    def test_bad_json_marker_and_disappearance_are_rejected(self):
        self.assertEqual(PhaseClock('sim', 60).observe(1, marker_error='JSON decode failure'), 'INVALID_PHASE_MARKER')
        c = PhaseClock('sim', 60); c.observe(1, started())
        self.assertEqual(c.observe(2), 'INVALID_PHASE_MARKER')

    def test_impossible_reported_worker_wall_is_rejected(self):
        c = PhaseClock('sim', 60)
        self.assertEqual(c.observe(1, started(), finished(800), True, 0), 'INVALID_PHASE_MARKER')

    def test_nonzero_exit_never_completes(self):
        c = PhaseClock('sim', 60)
        self.assertEqual(c.observe(1, started(), finished(.5), True, 15), 'PROCESS_NONZERO_EXIT')

    def test_throw_even_with_zero_exit_never_completes(self):
        c = PhaseClock('sim', 60)
        self.assertEqual(c.observe(1, started(), finished(.5, 'threw'), True, 0), 'WORKER_EXCEPTION')

    def test_shutdown_cap_is_independent_of_worker(self):
        c = PhaseClock('sim', 60)
        c.observe(.5); c.observe(1, started()); c.observe(2, started(), finished(1))
        self.assertEqual(c.observe(61, started(), finished(1)), 'SHUTDOWN_WALL_TIME_CAP')

    def test_exit_rechecks_markers_after_fast_completion_race(self):
        c = PhaseClock('sim', 60)
        proc = type('Exited', (), {'poll': lambda self: 0})()
        with patch.object(launcher, 'read_markers', side_effect=[(started(), None, None),
                (started(), finished(.5), None)]), patch.object(launcher.time, 'monotonic', return_value=1):
            _, _, result = launcher.observe_process(proc, None, c, 0)
        self.assertEqual(result, 'COMPLETE_ZERO_EXIT_VALID_MARKERS')

    def test_pid_reuse_is_not_killed_and_access_denied_is_not_clean(self):
        reused = FakeProcess(created=11)
        result = act_owned({(1, 10): reused}, 'kill')
        self.assertEqual(reused.actions, [])
        self.assertEqual(result[0]['identity_state'], 'reused')
        denied = FakeProcess(denied=True)
        self.assertEqual(identity_state(denied, 10)[0], 'unverified')
        self.assertEqual(act_owned({(1, 10): denied}, 'kill')[0]['result'], 'error')

    def test_cleanup_checks_descendants_even_after_root_exit(self):
        root, child = FakeProcess(1), FakeProcess(2)
        root.live = False
        entries = {(1, 10): root, (2, 10): child}
        with patch.object(launcher, 'remember', return_value=[]), patch.object(launcher.psutil, 'wait_procs'):
            result = launcher.cleanup(root, entries)
        self.assertTrue(result['complete'])
        self.assertEqual(child.actions, ['kill'])

    def test_cleanup_unverifiable_identity_is_failure(self):
        denied = FakeProcess(denied=True)
        with patch.object(launcher, 'remember', return_value=[]), patch.object(launcher.psutil, 'wait_procs'):
            result = launcher.cleanup(denied, {(1, 10): denied})
        self.assertFalse(result['complete'])
        self.assertEqual(result['unresolved_original_processes'][0]['state'], 'unverified')

    def test_cleanup_reap_exception_is_recorded_not_rethrown(self):
        process = FakeProcess()
        with patch.object(launcher, 'remember', return_value=[]), patch.object(launcher.psutil, 'wait_procs',
                side_effect=PermissionError('simulated reap denial')):
            result = launcher.cleanup(process, {(1, 10): process})
        self.assertFalse(result['complete'])
        self.assertIn('bounded reap', result['discovery_errors'][0])

    def test_suspend_freezes_root_before_new_descendants_and_stabilizes(self):
        root, child, late_child = FakeProcess(1), FakeProcess(2), FakeProcess(3)
        entries = {(1, 10): root}
        visits = []
        def discover(_root, owned):
            self.assertIn('suspend', root.actions)
            visits.append(1)
            owned[(2, 10)] = child
            if len(visits) >= 2: owned[(3, 10)] = late_child
            return []
        with patch.object(launcher, 'remember', side_effect=discover):
            frozen, _ = launcher.freeze_owned_tree(root, entries)
        self.assertEqual(len(frozen), 3)
        self.assertEqual(root.actions, ['suspend'])
        self.assertEqual(child.actions, ['suspend'])
        self.assertEqual(late_child.actions, ['suspend'])

if __name__ == '__main__':
    unittest.main(verbosity=2)
