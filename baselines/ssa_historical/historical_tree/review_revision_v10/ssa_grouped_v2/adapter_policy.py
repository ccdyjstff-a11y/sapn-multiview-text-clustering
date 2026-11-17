"""Pure scheduling policy. No model, process, data or filesystem imports."""
import math

RESOURCE_CAPS = {'RESOURCE_START_WAIT_CAP', 'RESOURCE_SUSPEND_WAIT_CAP'}
LOCAL_CAPS = RESOURCE_CAPS | {'WORKER_ACTIVE_TIME_CAP'}
RETURNED_STATUSES = {'RETURNED_FINITE', 'RETURNED_FINITE_CONTRACT_FAILED', 'RETURNED_NONFINITE_OR_INVALID'}


class ReceiptPolicy:
    def __init__(self):
        self.consecutive_resource_caps = 0
        self.last_resource_case = None

    def assess(self, receipt, eligible, validation_ber=None):
        """Called once per actual launcher dispatch, never for skipped slots."""
        status = receipt.get('status')
        previous = self.last_resource_case
        if status in RESOURCE_CAPS:
            self.consecutive_resource_caps += 1
            self.last_resource_case = receipt['case_id']
        else:
            self.consecutive_resource_caps = 0
            self.last_resource_case = None
        cleanup = receipt.get('cleanup', {})
        clean = cleanup.get('complete') is True
        lifecycle = receipt.get('attempt_lifecycle_ended') is True
        started = receipt.get('matlab_started') is True
        if not started:
            lifecycle = lifecycle and receipt.get('process_creation_attempted') is False and cleanup.get('no_process_created') is True
        else:
            lifecycle = lifecycle and receipt.get('process_creation_attempted') is True
        decision = dict(original_receipt_status=status, launcher_dispatched=True,
            matlab_started=started, terminal_attempt_lifecycle=bool(lifecycle and clean),
            worker_returned=receipt.get('worker', receipt.get('unqualified_worker_summary', {})).get('execution_returned') is True,
            consecutive_resource_caps=self.consecutive_resource_caps,
            previous_resource_cap_case=previous, disposition='GLOBAL_STOP', reason='UNVERIFIABLE_RECEIPT')
        if receipt.get('global_deadline_triggered') or status == 'GLOBAL_EXECUTION_WALL_CAP':
            decision['reason'] = 'GLOBAL_EXECUTION_WALL_CAP'
        elif receipt.get('interrupted'):
            decision['reason'] = 'INTERRUPTED'
        elif not clean:
            decision['reason'] = 'OWNED_CLEANUP_INCOMPLETE'
        elif not lifecycle:
            decision['reason'] = 'ATTEMPT_LIFECYCLE_UNVERIFIED'
        elif status in RESOURCE_CAPS and self.consecutive_resource_caps >= 2:
            decision['reason'] = 'RESOURCE_PERSISTENCE_STOP'
        elif status in LOCAL_CAPS:
            consistent = ((status=='RESOURCE_START_WAIT_CAP' and not started) or
                (status in {'WORKER_ACTIVE_TIME_CAP','RESOURCE_SUSPEND_WAIT_CAP'} and started
                    and receipt.get('clock',{}).get('status')==status))
            if consistent:
                decision.update(disposition='LOCAL_INELIGIBLE', reason=status)
            else:
                decision['reason']='CAP_LIFECYCLE_INCONSISTENT'
        elif status == 'COMPLETE_ZERO_EXIT_VALID_MARKERS' and receipt.get('exit_code') == 0:
            worker = receipt.get('worker', {})
            if not started or not worker.get('execution_returned') or worker.get('status') not in RETURNED_STATUSES:
                decision['reason'] = 'UNVERIFIABLE_WORKER_RETURN'
            elif eligible:
                if receipt.get('is_cv') and (type(validation_ber) not in (float, int) or not math.isfinite(validation_ber)):
                    decision['reason'] = 'INVALID_VALIDATION_BER'
                else:
                    decision.update(disposition='ELIGIBLE', reason='QUALIFIED_RETURN')
            else:
                decision.update(disposition='LOCAL_INELIGIBLE', reason='VERIFIED_MODEL_INELIGIBLE')
        else:
            decision['reason'] = status or 'UNKNOWN_LAUNCHER_STATE'
        return decision


def memory_gate(n, available_bytes, monotonic, sleep, global_deadline, sample=lambda event: None):
    """Injected-only gate; 60s continuous eligibility, 1800s total bounded wait."""
    threshold = (3 if n < 2000 else 4.5) * 2**30
    began, stable_since, count = monotonic(), None, 0
    last = began
    while True:
        now = monotonic()
        if not math.isfinite(now) or now < last:
            raise ValueError('Invalid monotonic clock')
        last = now
        elapsed = now - began
        if now >= global_deadline:
            status = 'GLOBAL_EXECUTION_WALL_CAP'
            break
        if elapsed >= 1800:
            status = 'GROUP_MEMORY_WAIT_CAP'
            break
        free = available_bytes()
        if type(free) not in (int, float) or not math.isfinite(free) or free < 0:
            raise ValueError('Invalid available-memory observation')
        count += 1
        if free >= threshold:
            if stable_since is None:
                stable_since = now
        else:
            stable_since = None
        stable = 0.0 if stable_since is None else now - stable_since
        sample(dict(elapsed_seconds=elapsed, available_bytes=free, continuous_qualified_seconds=stable))
        if stable >= 60:
            status = 'READY'
            break
        sleep(min(.5, 1800 - elapsed, global_deadline - now))
    return dict(status=status, n=n, threshold_bytes=int(threshold), required_stable_seconds=60,
        total_wait_cap_seconds=1800, elapsed_seconds=monotonic()-began, observations=count,
        native_fit_started=False, launcher_dispatched=False)
