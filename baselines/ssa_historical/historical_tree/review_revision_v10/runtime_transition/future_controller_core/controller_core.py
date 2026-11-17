"""Data-independent future scheduling logic; no process, disk or model I/O."""
from copy import deepcopy
from dataclasses import asdict, dataclass
import math

CANDIDATES = 18
FOLDS = 2
TIE_TOLERANCE = 1e-12
HALTING_STATUSES = {"GLOBAL_STOP", "CLEANUP_FAILED", "STARTUP_FAILED", "INCOMPLETE"}
RESOLVED_CV = {"ELIGIBLE", "LOCAL_INELIGIBLE", "SKIPPED_AFTER_LOCAL_INELIGIBLE"}


@dataclass(frozen=True)
class Group:
    corpus: str
    Q: int
    seeds: tuple


@dataclass(frozen=True)
class FitRequest:
    condition_id: str
    case_id: str
    corpus: str
    Q: int
    seed: int
    candidate: int
    fold: object  # 0/1 for CV, "final" for the selected final fit


@dataclass(frozen=True)
class FitResult:
    case_id: str
    status: str
    attempted: bool
    execution_complete: bool
    cleanup_complete: bool
    eligible: bool
    validation_BER: object = None
    irreversible_local_failure: bool = False
    reason: str = ""
    evidence_ref: object = None  # Opaque future runner reference, never opened here.


class ControllerFailure(RuntimeError):
    """Visible failure carrying calculated, uncommitted results in .result."""
    def __init__(self, message, result):
        super().__init__(message)
        self.result = deepcopy(result)


class CommitFailure(ControllerFailure):
    pass


def _case(condition, candidate, fold):
    suffix = "final" if fold == "final" else "f" + str(fold)
    return condition["condition_id"] + "_c" + str(candidate) + "_" + suffix


def _pending(case_id, candidate, fold):
    return dict(case_id=case_id, candidate=candidate, fold=fold, attempted=False,
        status="NOT_ATTEMPTED_PENDING", trigger_case=None, result=None)


def _plan(groups):
    conditions, seen = [], set()
    for group_index, group in enumerate(groups):
        if not isinstance(group, Group) or not isinstance(group.corpus, str) or not group.corpus:
            raise ValueError("Each group needs an explicit nonempty corpus")
        if type(group.Q) is not int or group.Q <= 0 or (group.corpus, group.Q) in seen:
            raise ValueError("Groups must have distinct (corpus, Q), with positive integer Q")
        seen.add((group.corpus, group.Q))
        if len(group.seeds) != 3 or any(type(seed) is not int for seed in group.seeds) or len(set(group.seeds)) != 3:
            raise ValueError("Exactly three distinct integer seeds must be explicit per group")
        for seed_index, seed in enumerate(group.seeds):
            condition = dict(condition_id=f"{group.corpus}_s{seed}_Q{group.Q}",
                group_index=group_index, seed_index=seed_index, corpus=group.corpus,
                Q=group.Q, seed=seed, status="NOT_ATTEMPTED_PENDING", committed=False,
                selection=dict(status="NOT_SELECTED_PENDING", index=None, rule=None))
            condition["cv"] = [_pending(_case(condition, candidate, fold), candidate, fold)
                for fold in range(FOLDS) for candidate in range(CANDIDATES)]
            condition["final"] = _pending(None, None, "final")
            conditions.append(condition)
    if not conditions:
        raise ValueError("At least one explicit group is required")
    if len({condition["condition_id"] for condition in conditions}) != len(conditions):
        raise ValueError("Condition identifiers collide")
    return dict(status="RUNNING", safe_stop_confirmed=False, plan_complete=False,
        planned_conditions=len(conditions), committed_conditions=0,
        conditions=conditions, stop_reason=None, trigger_case=None)


def _finish_pending(state, reason, trigger_case):
    """Keep every unrun logical slot, distinct from candidate-local skipping."""
    for condition in state["conditions"]:
        if condition["status"] == "NOT_ATTEMPTED_PENDING":
            condition["status"] = "NOT_ATTEMPTED_AFTER_" + reason
        elif condition["status"] == "RUNNING":
            condition["status"] = "INCOMPLETE_AFTER_" + reason
        if condition["selection"]["status"] == "NOT_SELECTED_PENDING":
            condition["selection"].update(status="NOT_SELECTED_AFTER_" + reason)
        for record in condition["cv"] + [condition["final"]]:
            if record["status"] == "NOT_ATTEMPTED_PENDING":
                record.update(status="NOT_ATTEMPTED_AFTER_" + reason, trigger_case=trigger_case)
    state.update(stop_reason=reason, trigger_case=trigger_case,
        plan_complete=all(condition["committed"] for condition in state["conditions"]))


def _request(condition, record):
    return FitRequest(condition["condition_id"], record["case_id"], condition["corpus"],
        condition["Q"], condition["seed"], record["candidate"], record["fold"])


def _callback_failure(state, callback, exc):
    state["failure"] = dict(callback=callback, exception_type=type(exc).__qualname__,
        exception_module=type(exc).__module__, message=str(exc),
        process_cleanup_performed=False)


def _interpret(request, result):
    if not isinstance(result, FitResult) or result.case_id != request.case_id:
        raise ValueError("Runner must return a FitResult for the requested case")
    if any(type(getattr(result, key)) is not bool for key in (
            "attempted", "execution_complete", "cleanup_complete", "eligible", "irreversible_local_failure")):
        raise ValueError("Runner lifecycle and eligibility flags must be bool")
    if result.status in HALTING_STATUSES:
        return result.status
    if result.status not in {"ELIGIBLE", "LOCAL_INELIGIBLE"}:
        raise ValueError("Unknown runner disposition")
    # Lifecycle failure always wins over a purported local disqualification.
    if not result.cleanup_complete:
        return "CLEANUP_FAILED"
    if not result.execution_complete:
        return "INCOMPLETE"
    if not result.attempted:
        raise ValueError("A completed result must have been attempted")
    if result.status == "LOCAL_INELIGIBLE":
        if result.eligible or not result.irreversible_local_failure:
            raise ValueError("Local skipping requires explicit irreversible ineligibility")
        return "LOCAL_INELIGIBLE"
    if not result.eligible or result.irreversible_local_failure:
        raise ValueError("Eligible result has contradictory flags")
    if request.fold != "final":
        value = result.validation_BER
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Eligible CV needs a finite BER in [0, 1]; it cannot be silently pruned")
    return "ELIGIBLE"


def _run_one(state, condition, record, runner):
    request = _request(condition, record)
    try:
        result = runner(request)
    except BaseException as exc:
        # Include KeyboardInterrupt/SystemExit at this explicit callback boundary;
        # re-raise visibly below with the original object as __cause__.
        # A raised callback provides no reliable fit-attempt fact.
        record.update(status="RUNNER_EXCEPTION", attempted=None)
        state.update(status="RUNNER_EXCEPTION", safe_stop_confirmed=False)
        _callback_failure(state, "runner", exc)
        _finish_pending(state, "RUNNER_EXCEPTION", request.case_id)
        raise ControllerFailure("Runner raised; attempt state is unknown", state) from exc
    try:
        disposition = _interpret(request, result)
    except Exception as exc:
        if isinstance(result, FitResult):
            record.update(result=asdict(result), attempted=result.attempted)
        record["status"] = "RUNNER_CONTRACT_ERROR"
        state.update(status="RUNNER_CONTRACT_ERROR", safe_stop_confirmed=False)
        _finish_pending(state, "RUNNER_CONTRACT_ERROR", request.case_id)
        raise ControllerFailure("Runner result violates the scheduler contract", state) from exc
    record.update(status=disposition, attempted=result.attempted, result=asdict(result))
    if disposition in HALTING_STATUSES:
        state.update(status="EXECUTION_HALTED", safe_stop_confirmed=False)
        _finish_pending(state, disposition, request.case_id)
        return False
    return True


def _select(records):
    if len(records) != CANDIDATES * FOLDS or any(record["status"] not in RESOLVED_CV for record in records):
        raise ValueError("All 18 candidates must be decided before selection")
    candidates, valid = [], []
    by_case = {record["case_id"]: record for record in records}
    for candidate in range(CANDIDATES):
        rows = [record for record in records if record["candidate"] == candidate]
        if len(rows) != FOLDS or [row["fold"] for row in rows] != list(range(FOLDS)):
            raise ValueError("Candidate fold identities differ")
        for row in rows:
            if row["status"] == "SKIPPED_AFTER_LOCAL_INELIGIBLE":
                trigger = by_case.get(row["trigger_case"])
                if not (trigger and trigger["candidate"] == candidate and trigger["fold"] < row["fold"]
                        and trigger["status"] == "LOCAL_INELIGIBLE" and row["attempted"] is False):
                    raise ValueError("A skipped fold lacks a valid earlier local trigger")
        if all(row["status"] == "ELIGIBLE" for row in rows):
            scores = [row["result"]["validation_BER"] for row in rows]
            if not all(type(value) in (int, float) and math.isfinite(value) for value in scores):
                raise ValueError("Eligible candidate needs finite validation scores")
            mean = sum(scores) / FOLDS
            valid.append((candidate, mean))
            candidates.append(dict(index=candidate, status="VALID", mean_BER=mean, scores=scores))
        else:
            if not any(row["status"] == "LOCAL_INELIGIBLE" for row in rows):
                raise ValueError("Invalid candidate was never locally decided")
            candidates.append(dict(index=candidate, status="INVALID_LOCAL", mean_BER=None, scores=None))
    if not valid:
        index, rule = 0, "all_validation_fits_failed_default_with_failure_flag"
    else:
        best = min(mean for _, mean in valid)
        tied = [index for index, mean in valid if abs(mean - best) <= TIE_TOLERANCE]
        index = 0 if 0 in tied else tied[0]
        rule = "minimum_balanced_pair_error"
    return dict(status="SELECTED", index=index, rule=rule, valid_candidates=len(valid),
        decided_candidates=CANDIDATES, candidates=candidates)


def run(groups, runner, commit, should_stop_after_condition=lambda condition: False):
    """Run an explicit plan through injected callbacks only.

    runner(FitRequest) -> FitResult.
    commit(condition_copy) -> True acknowledges successful persistence.
    should_stop_after_condition(committed_condition_copy) -> bool is called ONLY
    after commit returns True. No callback starts a new fit internally.
    On callback BaseException (including KeyboardInterrupt/SystemExit),
    ControllerFailure.result retains the calculated ledger, with
    safe_stop_confirmed=False, and __cause__ retains the original exception.
    No next fit is dispatched. This requires Python to reach the handler;
    it cannot survive a forced kill or crash and does not clean up processes.
    """
    state = _plan(tuple(groups))
    for condition in state["conditions"]:
        condition["status"] = "RUNNING"
        local_failures = {}
        for record in condition["cv"]:  # The prebuilt order is strictly fold-major.
            candidate = record["candidate"]
            if candidate in local_failures:
                record.update(status="SKIPPED_AFTER_LOCAL_INELIGIBLE", attempted=False,
                    trigger_case=local_failures[candidate])
                continue
            if not _run_one(state, condition, record, runner):
                return state
            if record["status"] == "LOCAL_INELIGIBLE":
                local_failures[candidate] = record["case_id"]
        condition["selection"] = _select(condition["cv"])
        selected = condition["selection"]["index"]
        condition["final"].update(candidate=selected, case_id=_case(condition, selected, "final"))
        if not _run_one(state, condition, condition["final"], runner):
            return state
        condition["status"] = "COMPUTED_COMPLETE_UNCOMMITTED"
        try:
            acknowledged = commit(deepcopy(condition))
            if acknowledged is not True:
                raise ValueError("commit must return True after successful persistence")
        except BaseException as exc:
            state.update(status="COMMIT_FAILED", safe_stop_confirmed=False)
            condition.update(commit_acknowledgement="NOT_RECEIVED_PERSISTENCE_UNKNOWN",
                persistence_may_be_partial=True)
            _callback_failure(state, "commit", exc)
            _finish_pending(state, "COMMIT_FAILED", condition["final"]["case_id"])
            raise CommitFailure("Commit failed; calculated condition remains uncommitted", state) from exc
        condition.update(status="COMMITTED", committed=True)
        state["committed_conditions"] += 1
        try:
            stop = should_stop_after_condition(deepcopy(condition))
            if type(stop) is not bool:
                raise ValueError("Boundary callback must return bool")
        except BaseException as exc:
            state.update(status="STOP_CALLBACK_FAILED", safe_stop_confirmed=False)
            _callback_failure(state, "should_stop_after_condition", exc)
            _finish_pending(state, "STOP_CALLBACK_FAILED", condition["final"]["case_id"])
            raise ControllerFailure("Boundary callback failed after commit", state) from exc
        if stop:
            state.update(status="SAFE_STOP_AT_CONDITION_BOUNDARY", safe_stop_confirmed=True)
            _finish_pending(state, "CONDITION_BOUNDARY_STOP", condition["final"]["case_id"])
            return state
    state.update(status="COMPLETED", plan_complete=True)
    return state
