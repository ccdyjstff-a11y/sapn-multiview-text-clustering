"""One bounded SSA comparison; default is read-only, selection uses paid pairs only."""
import argparse
import datetime
import json
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
from formal_common import *


def order(p):
    return [next(c for c in p['conditions'] if c['dataset'] == ds
        and c['seed'] == seed and c['Q'] == Q)
        for seed in [23, 24, 25] for Q in [160, 400] for ds in FIT_DATA]


def case_map(p):
    return {c['case_id']: c for c in p['cases']}


def serialized_scores(values):
    return [[float(v) if np.isfinite(v) else None for v in row] for row in values]


def must_stop(receipt):
    return bool(receipt.get('global_deadline_triggered') or receipt.get('interrupted')
        or receipt['status'] in ['OWNED_CLEANUP_INCOMPLETE', 'LAUNCHER_EXCEPTION',
            'INVALID_PHASE_MARKER', 'MISSING_PHASE_MARKER_ON_EXIT', 'STARTUP_WALL_TIME_CAP'])


def run(expected):
    no_oracle_guard()
    import query_protocol as qp
    import run_clock_qualification as runtime
    assert sha(ROOT / 'RUNTIME_PROTOCOL.json') == expected, 'Exact runtime SHA required'
    freeze = read(ROOT / 'FREEZE_RECEIPT.json')
    assert freeze['runtime_sha256'] == expected
    assert freeze['protocol_sha256'] == sha(ROOT / 'FROZEN_PROTOCOL.json')
    assert freeze['preflight']['passed'] is True
    assert not (ROOT / 'UNRESOLVED_OWNED_PROCESSES.json').exists()
    began = time.monotonic()
    deadline = began + 48 * 3600
    start = dict(start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        start_monotonic=began, deadline_monotonic=deadline, global_wall_seconds=48 * 3600,
        protocol_sha256=freeze['protocol_sha256'], runtime_protocol_sha256=expected,
        controller_pid=os.getpid(), fits=0)
    with (ROOT / 'EXECUTION_STARTED.json').open('x', encoding='utf8') as stream:
        json.dump(start, stream, indent=2)
    runtime.RUNTIME = read(ROOT / 'RUNTIME_PROTOCOL.json')
    runtime.RUNTIME_SHA = expected
    runtime.GLOBAL_DEADLINE = deadline
    p = protocol()  # All requests and shared source inputs checked inside the global deadline.
    cases = case_map(p)
    cv, final, selections = [], [], []
    dispatched = {}
    stopped, attempts = None, 0
    counts = dict(controller_attempts=0, matlab_started=0,
        worker_native_return_reported=0, strict_eligible=0)
    expected_cv = sum(len(c['eligible_folds']) * len(GRID)
        for c in p['conditions'] if not c['unsupported'])
    assert expected_cv <= COUNTS['maximum_cv_fits']
    for cond in p['conditions']:
        if cond['unsupported']:
            final.append(dict(condition_id=cond['id'], dataset=cond['dataset'],
                seed=cond['seed'], Q=cond['Q'], method=METHOD, status='INPUT_UNSUPPORTED',
                attempted=False, eligible=False, finite_partition_retained=False,
                case_id=None, reason=cond['unsupported_reason']))

    def attempt(case):
        nonlocal stopped, attempts
        if time.monotonic() >= deadline:
            stopped = stopped or 'GLOBAL_EXECUTION_WALL_CAP'
        if stopped:
            return not_attempted(case, 'NOT_ATTEMPTED_AFTER_' + stopped)
        assert attempts < COUNTS['maximum_total_fits']
        runtime.CURRENT_CASE = case
        attempts += 1
        counts['controller_attempts'] += 1
        dispatched[case['case_id']] = dict(case_id=case['case_id'], attempted=True,
            eligible=False, finite_partition_retained=False,
            status='CONTROLLER_DISPATCH_UNRESOLVED')
        receipt = runtime.run_one(p, case, 1)
        counts['matlab_started'] += int(bool(receipt.get('matlab_started')))
        worker = receipt.get('worker', receipt.get('unqualified_worker_summary', {}))
        counts['worker_native_return_reported'] += int(bool(worker.get('execution_returned')))
        result = outcome(case)
        dispatched[case['case_id']] = dict(result)
        counts['strict_eligible'] += int(result['eligible'])
        if must_stop(receipt):
            stopped = receipt['status']
        if time.monotonic() >= deadline:
            stopped = stopped or 'GLOBAL_EXECUTION_WALL_CAP'
        return result

    try:
        assert runtime.environment_gate(p)['passed'], 'Frozen real-bridge environment gate failed'
        for cond in order(p):
            data = paid(cond['query_input'], cond['n'])
            pairs, answers = data['pairs'][:cond['Q']], data['answers'][:cond['Q']]
            folds = cond['eligible_folds']
            scores = np.full((len(GRID), len(folds)), np.nan)
            local = []
            for fi, fold in enumerate(folds):
                for ci in range(len(GRID)):
                    case = cases[case_id(cond['dataset'], cond['seed'], cond['Q'], ci, fi)]
                    rec = attempt(case)
                    rec.update(condition_id=cond['id'], candidate=ci, fold=fi)
                    if rec['eligible']:
                        _, labels, _ = terminal(case)
                        held = np.asarray(fold['validation'], int)
                        scores[ci, fi] = qp.balanced_pair_error(labels, pairs[held], answers[held])
                    rec['validation_BER'] = float(scores[ci, fi]) if np.isfinite(scores[ci, fi]) else None
                    cv.append(rec)
                    local.append(rec)
                    dump(ROOT / 'CV_OUTCOMES.json', cv)
            complete = all(rec['attempted'] for rec in local)
            if complete:
                index, rule = qp.select_configuration(scores, default_index=0)
                selected = cases[case_id(cond['dataset'], cond['seed'], cond['Q'], index, 'final')]
                selection = dict(condition_id=cond['id'], status='SELECTED',
                    scores=serialized_scores(scores), index=int(index), config=GRID[index],
                    rule=rule, eligible_folds=len(folds),
                    valid_candidates=int(np.sum(np.all(np.isfinite(scores), axis=1))),
                    final_input_sha256=selected['sha256'])
                immutable(ROOT / (cond['id'] + '_selection.json'), selection)
                selections.append(selection)
                rec = attempt(selected)
            else:
                selection = dict(condition_id=cond['id'], status='INCOMPLETE_CV_NOT_SELECTED',
                    scores=serialized_scores(scores), index=None, config=None,
                    rule='not_selected_after_execution_stop', eligible_folds=len(folds))
                immutable(ROOT / (cond['id'] + '_selection.json'), selection)
                selections.append(selection)
                rec = dict(case_id=None, attempted=False, eligible=False,
                    finite_partition_retained=False, status='NOT_ATTEMPTED_AFTER_' + str(stopped))
            final.append(dict(rec, condition_id=cond['id'], dataset=cond['dataset'],
                seed=cond['seed'], Q=cond['Q'], method=METHOD))
            dump(ROOT / 'FINAL_OUTCOMES.json', final)
            dump(ROOT / 'SELECTIONS.json', selections)
            print(json.dumps(dict(condition_id=cond['id'], status=rec['status'],
                fit_calls=attempts, global_stop=stopped)), flush=True)
    except BaseException as error:
        reason = 'NOT_ATTEMPTED_AFTER_CONTROLLER_FAILURE'
        existing_cv = {(r['condition_id'], r['candidate'], r['fold']) for r in cv}
        existing_final = {r['condition_id'] for r in final}
        existing_selection = {r['condition_id'] for r in selections}
        for cond in order(p):
            for fi in range(len(cond['eligible_folds'])):
                for ci in range(len(GRID)):
                    if (cond['id'], ci, fi) not in existing_cv:
                        case = cases[case_id(cond['dataset'], cond['seed'], cond['Q'], ci, fi)]
                        retained = dispatched.get(case['case_id'], not_attempted(case, reason))
                        cv.append(dict(retained, condition_id=cond['id'],
                            candidate=ci, fold=fi, validation_BER=None))
            if cond['id'] not in existing_final:
                selection = next((s for s in selections if s['condition_id'] == cond['id']), None)
                selected_id = (case_id(cond['dataset'], cond['seed'], cond['Q'],
                    selection['index'], 'final') if selection and selection.get('index') is not None else None)
                retained = dispatched.get(selected_id, dict(case_id=selected_id,
                    attempted=False, eligible=False, finite_partition_retained=False, status=reason))
                final.append(dict(retained, condition_id=cond['id'], dataset=cond['dataset'],
                    seed=cond['seed'], Q=cond['Q'], method=METHOD))
            if cond['id'] not in existing_selection:
                selections.append(dict(condition_id=cond['id'],
                    status='CONTROLLER_FAILURE_NOT_COMPLETED', index=None, config=None,
                    rule='not_selected_after_controller_failure', eligible_folds=len(cond['eligible_folds'])))
        dump(ROOT / 'CV_OUTCOMES.json', cv)
        dump(ROOT / 'FINAL_OUTCOMES.json', final)
        dump(ROOT / 'SELECTIONS.json', selections)
        dump(ROOT / 'CONTROLLER_FAILURE.json', dict(error=repr(error), controller_attempts=attempts,
            counts=counts, logical_outcomes=len(final), CV_outcomes=len(cv), oracle_read=False,
            warning='Partial attempt receipts remain unresolved; independent technical audit is required before any global seal.'))
        raise
    assert len(cv) == expected_cv and len(final) == COUNTS['logical_outcomes']
    assert len(selections) == COUNTS['maximum_final_fits'] and attempts <= COUNTS['maximum_total_fits']
    dump(ROOT / 'EXECUTION_FINISHED.json', dict(status='FINISHED_WITH_COMPLETE_OUTCOME_LEDGER',
        protocol_sha256=sha(ROOT / 'FROZEN_PROTOCOL.json'), runtime_sha256=expected,
        fit_calls=attempts, counts=counts,
        counting_note='fit_calls counts controller dispatch attempts, including those with no MATLAB start; native optimization calls are not inferred.',
        CV_attempts=sum(r['attempted'] for r in cv), final_attempts=sum(r['attempted'] for r in final),
        logical_outcomes=len(final), global_stop=stopped, elapsed_wall_seconds=time.monotonic() - began,
        end_overshoot_seconds=max(0., time.monotonic() - deadline), oracle_read=False,
        scoring_performed=False, all_planned_fits_attempted=all(r['attempted'] for r in cv)
            and all(r['attempted'] for r in final if r['status'] != 'INPUT_UNSUPPORTED')))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute-runtime-sha')
    args = parser.parse_args()
    if args.execute_runtime_sha:
        run(args.execute_runtime_sha)
    else:
        path = ROOT / 'RUNTIME_PROTOCOL.json'
        print(json.dumps(dict(status='FROZEN_NOT_EXECUTED' if path.exists() else 'NOT_FROZEN_NO_EXECUTION',
            runtime_sha256=sha(path) if path.exists() else None,
            already_started=(ROOT / 'EXECUTION_STARTED.json').exists()), indent=2))


if __name__ == '__main__':
    main()
