"""Independent, data-free checks of the grouped controller's logical ledger.

This module does not import the scheduling implementation or access predictions.
Its inputs are already hash-bound JSON records; native evidence is checked by the
collector before this structural validation can authorize scoring.
"""
from copy import deepcopy
import math

RESOLVED = {'ELIGIBLE', 'LOCAL_INELIGIBLE', 'SKIPPED_AFTER_LOCAL_INELIGIBLE'}
ENDING = {'COMPLETED', 'SAFE_STOP_AT_CONDITION_BOUNDARY', 'EXECUTION_HALTED'}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def selected(records):
    require(len(records) == 36, 'A condition must retain 36 CV slots')
    by_key = {(r['candidate'], r['fold']): r for r in records}
    require(set(by_key) == {(c, f) for c in range(18) for f in range(2)}, 'CV identities differ')
    candidates, valid = [], []
    for candidate in range(18):
        rows = [by_key[candidate, f] for f in range(2)]
        require(all(r['status'] in RESOLVED for r in rows), 'Unresolved CV cannot select')
        for row in rows:
            if row['status'] == 'SKIPPED_AFTER_LOCAL_INELIGIBLE':
                prior = by_key[candidate, 0]
                require(row['fold'] == 1 and row['attempted'] is False and row['result'] is None
                        and row['trigger_case'] == prior['case_id'] and prior['status'] == 'LOCAL_INELIGIBLE',
                        'Skipped fold has no earlier terminal local trigger')
        if all(r['status'] == 'ELIGIBLE' for r in rows):
            scores = [r['result']['validation_BER'] for r in rows]
            require(all(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1 for x in scores),
                    'Eligible CV needs finite BER in [0,1]')
            mean = sum(scores) / 2
            valid.append((candidate, mean))
            candidates.append(dict(index=candidate, status='VALID', mean_BER=mean, scores=scores))
        else:
            require(any(r['status'] == 'LOCAL_INELIGIBLE' for r in rows), 'Invalid candidate lacks a local failure')
            candidates.append(dict(index=candidate, status='INVALID_LOCAL', mean_BER=None, scores=None))
    if valid:
        best = min(v for _, v in valid)
        tied = [i for i, v in valid if abs(v-best) <= 1e-12]
        index = 0 if 0 in tied else min(tied)
        rule = 'minimum_balanced_pair_error'
    else:
        index, rule = 0, 'all_validation_fits_failed_default_with_failure_flag'
    return dict(status='SELECTED', index=index, rule=rule, valid_candidates=len(valid),
                decided_candidates=18, candidates=candidates)


def check_result(row):
    result = row['result']
    require(isinstance(result, dict) and result['case_id'] == row['case_id'], 'Missing or wrong result identity')
    flags = ['attempted', 'execution_complete', 'cleanup_complete', 'eligible', 'irreversible_local_failure']
    require(all(type(result.get(k)) is bool for k in flags), 'Non-Boolean native lifecycle flag')
    require(type(row['attempted']) is bool and row['attempted'] == result['attempted'], 'Attempt fact differs')
    require(row['status'] == result['status'], 'Native disposition and scheduler status differ')
    if row['status'] in {'ELIGIBLE', 'LOCAL_INELIGIBLE'}:
        require(result['attempted'] and result['execution_complete'] and result['cleanup_complete'],
                'A local disposition requires a completed, cleaned attempt')
        require(result['eligible'] == (row['status'] == 'ELIGIBLE'), 'Eligibility contradicts disposition')
        require(result['irreversible_local_failure'] == (row['status'] == 'LOCAL_INELIGIBLE'),
                'Candidate-local failure is not irreversible')
    else:
        require(row['status'] == 'GLOBAL_STOP' and not result['eligible'], 'Unsupported or unresolved terminal disposition')
        # A failed lifecycle may be recorded, but the collector must separately
        # prove current process exit. It is never converted into a local skip.


def validate_state(state, protocol):
    require(state['status'] in ENDING, 'Failure/forced abort lacks a scoreable terminal controller ledger')
    expected = [(g['corpus'], g['Q'], seed) for g in protocol['groups'] for seed in g['seeds']]
    conditions = state['conditions']
    require(len(expected) == len(conditions) == 24, 'Supported condition plan differs')
    require(state['planned_conditions'] == 24, 'Planned-condition count differs')
    committed, stopped, stop_record, all_results = 0, False, None, []
    boundary = state['status'] == 'SAFE_STOP_AT_CONDITION_BOUNDARY'
    stop_suffix = 'CONDITION_BOUNDARY_STOP' if boundary else 'GLOBAL_STOP'
    for ordinal, (condition, (dataset, budget, seed)) in enumerate(zip(conditions, expected)):
        cid = f'{dataset}_s{seed}_Q{budget}'
        require((condition['condition_id'], condition['corpus'], condition['Q'], condition['seed'],
                 condition['group_index'], condition['seed_index']) == (cid,dataset,budget,seed,ordinal//3,ordinal%3),
                'Condition order/identity changed')
        require(type(condition['committed']) is bool, 'Commit flag is not Boolean')
        cv = condition['cv']
        require(len(cv) == 36, 'Missing logical CV slots')
        local_failures = {}
        for position, row in enumerate(cv):
            candidate, fold = position % 18, position // 18
            case_id = f'{cid}_c{candidate}_f{fold}'
            require((row['case_id'], row['candidate'], row['fold']) == (case_id,candidate,fold), 'CV order/identity differs')
            status = row['status']
            if row['result'] is not None:
                require(not stopped, 'A fit appears after a global/boundary stop')
                check_result(row)
                all_results.append(row)
                if status == 'GLOBAL_STOP':
                    stopped, stop_record = True, row
                elif status == 'LOCAL_INELIGIBLE':
                    local_failures[candidate] = case_id
                require(row['trigger_case'] is None, 'An actual result has a skip trigger')
            elif status == 'SKIPPED_AFTER_LOCAL_INELIGIBLE':
                require(not stopped and candidate in local_failures and fold == 1
                        and row['trigger_case'] == local_failures[candidate] and row['attempted'] is False,
                        'Local skip is not supported by a prior local failure')
            else:
                require(status == 'NOT_ATTEMPTED_AFTER_' + stop_suffix and row['attempted'] is False,
                        'Unexplained unattempted CV slot')
                require(stopped or boundary, 'Pending slot before any stop')
                stopped = True
                require(row['trigger_case'] == state['trigger_case'], 'Global pending trigger differs')
        resolved = all(r['status'] in RESOLVED for r in cv)
        selection = condition['selection']
        if resolved:
            require(selection == selected(cv), 'Selection was not reconstructed from all decided candidates')
            expected_case = f"{cid}_c{selection['index']}_final"
            require((condition['final']['case_id'], condition['final']['candidate']) == (expected_case,selection['index']),
                    'Selected final input differs')
        else:
            require(selection == dict(status='NOT_SELECTED_AFTER_'+stop_suffix,index=None,rule=None),
                    'Incomplete CV must remain unselected')
            require(condition['final']['case_id'] is None and condition['final']['candidate'] is None,
                    'Unselected condition has a final case')
        final = condition['final']
        require(final['fold'] == 'final', 'Final slot type differs')
        if final['result'] is not None:
            require(resolved and not stopped, 'Final dispatched without complete CV or after stop')
            check_result(final); all_results.append(final)
            if final['status'] == 'GLOBAL_STOP':
                stopped, stop_record = True, final
        else:
            require(final['status'] == 'NOT_ATTEMPTED_AFTER_'+stop_suffix and final['attempted'] is False
                    and final['trigger_case'] == state['trigger_case'], 'Unexplained missing final')
        if condition['committed']:
            require(not stopped and ordinal == committed and condition['status'] == 'COMMITTED',
                    'Committed conditions must form a complete prefix')
            require(resolved and final['status'] in {'ELIGIBLE','LOCAL_INELIGIBLE'}, 'Commit precedes terminal final')
            committed += 1
        else:
            expected_status = ('INCOMPLETE_AFTER_GLOBAL_STOP' if stop_record and
                stop_record['case_id'].startswith(cid+'_') else 'NOT_ATTEMPTED_AFTER_'+stop_suffix)
            require(condition['status'] == expected_status, 'Uncommitted condition status differs')
    require(state['committed_conditions'] == committed and state['plan_complete'] is (committed==24), 'Commit counters differ')
    require(state['safe_stop_confirmed'] is boundary, 'Safe-stop flag is not a boundary-stop fact')
    if state['status'] == 'COMPLETED':
        require(committed == 24 and not stopped and state['stop_reason'] is None and state['trigger_case'] is None,
                'Completed plan is incomplete')
    elif boundary:
        require(committed > 0 and stop_record is None and state['stop_reason'] == 'CONDITION_BOUNDARY_STOP'
                and state['trigger_case'] == conditions[committed-1]['final']['case_id'], 'Boundary-stop identity differs')
    else:
        require(stop_record is not None and state['stop_reason'] == 'GLOBAL_STOP'
                and state['trigger_case'] == stop_record['case_id'], 'Global-stop trigger differs')
    return dict(committed_conditions=committed, results=all_results, trigger=stop_record)


def expected_commit_payload(condition):
    prior = deepcopy(condition)
    prior['status'] = 'COMPUTED_COMPLETE_UNCOMMITTED'
    prior['committed'] = False
    return prior


def primary_eligible(row):
    return row.get('eligible') is True and row.get('condition_committed') is True
