"""Plan, execute, or verify this revision's finite completion scope (no test scoring)."""
import argparse
from pathlib import Path
from completion_common import (require, read, sha, immutable, case_id, select, identity,
                               controller_lock, check_stop, own_sources)
from native_backend import Context, NativeBackend

def execution_plan(ctx):
    ssa = ctx.spec['methods']['SSA']; pcbg = ctx.spec['methods']['PCBG']
    cv = [{'case_id': case_id(c, ci, f), 'condition_id': c['id'], 'candidate': ci, 'fold': f,
           'source_carrier': case_id(c, 0, f), 'effective_request_config': cfg,
           'whole_process_cap_seconds': 600}
          for c in ssa['conditions'] for ci, cfg in enumerate(ssa['grid']) for f in (0, 1)]
    finals = [{'condition_id': c['id'], 'source_carrier': case_id(c, 0, 'final'),
               'candidate_rule': 'global minimum complete-fold paid BER; default-first within 1e-12',
               'possible_cases': [case_id(c, i, 'final') for i in range(18)],
               'whole_process_cap_seconds': 600} for c in ssa['conditions']]
    new = []
    for c in pcbg['conditions']:
        for case in ctx.plans['PCBG']['cases']:
            if case['condition_id'] == c['id'] and case['fold'] != 'final': new.append(case['id'])
        new.append(case_id(c, 0, 'final'))
    value = {'schema': 'portable-completion-execution-plan-v1', 'status': 'PLAN_ONLY_NO_OPTIMIZATION',
             'input_manifest_sha256': sha(ctx.work / 'INPUT_MANIFEST.json'), 'own_sources': own_sources(),
             'SSA': {'CV_decisions': cv, 'conditional_finals': finals, 'maximum_fits': 555,
                     'terminal_failure_skips_later_fold_only': True, 'prune_by_BER': False},
             'PCBG': {'conditions': [c['id'] for c in pcbg['conditions']], 'retained_CV': 23,
                      'ordered_new_cases': new, 'startup_seconds': 300, 'worker_seconds': 1800,
                      'shutdown_seconds': 60, 'finite_wrong_k_not_retried': True},
             'class_labels_loaded': False, 'matlab_started': False,
             'scope': 'New optimizer replay identity; not identical historical process timing or manuscript score replacement'}
    value['portable_plan_identity'] = identity(value)
    return value

class Ledger:
    def __init__(self, root, readonly=False):
        self.root, self.readonly = Path(root), readonly
    def put(self, name, value):
        path = self.root / name
        if self.readonly: require(path.is_file() and read(path) == value, 'Recorded decision differs/missing: ' + name)
        else: immutable(path, value)
        return sha(path)

def run_ssa(spec, backend, ledger, binding, stop=lambda: None, make_seal=True):
    conditions = []
    for c in spec['conditions']:
        scores = []; hashes = {}
        for candidate in range(len(spec['grid'])):
            failed = None; row = []
            for fold in (0, 1):
                stop(); cid = case_id(c, candidate, fold)
                if failed is not None:
                    result = {'case_id': cid, 'condition_id': c['id'], 'candidate': candidate, 'fold': fold,
                              'attempted': False, 'eligible': False, 'validation_BER': None,
                              'classification': 'skipped_after_candidate_terminal_failure', 'trigger_case': failed,
                              'full_class_labels_loaded': False}
                else:
                    result = backend.fit(c, candidate, fold)
                    require(result['case_id'] == cid, 'Backend returned wrong SSA case')
                    if not result['eligible']: failed = cid
                row.append(result['validation_BER'] if result['eligible'] else None)
                hashes[cid] = ledger.put('fits/' + cid + '.json', result)
            scores.append(row)
        chosen, means, fallback = select(scores)
        selection = {'condition_id': c['id'], 'selected_candidate': chosen, 'config': spec['grid'][chosen],
                     'scores': scores, 'mean_query_BER': means, 'fallback': fallback,
                     'fit_decision_hashes': hashes, 'run_binding': binding, 'full_class_labels_loaded': False}
        selection_sha = ledger.put('selections/' + c['id'] + '.json', selection)
        stop(); final = backend.fit(c, chosen, 'final')
        require(final['case_id'] == case_id(c, chosen, 'final'), 'Backend returned wrong SSA final')
        final_sha = ledger.put('fits/' + final['case_id'] + '.json', final)
        record = {'condition_id': c['id'], 'status': 'COMPLETE', 'eligible': final['eligible'],
                  'final_case': final['case_id'], 'selection_sha256': selection_sha,
                  'final_decision_sha256': final_sha, 'run_binding': binding, 'full_class_labels_loaded': False}
        digest = ledger.put('conditions/' + c['id'] + '.json', record)
        conditions.append({'condition_id': c['id'], 'sha256': digest})
    return seal(ledger, 'SSA', conditions, binding, make_seal)

def run_pcbg(spec, backend, old, ledger, binding, stop=lambda: None, make_seal=True):
    conditions = []
    for c in spec['conditions']:
        rows = []; scores = []
        for candidate in range(3):
            values = []
            for fold in (0, 1):
                stop(); cid = case_id(c, candidate, fold)
                r = old[cid] if cid in old else backend.fit(c, candidate, fold)
                require(r['case_id'] == cid, 'Backend returned wrong PCBG CV case')
                rows.append(r); values.append(r['validation_BER'] if r['eligible'] else None)
            scores.append(values)
        chosen, means, fallback = select(scores)
        require(chosen == 0, 'New selection outside frozen five-fit plan; stop for a separately reviewed plan')
        selection = {'condition_id': c['id'], 'candidate': chosen, 'anchor_rate': spec['grid'][chosen],
                     'mean_query_BER': means, 'default_fallback': fallback, 'CV': rows,
                     'run_binding': binding, 'full_class_labels_loaded': False}
        digest = ledger.put('selections/' + c['id'] + '.json', selection)
        stop(); final = backend.fit(c, 0, 'final')
        require(final['case_id'] == case_id(c, 0, 'final'), 'Backend returned wrong PCBG final')
        record = {'condition_id': c['id'], 'selection_sha256': digest, 'final': final,
                  'run_binding': binding, 'full_class_labels_loaded': False}
        digest = ledger.put('conditions/' + c['id'] + '.json', record)
        conditions.append({'condition_id': c['id'], 'sha256': digest})
    return seal(ledger, 'PCBG', conditions, binding, make_seal)

def seal(ledger, method, conditions, binding, make_seal=True):
    value = {'status': 'NEW_PORTABLE_PREDICTIONS_SEALED_BEFORE_EXTERNAL_SCORING', 'method': method,
             'conditions': conditions, 'run_binding': binding, 'full_class_labels_loaded': False,
             'not_a_historical_score_replacement': True}
    if make_seal: ledger.put('GLOBAL_PREDICTIONS_SEALED.json', value)
    return value

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['plan', 'run', 'verify'], nargs='?', default='plan')
    for key in ('release-root', 'work'): p.add_argument('--' + key, required=True)
    p.add_argument('--method', choices=['SSA', 'PCBG'])
    p.add_argument('--matlab'); p.add_argument('--ssa-runtime'); p.add_argument('--pcbg-author')
    a = p.parse_args(); ctx = Context(a.release_root, a.work)
    if a.action == 'plan':
        plan = execution_plan(ctx)
        immutable(ctx.work / 'EXECUTION_PLAN.json', plan)
        # Paid-CV evidence is already public; no new optimizer output or test score is opened.
        require(len(ctx.old_pcbg()) == 23, 'Old CV evidence verification failed')
        print(plan['portable_plan_identity'], 'PLAN_ONLY: 540 SSA CV decisions + 15 finals; 5 PCBG fits')
        return
    require(a.method and a.matlab, 'run/verify require --method and --matlab')
    readonly = (a.action == 'verify' or
                (ctx.work / a.method / 'ledger/GLOBAL_PREDICTIONS_SEALED.json').exists())
    with controller_lock(ctx.work):
        backend = NativeBackend(ctx, a.method, a.matlab, a.ssa_runtime, a.pcbg_author, readonly)
        binding = sha(ctx.work / a.method / 'RUN_BINDING.json')
        ledger = Ledger(ctx.work / a.method / 'ledger', readonly)
        stop = (lambda: None) if readonly else (lambda: check_stop(ctx.work))
        if a.method == 'SSA': result = run_ssa(ctx.spec['methods']['SSA'], backend, ledger, binding, stop, False)
        else: result = run_pcbg(ctx.spec['methods']['PCBG'], backend, ctx.old_pcbg(), ledger, binding, stop, False)
        # Re-read every native receipt/partition and recompute paid selection before accepting a seal.
        ctx = Context(a.release_root, a.work)
        backend = NativeBackend(ctx, a.method, a.matlab, a.ssa_runtime, a.pcbg_author, True)
        ledger.readonly = True
        if a.method == 'SSA': checked = run_ssa(ctx.spec['methods']['SSA'], backend, ledger, binding, make_seal=False)
        else: checked = run_pcbg(ctx.spec['methods']['PCBG'], backend, ctx.old_pcbg(), ledger, binding, make_seal=False)
        require(checked == result, 'Second-pass evidence verification differs')
        ledger.readonly = readonly
        ledger.put('GLOBAL_PREDICTIONS_SEALED.json', checked)
        print(result['status'], a.method, len(result['conditions']))

if __name__ == '__main__':
    main()
