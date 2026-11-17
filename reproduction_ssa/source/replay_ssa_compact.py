"""Portable grouped-S2 evidence replay. No fitting, class truth or OS process scans."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
sys.dont_write_bytecode=True


def _local(name):
    path=Path(__file__).resolve().with_name(name+'.py')
    spec=importlib.util.spec_from_file_location('grouped_compact_'+name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


P=_local('portable_primitives')
L=_local('ledger_validation')
POLICY=_local('receipt_policy')
for _name in ['np','sha','read','member','identity','npz_members','same_score','folds_rebuilt','hard_ber','receipt_eligible','insist','GRID','SEEDS','SUPPORTED','EXPECTED_SCOPE']:
    globals()[_name]=getattr(P,_name)

def verify(export_root, expected_manifest_sha=None):
    base = Path(export_root).resolve()
    manifest_path = base / 'EXPORT_MANIFEST.json'
    manifest_sha = sha(manifest_path)
    if expected_manifest_sha is not None:
        insist(manifest_sha == expected_manifest_sha, 'Manifest identity changed')
    manifest = read(manifest_path)
    insist(manifest['kind'] == 'SSA_COMPACT_EXPORT'
           and manifest['status'] == 'EXPORTED_WITH_SOURCE_IDENTITY_CHECKS', 'Wrong manifest kind/status')
    insist(all(manifest[key] is True for key in (
        'no_model_calls', 'no_oracle_arrays_read', 'no_evaluation_metrics_decoded')), 'Wrong evidence boundary')
    insist(manifest['logical_counts'] == EXPECTED_SCOPE, 'Wrong fixed formal scope')
    insist(manifest.get('schema')==2 and manifest.get('batch')=='SSA_GROUPED_V2' and manifest.get('prior_S1_results_imported') is False, 'Independent S2 export required')
    names = manifest['members_sha256']
    insist(len({member(base, key) for key in names}) == len(names), 'Duplicate resolved member')
    for name, digest in names.items():
        insist(sha(member(base, name)) == digest, 'Changed exported member: ' + name)
    insist(all(name in names for name in manifest['public_source']), 'Unbound public source')
    insist(manifest['exporter_sha256'] == names['source/export_ssa_compact.py'], 'Exporter source identity')
    insist(sha(Path(__file__)) == names['source/replay_ssa_compact.py'], 'Replay source differs from packet')

    def get(name):
        insist(name in names, 'Unbound export member: ' + name)
        return read(member(base, name))

    protocol = get('evidence/FROZEN_PROTOCOL.json')
    runtime = get('evidence/RUNTIME_PROTOCOL.json')
    seal = get('evidence/GLOBAL_PREDICTIONS_SEALED.json')
    post = get('evidence/POST_SCORE_AUDIT.json')
    evaluation_manifest = get('evidence/EVALUATION_MANIFEST.json')
    report = get('evidence/REPORT_MANIFEST.json')
    insist(protocol.get('batch')=='SSA_GROUPED_V2' and protocol.get('prior_batch_results_imported') is False, 'S2 protocol required')
    insist(seal.get('batch')=='SSA_GROUPED_V2' and seal.get('prior_batch_results_imported') is False
        and report.get('batch')=='SSA_GROUPED_V2' and report.get('prior_S1_results_imported') is False, 'S2 seal/report required')
    insist(post.get('all_CV_and_selection_records_recollected') is True, 'S2 collector audit required')
    seal_members = {key.replace('\\', '/'): value for key, value in seal['members_sha256'].items()}
    insist(len(seal_members) == len(seal['members_sha256']), 'Duplicate seal identity')
    for key, name in [('protocol_sha256', 'FROZEN_PROTOCOL.json'),
                      ('runtime_sha256', 'RUNTIME_PROTOCOL.json'),
                      ('global_seal_sha256', 'GLOBAL_PREDICTIONS_SEALED.json'),
                      ('post_score_audit_sha256', 'POST_SCORE_AUDIT.json'),
                      ('evaluation_manifest_sha256', 'EVALUATION_MANIFEST.json'),
                      ('report_manifest_sha256', 'REPORT_MANIFEST.json')]:
        insist(manifest[key] == names['evidence/' + name], 'Manifest evidence mismatch: ' + key)
    insist(seal['status'] == 'SEALED' and seal['protocol_sha256'] == manifest['protocol_sha256']
           and seal['runtime_sha256'] == manifest['runtime_sha256'], 'Seal identity mismatch')
    insist(runtime['original_scientific_protocol_sha256'] == manifest['protocol_sha256'], 'Runtime protocol identity')
    insist(post['passed'] is True and post['kind'] == 'POST_SCORE'
           and post['all42_rows_checked'] is True and post['all14_groups_checked'] is True, 'Incomplete POST_SCORE')
    for key in ('protocol_sha256', 'runtime_sha256', 'global_seal_sha256',
                'evaluation_manifest_sha256', 'evaluation_sha256', 'summary_sha256'):
        insist(post[key] == manifest[key], 'POST_SCORE binding mismatch: ' + key)
    insist(evaluation_manifest['status'] == 'SCORED_PENDING_INDEPENDENT_REVIEW', 'Wrong scoring status')
    for key in ('protocol_sha256', 'global_seal_sha256', 'evaluation_sha256', 'summary_sha256'):
        insist(evaluation_manifest[key] == manifest[key], 'Evaluation binding mismatch: ' + key)
    insist(report['status'] == 'AUDIT_GATED_COMPARISON_REPORT'
           and report['logical_rows'] == 42 and report['corpus_query_groups'] == 14
           and report['fit_diagnostic_rows'] == 906, 'Wrong comparison scope')
    for key in ('protocol_sha256', 'global_seal_sha256', 'post_score_audit_sha256',
                'evaluation_manifest_sha256', 'evaluation_sha256', 'summary_sha256'):
        insist(report[key] == manifest[key], 'Comparison binding mismatch: ' + key)
    insist(report['runtime_protocol_sha256'] == manifest['runtime_sha256'], 'Comparison runtime binding')
    for name in ('FROZEN_PROTOCOL.json', 'RUNTIME_PROTOCOL.json', 'EXECUTION_ENDED.json',
                 'CV_OUTCOMES.json', 'FINAL_OUTCOMES.json', 'SELECTIONS.json', 'EXECUTION_STARTED.json', 'CONTROLLER_LEDGER.json', 'FREEZE_RECEIPT.json'):
        insist(seal_members[name] == names['evidence/' + name], 'Export differs from seal: ' + name)

    cv, final, selections = [get('evidence/' + name) for name in
                            ('CV_OUTCOMES.json', 'FINAL_OUTCOMES.json', 'SELECTIONS.json')]
    cases = {case['case_id']: case for case in protocol['cases']}
    conditions = {cond['id']: cond for cond in protocol['conditions']}
    insist(len(cases) == len(protocol['cases']) == 1296 and len(conditions) == len(protocol['conditions']), 'Protocol key count')
    insist(protocol['grid'] == GRID, 'Changed limited configuration grid')
    insist(len(final) == len(conditions) == protocol['counts']['logical_outcomes'] == seal['logical_outcomes'] == 42, 'Final denominator')
    expected_cv = {case['case_id'] for case in protocol['cases'] if case['fold'] != 'final'}
    insist(len(cv) == len(expected_cv) == protocol['counts']['maximum_cv_fits'] == seal['CV_outcomes'] == 864, 'CV denominator')
    insist({row['case_id'] for row in cv} == expected_cv and len({row['case_id'] for row in cv}) == len(cv), 'CV mapping')
    insist({row['condition_id'] for row in final} == set(conditions)
           and len({row['condition_id'] for row in final}) == len(final), 'Final mapping')
    expected_conditions = {(ds, seed, q) for ds, seeds in SEEDS.items() for seed in seeds for q in (160, 400)}
    insist({(cond['dataset'], cond['seed'], cond['Q']) for cond in conditions.values()} == expected_conditions, 'Dataset/seed/budget scope')
    budgets = sorted({cond['Q'] for cond in conditions.values()})

    paid = {}
    expected_paid = {(ds, seed) for ds, seeds in SEEDS.items() for seed in seeds}
    insist({(ds, int(seed)) for ds, card in protocol['datasets'].items()
            for seed in card['query_inputs']} == expected_paid, 'Frozen paid stream scope')
    for stream in get('PAID_STREAMS.json'):
        dataset, seed = stream['dataset'], stream['seed']
        insist((dataset, seed) in expected_paid and (dataset, seed) not in paid, 'Paid mapping')
        original = protocol['datasets'][dataset]['query_inputs'][str(seed)]
        insist(stream['source'] == original and stream['n'] == protocol['datasets'][dataset]['n'], 'Paid source identity')
        rel = stream['projection']['file']
        insist(names[rel] == stream['projection']['sha256'], 'Paid file hash')
        raw, arrays = npz_members(member(base, rel), ('pool.npy', 'test.npy', 'pairs.npy', 'answers.npy'))
        insist(set(stream['members']) == set(raw), 'Paid member map')
        for name, payload in raw.items():
            insist(stream['members'][name] == dict(sha256=hashlib.sha256(payload).hexdigest(),
                                                  byte_length=len(payload)), 'Paid member byte identity')
        n = stream['n']
        pool, test, pairs, answers = [arrays[key] for key in ('pool', 'test', 'pairs', 'answers')]
        insist(pool.ndim == test.ndim == 1 and pool.dtype.kind in 'iu' and test.dtype.kind in 'iu', 'Node array schema')
        insist(len(np.unique(pool)) == len(pool) and np.all((pool >= 0) & (pool < n)), 'Invalid pool')
        insist(np.array_equal(test, np.setdiff1d(np.arange(n), pool)), 'Test/pool partition')
        insist(pairs.shape == (max(budgets), 2) and pairs.dtype.kind in 'iu' and np.isin(pairs, pool).all(), 'Paid pairs schema')
        insist(np.all(pairs[:, 0] < pairs[:, 1]) and len(np.unique(pairs, axis=0)) == len(pairs), 'Duplicate/self/reversed pair')
        insist(answers.shape == (len(pairs),) and np.isin(answers, [-1, 1]).all(), 'Answers schema')
        paid[(dataset, seed)] = arrays
    insist(set(paid) == expected_paid, 'Missing paid streams')

    def projected(card, expected_member):
        rel = card['file']
        insist(names[rel] == card['sha256'] and card['member'] == expected_member, 'Array file binding')
        raw, arrays = npz_members(member(base, rel), [expected_member])
        array = arrays[expected_member[:-4]]
        insist(identity(array) == card['original_array'] == card['exported_array'], 'Shape/dtype/array identity')
        insist(hashlib.sha256(raw[expected_member]).hexdigest() == card['npy_member_sha256']
               and len(raw[expected_member]) == card['npy_member_byte_length'], 'Array NPY identity')
        return array

    partitions = {}
    receipts = {}
    counts = dict(CV_arrays=0, final_arrays=0, final_consensus_arrays=0)
    attempts = dict(controller_attempts=0, matlab_started=0, worker_native_return_reported=0, strict_eligible=0)
    for phase, rows in [('CV', cv), ('FINAL', final)]:
        mapping = get(phase + '_PARTITIONS.json')
        insist(len(mapping) == len(rows), 'Partition logical denominator')
        for record, row in zip(mapping, rows):
            insist(record['outcome'] == row, 'Partition/outcome mapping mismatch')
            case_id, source, partition = row.get('case_id'), record['source'], record['partition']
            consensus = record.get('consensus')
            array = None
            if case_id is None:
                insist(source is None and not row['attempted'] and not row['eligible'], 'Null-case state')
            else:
                case = cases[case_id]
                insist(source['case_id'] == case_id and source['n'] == case['n']
                       and source['input_sha256'] == case['sha256'], 'Case identity')
            insist((partition is not None) == bool(row['finite_partition_retained']), 'Finite mapping mismatch')
            if partition is not None:
                insist(row['attempted'] and source['native_mat'] is not None, 'Unbound partition')
                array = projected(partition, 'partition.npy')
                insist(array.dtype.kind in 'iuf' and array.size == case['n'] and np.isfinite(array).all()
                       and np.all(array >= 1) and np.all(array == np.floor(array)), 'Finite partition schema')
                insist(case_id not in partitions, 'Duplicate partition case')
                partitions[case_id] = array.reshape(-1)
                counts['CV_arrays' if phase == 'CV' else 'final_arrays'] += 1
            if row['attempted']:
                attempts['controller_attempts'] += 1
                card = source['receipt']
                receipt = get(card['file'])
                insist(card['sha256'] == row['receipt_sha256'] == names[card['file']]
                       == seal_members[card['original_member']], 'Receipt binding')
                insist(card['original_member'] == 'runs/' + case_id + '/repeat1/launcher_receipt.json', 'Wrong receipt path')
                insist(receipt['case_id'] == case_id and receipt['repeat'] == 1
                       and receipt['input_sha256'] == case['sha256']
                       and receipt['protocol_sha256'] == manifest['protocol_sha256']
                       and receipt['runtime_protocol_sha256'] == manifest['runtime_sha256'], 'Receipt case/protocol')
                worker = receipt.get('worker', receipt.get('unqualified_worker_summary', {}))
                eligible = receipt_eligible(receipt, array, case['c'])
                insist(row['finite_partition_retained'] == bool(worker.get('finite_partition_retained')), 'Retained partition claim differs')
                receipts[case_id] = receipt
                attempts['matlab_started'] += int(bool(receipt.get('matlab_started')))
                attempts['worker_native_return_reported'] += int(bool(worker.get('execution_returned')))
                attempts['strict_eligible'] += int(eligible)
                mat = source['native_mat']
                if 'native_outputs.mat' in receipt['output_hashes']:
                    insist(mat is not None and mat['original_member'] == 'runs/' + case_id + '/repeat1/native_outputs.mat'
                           and mat['field'] == 'label' and mat['sha256'] == receipt['output_hashes']['native_outputs.mat']
                           == seal_members[mat['original_member']], 'MAT provenance mismatch')
                else:
                    insist(mat is None, 'Invented MAT provenance')
            else:
                insist(not row['eligible'] and partition is None and consensus is None, 'Unattempted array/eligibility')
                insist(row['status'] in {'INPUT_UNSUPPORTED','GLOBAL_STOP','SKIPPED_AFTER_LOCAL_INELIGIBLE'} or row['status'].startswith('NOT_ATTEMPTED_AFTER_'), 'Unknown unattempted state')
                if source is not None:
                    insist(source['receipt'] is None and source['native_mat'] is None, 'Unattempted artifact')
            insist(not row['eligible'] or partition is not None, 'Eligible without array')
            if phase == 'CV':
                insist(consensus is None and record['consensus_status'] == 'not_applicable_CV', 'CV consensus outside export scope')
            elif consensus is None:
                expected_status = ('no_retained_partition' if partition is None else
                                   'unexpected_consensus_shape_or_dtype_not_exported')
                insist(record['consensus_status'] == expected_status, 'Missing-consensus status differs')
            if consensus is not None:
                insist(row['attempted'] and source['native_mat'] is not None
                       and source['final_consensus_field'] == 'V_star'
                       and record['consensus_status'] == 'exported', 'Unbound final consensus')
                values = projected(consensus, 'V_star.npy')
                insist(values.dtype.kind in 'iufc' and values.shape == (case['n'], case['c']), 'Final consensus shape/type')
                if np.isrealobj(values) and np.isfinite(values).all() and array is not None:
                    insist(np.array_equal(np.argmax(values, axis=1) + 1, array.reshape(-1)), 'Consensus/label mismatch')
                counts['final_consensus_arrays'] += 1
    insist(counts == manifest['counts'], 'Actual array counts')
    insist(manifest['source_identity_checks'] == dict(
        partition_arrays_compared=counts['CV_arrays'] + counts['final_arrays'],
        final_consensus_arrays_compared=counts['final_consensus_arrays'],
        paid_NPY_payloads_preserved=84), 'Recorded projection-check counts')
    insist(manifest['logical_counts'] == dict(CV=len(cv), final=len(final), paid_streams=len(paid)), 'Logical counts')
    bindings=manifest['implementation_source_bindings']
    insist(bindings['ledger_validation']==seal['reporting_source_sha256']['ledger_validation.py']
        ==names['source/ledger_validation.py']==sha(Path(L.__file__)), 'Collector structural source differs')
    adapter_sources=[digest for name,digest in protocol['source_hashes'].items()
        if name.replace('\\','/').endswith('/adapter_policy.py')]
    insist(adapter_sources==[bindings['receipt_policy']] and bindings['receipt_policy']
        ==names['source/receipt_policy.py']==sha(Path(POLICY.__file__)), 'Frozen receipt policy differs')
    insist(sha(Path(P.__file__))==names['source/portable_primitives.py'],'Primitive source differs')
    replay=replay_scheduler(protocol,get('evidence/CONTROLLER_LEDGER.json'),cv,final,selections,
        get('evidence/EXECUTION_STARTED.json'),get('evidence/EXECUTION_ENDED.json'),
        get('SCHEDULER_MEMBERS.json'),get,names,seal_members,paid,partitions,receipts)
    return dict(kind='SSA_COMPACT_REPLAY',batch='SSA_GROUPED_V2',passed=True,
        manifest_sha256=manifest_sha,external_manifest_hash_supplied=expected_manifest_sha is not None,
        counts=counts,logical_counts=manifest['logical_counts'],selections_checked=24,
        **replay,metrics_recomputed=False,budget_BER_recomputed=True,fitting_performed=False,
        full_truth_reads=0,full_matrix_qualification_recomputed=False,process_exit_rechecked=False,
        original_MAT_array_equality_independently_rechecked=False,
        limitation='Paid-query/selection/short-circuit/commit replay only. Original MAT projection equality is checked at export; '
            'dense states, physical memory observations, process exit, and class/test metrics are not recomputed.')


def replay_condition_budget(condition,meta,data,partitions):
    """Small independently testable unit, retaining unresolved/local/final roles."""
    from copy import deepcopy
    rows=deepcopy(condition['cv'])
    pairs,answers=data['pairs'][:meta['Q']],data['answers'][:meta['Q']]
    for row in rows:
        if row['status']=='ELIGIBLE':
            held=np.asarray(meta['eligible_folds'][row['fold']]['validation'],dtype=int)
            value=hard_ber(partitions[row['case_id']],pairs[held],answers[held])
            insist(same_score(row['result']['validation_BER'],value),'Paid validation BER differs')
            row['result']['validation_BER']=value
    if all(row['status'] in L.RESOLVED for row in rows):
        expected=L.selected(rows)
        insist(expected['index']==condition['selection']['index']
            and expected['rule']==condition['selection']['rule']
            and expected['valid_candidates']==condition['selection']['valid_candidates'],
            'Recomputed paid BER changes selection/default')
    else:
        insist(condition['selection']['index'] is None,'An unresolved condition selected a candidate')
    final=condition['final'];partition=partitions.get(final['case_id'])
    primary=bool(final['status']=='ELIGIBLE' and condition['committed'])
    return dict(condition_id=condition['condition_id'],dataset=condition['corpus'],Q=condition['Q'],seed=condition['seed'],
        condition_committed=condition['committed'],status=final['status'],final_attempted=final['attempted'],
        finite_partition_retained=partition is not None,primary_eligible=primary,
        selection_status=condition['selection']['status'],selected_index=condition['selection']['index'],
        selection_rule=condition['selection']['rule'],
        training_BER=None if partition is None else hard_ber(partition,pairs,answers),
        training_BER_role='primary_eligible' if primary else 'diagnostic_ineligible_finite' if partition is not None else 'missing')


def _scheduler_allowed(name):
    roots={'FROZEN_PROTOCOL.json','RUNTIME_PROTOCOL.json','FREEZE_RECEIPT.json',
        'EXECUTION_STARTED.json','EXECUTION_ENDED.json','CONTROLLER_LEDGER.json',
        'CV_OUTCOMES.json','FINAL_OUTCOMES.json','SELECTIONS.json',
        'STOP_AFTER_CONDITION.json','STOP_REQUEST_ACKNOWLEDGED.json'}
    return (name in roots or (name.endswith('.json') and name.split('/')[0] in {'guards','dispatches','decisions','conditions'})
        or (name.startswith('runs/') and name.endswith(('/repeat1/RUN_IDENTITY.json','/repeat1/launcher_receipt.json'))))


def replay_scheduler(protocol,state,raw_cv,raw_final,raw_selections,start,end,mapping,get,names,seal_members,paid,partitions,receipts):
    checked=L.validate_state(state,protocol)
    execution={k:start[k] for k in ['run_id','protocol_sha256','runtime_sha256']}
    insist(all(end[k]==v for k,v in execution.items()),'S2 start/end identity')
    insist(end['logical_counts']==dict(CV=864,final=42,selections=24) and end['no_class_scoring'] is True,'S2 end boundary')
    expected_stop=checked['trigger']['result']['reason'] if checked['trigger'] else state['stop_reason']
    insist(state['scheduler_stop_reason']==expected_stop,'S2 scheduling trigger reason')
    for key in ['status','safe_stop_confirmed','plan_complete','scheduler_stop_reason','condition_commits']:
        insist(end[key]==state[key],'S2 end/ledger differs: '+key)
    insist(set(mapping)=={name for name in seal_members if _scheduler_allowed(name)},'Missing/extra scheduler metadata')
    canonical=lambda value:str(value).replace('\\','/').casefold()
    by_path={canonical(card['original_absolute_path']):original for original,card in mapping.items()}
    insist(len(by_path)==len(mapping),'Ambiguous original path mapping')
    for original,card in mapping.items():
        insist(card['sha256']==seal_members[original]==names[card['file']],'Scheduler member source identity')
    def original(name):
        insist(name in mapping,'Unexported scheduler member: '+name)
        return get(mapping[name]['file'])
    def bound(ref,execution_required=True):
        key=by_path.get(canonical(ref['path']))
        insist(key is not None and mapping[key]['sha256']==ref['sha256'],'Unbound original reference')
        value=original(key)
        if execution_required:
            insist(all(value.get(k)==v for k,v in execution.items()),'Scheduler reference execution differs')
        return value
    conditions={c['id']:c for c in protocol['conditions']};cases={c['case_id']:c for c in protocol['cases']}
    for meta in conditions.values():
        data=paid[(meta['dataset'],meta['seed'])]
        all_folds,folds=folds_rebuilt(data['pool'],data['pairs'][:meta['Q']],meta['fold_partition_seed'],data['answers'][:meta['Q']])
        insist(all_folds==meta['all_folds'] and folds==meta['eligible_folds'],'Original node-disjoint fold identities')
    policy=POLICY.ReceiptPolicy()
    counts=dict(adapter_calls=0,launcher_dispatches=0,matlab_started=0,terminal_attempt_lifecycles=0)
    seen_groups=set();guard_names=set();attempts=set();decision_ids=set();decisions={};outcomes=[]
    for condition in state['conditions']:
        cid=condition['condition_id'];meta=conditions[cid];data=paid[(meta['dataset'],meta['seed'])]
        for row in condition['cv']+[condition['final']]:
            result=row['result']
            if result is None:continue
            counts['adapter_calls']+=1;case=cases[row['case_id']]
            ref=result['evidence_ref'];evidence=bound(ref);group=(meta['dataset'],meta['Q'])
            if group not in seen_groups:
                if ref['kind']!='PRE_DISPATCH_DEADLINE':
                    guard_name=f"guards/{group[0]}_Q{group[1]}.json";guard=original(guard_name);guard_names.add(guard_name)
                    insist(all(guard.get(k)==v for k,v in execution.items()),'Group guard execution identity')
                    insist((guard['corpus'],guard['Q'],guard['n'])==(meta['dataset'],meta['Q'],meta['n']),'Group guard condition')
                    insist(guard['threshold_bytes']==int((3 if meta['n']<2000 else 4.5)*2**30)
                        and guard['required_stable_seconds']==60 and guard['total_wait_cap_seconds']==1800,'Group guard policy')
                    insist(guard['native_fit_started'] is False and guard['launcher_dispatched'] is False,'Guard falsely claims a fit')
                    if result['attempted']:insist(guard['status']=='READY','Attempt preceded memory qualification')
                seen_groups.add(group)
            if not result['attempted']:
                insist(ref['kind'] in {'GROUP_GUARD','PRE_DISPATCH_DEADLINE'} and row['status']=='GLOBAL_STOP','Unattempted result is not a guard')
                guard_names.add(by_path[canonical(ref['path'])])
                insist(evidence['status']==result['reason'] and evidence['native_fit_started'] is False
                    and evidence['launcher_dispatched'] is False,'Guard attempt/reason differs')
                insist(case['case_id'] not in receipts,'Unattempted guard has a native receipt')
                continue
            case_id=case['case_id'];attempts.add(case_id);counts['launcher_dispatches']+=1
            dispatch=original('dispatches/'+case_id+'.json')
            insist(all(dispatch.get(k)==v for k,v in execution.items()) and dispatch['case_id']==case_id
                and dispatch['input_sha256']==case['sha256'] and dispatch['ordinal']==counts['launcher_dispatches'],'Native dispatch identity/order')
            receipt=receipts[case_id];labels=partitions.get(case_id)
            insist(original('runs/'+case_id+'/repeat1/launcher_receipt.json')==receipt,'Receipt mapping')
            native_identity=original('runs/'+case_id+'/repeat1/RUN_IDENTITY.json')
            insist(native_identity==dict(run_id=receipt['run_id'],case_id=case_id,repeat=1,input_sha256=case['sha256'],
                scientific_protocol_sha256=execution['protocol_sha256'],runtime_protocol_sha256=execution['runtime_sha256']),'Native run identity')
            insist(receipt.get('attempt_lifecycle_ended') is True and receipt.get('cleanup',{}).get('complete') is True,'Unclosed native attempt')
            numeric_eligible=receipt_eligible(receipt,labels,case['c']);ber=None
            if row['fold']!='final' and numeric_eligible:
                held=np.asarray(meta['eligible_folds'][row['fold']]['validation'],dtype=int)
                ber=hard_ber(labels,data['pairs'][:meta['Q']][held],data['answers'][:meta['Q']][held])
            expected=policy.assess(dict(receipt,is_cv=row['fold']!='final'),numeric_eligible,ber)
            decision=original('decisions/'+case_id+'.json');decision_ids.add(case_id);decisions[case_id]=decision
            insist(all(decision.get(k)==v for k,v in execution.items()),'Native decision execution identity')
            insist(all(decision.get(k)==v for k,v in expected.items()),'Resource streak/native disposition differs')
            insist(decision['case_id']==case_id and bound(dict(path=decision['receipt_path'],sha256=decision['receipt_sha256']),execution_required=False)==receipt,
                'Native decision receipt differs')
            insist(decision['finite_partition_retained']==(labels is not None) and decision['numerical_eligible']==numeric_eligible
                and same_score(decision['validation_BER'],ber),'Decision partition or paid BER differs')
            counts['matlab_started']+=int(expected['matlab_started']);counts['terminal_attempt_lifecycles']+=int(expected['terminal_attempt_lifecycle'])
            insist(result['execution_complete']==expected['terminal_attempt_lifecycle']
                and result['cleanup_complete']==receipt['cleanup']['complete'],'Scheduler lifecycle differs')
            if ref['kind']=='POST_DECISION_DEADLINE':
                guard_names.add(by_path[canonical(ref['path'])])
                insist(evidence['case_id']==case_id and evidence['status']=='GLOBAL_EXECUTION_WALL_CAP'
                    and result['status']=='GLOBAL_STOP' and result['reason']=='GLOBAL_EXECUTION_WALL_CAP','Deadline override identity')
                insist(bound(evidence['native_decision'])==decision,'Deadline native evidence differs')
            else:
                insist(ref['kind']=='NATIVE_DECISION' and evidence==decision and result['status']==expected['disposition']
                    and result['reason']==expected['reason'] and same_score(result['validation_BER'],ber),'Native scheduler result differs')
        if condition['committed']:
            insist(bound(state['condition_commits'][cid])['condition']==L.expected_commit_payload(condition),'Incomplete or altered commit payload')
        outcomes.append(replay_condition_budget(condition,meta,data,partitions))
    insist(counts==state['adapter_counts']==end['counts'],'Actual adapter/launcher/native counters differ')
    insist(attempts==set(receipts),'Native receipts outside the actual dispatch set')
    insist({name for name in mapping if name.startswith('guards/')}==guard_names,'Guard set differs')
    for folder,ids in [('dispatches',attempts),('decisions',decision_ids),('conditions',set(state['condition_commits']))]:
        insist({name for name in mapping if name.startswith(folder+'/')}=={folder+'/'+cid+'.json' for cid in ids},'Extra/missing '+folder)
    insist(set(state['condition_commits'])=={c['condition_id'] for c in state['conditions'] if c['committed']},'Commit-reference denominator')
    if state['status']=='SAFE_STOP_AT_CONDITION_BOUNDARY':
        ack=original('STOP_REQUEST_ACKNOWLEDGED.json');request=ack['request'];req=bound(request)
        insist(all(ack.get(k)==v for k,v in execution.items()) and req==request['request'] and type(req['schema']) is int
            and req['schema']==1 and req['request']=='STOP_AFTER_CONDITION','Boundary request identity')
        last=state['conditions'][checked['committed_conditions']-1]
        insist(ack['condition_id']==last['condition_id'] and ack['condition_commit']==state['condition_commits'][last['condition_id']],
            'Boundary acknowledgement differs from final committed condition')
    else:insist('STOP_REQUEST_ACKNOWLEDGED.json' not in mapping,'Unexpected boundary acknowledgement')
    # Reconstruct exactly the raw 864/42/24 ledgers, preserving scheduler status.
    from copy import deepcopy
    expected_cv=[];expected_final=[];expected_selections=[]
    for condition in state['conditions']:
        metadata=dict(condition_id=condition['condition_id'],dataset=condition['corpus'],Q=condition['Q'],seed=condition['seed'])
        for row in condition['cv']+[condition['final']]:
            item=deepcopy(row);item.update(metadata);decision=decisions.get(row['case_id'],{})
            item.update(eligible=row['status']=='ELIGIBLE',finite_partition_retained=decision.get('finite_partition_retained',False),
                receipt_sha256=decision.get('receipt_sha256'),condition_committed=condition['committed'],
                validation_BER=decision.get('validation_BER') if row['status']=='ELIGIBLE' and row['fold']!='final' else None)
            (expected_final if row['fold']=='final' else expected_cv).append(item)
        expected_selections.append(dict(**metadata,**deepcopy(condition['selection']),condition_committed=condition['committed']))
    for meta in protocol['conditions']:
        if meta['unsupported']:
            expected_final.append(dict(condition_id=meta['id'],dataset=meta['dataset'],Q=meta['Q'],seed=meta['seed'],case_id=None,
                status='INPUT_UNSUPPORTED',reason=meta['unsupported_reason'],attempted=False,eligible=False,
                finite_partition_retained=False,condition_committed=False))
            outcomes.append(dict(condition_id=meta['id'],dataset=meta['dataset'],Q=meta['Q'],seed=meta['seed'],
                status='INPUT_UNSUPPORTED',condition_committed=False,primary_eligible=False,training_BER=None,training_BER_role='missing'))
    insist(expected_cv==raw_cv and expected_final==raw_final and expected_selections==raw_selections,'Raw logical ledger reconstruction differs')
    groups=[]
    for ds in protocol['datasets']:
        for q in [160,400]:
            rows=[row for row in outcomes if row['dataset']==ds and row['Q']==q]
            insist(len(rows)==3,'Missing planned group denominator')
            groups.append(dict(dataset=ds,Q=q,n_planned=3,committed=sum(r['condition_committed'] for r in rows),
                primary_eligible=sum(r['primary_eligible'] for r in rows),complete_three_seed_primary=all(r['primary_eligible'] for r in rows)))
    return dict(receipt_counts=counts,committed_conditions=checked['committed_conditions'],
        pruned_CV=sum(r['status']=='SKIPPED_AFTER_LOCAL_INELIGIBLE' for r in raw_cv),
        controller_status=state['status'],scheduler_stop_reason=state['scheduler_stop_reason'],
        outcomes=outcomes,groups=groups,all42_rows_checked=True,all14_groups_checked=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root');parser.add_argument('--manifest-sha');parser.add_argument('--output');args=parser.parse_args()
    if args.root is None:
        insist(args.manifest_sha is None and args.output is None,'--root required')
        print(json.dumps(dict(status='READ_ONLY_PREPARATION',scientific_evidence_opened=False)));return
    result=verify(args.root,args.manifest_sha)
    if args.output:
        output=Path(args.output).resolve();insist(not output.is_relative_to(Path(args.root).resolve()),'Keep replay output outside packet')
        with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k not in {'outcomes','groups'}},indent=2))


if __name__=='__main__':main()
