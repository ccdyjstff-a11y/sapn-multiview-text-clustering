"""Pure metadata qualification and fixed-scope aggregation; no filesystem access."""
import math
import re
import statistics

METRICS = ('ACC', 'NMI', 'ARI', 'training_BER', 'unqueried_BER')
CATEGORIES = {'strict_eligible', 'finite_wrong_k', 'finite_other_contract',
              'input_unsupported', 'not_attempted', 'runtime_failure', 'invalid_partition'}
PROJECTION_SCHEMA = 'sapn-non-k-worker-projection-v1'
WORKER_FIELDS = ('case_id', 'status', 'execution_returned', 'finite_partition_retained',
                 'numeric_outputs_finite', 'output_shapes_valid', 'n', 'c', 'strict_c',
                 'benchmark_contract_met', 'sample_partition_count')
SSA_FIELDS = ('views', 'state_checks', 'initial_CL_nonzeros', 'final_CL_max_abs',
              'full_class_truth_used', 'clustering_scores_computed')
PCBG_FIELDS = ('graph_component_count', 'minimum_P', 'minimum_alpha',
               'alpha_simplex_residual', 'native_label_partition_matches_graph',
               'initializer_repeated_B_exact')
STATE_FIELDS = ('all_finite', 'all_shapes_valid', 'nonnegative_where_required',
                'all_membership_rows_positive')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def integer(value):
    return finite(value) and value == int(value)


def project_worker(method, worker):
    """Never propagate paths, RNG, source inputs, or unneeded diagnostics."""
    fields = WORKER_FIELDS + (SSA_FIELDS if method == 'SSA' else PCBG_FIELDS)
    result = {key: worker[key] for key in fields if key in worker}
    if isinstance(result.get('state_checks'), dict):
        result['state_checks'] = {key: result['state_checks'][key]
                                  for key in STATE_FIELDS if key in result['state_checks']}
    return safe_projection(result)


def safe_projection(value, field=''):
    """Malformed values remain ineligible without leaking paths or JSON NaN."""
    if field == '':
        if not isinstance(value, dict):
            return None
        allowed = set(WORKER_FIELDS + SSA_FIELDS + PCBG_FIELDS) | {
            'schema', 'exit_code', 'cleanup_complete', 'matlab_exit_code', 'timed_out',
            'failure', 'input_and_runtime_hashes_unchanged', 'class_labels_loaded',
            'worker_status', 'started_marker_bound', 'finished_marker_bound'}
        return {key: safe_projection(item, key) for key, item in value.items() if key in allowed}
    if field == 'state_checks':
        if not isinstance(value, dict):
            return None
        return {key: safe_projection(value[key], key) for key in STATE_FIELDS if key in value}
    if field == 'initial_CL_nonzeros':
        return [item if finite(item) else None for item in value] if isinstance(value, list) else None
    # A malformed/non-null failure must stay a failure, not become valid null.
    if field == 'failure':
        return None if value is None else 'REPORTED_FAILURE'
    if isinstance(value, (dict, list)):
        return None
    if value is None or type(value) is bool or finite(value):
        return value
    if field == 'case_id' and isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_]+', value):
        return value
    if field in ('status', 'worker_status', 'schema'):
        allowed = {'RETURNED_FINITE', 'RETURNED_FINITE_CONTRACT_FAILED', 'legacy_launcher',
                   'ssa_portable', 'pcbg_extension', 'COMPLETE_ZERO_EXIT_VALID_MARKERS',
                   'EXECUTED', 'PROCESS_EXITED'}
        return value if isinstance(value, str) and value in allowed else 'UNKNOWN'
    return None


def qualification(condition, row, metadata):
    """Uses identities/contracts only. No metric field is inspected here."""
    reasons = []
    method, cid = row['method'], condition['id']
    result = dict(method=method, condition_id=cid, requested_k=condition['c'],
                  sample_partition_count=None, graph_component_count=None,
                  original_strict_c=None, non_k_qualified=False,
                  paired_qualified=False, exclusion_reasons=reasons)
    if not condition['supported']:
        reasons.append('input_unsupported'); return result
    if row['outcome_category'] not in ('strict_eligible', 'finite_wrong_k'):
        reasons.append('original_category_excluded:' + row['outcome_category']); return result
    if not isinstance(metadata, dict) or metadata.get('schema') != PROJECTION_SCHEMA:
        reasons.append('missing_or_unknown_projection_schema'); return result
    if metadata.get('unavailable_reason'):
        reasons.append('qualification_metadata_unavailable:' + metadata['unavailable_reason']); return result
    if metadata.get('method') != method or metadata.get('condition_id') != cid:
        reasons.append('projection_identity_mismatch'); return result
    w, rt = metadata.get('worker'), metadata.get('runtime')
    if not isinstance(w, dict) or not isinstance(rt, dict):
        reasons.append('worker_or_runtime_metadata_missing'); return result

    def check(ok, field):
        if not ok:
            reasons.append(field)

    def true(obj, key):
        check(obj.get(key) is True, key + ':requires_boolean_true')

    def false(obj, key):
        check(obj.get(key) is False, key + ':requires_boolean_false')

    for key in ('execution_returned', 'finite_partition_retained',
                'numeric_outputs_finite', 'output_shapes_valid'):
        true(w, key)
    check(w.get('status') in ('RETURNED_FINITE', 'RETURNED_FINITE_CONTRACT_FAILED'), 'unknown_or_invalid_worker_status')
    check(isinstance(metadata.get('case_id'), str) and metadata['case_id'].startswith(cid + '_c')
          and metadata['case_id'].endswith('_final'), 'invalid_selected_final_case')
    check(w.get('case_id') == metadata.get('case_id'), 'worker_case_id_mismatch')
    for key in ('n', 'c'):
        check(integer(w.get(key)) and w[key] == condition[key], 'worker_' + key + '_mismatch')
    count = w.get('sample_partition_count')
    count_ok = integer(count) and 1 <= count <= condition['n']
    check(count_ok, 'invalid_sample_partition_count')
    if count_ok:
        result['sample_partition_count'] = int(count)
    check(integer(row.get('occupied_clusters')) and row['occupied_clusters'] == count,
          'sample_count_differs_from_scored_partition')
    check(type(w.get('strict_c')) is bool, 'strict_c:requires_boolean')
    if type(w.get('strict_c')) is bool:
        result['original_strict_c'] = w['strict_c']
    check(type(w.get('benchmark_contract_met')) is bool, 'benchmark_contract_met:requires_boolean')

    kind = rt.get('schema')
    if kind == 'legacy_launcher':
        check(rt.get('status') == 'COMPLETE_ZERO_EXIT_VALID_MARKERS', 'legacy_execution_not_successful')
        check(integer(rt.get('exit_code')) and rt['exit_code'] == 0, 'legacy_nonzero_or_missing_exit')
        true(rt, 'cleanup_complete')
    elif kind == 'ssa_portable' and method == 'SSA':
        check(rt.get('status') == 'EXECUTED', 'ssa_execution_not_successful')
        check(integer(rt.get('matlab_exit_code')) and rt['matlab_exit_code'] == 0, 'ssa_nonzero_or_missing_exit')
        false(rt, 'timed_out'); false(rt, 'class_labels_loaded')
        check('failure' in rt and rt['failure'] is None, 'ssa_execution_failure_or_missing_field')
        true(rt, 'input_and_runtime_hashes_unchanged')
        check(rt.get('worker_status') == w.get('status'), 'ssa_runtime_worker_status_mismatch')
    elif kind == 'pcbg_extension' and method == 'PCBG':
        check(rt.get('status') == 'PROCESS_EXITED', 'pcbg_execution_not_successful')
        check(integer(rt.get('exit_code')) and rt['exit_code'] == 0, 'pcbg_nonzero_or_missing_exit')
        for key in ('started_marker_bound', 'finished_marker_bound', 'finite_partition_retained'):
            true(rt, key)
        check(rt.get('worker_status') == w.get('status'), 'pcbg_runtime_worker_status_mismatch')
    else:
        reasons.append('unknown_runtime_schema')

    if method == 'SSA':
        states = w.get('state_checks')
        if not isinstance(states, dict):
            reasons.append('state_checks_missing')
        else:
            for key in STATE_FIELDS:
                true(states, key)
        views, initial = w.get('views'), w.get('initial_CL_nonzeros')
        check(integer(views) and views > 0, 'invalid_views')
        check(isinstance(initial, list) and integer(views) and len(initial) == views
              and all(finite(value) and value == 0 for value in initial), 'initial_CL_zero_lock_not_verified')
        check(finite(w.get('final_CL_max_abs')) and w['final_CL_max_abs'] == 0,
              'final_CL_zero_lock_not_verified')
        false(w, 'full_class_truth_used'); false(w, 'clustering_scores_computed')
        check(count_ok and count <= condition['c'], 'SSA_count_outside_native_label_domain')
        expected_k = count == condition['c'] if count_ok else None
    elif method == 'PCBG':
        graph = w.get('graph_component_count')
        graph_ok = integer(graph) and graph >= 1
        check(graph_ok, 'invalid_graph_component_count')
        if graph_ok:
            result['graph_component_count'] = int(graph)
        for key in ('native_label_partition_matches_graph', 'initializer_repeated_B_exact'):
            true(w, key)
        for key in ('minimum_P', 'minimum_alpha'):
            check(finite(w.get(key)) and w[key] >= -1e-12, key + ':nonnegative_tolerance_failed')
        check(finite(w.get('alpha_simplex_residual')) and 0 <= w['alpha_simplex_residual'] <= 1e-8,
              'alpha_simplex_residual:invalid_or_above_tolerance')
        expected_k = count == condition['c'] and graph == condition['c'] if count_ok and graph_ok else None
    else:
        reasons.append('unknown_method'); expected_k = None
    check(expected_k is not None and w.get('strict_c') is expected_k, 'strict_c_count_predicate_mismatch')
    if row['outcome_category'] == 'strict_eligible':
        check(row['eligible'] is True and w.get('strict_c') is True
              and w.get('benchmark_contract_met') is True and w.get('status') == 'RETURNED_FINITE',
              'strict_primary_category_inconsistent')
    else:
        check(row['eligible'] is False and w.get('strict_c') is False
              and w.get('benchmark_contract_met') is False
              and w.get('status') == 'RETURNED_FINITE_CONTRACT_FAILED', 'wrong_k_category_inconsistent')
    result['non_k_qualified'] = not reasons
    if row['SAPN_eligible'] is not True:
        reasons.append('original_SAPN_not_eligible')
    result['paired_qualified'] = result['non_k_qualified'] and row['SAPN_eligible'] is True
    return result


def validate_rows(protocol, rows):
    need(protocol.get('schema') == 'sapn-exact-k-only-sensitivity-v1', 'Unknown protocol schema')
    expected = {(method, c['id']): c for method, spec in protocol['methods'].items()
                for c in spec['conditions']}
    need(len(expected) == len(rows) == 84, 'All 84 planned slots are required')
    actual = {(r['method'], r['condition_id']): r for r in rows}
    need(len(actual) == 84 and set(actual) == set(expected), 'Duplicate or missing planned slot')
    for key, condition in expected.items():
        r = actual[key]
        for name in ('dataset', 'seed', 'Q', 'n', 'extension_condition'):
            need(type(r.get(name)) is type(condition[name]) and r[name] == condition[name], 'Seed-row identity mismatch: ' + name)
        need(r.get('outcome_category') in CATEGORIES, 'Unknown outcome category')
        need(type(r.get('eligible')) is bool and type(r.get('SAPN_eligible')) is bool, 'Eligibility must be boolean')
        need(condition['supported'] == (r['outcome_category'] != 'input_unsupported'), 'Supported-input scope mismatch')
        for prefix in ('baseline', 'SAPN'):
            for metric in METRICS:
                field = prefix + '_' + metric
                need(field in r and (r[field] is None or finite(r[field])), 'Missing or nonfinite metric field: ' + field)
    return expected, actual


def aggregate(protocol, rows, metadata):
    conditions, indexed = validate_rows(protocol, rows)
    need(set(metadata) == set(indexed), 'Projection must retain every planned slot')
    output = []
    for key, c in conditions.items():
        r = indexed[key]
        q = qualification(c, r, metadata[key])
        item = {name: r[name] for name in ('method', 'condition_id', 'dataset', 'seed', 'Q',
                'n', 'extension_condition', 'outcome_category', 'eligible', 'SAPN_eligible', 'occupied_clusters')}
        item.update(q)
        item['overlaps_original_two_group_PCBG_scope'] = (r['method'] == 'PCBG' and
            r['dataset'] in ('3Sources', 'CiteSeer') and r['Q'] == 160)
        for metric in METRICS:
            x, y = r['baseline_' + metric], r['SAPN_' + metric]
            item['baseline_' + metric], item['SAPN_' + metric] = x, y
            item['paired_delta_' + metric] = x - y if q['paired_qualified'] and x is not None and y is not None else None
        output.append(item)
    groups = []
    for method in ('PCBG', 'SSA'):
        for dataset in protocol['supported_corpora']:
            for budget in protocol['budgets']:
                selected = [r for r in output if (r['method'], r['dataset'], r['Q']) == (method, dataset, budget)]
                need(len(selected) == 3, 'Group must contain all three planned seeds')
                g = dict(method=method, dataset=dataset, Q=budget, planned_pairs=3,
                         primary_eligible_pairs=sum(r['eligible'] and r['SAPN_eligible'] for r in selected),
                         non_k_qualified_runs=sum(r['non_k_qualified'] for r in selected),
                         qualified_pairs=sum(r['paired_qualified'] for r in selected))
                for metric in METRICS:
                    values = [r['paired_delta_' + metric] for r in selected if r['paired_delta_' + metric] is not None]
                    g[metric + '_pairs'] = len(values)
                    g[metric + '_mean'] = statistics.mean(values) if len(values) == 3 else None
                    g[metric + '_sample_sd'] = statistics.stdev(values) if len(values) == 3 else None
                groups.append(g)
    need(len(groups) == 16, 'Expected 16 supported groups')
    return output, groups


def table(groups, lang):
    need(lang in ('en', 'zh') and len(groups) == 16, 'Wrong table scope/language')
    caption = ('Fixed-selection sensitivity to removing only the native exact-$k$ gate. '
               'All other execution and worker requirements are retained. Differences are baseline minus SAPN-MV '
               '(percentage points), mean $\\pm$ sample SD across three paired seeds; a metric is omitted unless all three pairs are available. '
               'The pairs column counts metadata-qualified pairs. Positive ACC/NMI/ARI and negative BER differences favour the baseline. '
               'This secondary analysis leaves the primary comparison and CV selection unchanged; its two historical PCBG groups overlap the earlier diagnostic.'
               if lang == 'en' else
               '仅移除原生 exact-$k$ 门限的固定选参敏感性分析，其余执行与 worker 资格条款保持不变。'
               '差值为基线减 SAPN-MV（百分点），报告三个配对种子的均值 $\\pm$ 样本标准差；任一指标不足三个完整配对则不报告汇总。'
               'pairs 列为通过元数据资格检查的配对数。ACC/NMI/ARI 正差及 BER 负差有利于基线。'
               '本次次级分析不改变主比较或 CV 选择；其中两个历史 PCBG 组与先前诊断范围重叠。')
    lines = [r'\begin{table}[htbp]', r'\centering', r'\fontsize{9}{11}\selectfont',
             r'\setlength{\tabcolsep}{3pt}', r'\caption{' + caption + '}',
             r'\label{tab:exact-k-sensitivity}', r'\begin{tabular}{llrcccccc}', r'\toprule',
             r'Method & Corpus & $Q$ & pairs & $\Delta$ACC & $\Delta$NMI & $\Delta$ARI & $\Delta$train BER & $\Delta$test BER \\', r'\midrule']
    for g in groups:
        cells = [g['method'], g['dataset'], str(g['Q']), str(g['qualified_pairs']) + '/3']
        for metric in METRICS:
            cells.append('--' if g[metric + '_mean'] is None else
                         f"${100*g[metric + '_mean']:+.2f} \\pm {100*g[metric + '_sample_sd']:.2f}$")
        lines.append(' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    return '\n'.join(lines)
