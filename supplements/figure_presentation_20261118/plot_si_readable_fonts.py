"""Typography-only redraw of supplementary figures 13, 15, 16 and 17.

Reuse the release's study-authored drawing functions and public, complete CSVs.
The adapter has no fit, selection, label-scoring or native-output dependency.
Only text below 9 pt is enlarged. Scientific artists and axes are compared
before and after the change. Missing values and original sample SD are retained.
Run: python -B plot_si_readable_fonts.py --release-root RELEASE --output NEW_DIR
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import ast
import csv
import hashlib
import json
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator

DATASETS = ['3Sources', 'BBC', 'Reuters', 'Cora', 'CiteSeer', '20NG', 'AGNews']
STEMS = ['fig13_reference_filter_risk', 'fig15_supervision_repetition',
         'fig16_signed_ablation_repetition', 'fig17_reference_repetition']
SOURCE_MAP = Path(__file__).with_name('SI_FONT_SOURCE_BINDING.json')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                  allow_nan=False) + '\n', encoding='utf8')


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def optional_float(value):
    return None if value in ('', 'None') else float(value)


def boolean(value):
    if value not in ('True', 'False'):
        raise ValueError('Unexpected boolean in frozen CSV')
    return value == 'True'


def typed_points(rows):
    result = []
    for original in rows:
        row = dict(original)
        for key in ('seed', 'Q'):
            row[key] = int(row[key])
        for key in ('eligible', 'plotted'):
            row[key] = boolean(row[key])
        for key in ('value_percent', 'full_metric_fraction',
                    'comparison_metric_fraction', 'delta_pp'):
            if key in row:
                row[key] = optional_float(row[key])
        result.append(row)
    return result


def verify_inputs(repo):
    binding = json.loads(SOURCE_MAP.read_text(encoding='utf8'))
    for record in binding['inputs']:
        if sha(repo / record['path']) != record['sha256']:
            raise ValueError('Source identity changed: ' + record['path'])
    return binding


def select_functions(path, names, namespace, adapt15=False, truncate13=False):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    selected = []
    adaptation_count = 0
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in names:
            continue
        if adapt15 and node.name == 'draw15':
            # The public figure CSV already stores ACC in percent, whereas the
            # historical loader supplied fractions. Avoid a lossy /100*100
            # round trip: replace exactly this input-unit expression, keeping
            # every plotting call, mark and coordinate convention unchanged.
            class PercentInput(ast.NodeTransformer):
                def visit_Assign(self, item):
                    nonlocal adaptation_count
                    expected = 'value = None if value is None else 100 * value'
                    if ast.unparse(item) == expected:
                        adaptation_count += 1
                        return ast.copy_location(ast.parse(
                            "value = record['metrics_percent']['ACC']").body[0], item)
                    return self.generic_visit(item)
            node = PercentInput().visit(node)
        if truncate13 and node.name == 'build':
            # Stop just before the original draw/export; the wrapper performs
            # the same final alignment gate after its text-only adjustment.
            cut = next(i for i, item in enumerate(node.body)
                       if ast.unparse(item) == 'fig.canvas.draw()')
            node.body = node.body[:cut] + [ast.Return(value=ast.Tuple(
                elts=[ast.Name(id=n, ctx=ast.Load()) for n in ('fig', 'rows', 'checks')],
                ctx=ast.Load()))]
        selected.append(node)
    if {node.name for node in selected} != set(names):
        raise ValueError('Unexpected public plotting function inventory')
    if adapt15 and adaptation_count != 1:
        raise ValueError('Percent-unit input adapter did not match exactly once')
    code = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(code, str(path), 'exec'), namespace)


def array(value):
    a = np.asarray(value, dtype=float)
    # NaN segments mean a missing aggregate; preserve their masks explicitly.
    return {'shape': list(a.shape), 'finite_values': np.nan_to_num(a).tolist(),
            'nan_mask': np.isnan(a).tolist(), 'inf_mask': np.isinf(a).tolist()}


def scientific_trace(fig):
    axes = []
    for ax in fig.axes:
        lines = [dict(xy=array(line.get_xydata()), color=line.get_color(),
                      marker=line.get_marker(), linestyle=line.get_linestyle(),
                      linewidth=line.get_linewidth(), markersize=line.get_markersize(),
                      alpha=line.get_alpha()) for line in ax.lines]
        collections = []
        for item in ax.collections:
            value = dict(kind=type(item).__name__, offsets=array(item.get_offsets()),
                         facecolors=array(item.get_facecolors()),
                         edgecolors=array(item.get_edgecolors()),
                         linewidths=array(item.get_linewidths()), alpha=item.get_alpha(),
                         paths=[array(p.vertices) for p in item.get_paths()])
            if hasattr(item, 'get_segments'):
                value['segments'] = [array(s) for s in item.get_segments()]
            if hasattr(item, 'get_sizes'):
                value['sizes'] = array(item.get_sizes())
            collections.append(value)
        axes.append(dict(position=list(ax.get_position().bounds),
                         xlim=list(ax.get_xlim()), ylim=list(ax.get_ylim()),
                         xscale=ax.get_xscale(), yscale=ax.get_yscale(),
                         xticks=array(ax.get_xticks()), yticks=array(ax.get_yticks()),
                         xlabels=[t.get_text() for t in ax.get_xticklabels()],
                         ylabels=[t.get_text() for t in ax.get_yticklabels()],
                         lines=lines, collections=collections))
    return dict(size_inches=list(fig.get_size_inches()), axes=axes)


def save_font_revision(fig, stem, out, alignment, rows, summaries, args):
    fig.canvas.draw()
    old_trace = scientific_trace(fig)
    plt.rcParams.update({'font.family': 'sans-serif',
                         'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
                         'font.size': 9, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    old_text = [(t, t.get_text(), t.get_fontsize(), t.get_position())
                for t in fig.findobj(match=Text)]
    changed = []
    for text, label, size, position in old_text:
        if size < 9:
            text.set_fontsize(9)
            if label:
                changed.append(dict(text=label, before_pt=size, after_pt=9))
    fig.canvas.draw()
    new_trace = scientific_trace(fig)
    if old_trace != new_trace:
        raise ValueError('Font edit changed scientific artist geometry')
    automatic_positions = []
    for text, label, size, position in old_text:
        if text.get_text() != label:
            raise ValueError('Font edit changed a text label')
        if text.get_position() != position:
            automatic_positions.append(dict(text=label, before=list(position),
                                            after=list(text.get_position())))
    result = alignment(fig, json_out=out / (stem + '.alignment.json'),
        overlay_svg=out / (stem + '.alignment.svg'), tolerance_pt=1.5,
        gutter_tolerance_pt=1.5, strict=True, require_panel_labels=True,
        axes=[ax for ax in fig.axes if ax.axison])
    # Preserve the original 180-mm canvas and axes rectangles: no tight bbox.
    fig.savefig(out / (stem + '.pdf'))
    fig.savefig(out / (stem + '.svg'))
    fig.savefig(out / (stem + '.png'), dpi=600)
    fig.savefig(out / (stem + '.tiff'), dpi=600,
                pil_kwargs={'compression': 'tiff_lzw'})
    # Final page-width preview, rendered with the same Matplotlib backend.
    fig.savefig(out / (stem + '.page166mm.png'), dpi=360 * 166 / 180)
    dump(out / (stem + '.geometry.json'), new_trace)
    dump(out / (stem + '.font_ledger.json'), dict(font_changes=changed,
        matplotlib_automatic_text_position_updates=automatic_positions))
    visible = [t.get_fontsize() for t in fig.findobj(match=Text)
               if t.get_visible() and t.get_text().strip()]
    record = dict(figure=stem, source_rows=len(rows), source_aggregates=len(summaries),
                  exclusions=0, body_text_minimum_source_pt=min(visible),
                  body_text_minimum_at_166mm_pt=min(visible) * 166 / 180,
                  mathematical_scripts_may_be_smaller=stem == STEMS[0],
                  text_strings_unchanged=True, manual_text_anchor_edits=0,
                  automatic_text_position_updates=len(automatic_positions),
                  scientific_artists_exactly_equal=True,
                  axes_rectangles_limits_ticks_symbols_unchanged=True,
                  alignment=result['verdict'], font_changes=len(changed))
    plt.close(fig)
    return record


def build15_inputs(repo):
    raw = read_csv(repo / 'tables/figures/fig15_supervision_repetition_source.csv')
    points = typed_points(raw)
    summaries = read_csv(repo / 'tables/figures/fig15_supervision_repetition_summary.csv')
    records = [dict(dataset=p['dataset'], seed=p['seed'], Q=p['Q'], method=p['method'],
                    eligible=p['eligible'], status=p['status'],
                    prediction_sha256=p['prediction_sha256'],
                    metrics_raw={'ACC': None}, metrics_percent={'ACC': p['value_percent']})
               for p in points]
    cells = []
    for s in summaries:
        rr = [p for p in points if (p['dataset'], p['Q'], p['method']) ==
              (s['dataset'], int(s['Q']), s['method'])]
        if [p['seed'] for p in rr] != list(range(201, 208)):
            raise ValueError('Incomplete seven-seed public figure cell')
        mean, sd = optional_float(s['mean_percent']), optional_float(s['sd_percent'])
        eligible = sum(p['eligible'] for p in rr)
        if eligible != int(s['eligible']):
            raise ValueError('Public point/summary eligibility mismatch')
        if mean is not None:
            values = [p['value_percent'] for p in rr]
            if eligible != 7 or not np.allclose([mean, sd],
                    [np.mean(values), np.std(values, ddof=1)], rtol=0, atol=1e-12):
                raise ValueError('Public mean/sample-SD mismatch')
        cells.append(dict(dataset=s['dataset'], Q=int(s['Q']), method=s['method'],
                          planned=int(s['planned']), eligible=eligible,
                          seeds=[p['seed'] for p in rr], metrics={'ACC': dict(
                          primary_mean_percent=mean, primary_sd_percent=sd)}))
    if len(points) != 315 or len(cells) != 45:
        raise ValueError('Unexpected public supervision coverage')
    return {'records': records, 'cells': cells}, points, summaries


def build_effect_inputs(repo, stem):
    points = typed_points(read_csv(repo / ('tables/figures/' + stem + '_source.csv')))
    raw = read_csv(repo / ('tables/figures/' + stem + '_summary.csv'))
    cases = {}
    for s in raw:
        key = (s['dataset'], int(s['Q']), s['comparison'], s['metric'])
        pp = [p for p in points if (p['dataset'], p['Q'], p['comparison'], p['metric']) == key]
        if [p['seed'] for p in pp] != list(range(201, 208)):
            raise ValueError('Incomplete paired-seed public figure cell')
        direction = -1 if s['direction'] == 'Comparison minus full' else 1
        for p in pp:
            expected = None if p['full_metric_fraction'] is None or p['comparison_metric_fraction'] is None else direction * 100 * (p['full_metric_fraction'] - p['comparison_metric_fraction'])
            if (expected is None) != (p['delta_pp'] is None) or (expected is not None and
                    abs(expected - p['delta_pp']) > 1e-12):
                raise ValueError('Paired direction or source arithmetic mismatch')
            if p['direction'] != s['direction']:
                raise ValueError('Mixed effect directions')
        mean, sd = optional_float(s['mean_pp']), optional_float(s['sd_pp'])
        if mean is not None:
            if not all(p['eligible'] and p['delta_pp'] is not None for p in pp):
                raise ValueError('Incomplete aggregate must remain missing')
            values = [p['delta_pp'] for p in pp]
            if not np.allclose([mean, sd], [np.mean(values), np.std(values, ddof=1)],
                               rtol=0, atol=1e-12):
                raise ValueError('Public mean/sample-SD mismatch')
        cases[key] = dict(dataset=key[0], Q=key[1], comparison=key[2], metric=key[3],
                          points=pp, planned=int(s['planned']), eligible=int(s['eligible']),
                          mean_pp=mean, sd_pp=sd)
        if cases[key]['eligible'] != sum(p['eligible'] for p in pp):
            raise ValueError('Pair eligibility count mismatch')
    expected = (525, 75) if stem == STEMS[2] else (630, 90)
    if (len(points), len(cases)) != expected:
        raise ValueError('Unexpected public paired-effect coverage')
    return cases, points, raw


def main():
    if not __debug__:
        raise RuntimeError('Do not disable source assertions with -O or -OO')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--skill-scripts', type=Path,
                        help='Optional independently installed full alignment QA')
    args = parser.parse_args()
    repo, out = args.release_root.resolve(), args.output.resolve()
    if out.exists():
        raise ValueError('Output must be a new directory')
    if out == repo or repo in out.parents:
        raise ValueError('Do not write into the release')
    binding = verify_inputs(repo)
    out.mkdir(parents=True)
    sys.path.insert(0, str(args.skill_scripts or (repo / 'figures')))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    namespace = dict(Path=Path, np=np, plt=plt, Line2D=Line2D, MaxNLocator=MaxNLocator,
                     sys=sys, csv=csv, hashlib=hashlib, json=json, ROOT=out,
                     STRENGTHS=[.1, .5, .9], COLORS=['#0072B2', '#D55E00', '#009E73'])
    select_functions(repo / 'figures/plot_reference_risk.py',
                     ['response', 'threshold', 'numerical_checks', 'build'],
                     namespace, truncate13=True)
    fig, analytic_rows, checks = namespace['build'](SimpleNamespace(output_dir=out,
        skill_scripts=repo / 'figures',
        original_csv=repo / 'tables/figures/fig13_reference_filter_source.csv'))
    results = [save_font_revision(fig, STEMS[0], out, require_matplotlib_panel_alignment,
                                 analytic_rows, [], args)]
    dump(out / 'fig13_analytic_checks.json', checks)
    namespace = dict(np=np, plt=plt, Line2D=Line2D, MaxNLocator=MaxNLocator,
        WIDTH_MM=180, DEVELOPMENT=DATASETS[:5], METRICS=['ACC', 'NMI', 'ARI'],
        NAMES={d: 'Reuters-local' if d == 'Reuters' else d for d in DATASETS},
        METHODS=['SAPN-MV', 'SAPN-EQ', 'CMVNMF'],
        COLORS={'SAPN-MV': '#0072B2', 'SAPN-EQ': '#D55E00', 'CMVNMF': '#009E73',
                'graph-only-CV': '#0072B2', 'fixed-CV': '#D55E00',
                'without_correction_graph': '#009E73'},
        REMOVALS=['without_consensus_pair', 'without_private_pair', 'without_all_pair',
                  'without_ML', 'without_CL'],
        CONTROLS=['graph-only-CV', 'fixed-CV', 'without_correction_graph'],
        CONTROL_LABELS=['Full − graph only', 'Full − fixed-CV', 'Full − no correction graph'])
    select_functions(repo / 'figures/historical/build_extension_figures.py',
        ['setup', 'panel_label', 'statistics_legend', 'record_lookup', 'effect_extent',
         'effect_marks', 'effects_source', 'draw15', 'draw16', 'draw17'], namespace, adapt15=True)
    namespace['setup']()
    group, expected_points, expected_summaries = build15_inputs(repo)
    expected_case_points = expected_points

    def export_adapter(fig, stem, points, summaries, evidence):
        if points != expected_case_points:
            raise ValueError('Drawing omitted, reordered or altered frozen source rows')
        normalized = []
        for original in expected_summaries:
            row = dict(original)
            for key in ('Q', 'planned', 'eligible'):
                row[key] = int(row[key])
            for key in ('mean_percent', 'sd_percent', 'mean_pp', 'sd_pp'):
                if key in row:
                    row[key] = optional_float(row[key])
            normalized.append(row)
        if summaries != normalized:
            raise ValueError('Drawing changed original complete-cell summaries')
        return save_font_revision(fig, stem, out, require_matplotlib_panel_alignment,
                                  points, summaries, args)

    namespace['export_and_audit'] = export_adapter
    results.append(namespace['draw15']({'new_repetition': group}))
    for stem, name in zip(STEMS[2:], ['draw16', 'draw17']):
        namespace['setup']()
        cases, expected_case_points, expected_summaries = build_effect_inputs(repo, stem)

        def frozen_paired_case(group, lookup, ds, Q, comparison, metric, direction=1):
            case = cases[ds, Q, comparison, metric]
            expected_direction = 'Full minus comparison' if direction == 1 else 'Comparison minus full'
            if any(p['direction'] != expected_direction for p in case['points']):
                raise ValueError('Drawing requests a different contrast direction')
            return case

        namespace['paired_case'] = frozen_paired_case
        results.append(namespace[name]({'new_repetition': {'records': []}}))
    verify_inputs(repo)
    dump(out / 'SI_FONT_REDRAW_QA.json', dict(status='RENDERED_EXTERNAL_QA_PENDING',
        script_sha256=sha(__file__), source_binding_sha256=sha(SOURCE_MAP),
        input_sources=binding['inputs'], matplotlib=matplotlib.__version__, numpy=np.__version__,
        no_fit_selection_or_scoring=True, no_new_hypothesis_test=True,
        source_csvs_unchanged=True, original_figure14_untouched=True, figures=results))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
