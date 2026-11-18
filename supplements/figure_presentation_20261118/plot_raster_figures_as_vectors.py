"""Re-export two historical plots as vectors, using original public evidence.

No raster image is enlarged or traced. The original drawing functions operate
on released JSON, independently cross-checked against the public plot CSVs.
The original SVG plot rectangles and complete scientific paths are hash-bound
in the companion manifest. Text is enlarged to 9 pt; no data mark is changed.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import ast
import csv
import json
import re
import sys
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
from plot_si_readable_fonts import scientific_trace, sha, dump

MANIFEST = Path(__file__).with_name('RASTER_VECTOR_SOURCE_BINDING.json')
NS = {'s': 'http://www.w3.org/2000/svg'}
XLINK = '{http://www.w3.org/1999/xlink}href'
DATA = ['3Sources', 'BBC', 'Reuters', 'Cora', 'CiteSeer']


def csv_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def svg_scientific_paths(path):
    tree = ET.parse(path)
    root = tree.getroot()
    definitions = {x.get('id'): x for x in root.iter() if x.get('id')}
    result = []
    for axes in root.findall('.//s:g', NS):
        if not axes.get('id', '').startswith('axes_'):
            continue
        groups = []
        for child in axes:
            if not child.get('id', '').startswith(('line2d_', 'PathCollection_')):
                continue
            marks = []
            for item in child.iter():
                tag = item.tag.rsplit('}', 1)[-1]
                if tag == 'path' and not item.get('id'):
                    marks.append(dict(kind='path', d=item.get('d'), style=item.get('style', '')))
                if tag == 'use':
                    target = definitions[item.get(XLINK).lstrip('#')]
                    marks.append(dict(kind='use', d=target.get('d'), x=float(item.get('x', '0')),
                                      y=float(item.get('y', '0')), style=item.get('style', '')))
            groups.append(marks)
        result.append(groups)
    return result


def compare_paths(old, new):
    maxima = []
    count = 0
    if len(old) != len(new):
        raise ValueError('Original and new SVG panel counts differ')
    pattern = r'[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?'
    for old_panel, new_panel in zip(old, new):
        if len(old_panel) != len(new_panel):
            raise ValueError('Scientific group count changed')
        for old_group, new_group in zip(old_panel, new_panel):
            if len(old_group) != len(new_group):
                raise ValueError('Scientific mark count changed')
            for a, b in zip(old_group, new_group):
                count += 1
                if a['kind'] != b['kind'] or a['style'] != b['style']:
                    raise ValueError('Scientific mark type or style changed')
                av, bv = [float(x) for x in re.findall(pattern, a['d'])], [float(x) for x in re.findall(pattern, b['d'])]
                if re.sub(pattern, '#', a['d']) != re.sub(pattern, '#', b['d']):
                    raise ValueError('Scientific vector path command changed')
                if a['kind'] == 'use':
                    av += [a['x'], a['y']]
                    bv += [b['x'], b['y']]
                error = max([0.] + [abs(x-y) for x, y in zip(av, bv)])
                if len(av) != len(bv) or error > 2e-5:
                    raise ValueError('Scientific SVG coordinates changed: ' + str(error))
                maxima.append(error)
    return dict(scientific_marks=count, maximum_coordinate_error_pt=max(maxima),
                coordinate_tolerance_pt=2e-5, styles_symbols_path_commands_identical=True)


def load_functions(source, names, namespace, adapt_location=False):
    selected = []
    for node in ast.parse(source.read_text(encoding='utf-8-sig')).body:
        if not isinstance(node, ast.FunctionDef) or node.name not in names:
            continue
        if adapt_location and node.name == 'location':
            # The archival private audit file is not in the public release.
            # Replace only its existence/status guards with the independently
            # checked released JSON/CSV identities; do not synthesize an audit.
            if not (isinstance(node.body[1], ast.If) and isinstance(node.body[2], ast.Assert)
                    and 'INDEPENDENT_FULL_AUDIT.json' in ast.unparse(node.body[1])
                    and 'INDEPENDENT_FULL_AUDIT.json' in ast.unparse(node.body[2])):
                raise ValueError('Historical audit-guard adapter did not match')
            node.body = node.body[:1] + node.body[3:]
        selected.append(node)
    if {n.name for n in selected} != set(names):
        raise ValueError('Unexpected original function inventory')
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])),
                 str(source), 'exec'), namespace)


def check_inputs(repo, binding):
    for item in binding['inputs']:
        if sha(repo / item['path']) != item['sha256']:
            raise ValueError('Public input hash mismatch: ' + item['path'])
    if sha(Path(__file__).with_name('plot_si_readable_fonts.py')) != binding['helper_sha256']:
        raise ValueError('Artist-trace helper changed')
    location = json.loads((repo / 'evidence/original/coverage_location_control_v2/summary.json').read_text())
    for row in csv_rows(repo / 'figures/source_data/fig08_reference_strengths.csv'):
        cell = next(c for c in location['cells'] if (c['dataset'], c['Q'], c['method']) ==
                    (row['dataset'], int(row['Q']), 'coverage'))
        i = int(row['position_in_source_block'])
        if (cell['u_expected'][i], cell['u_actual'][i]) != (float(row['u_expected']), float(row['u_actual'])):
            raise ValueError('Location-strength JSON/CSV mismatch')
    for row in csv_rows(repo / 'figures/source_data/fig08_paired_ACC.csv'):
        cell = next(c for c in location['paired'] if (c['dataset'], c['Q']) ==
                    (row['dataset'], int(row['Q'])))
        if cell['ACC']['deltas_pp'][int(row['position_in_source_block'])] != float(row['paired_ACC_difference_pp']):
            raise ValueError('Location paired-delta JSON/CSV mismatch')
        if cell['ACC']['mean_pp'] != float(row['recorded_group_mean_pp']):
            raise ValueError('Location group-mean JSON/CSV mismatch')
    if len(location['paired']) != 10:
        raise ValueError('Unexpected location coverage')
    return location


def export(fig, stem, out, binding, align, full_helper):
    original = binding['figures'][stem]
    width, height = original['size_pt']
    # Restore the exact canvas/plot rectangles from the old vector. Figure 08
    # originally used a tight bounding box; this avoids recomputing that crop
    # after font changes and preserves all scientific page coordinates.
    fig.set_size_inches(width/72, height/72)
    fig.set_layout_engine(None)
    for ax, (x, y, w, h) in zip(fig.axes, original['axes_rectangles_pt']):
        ax.set_position([x/width, (height-y-h)/height, w/width, h/height])
    fig.canvas.draw()
    before = scientific_trace(fig)
    texts = [(t, t.get_text(), t.get_fontsize()) for t in fig.findobj(match=Text)]
    for text, label, size in texts:
        if size < 9:
            text.set_fontsize(9)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'], 'font.size': 9,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig.canvas.draw()
    after = scientific_trace(fig)
    if before != after or any(t.get_text() != label for t, label, size in texts):
        raise ValueError('Typography changed scientific artists or text content')
    # Titles contain the original (a)/(b) labels. Measure them directly because
    # the generic helper detects standalone bold panel letters only.
    titles = [ax.get_title(loc='left') for ax in fig.axes]
    if not (titles[0].startswith('(a)') and titles[1].startswith('(b)')):
        raise ValueError('Original title-integrated panel labels missing')
    title_y = [ax._left_title.get_window_extent().y1 / fig.dpi * 72 for ax in fig.axes]
    if abs(title_y[0]-title_y[1]) > 1.5:
        raise ValueError('Title panel labels are not aligned')
    rects = original['axes_rectangles_pt']
    if abs(rects[0][1]-rects[1][1]) > 1.5 or abs(rects[0][3]-rects[1][3]) > 1.5:
        raise ValueError('Comparable panels have different top or bottom edges')
    unequal = stem == 'fig08_location_control'
    options = dict(exemptions=[dict(panels=['a', 'b'], checks=['panel-width'],
        reason='Original reference-strength and paired-ACC panels retain their 1:1.7 width ratio')] if unequal else [])
    if unequal and not full_helper:
        # The small public checker only supports equal-width inferred rows.
        # Exact top/bottom/height checks above retain the original unequal row.
        options['row_groups'] = []
    alignment = align(fig, json_out=out/(stem+'.alignment.json'),
        overlay_svg=out/(stem+'.alignment.svg'), strict=True, tolerance_pt=1.5,
        gutter_tolerance_pt=1.5, require_panel_labels=False, **options)
    fig.savefig(out/(stem+'.pdf'))
    fig.savefig(out/(stem+'.svg'))
    fig.savefig(out/(stem+'.png'), dpi=600)
    fig.savefig(out/(stem+'.tiff'), dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
    fig.savefig(out/(stem+'.page166mm.png'), dpi=360*166/(width/72*25.4))
    svg_qa = compare_paths(original['scientific_svg_paths'], svg_scientific_paths(out/(stem+'.svg')))
    dump(out/(stem+'.geometry.json'), after)
    dump(out/(stem+'.font_ledger.json'), [dict(text=label, before_pt=size, after_pt=t.get_fontsize())
                                         for t, label, size in texts if label and t.get_fontsize()!=size])
    minimum = min(t.get_fontsize() for t in fig.findobj(match=Text) if t.get_visible() and t.get_text().strip())
    result = dict(figure=stem, scientific_trace_exact=True, original_svg=svg_qa,
                  canvas_width_mm=width/72*25.4, body_font_source_pt=minimum,
                  body_font_at_166mm_pt=minimum*166/(width/72*25.4),
                  alignment=alignment['verdict'], panel_title_y_difference_pt=abs(title_y[0]-title_y[1]),
                  original_unequal_panel_widths_retained=unequal)
    plt.close(fig)
    return result


def main():
    if not __debug__:
        raise RuntimeError('Do not run with -O/-OO')
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--skill-scripts', type=Path)
    args = ap.parse_args()
    repo, out = args.release_root.resolve(), args.output.resolve()
    if out.exists() or out == repo or repo in out.parents:
        raise ValueError('Use a new output directory outside the release')
    binding = json.loads(MANIFEST.read_text(encoding='utf8'))
    check_inputs(repo, binding)
    out.mkdir(parents=True)
    sys.path.insert(0, str(args.skill_scripts or repo/'figures'))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    results = []
    namespace = dict(np=np, plt=plt, Line2D=Line2D, MaxNLocator=MaxNLocator,
        DATA=DATA, SEEDS=[23,24,25], DISPLAY={'Reuters':'Reuters-local'},
        COLORS={'SAPN-MV':'#0072B2','SAPN-EQ':'#D55E00'},
        read=lambda name: json.loads((repo/'evidence/original'/name).read_text()))
    def save_noise(fig, stem, sources, records, *unused):
        rows = csv_rows(repo/'figures/source_data/fig06_paired_changes.csv')
        if len(rows) != 60 or len(records) != 20:
            raise ValueError('Wrong-answer source coverage changed')
        for row in rows:
            rec = next(r for r in records if (r['dataset'],r['method'],r['wrong_fraction']) ==
                       (row['dataset'],row['method'],float(row['wrong_fraction'])))
            if (rec['paired_deltas_pp'][rec['seeds'].index(int(row['seed']))],rec['mean_pp']) != (
                    float(row['paired_ACC_change_pp']),float(row['recorded_group_mean_pp'])):
                raise ValueError('Wrong-answer plotted values differ from public CSV')
        results.append(export(fig,stem,out,binding,require_matplotlib_panel_alignment,bool(args.skill_scripts)))
    namespace['save'] = save_noise
    load_functions(repo/'figures/historical/build_evidence_figures.py',
                   ['one','style_axes','name','wrong_answer_figure'],namespace)
    plt.rcdefaults()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':8.2,
        'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7.4,'legend.fontsize':7.2,
        'axes.linewidth':.6,'lines.linewidth':1,'svg.fonttype':'path',
        'svg.hashsalt':'SAPN-submission-revision-v3','pdf.fonttype':42,'ps.fonttype':42,
        'savefig.facecolor':'white','axes.unicode_minus':True})
    namespace['wrong_answer_figure']()
    namespace = dict(np=np,plt=plt,MAIN=repo/'evidence/original',DATA=DATA,
        COLORS=['#0072B2','#D55E00','#009E73','#7A5195','#AA4499'],
        read=lambda path: json.loads(path.read_text()),
        save=lambda fig,stem,*unused:results.append(export(fig,stem,out,binding,
            require_matplotlib_panel_alignment,bool(args.skill_scripts))))
    load_functions(repo/'figures/historical/build_new_figures.py',['location'],namespace,adapt_location=True)
    plt.rcdefaults()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,
        'axes.titlesize':9,'legend.fontsize':7,'svg.fonttype':'path',
        'axes.spines.top':False,'axes.spines.right':False})
    namespace['location']()
    check_inputs(repo,binding)
    dump(out/'RASTER_VECTOR_REDRAW_QA.json',dict(status='RENDERED_EXTERNAL_QA_PENDING',
        plotter_sha256=sha(__file__),manifest_sha256=sha(MANIFEST),
        no_fit_selection_scoring_or_raster_resampling=True,public_inputs_verified=True,
        source_csv_rows={'fig06':60,'fig08_strengths':30,'fig08_paired_ACC':30},
        figures=results))
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
