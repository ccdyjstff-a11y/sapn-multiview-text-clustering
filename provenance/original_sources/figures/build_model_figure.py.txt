"""SAPN model-dependency schematic, drawn from the stated objective.

Contract: show fixed-share supervision, endpoint-dependent coverage, and the
reference penalty without implying a sequential solver or a causal effect.
Python vector artwork; 180 x 130 mm; Arial 8 pt; no empirical observations.
The inputs are repeated in the second tier solely to keep the edges legible.
The source model and frozen experiment implementation are never modified.
"""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path as MplPath

ROOT = Path(__file__).resolve().parent
TOOLS = Path('C:/Users/LENOVO/.codex/skills/nature-figure/scripts')
sys.path.insert(0, str(TOOLS))
from audit_panel_alignment import require_matplotlib_panel_alignment


def main():
    target, qa = ROOT / 'figures', ROOT / 'figure_qa'
    target.mkdir(exist_ok=True); qa.mkdir(exist_ok=True)
    stem = 'fig00_model_v4'
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
        'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic',
        'mathtext.bf': 'Arial:bold', 'axes.linewidth': .65,
    })
    fig, ax = plt.subplots(figsize=(180 / 25.4, 130 / 25.4))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set(xlim=(0, 180), ylim=(0, 130)); ax.axis('off')
    ink, line = '#25313C', '#61717F'
    fills = {'input': '#FFFFFF', 'fixed': '#EEF2F5', 'fit': '#EAF0F5', 'reference': '#DAECE7'}
    boxes, texts, edges = {}, [], []

    def text(key, x, y, label, size=8, bold=False, align='center'):
        t = ax.text(x, y, label, fontsize=size, color=ink, ha=align, va='center',
                    fontweight='bold' if bold else 'normal')
        t.set_gid('t:' + key); texts.append((key, t)); return t

    def box(key, x, y, w, h, labels, family='fixed'):
        p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.1',
                          facecolor=fills[family], edgecolor=line, linewidth=.7)
        p.set_gid('n:' + key); ax.add_patch(p); boxes[key] = (x, y, w, h)
        spacing = 4.2 if len(labels) < 4 else 4.5
        for index, label in enumerate(labels):
            yy = y + h / 2 + (len(labels) - 1) * spacing / 2 - index * spacing
            text(key + ':' + str(index), x + w / 2, yy, label, bold=index == 0)

    def arrow(key, points, double=False, color=line, linewidth=.85):
        path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1))
        patch = FancyArrowPatch(path=path, arrowstyle='<->' if double else '-|>',
                                mutation_scale=7.5, linewidth=linewidth, color=color,
                                capstyle='round', joinstyle='round')
        patch.set_gid('e:' + key); ax.add_patch(patch)
        edges.append({'id': key, 'points_mm': points, 'direction': 'joint coupling' if double else 'dependency'})

    text('tier_fixed', 5, 124, 'Fixed inputs and reference geometry', bold=True, align='left')
    box('views_fixed', 5, 96, 31, 18, ['Aligned views', r'$X_1,\ldots,X_V$'], 'input')
    box('reference', 49, 106, 40, 16, ['Unsupervised warm start', r'Frozen reference $R$'])
    box('graph', 49, 82, 40, 16, ['Unlabeled graph', r'Fixed Laplacian $L$'])
    box('queries_fixed', 5, 64, 31, 18, ['Queried pair answers', r'$\mathcal{P},\mathcal{N}$'], 'input')
    box('coverage', 105, 64, 36, 18, ['Uncovered fraction', r'$u=1-|\mathcal{C}|/n$'])
    box('penalty', 146, 96, 31, 23, ['Reference penalty', r'$R_u(H)$', r'Anchor to $R$', r'Smooth $H$ and $H-R$'], 'reference')
    arrow('views_to_reference', [(36, 109), (49, 114)])
    arrow('views_to_graph', [(36, 101), (42, 101), (42, 90), (49, 90)])
    arrow('endpoints_to_coverage', [(36, 73), (105, 73)])
    text('endpoints_label', 69, 76.5, r'Query endpoints $\mathcal{E}$')
    arrow('graph_to_coverage', [(89, 90), (123, 90), (123, 82)])
    arrow('reference_to_penalty', [(89, 114), (146, 114)])
    arrow('graph_to_penalty', [(89, 90), (138, 90), (138, 104), (146, 104)])
    arrow('coverage_to_penalty', [(141, 73), (151, 73), (151, 96)])
    text('coverage_note', 105, 58, r'$\mathcal{C}$: endpoints and their one-hop graph neighbors', size=7.5)

    text('tier_fit', 5, 58, 'Joint constrained optimization', bold=True, align='left')
    box('views_fit', 5, 32, 31, 18, ['Same aligned views', r'$X_1,\ldots,X_V$'], 'input')
    box('private', 62, 29, 43, 24, ['View-specific factors', r'$U_v,H_v;\quad X_v\approx U_vH_v,\ U_vH$',
                                 'Weighted reconstruction', r'Endpoint simplex on $H_v$'], 'fit')
    box('consensus', 123, 32, 31, 18, ['Consensus', r'Simplex columns $H$'], 'fit')
    box('clusters', 160, 32, 17, 18, ['Clusters', 'Argmax H'], 'input')
    box('signed', 5, 8, 50, 17, ['Same queried pair answers', 'Fixed-share signed losses', 'Outside reconstruction weights'], 'input')
    arrow('views_to_factors', [(36, 41), (62, 41)])
    arrow('joint_coupling', [(105, 41), (123, 41)], double=True)
    text('coupling_label', 114, 45.5, 'Coupling', size=7.5)
    arrow('signed_to_private', [(55, 20), (83.5, 20), (83.5, 29)])
    text('private_share', 69, 24, r'$\eta/V$')
    arrow('signed_to_consensus', [(55, 12), (138.5, 12), (138.5, 32)])
    text('consensus_share', 111, 16, r'$1-\eta$')
    arrow('reference_penalty_to_consensus', [(170, 96), (170, 61), (138.5, 61), (138.5, 50)], linewidth=1.1)
    arrow('consensus_to_clusters', [(154, 41), (160, 41)], linewidth=1.1)
    text('scope', 5, 3, 'Arrows denote dependencies; coupled factors are optimized jointly.', size=7.5, align='left')

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    containment = []
    for key, t in texts:
        owner = key.split(':')[0]
        if owner not in boxes: continue
        x, y, w, h = boxes[owner]
        bbox = t.get_window_extent(renderer).transformed(ax.transData.inverted())
        okay = bbox.x0 >= x + .8 and bbox.x1 <= x + w - .8 and bbox.y0 >= y + .8 and bbox.y1 <= y + h - .8
        containment.append({'text': key, 'box': owner, 'inside_padding': bool(okay), 'bounds_mm': list(bbox.extents)})
        if not okay: raise AssertionError('Label does not fit its allocated box: ' + key)
    require_matplotlib_panel_alignment(fig, json_out=qa / (stem + '.alignment.json'),
        overlay_svg=qa / (stem + '.alignment.svg'), tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    for extension in ['pdf', 'svg']:
        fig.savefig(target / (stem + '.' + extension), metadata={'Creator': 'Matplotlib; SAPN model dependency schematic'})
    # Matplotlib's optional external SVG 1.1 DTD is unnecessary for this fully
    # self-contained vector artwork. Removing it changes no drawn geometry.
    svg_path = target / (stem + '.svg')
    svg = re.sub(r'<!DOCTYPE[^>]*>\s*', '', svg_path.read_text(encoding='utf-8'))
    # SVG pixels and unitless presentation values are both user-space units.
    # Expose the sizes as presentation attributes: the SVG auditor otherwise
    # converts CSS px to points a second time despite this point-sized viewBox.
    def expose_font(match):
        style = match.group(1)
        font = re.search(r'font-size:\s*([0-9.]+)px;?', style)
        if font is None: return match.group(0)
        return 'font-size="' + font.group(1) + '" style="' + style[:font.start()] + style[font.end():] + '"'
    svg = re.sub(r'style="([^"]*)"', expose_font, svg)
    svg_path.write_text(svg, encoding='utf-8')
    fig.savefig(target / (stem + '.png'), dpi=600)
    fig.savefig(target / (stem + '.tiff'), dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)
    audits = {}
    commands = [
        ('source', [str(TOOLS / 'validate_figure.py'), str(Path(__file__)), '--json']),
        ('text', [str(TOOLS / 'audit_pdf_text.py'), str(target / (stem + '.pdf')), '--min-pt', '5', '--json']),
        ('svg', [str(TOOLS / 'audit_svg_schematic.py'), str(svg_path), '--min-pt', '5', '--max-pt', '8', '--json']),
        ('collision', [str(TOOLS / 'audit_figure_collisions.py'), str(target / (stem + '.pdf')),
                       '--json-out', str(qa / (stem + '.collision-audit.json')),
                       '--overlay-pdf', str(qa / (stem + '.collision-audit.pdf'))]),
    ]
    # The supplied SVG audit refers to math in its rotation parser without an
    # import. Supply that standard-library dependency without changing its code.
    bootstrap = ('import math,runpy,sys;sys.path.insert(0,sys.argv.pop(1));'
                 'p=sys.argv.pop(1);sys.argv[0]=p;runpy.run_path(p,init_globals={"math":math},run_name="__main__")')
    for name, arguments in commands:
        run = subprocess.run([sys.executable, '-X', 'utf8', '-c', bootstrap, str(ROOT / 'qa_dependencies'), *arguments],
                             capture_output=True, text=True, encoding='utf-8')
        (qa / (stem + '_' + name + '_stdout.txt')).write_text(run.stdout, encoding='utf-8')
        (qa / (stem + '_' + name + '_stderr.txt')).write_text(run.stderr, encoding='utf-8')
        audits[name] = json.loads((qa / (stem + '.collision-audit.json')).read_text(encoding='utf-8')) if name == 'collision' else json.loads(run.stdout)
        if name != 'collision':
            (qa / (stem + '_' + name + '_audit.json')).write_text(json.dumps(audits[name], indent=2), encoding='utf-8')
        if run.returncode != 0: raise AssertionError('Rendered QA failed: ' + name)
    manuscript = (ROOT / 'paper_en.texpart').read_text(encoding='utf-8')
    model_text = manuscript.split(r'\section{Model}', 1)[1].split(r'\section{Optimization and properties}', 1)[0]
    source = {
        'figure': stem, 'type': 'model dependency schematic; no experimental data',
        'approved_brief': 'Explicit parent task specifies all displayed dependencies and Python/vector output.',
        'width_mm': 180, 'height_mm': 130, 'base_font_pt': 8,
        'nodes': boxes, 'edges': edges, 'text': {key: t.get_text() for key, t in texts},
        'boxes_fit_measured_labels': containment,
        'model_section_sha256': hashlib.sha256(model_text.encode()).hexdigest(),
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'automated_audits': audits,
        'outputs_sha256': {ext: hashlib.sha256((target / (stem + '.' + ext)).read_bytes()).hexdigest()
                           for ext in ['pdf', 'svg', 'png', 'tiff']},
        'mapping': {
            'views_to_reference': 'Unlabeled aligned NMF factors yield the simplex-projected frozen initial consensus R.',
            'views_to_graph': 'Eq. graph: average normalized positive mutual cosine graph Laplacians.',
            'endpoints_to_coverage': 'Eq. reference: C contains actual training-query endpoints and their one-hop neighbors.',
            'graph_to_coverage': 'Neighborhoods are obtained from the support of the fixed L.',
            'R_L_u_to_penalty': 'Eq. reference: ordinary H smoothing, Frobenius proximity to R, and smoothing of H-R.',
            'signed_to_private': 'Eq. objective: b*eta/V times each D(H_v), independent of learned reconstruction weights; endpoints also set the H_v domain including initialization.',
            'signed_to_consensus': 'Eq. objective: b*(1-eta)*D(H), independent of learned reconstruction weights.',
            'joint_coupling': 'Eq. energy: the H_v-H penalty is jointly optimized; the common reconstruction also uses U_v H.',
            'consensus_to_clusters': 'The cluster output is the columnwise argmax of H.'
        },
        'scope': [
            'The same views and queried pair answers are repeated in the second tier for legibility, not separate fits.',
            'Shares label allocation factors; the common multiplier b and relation coefficient are omitted from the diagram.',
            'Soft orthogonality, basis shrinkage, weight updates, and individual solver stages are omitted; this is not the complete objective or an execution-order flowchart.',
            'No empirical observation, estimated effect, or guarantee of improved accuracy is encoded.',
            'Style-only inheritance from the companion analytic figure; all schematic geometry is new.'
        ],
        'caption_en': 'Model dependencies in SAPN. Unlabeled warm starts initialize H, H_v, and U_v and provide the frozen reference R; aligned views also determine the fixed graph L. Queried endpoints and graph neighborhoods determine the uncovered fraction u. The reference penalty, defined by R, L, and u, acts on H. Fixed supervision shares act on H and H_v independently of learned reconstruction weights; the common multiplier b is omitted from the displayed shares. The endpoint simplex domain of H_v applies from initialization onward. The double-headed arrow denotes joint coupling, and H yields clusters by columnwise argmax. Inputs are repeated for readability. Arrows indicate dependencies, not an execution sequence or causal effects.'
    }
    (qa / (stem + '_contract.json')).write_text(json.dumps(source, ensure_ascii=False, indent=2), encoding='utf-8')
    print(target / (stem + '.pdf'))


if __name__ == '__main__':
    main()
