"""Paired ACC changes for the prespecified v5 mechanism experiment.

Figure question: does fixed private supervision help, and does correction
smoothing retain value after the simpler model is independently selected?
Four panels show the two independently selected controls at two query budgets.
The additional fixed-coefficient control remains in the full table and source
CSV. All planned seeds appear; failed outputs never enter primary means.
Python/Matplotlib; 183 mm wide; mean and sample SD, not confidence intervals.
"""
from pathlib import Path
import argparse, csv, hashlib, json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from evidence_gate import require_mechanisms

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT.parent / 'final_improvement_v5' / 'mechanisms'
DATA = ['3Sources', 'BBC', 'Reuters', 'Cora', 'CiteSeer']
CONTROLS = ['private-coupled-CV', 'private-coupled-at-full-lambda', 'no-correction-CV']
PLOT_CONTROLS = ['private-coupled-CV', 'no-correction-CV']
TITLES = ['Coupled: independent selection', 'Coupled: full-model coefficient',
          'No correction: independent selection']


def load_verified_pairs():
    require_mechanisms()
    ready = EVIDENCE / 'RESULTS_READY.json'
    if not ready.exists():
        raise RuntimeError('Numerical and result verification must finish before rendering')
    summary = json.loads((EVIDENCE / 'summary.json').read_text(encoding='utf-8'))
    with (EVIDENCE / 'per_run.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    lookup = {(r['dataset'], int(r['seed']), int(r['budget']), r['method']): r for r in rows}
    assert len(rows) == len(lookup) == 120
    result = []
    for item in summary['comparisons']:
        ds, q = item['dataset'], item['budget']
        control = item['comparison'].removeprefix('full-CV minus ')
        assert control in CONTROLS
        assert [p['seed'] for p in item['pairs']] == [301, 302, 303]
        calculated = []
        for pair in item['pairs']:
            seed = pair['seed']
            a, b = lookup[ds, seed, q, 'full-CV'], lookup[ds, seed, q, control]
            eligible = a['converged'].lower() == 'true' and b['converged'].lower() == 'true'
            assert eligible == pair['eligible']
            delta = None if not a['ACC'] or not b['ACC'] else 100 * (float(a['ACC']) - float(b['ACC']))
            if delta is not None:
                assert abs(delta - pair['delta']['ACC']) < 1e-10
            calculated.append(dict(dataset=ds, budget=q, control=control, seed=seed,
                                   eligible=eligible, delta_ACC_pp=delta))
        if item['complete']:
            assert all(p['eligible'] for p in calculated)
            values = [p['delta_ACC_pp'] for p in calculated]
            assert abs(np.mean(values) - item['statistics']['ACC']['mean_pp']) < 1e-10
            assert abs(np.std(values, ddof=1) - item['statistics']['ACC']['sd_pp']) < 1e-10
        result.extend(calculated)
    assert len(result) == 90
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alignment-module', type=Path, required=True,
                        help='Directory containing audit_panel_alignment.py')
    args = parser.parse_args()
    sys.path.insert(0, str(args.alignment_module))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    records = load_verified_pairs()
    out = ROOT / 'figures'
    out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'], 'font.size': 7,
        'axes.labelsize': 7, 'axes.titlesize': 7, 'xtick.labelsize': 6.5,
        'ytick.labelsize': 7, 'legend.fontsize': 6.5, 'pdf.fonttype': 42,
        'ps.fonttype': 42, 'svg.fonttype': 'none', 'axes.linewidth': .55,
        'xtick.major.width': .5, 'ytick.major.width': .5, 'savefig.dpi': 600})
    fig, axes = plt.subplots(2, 2, figsize=(183 / 25.4, 110 / 25.4), sharey=True)
    fig.subplots_adjust(left=.17, right=.985, bottom=.13, top=.89, wspace=.24, hspace=.60)
    colors = ['#346A8A', '#407B6B']
    spans = {}
    for control in PLOT_CONTROLS:
        all_values = [r['delta_ACC_pp'] for r in records if r['control']==control and r['delta_ACC_pp'] is not None]
        span = max(1., max(abs(v) for v in all_values))
        means_and_sd=[]
        for q in [160, 400]:
            for ds in DATA:
                cell = [r for r in records if r['dataset'] == ds and r['budget'] == q and r['control'] == control]
                if all(r['eligible'] for r in cell):
                    values = [r['delta_ACC_pp'] for r in cell]
                    means_and_sd.append(abs(np.mean(values)) + np.std(values, ddof=1))
        spans[control] = max(span, max(means_and_sd, default=1.)) * 1.18
    for i, q in enumerate([160, 400]):
        for j, control in enumerate(PLOT_CONTROLS):
            ax = axes[i, j]
            ax.axvline(0, color='#A5A5A5', linewidth=.7, zorder=0)
            for y, ds in enumerate(DATA):
                cell = [r for r in records if r['dataset'] == ds and r['budget'] == q and r['control'] == control]
                assert len(cell) == 3
                for z, r in enumerate(cell):
                    if r['delta_ACC_pp'] is not None:
                        ax.scatter(r['delta_ACC_pp'], y + (z - 1) * .12, s=11,
                            marker='o' if r['eligible'] else 'x', color='#9CA4AA',
                            linewidths=.6, alpha=.85, zorder=2)
                if all(r['eligible'] for r in cell):
                    values = [r['delta_ACC_pp'] for r in cell]
                    ax.errorbar(np.mean(values), y, xerr=np.std(values, ddof=1), fmt='D',
                        markersize=3.5, color=colors[j], capsize=2, elinewidth=.8, zorder=3)
                else:
                    ax.text(.98, y, f"n={sum(r['eligible'] for r in cell)}/3", fontsize=6,
                            transform=ax.get_yaxis_transform(), ha='right', va='center')
            ax.set_xlim(-spans[control], spans[control])
            ax.set_ylim(4.55, -.55)
            ax.set_yticks(range(5), [ds if ds != 'Reuters' else 'Reuters-local' for ds in DATA])
            ax.tick_params(axis='y', length=0)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.grid(axis='y', color='#ECEFF1', linewidth=.5, zorder=-1)
            ax.spines[['top', 'right', 'left']].set_visible(False)
            ax.text(-.08, 1.08, chr(ord('a') + i * 2 + j), transform=ax.transAxes,
                    fontsize=8, fontweight='bold', va='bottom')
            title = ['Private coupling\nindependent selection',
                     'No correction smoothing\nindependent selection'][j]
            ax.set_title(title, pad=10)
            ax.set_xlabel('Full minus control ACC (pp)')
            if j == 0:
                ax.text(-.36, .5, f'Q = {q}', transform=ax.transAxes,
                        fontsize=7.5, rotation=90, rotation_mode='anchor', ha='center', va='center')
    fig.canvas.draw()
    alignment = require_matplotlib_panel_alignment(fig, axes=list(axes.ravel()),
        panel_ids=list('abcd'), require_panel_labels=True,
        json_out=out / 'fig18_mechanism_controls.alignment.json')
    fig.savefig(out / 'fig18_mechanism_controls.pdf', dpi=600)
    fig.savefig(out / 'fig18_mechanism_controls.svg', dpi=600)
    fig.savefig(out / 'fig18_mechanism_controls.png', dpi=600)
    plt.close(fig)
    with (out / 'fig18_mechanism_controls_source.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    manifest = dict(question='Private allocation and correction smoothing under fair selection',
        scope='New seeds on previously used development corpora; no new-corpus confirmation',
        statistic='Full-CV minus control ACC in percentage points; mean and sample SD across three matched seeds',
        exclusions='No planned row excluded; nonconverged pairs shown as crosses where evaluable, no incomplete primary mean',
        plotted_points=60, source_csv_points=90, planned_seeds=[301,302,303], budgets=[160,400],
        scales='Symmetric limits differ by control column; both budgets share each column scale. All points and mean +/- SD retained.',
        supplementary_control='Fixed-full-coefficient coupling retained in complete table/source CSV; 29/30 selected coefficients match independent coupling.',
        source_hashes={name:hashlib.sha256((EVIDENCE/name).read_bytes()).hexdigest()
                       for name in ['per_run.csv','summary.json','RESULTS_READY.json']})
    (out / 'fig18_mechanism_controls.manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'status':'rendered','planned_points':len(records)}))


if __name__ == '__main__':
    main()
