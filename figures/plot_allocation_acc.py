#!/usr/bin/env python3
"""Reproduce the allocation figure from the unmodified 400-row source CSV.

Run: python plot_allocation_acc.py
Dependencies: Python >=3.10, numpy, matplotlib; bundled alignment audit helper.
The source copies are byte-identical to the archived CSV members and verified
against source_provenance.json before plotting. No experiments are rerun.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import os, sys
sys.path.insert(0, os.environ.get('SAPN_FIGURE_QA_TOOLS', str(Path(__file__).resolve().parent)))
from audit_panel_alignment import require_matplotlib_panel_alignment

HERE = Path(__file__).resolve().parent
DATASETS = ['3Sources', 'BBC', 'Reuters', 'Cora', 'CiteSeer']
COLORS = ['#286B96', '#BA6A32']
MARKERS = ['o', '^']
PAIRED = 'acc_paired_differences_400.csv'
SUMMARY = 'acc_paired_directions_40.csv'


def read_verified_sources(source_dir: Path):
    provenance = json.loads((HERE / 'source_provenance.json').read_text(encoding='utf-8'))
    tables = {}
    for name in [PAIRED, SUMMARY]:
        raw = (source_dir / name).read_bytes()
        expected = provenance['members']['evidence/final/alloc/' + name]['sha256']
        assert hashlib.sha256(raw).hexdigest() == expected, f'Changed source bytes: {name}'
        tables[name] = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
    rows = tables[PAIRED]
    assert len(rows) == 400
    groups = {}
    unique = set()
    for row in rows:
        key = (row['dataset'], int(row['budget']), row['comparator'])
        pair_key = (*key, int(row['seed']))
        assert pair_key not in unique, f'Duplicate pair: {pair_key}'
        unique.add(pair_key)
        assert row['eligible'] == 'True', f'Ineligible source row: {pair_key}'
        value = float(row['Full_minus_control_ACC_pp'])
        assert np.isfinite(value)
        groups.setdefault(key, []).append((int(row['seed']), value))
    assert len(groups) == 40
    summaries = []
    for reference in tables[SUMMARY]:
        key = (reference['dataset'], int(reference['budget']), reference['comparator'])
        seed_values = sorted(groups[key])
        seeds = [s for s, _ in seed_values]
        values = np.array([v for _, v in seed_values])
        assert len(values) == 10 and len(set(seeds)) == 10
        assert int(reference['eligible_pairs']) == int(reference['planned_pairs']) == 10
        mean, sd = float(values.mean()), float(values.std(ddof=1))
        assert np.isclose(mean, float(reference['mean_ACC_difference_pp']), atol=1e-12, rtol=0)
        assert np.isclose(sd, float(reference['sample_SD_pp']), atol=1e-12, rtol=0)
        tolerance = float(reference['ACC_tie_tolerance_pp'])
        wins = int(np.count_nonzero(values > tolerance))
        losses = int(np.count_nonzero(values < -tolerance))
        ties = 10 - wins - losses
        assert (wins, ties, losses) == tuple(int(reference[k]) for k in ['wins', 'ties', 'losses'])
        summaries.append(dict(dataset=key[0], budget=key[1], comparator=key[2],
                              n=10, mean_ACC_difference_pp=mean, sample_SD_pp=sd,
                              wins=wins, ties=ties, losses=losses,
                              min_pp=float(values.min()), max_pp=float(values.max()),
                              inside_zoom=int(np.count_nonzero(abs(values) <= 2))))
    assert len(summaries) == 40
    return groups, summaries


def draw_figure(groups, summaries, output_dir: Path, qa_dir: Path):
    matplotlib.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 8.7, 'axes.titlesize': 9.5, 'axes.labelsize': 8.7,
        'xtick.labelsize': 8.7, 'ytick.labelsize': 8.7,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.major.size': 2, 'ytick.major.size': 2,
        'svg.fonttype': 'none', 'pdf.fonttype': 42,
        'legend.frameon': False, 'savefig.facecolor': 'white',
    })
    width_mm, height_mm = 180, 190
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), facecolor='white')
    full_limits = (-22, 24)
    zoom_limits = (-2, 2)
    specs = [('a', 160, 'CV', 'Independent CV', 14, 137, 103),
             ('b', 160, 'at-full-lambda', 'Full-selected λ', 101, 137, 103),
             ('c', 400, 'CV', 'Independent CV', 14, 53, 19),
             ('d', 400, 'at-full-lambda', 'Full-selected λ', 101, 53, 19)]
    axes, ids, counts = [], [], []
    raw_point_total = 0
    for letter, budget, suffix, title, x_mm, full_y, zoom_y in specs:
        full = fig.add_axes([x_mm / width_mm, full_y / height_mm, 72 / width_mm, 34 / height_mm])
        zoom = fig.add_axes([x_mm / width_mm, zoom_y / height_mm, 72 / width_mm, 26 / height_mm])
        axes.extend([full, zoom]); ids.extend([letter + '-full', letter + '-zoom'])
        full.annotate(letter, xy=(0, 1), xycoords='axes fraction', xytext=(-18, 10),
                      textcoords='offset points', fontsize=10, weight='bold', annotation_clip=False)
        full.set_title(f'{title}, Q = {budget}', loc='left', pad=10, fontsize=9.5)
        full.set_title('Full range', loc='right', pad=10, fontsize=8.7, color='#555555')
        zoom.set_title('Near-zero detail: −2 to +2 pp', loc='left', pad=5, fontsize=8.7)
        for ax, lim, yticks in [(full, full_limits, [-20, -10, 0, 10, 20]),
                               (zoom, zoom_limits, [-2, -1, 0, 1, 2])]:
            ax.set_xlim(-0.55, 4.55); ax.set_ylim(*lim); ax.set_yticks(yticks)
            ax.set_xticks(np.arange(5))
            ax.set_axisbelow(True)
            ax.grid(axis='y', color='#E7E7E7', linewidth=0.4)
            ax.axhline(0, color='#666666', lw=0.75, zorder=1)
            ax.tick_params(axis='y', pad=2)
            if x_mm == 14:
                ax.set_ylabel('ΔACC (pp)', labelpad=3)
        full.set_xticklabels([]); full.tick_params(axis='x', length=0)
        zoom.set_xticklabels(DATASETS)
        full.axhspan(-2, 2, color='#E9EEF2', alpha=0.8, zorder=0)
        zoom.set_facecolor('#F8FAFC')
        for di, dataset in enumerate(DATASETS):
            for ci, stem in enumerate(['consensus-only', 'views-only']):
                values = np.array([v for _, v in sorted(groups[(dataset, budget, stem + '-' + suffix)])])
                mean, sd = float(values.mean()), float(values.std(ddof=1))
                assert full_limits[0] < min(values.min(), mean - sd)
                assert full_limits[1] > max(values.max(), mean + sd)
                center = di + [-0.18, 0.18][ci]
                # Deterministic horizontal jitter reveals exact ties without changing y values.
                x = center + np.linspace(-0.11, 0.11, 10)
                full.scatter(x, values, s=10, marker=MARKERS[ci], facecolors='none',
                             edgecolors=COLORS[ci], linewidths=0.55, alpha=0.7, zorder=3)
                full.errorbar(center, mean, yerr=sd, fmt=MARKERS[ci], color=COLORS[ci],
                              markersize=4.1, markeredgecolor='white', markeredgewidth=0.45,
                              elinewidth=1.0, capsize=2.5, capthick=0.85, zorder=4)
                # Only the *secondary display* is restricted. All ten values
                # remain in the full view and in both views' summary statistics.
                visible = np.abs(values) <= 2
                zoom.scatter(x[visible], values[visible], s=10, marker=MARKERS[ci],
                             facecolors='none', edgecolors=COLORS[ci], linewidths=0.55,
                             alpha=0.7, zorder=3)
                lower, upper = mean - sd, mean + sd
                if upper >= -2 and lower <= 2:
                    zoom.vlines(center, max(-2, lower), min(2, upper), color=COLORS[ci],
                                linewidth=1.0, zorder=4)
                    for end in [lower, upper]:
                        if -2 <= end <= 2:
                            zoom.hlines(end, center - 0.055, center + 0.055,
                                        color=COLORS[ci], linewidth=0.85, zorder=4)
                    for end, direction in [(lower, -1), (upper, 1)]:
                        if abs(end) > 2 and end * direction > 2:
                            zoom.annotate('', xy=(center, direction * 1.98),
                                          xytext=(center, direction * 1.67),
                                          arrowprops={'arrowstyle': '-|>', 'lw': 0.8,
                                                      'color': COLORS[ci], 'mutation_scale': 6,
                                                      'shrinkA': 0, 'shrinkB': 0}, zorder=5)
                if -2 <= mean <= 2:
                    zoom.plot(center, mean, marker=MARKERS[ci], color=COLORS[ci],
                              markersize=4.1, markeredgecolor='white',
                              markeredgewidth=0.45, zorder=5, linestyle='none')
                raw_point_total += len(values)
                counts.append({'panel': letter, 'dataset': dataset, 'comparator': stem,
                               'full_view_pairs': 10,
                               'zoom_pairs_within_limits': int(np.count_nonzero(abs(values) <= 2)),
                               'mean_pp': mean, 'sample_sd_pp': sd})
    assert raw_point_total == 400
    legend = [Line2D([], [], marker=MARKERS[i], color=COLORS[i], lw=0,
                     markersize=4.4, label=name)
              for i, name in enumerate(['Consensus only', 'Views only'])]
    fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(0.5, 0.989),
               ncol=2, handletextpad=0.5, columnspacing=2.2, fontsize=9)
    fig.text(0.5, 0.04, 'Open marks: paired seeds   •   Filled marks and bars: mean ± sample SD',
             ha='center', va='center', fontsize=8.7)
    fig.canvas.draw()
    alignment = require_matplotlib_panel_alignment(
        fig, axes=axes, panel_ids=ids,
        row_groups=[['a-full', 'b-full'], ['a-zoom', 'b-zoom'],
                    ['c-full', 'd-full'], ['c-zoom', 'd-zoom']],
        column_groups=[['a-full', 'a-zoom', 'c-full', 'c-zoom'],
                       ['b-full', 'b-zoom', 'd-full', 'd-zoom']],
        exemptions=[{'panels': ids, 'checks': ['vertical-gutter'],
                     'reason': 'Each scientific panel intentionally groups its full view with its subordinate zoom; the gap between scientific panels is larger.'}],
        json_out=qa_dir / 'fig21_allocation_new10_acc.alignment.json',
        overlay_svg=qa_dir / 'fig21_allocation_new10_acc.alignment.svg',
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    stem = output_dir / 'fig21_allocation_new10_acc'
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(stem.with_suffix('.' + ext), dpi=300)
    # Independent panel previews from the same authoritative Matplotlib canvas.
    from matplotlib.transforms import Bbox
    for letter, _, _, _, x_mm, full_y, zoom_y in specs:
        bbox = Bbox.from_extents((x_mm - 13) / 25.4, (zoom_y - 8) / 25.4,
                                (x_mm + 74) / 25.4, (full_y + 42) / 25.4)
        fig.savefig(qa_dir / f'panel_{letter}.png', dpi=300, bbox_inches=bbox)
    plt.close(fig)
    (qa_dir / 'data_integrity.json').write_text(json.dumps({
        'total_source_rows': 400, 'total_full_view_seed_marks': raw_point_total,
        'excluded_rows': 0, 'groups': 40, 'pairs_per_group': 10,
        'full_view_limits_pp': full_limits, 'zoom_view_limits_pp': zoom_limits,
        'all_full_view_raw_points_and_sd_extents_inside_limits': True,
        'center': 'arithmetic mean of the ten paired differences',
        'uncertainty': 'sample standard deviation, ddof=1; not a confidence interval',
        'all_40_summaries_match_archived_directions': True,
        'panels': counts}, indent=2), encoding='utf-8')
    with (qa_dir / 'recomputed_40_summaries.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    print(json.dumps({'figure': str(stem.with_suffix('.pdf')), 'paired_points': 400,
                      'groups': 40, 'alignment': alignment.get('verdict', alignment.get('status'))}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=HERE.parent / 'evidence/final/alloc')
    parser.add_argument('--output-dir', type=Path, default=HERE / 'generated')
    parser.add_argument('--qa-dir', type=Path, default=HERE / 'qa')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True); args.qa_dir.mkdir(parents=True, exist_ok=True)
    groups, summaries = read_verified_sources(args.source_dir)
    draw_figure(groups, summaries, args.output_dir, args.qa_dir)


if __name__ == '__main__':
    main()
