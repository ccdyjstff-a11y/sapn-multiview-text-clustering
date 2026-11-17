"""Analytic spectral responses and conditional risk, with no empirical data.

The figure contract is stored alongside this script. All 501 original positive
frequencies and all three original u values are retained. Uncertainty bars are
inapplicable to deterministic formulas. Nothing here selects a parameter or
evaluates the fitted SAPN estimator.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
STEM = 'fig13_reference_filter_risk'
STRENGTHS = [.1, .5, .9]
COLORS = ['#0072B2', '#D55E00', '#009E73']


def response(u, ell):
    full = u * (1 + ell) / (u + ell)
    nocorr = u / (u + (1 - u) * ell)
    difference = u * (1 - u) * ell**2 / ((u + ell) * (u + (1-u) * ell))
    return full, nocorr, difference


def threshold(u, ell):
    # Algebraically equivalent to (f+g)/(2-f-g), without low-frequency
    # cancellation. Only 0<u<1 and ell>0 belong to the strict-risk comparison.
    assert 0 < u < 1 and np.all(ell > 0)
    return u * (2*u + 2*ell + (1-u)*ell**2) / ((1-u)*ell*(2*u+(2-u)*ell))


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')


def numerical_checks(frequencies, original_csv):
    errors = {'response_difference': 0., 'threshold_equivalence_relative': 0.,
              'risk_identity': 0., 'threshold_crossing': 0.}
    tests = 0
    for u in STRENGTHS:
        f, g, d = response(u, frequencies)
        t = threshold(u, frequencies)
        assert np.all((0 < g) & (g < f) & (f < 1))
        assert np.all(f >= u) and np.all(d > 0)
        assert np.all(np.diff(t) < 0)
        errors['response_difference'] = max(errors['response_difference'], float(np.max(abs(f-g-d))))
        errors['threshold_equivalence_relative'] = max(errors['threshold_equivalence_relative'], float(np.max(abs(t-(f+g)/(2-f-g))/t)))
        for ratio in [.5, 1., 2.]:
            A, B = ratio*t, 1.
            actual = (1-f)**2*A+f*f*B - ((1-g)**2*A+g*g*B)
            factorized = (f-g)*((f+g)*B-(2-f-g)*A)
            errors['risk_identity'] = max(errors['risk_identity'], float(np.max(abs(actual-factorized))))
            if ratio < 1: assert np.all(actual > 0)
            elif ratio > 1: assert np.all(actual < 0)
            else: errors['threshold_crossing'] = max(errors['threshold_crossing'], float(np.max(abs(actual))))
            tests += len(frequencies)
        assert np.array_equal(np.asarray(response(u, 0.)[:2]), [1., 1.])
        f1, g1, d1 = response(1., frequencies)
        assert np.array_equal(f1, g1) and np.all(d1 == 0)
        limit = u/(2-u)
        assert abs(threshold(u, np.asarray(1e12))-limit) < 1e-10
        assert threshold(u, np.asarray(1e-12)) > 1e10
    assert errors['response_difference'] < 1e-14
    assert errors['threshold_equivalence_relative'] < 1e-11
    assert errors['risk_identity'] < 1e-12 and errors['threshold_crossing'] < 1e-12
    old = list(csv.DictReader(original_csv.open(encoding='utf8')))
    assert len(old) == len(STRENGTHS)*len(frequencies)
    old_max = 0.
    for row in old:
        u, ell = float(row['u']), float(row['frequency'])
        f, g, d = response(u, ell)
        old_max = max(old_max, abs(f-float(row['full'])), abs(g-float(row['removed'])), abs(d-float(row['difference'])))
    assert old_max < 1e-14
    # Independently supplied two-node examples, not new performance results.
    examples = []
    f, g, _ = response(.5, 1.)
    assert abs(float(threshold(.5, np.asarray(1.)))-1.4) < 1e-14
    for A, B, expected in [(.125, .02, [.02277777777777778, .03625]), (.005, .08, [.03611111111111111, .02125])]:
        values = [(1-f)**2*A+f*f*B, (1-g)**2*A+g*g*B]
        assert np.max(abs(np.asarray(values)-expected)) < 1e-14
        examples.append(dict(u=.5, frequency=1., A=A, B=B, ratio=A/B, threshold=1.4,
                             full_mse=values[0], nocorr_mse=values[1]))
    return dict(status='PASS', error_maxima=errors, risk_identity_comparisons=tests,
                old_source_rows_checked=len(old), old_source_max_error=old_max,
                original_source_sha256=hashlib.sha256(original_csv.read_bytes()).hexdigest(),
                supplied_two_node_examples=examples,
                boundaries='Zero frequency and u=1 agree exactly. u=0 and B=0 excluded from threshold plot. Low positive-frequency divergence and high-frequency limit checked.')


def build(args):
    global ROOT
    ROOT = args.output_dir.resolve()
    ROOT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.skill_scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    qa = ROOT/'qa'
    qa.mkdir(exist_ok=True)
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8.5, 'axes.labelsize': 8.5, 'axes.titlesize': 9,
        'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'legend.fontsize': 8.5,
        'pdf.fonttype': 42, 'svg.fonttype': 'none', 'mathtext.fontset': 'dejavusans',
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .65,
        'lines.linewidth': 1.45, 'legend.frameon': False,
    })
    frequencies = np.logspace(-2, 2, 501)
    assert not np.any(frequencies <= 0) and np.all(np.diff(frequencies) > 0)
    checks = numerical_checks(frequencies, args.original_csv)
    fig = plt.figure(figsize=(180/25.4, 130/25.4))
    grid = fig.add_gridspec(2, 2, left=.105, right=.975, bottom=.180, top=.855,
                           wspace=.30, hspace=.58, height_ratios=[1, 1])
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])]
    rows = []
    for u, color in zip(STRENGTHS, COLORS):
        full, nocorr, diff = response(u, frequencies)
        risk = threshold(u, frequencies)
        axes[0].plot(frequencies, full, color=color)
        axes[0].plot(frequencies, nocorr, color=color, linestyle=(0, (4, 2)))
        axes[1].plot(frequencies, diff, color=color)
        axes[2].plot(frequencies, risk, color=color)
        rows.extend(dict(u=u, frequency=ell, full=f, nocorr=g, difference=d, threshold=t,
                         high_frequency_threshold_limit=u/(2-u))
                    for ell, f, g, d, t in zip(frequencies, full, nocorr, diff, risk))
    titles = ['Frequency response', 'Additional reference retention', 'Risk threshold for a fixed soft target']
    for ax, letter, title in zip(axes, ['a', 'b', 'c'], titles):
        ax.set_xscale('log')
        ax.set_xlim(.01, 100)
        ax.set_xticks([.01, .1, 1, 10, 100], labels=['0.01', '0.1', '1', '10', '100'])
        ax.set_xlabel(r'Graph frequency $\ell$', labelpad=4)
        ax.set_title(title, loc='left', pad=8)
        ax.annotate(letter, xy=(0, 1), xycoords='axes fraction', xytext=(-29, 7),
                    textcoords='offset points', ha='left', va='bottom', weight='bold', fontsize=10)
        ax.grid(axis='y', linewidth=.4, color='#DBDEE3', zorder=0)
        ax.minorticks_off()
    for ax in axes[:2]:
        ax.set_ylim(0, 1.025)
        ax.set_yticks([0, .25, .5, .75, 1], labels=['0', '0.25', '0.5', '0.75', '1'])
    axes[0].set_ylabel('Retained amplitude', labelpad=4)
    axes[1].set_ylabel(r'$f_u(\ell)-g_u(\ell)$', labelpad=4)
    axes[2].set_yscale('log')
    axes[2].set_ylim(.035, 1500)
    axes[2].set_yticks([.1, 1, 10, 100, 1000], labels=['0.1', '1', '10', '100', '1000'])
    axes[2].minorticks_off()
    axes[2].set_ylabel(r'Threshold $T$ for $A/B$', labelpad=5)
    axes[2].grid(False)
    axes[2].text(2., 800, 'For the curve with the same u:\nAbove: Full lower MSE\nBelow: NoCorr lower MSE\nOn the curve: equal MSE',
                 fontsize=8.5, va='top', linespacing=1.4)
    axes[2].text(.02, .065, r'As $\ell\to\infty$:  $T\to u/(2-u)$', fontsize=8.5, va='bottom')
    colors = [Line2D([0], [0], color=c, label=f'u = {u:g}') for u, c in zip(STRENGTHS, COLORS)]
    styles = [Line2D([0], [0], color='#333333', label='Full'),
              Line2D([0], [0], color='#333333', linestyle=(0, (4, 2)), label='NoCorr')]
    fig.legend(handles=colors, loc='upper center', bbox_to_anchor=(.53, 1.0), ncol=3,
               columnspacing=2.6, handlelength=2.3)
    fig.legend(handles=styles, loc='upper center', bbox_to_anchor=(.53, .953), ncol=2,
               columnspacing=2.6, handlelength=2.3)
    fig.text(.105, .060, r'Panel c: zero-mean reference error; $0<u<1$, $\ell>0$, $B>0$; per-frequency MSE only.', fontsize=8.5)
    fig.text(.105, .025, r'At $\ell=0$ or $u=1$: equal MSE.  $u=0$ is excluded (graph-only solution need not be unique).', fontsize=8.5)
    fig.canvas.draw()
    alignment = require_matplotlib_panel_alignment(fig, json_out=qa/'alignment.json',
        overlay_svg=qa/'alignment.svg', tolerance_pt=1.5, gutter_tolerance_pt=1.5,
        require_panel_labels=True, strict=True)
    fig.savefig(ROOT/f'{STEM}.pdf', metadata={'Creator': 'Matplotlib; analytic isolated reference-filter formulas'})
    fig.savefig(ROOT/f'{STEM}.svg', metadata={'Creator': 'Matplotlib; analytic isolated reference-filter formulas'})
    fig.savefig(ROOT/f'{STEM}.png', dpi=300)
    plt.close(fig)
    csv_path = ROOT/f'{STEM}_source.csv'
    with csv_path.open('w', newline='', encoding='utf8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    checks.update(source_rows=len(rows), empirical_observations=0, exclusions=0,
                  source_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  width_mm=180, height_mm=130, preview_dpi=300,
                  alignment_verdict=alignment['verdict'])
    save_json(ROOT/'NUMERICAL_CHECKS.json', checks)
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--skill-scripts', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent/'generated')
    parser.add_argument('--original-csv', type=Path, default=Path(__file__).resolve().parents[1]/'tables/figures/fig13_reference_filter_source.csv')
    build(parser.parse_args())
