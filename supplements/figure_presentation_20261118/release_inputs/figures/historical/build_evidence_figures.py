"""Author-owned function snapshot; consumed by the portable replay entry.
Original function ASTs and their immutable public origins are bound in SOURCE_PROVENANCE.json.
This file is not a standalone entry point.
"""

def one(rows, **conditions):
    matches = [r for r in rows if all(r.get(k) == v for k, v in conditions.items())]
    assert len(matches) == 1, (conditions, len(matches))
    return matches[0]

def style_axes(ax, grid="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid, color="#DDDDDD", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=0.6)

def name(ds):
    return DISPLAY.get(ds, ds)

def wrong_answer_figure():
    source = "noisy_query_summary.json"
    d = read(source); assert not d["pending"] and len(d["blocks"]) == 15
    records = []
    methods = [("SAPN-original-MV", "SAPN-MV"), ("SAPN-original-EQ", "SAPN-EQ")]
    for method, label in methods:
        for frac in [.1, .2]:
            for ds in DATA:
                values = []
                for seed in SEEDS:
                    clean = one(d["results"], dataset=ds, seed=seed, method=method, fraction=0.)
                    noisy = one(d["results"], dataset=ds, seed=seed, method=method, fraction=frac)
                    assert clean["converged"] and noisy["converged"]
                    values.append(100 * (noisy["ACC"] - clean["ACC"]))
                records.append(dict(dataset=ds, method=label, wrong_fraction=frac, seeds=SEEDS,
                                    paired_deltas_pp=values, mean_pp=float(np.mean(values))))
    lo = 5 * np.floor(min(v for r in records for v in r["paired_deltas_pp"]) / 5)
    hi = max(0, 5 * np.ceil(max(v for r in records for v in r["paired_deltas_pp"]) / 5))
    fig, axes = plt.subplots(1, 2, figsize=(178 / 25.4, 98 / 25.4), sharey=True, sharex=True)
    for j, (ax, (_, method)) in enumerate(zip(axes, methods)):
        for frac, marker, shift in [(0.1, "o", -.15), (0.2, "s", .15)]:
            for i, ds in enumerate(DATA):
                r = one(records, dataset=ds, method=method, wrong_fraction=frac)
                for smalloff, value in zip([-.035, 0, .035], r["paired_deltas_pp"]):
                    ax.plot(value, i + shift + smalloff, marker=marker, ms=2.6, color=COLORS[method],
                            alpha=.45, mfc="white", mew=.7, ls="none")
                ax.plot(r["mean_pp"], i + shift, marker=marker, ms=5, color=COLORS[method],
                        mfc=COLORS[method] if frac == .1 else "white", mew=1, ls="none")
        ax.axvline(0, color="#555555", lw=.8)
        ax.set_title(f"({chr(97+j)}) {method}", loc="left")
        ax.set(xlim=(lo-1, hi+1), xlabel="Noisy − clean answers (ACC, pp)")
        ax.xaxis.set_major_locator(MaxNLocator(6)); style_axes(ax, "x")
    axes[0].set_yticks(range(5), [name(ds) for ds in DATA]); axes[0].set_ylim(4.5, -.5)
    handles = [Line2D([], [], marker="o", color="#333333", ms=5, ls="none", label="10% wrong: mean"),
               Line2D([], [], marker="s", mfc="white", mec="#333333", ms=5, ls="none", label="20% wrong: mean"),
               Line2D([], [], marker="o", mfc="white", mec="#888888", ms=2.6, ls="none", label="Small markers: paired seeds")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.55, .005), fontsize=7.1)
    fig.subplots_adjust(left=.14, right=.985, bottom=.23, top=.92, wspace=.15)
    save(fig, "fig06_wrong_answer_sensitivity", [source], records,
         "Wrong-answer sensitivity at Q=400, relative to the same seed's clean-answer result. Exactly 40 or 80 nested randomly selected answers are flipped for 10% or 20% corruption, respectively, with identical flips across methods; both fitting and within-budget validation use the corrupted answers. Small markers retain all three paired seed changes and larger markers show their mean. Native SAPN-EQ is shown alongside SAPN-MV on the same scale. The full 3Sources deterioration is retained. These perturbations do not establish robustness to arbitrary errors, and a positive difference on an individual corpus does not show that incorrect supervision is beneficial.",
         "Two dot-plot panels show large negative accuracy changes on 3Sources under incorrect answers, while responses on the other corpora are heterogeneous. Every seed is visible and the same scale is used for both pipelines.",
         "Supplement: explicit robustness boundary; cite the 3Sources failure in the main discussion.",
         ["Subtract the clean result separately for each method, corpus and seed.", "Use all 10% and 20% corruption outcomes, including positive and strongly negative differences.", "No data-dependent seed selection or axis truncation."])
