"""Rebuild seven evidence-bound manuscript figures from frozen SAPN records.

No fitting, parameter selection, class loading, or modification of source files.
SVG glyphs are outlines for portable appearance. PNGs are 600 dpi at the stated
physical size. Journal-specific requirements remain to be checked by the author.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from PIL import Image

OUT = Path(__file__).resolve().parent
MAIN = OUT.parent.parent
DATA = ["3Sources", "BBC", "Reuters", "Cora", "CiteSeer"]
SIX = DATA + ["20NG"]
DISPLAY = {"Reuters": "Reuters-local", "20NG": "20NG (confirmation)"}
SEEDS = [23, 24, 25]
METHODS = ["SAPN-MV", "SAPN-EQ", "CMVNMF", "MVCSC"]
COLORS = {"SAPN-MV": "#0072B2", "SAPN-EQ": "#D55E00",
          "CMVNMF": "#007A5E", "MVCSC": "#8A4D91"}
MARKERS = {"SAPN-MV": "o", "SAPN-EQ": "s", "CMVNMF": "^", "MVCSC": "D"}
STYLES = {"SAPN-MV": "-", "SAPN-EQ": "--", "CMVNMF": "-.", "MVCSC": ":"}
DATA_COLORS = ["#0072B2", "#D55E00", "#007A5E", "#8A4D91", "#555555"]
SOURCES: dict[str, dict] = {}
FIGURES: dict[str, dict] = {}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name: str):
    path = MAIN / name
    SOURCES[name] = {"absolute_path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}
    return json.loads(path.read_text(encoding="utf-8"))


def one(rows, **conditions):
    matches = [r for r in rows if all(r.get(k) == v for k, v in conditions.items())]
    assert len(matches) == 1, (conditions, len(matches))
    return matches[0]


def same(a, b, atol=1e-10):
    assert np.allclose(a, b, rtol=0, atol=atol), (a, b)


def style_axes(ax, grid="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid, color="#DDDDDD", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=0.6)


def name(ds):
    return DISPLAY.get(ds, ds)


def save(fig, stem, sources, data, caption, alt, placement, transformations):
    outputs = {}
    for ext in ["svg", "png"]:
        target = OUT / (stem + "." + ext)
        fig.savefig(target, dpi=600, facecolor="white", transparent=False,
                    metadata={"Creator": "SAPN evidence figure builder"} if ext == "svg" else None)
        outputs[ext] = {"file": target.name, "sha256": sha(target), "bytes": target.stat().st_size}
    with Image.open(OUT / (stem + ".png")) as im:
        outputs["png"].update(pixels=list(im.size), mode=im.mode, dpi=list(im.info.get("dpi", [])),
                              opaque=(im.mode != "RGBA" or im.getchannel("A").getextrema() == (255, 255)))
    FIGURES[stem] = {
        "sources": sources, "data": data, "caption_en": caption, "alt_text_en": alt,
        "recommended_placement": placement, "transformations": transformations,
        "width_mm": float(fig.get_figwidth() * 25.4), "height_mm": float(fig.get_figheight() * 25.4),
        "outputs": outputs,
    }
    plt.close(fig)


def budget_figure():
    dev_name = "annotation_budget_effects_summary.json"
    con_name = "confirmation_summary.json"
    dev, conf = read(dev_name), read(con_name)
    assert len(dev["cells"]) == 60 and conf["passed"] and conf["final_nonconverged"] == 0
    for source, expected in dev["source_hashes"].items():
        read(source)
        assert SOURCES[source]["sha256"] == expected
    evaluation = read("confirmation_frozen_v1/evaluation.json")
    assert sha(MAIN / "confirmation_frozen_v1/evaluation.json") == conf["evaluation_sha256"]
    dev_methods = dict(zip(METHODS, ["MV-free", "EQ-original-free", "CMVNMF", "MVCSC"]))
    con_methods = dict(zip(METHODS, ["SAPN-original-MV", "SAPN-original-EQ", "CMVNMF", "MVCSC"]))
    records = []
    for ds in SIX:
        for m in METHODS:
            for q in [0, 160, 400]:
                if ds != "20NG":
                    r = one(dev["cells"], dataset=ds, method=dev_methods[m], total_queries=q)
                    mean, sd, failed = r["mean_ACC"], r["sd_ACC"], r["final_nonconverged"]
                else:
                    r = one(conf["summaries"], method=con_methods[m], total_queries=q)
                    rr = [x for x in evaluation if x["method"] == con_methods[m] and x["total_queries"] == q]
                    assert sorted(x["seed"] for x in rr) == [41, 42, 43]
                    values = np.array([100 * x["ACC"] for x in rr])
                    same([values.mean(), values.std(ddof=1)], [r["mean_ACC"], r["sd_ACC"]])
                    mean, sd, failed = r["mean_ACC"], r["sd_ACC"], r["nonconverged"]
                records.append(dict(dataset=ds, method=m, Q=q, mean_ACC_percent=mean,
                                    sample_sd_percent=sd, final_nonconverged=failed))
    fig, axes = plt.subplots(2, 3, figsize=(178 / 25.4, 127 / 25.4), sharex=True, sharey=True)
    for index, (ax, ds) in enumerate(zip(axes.flat, SIX)):
        for m in METHODS:
            rr = [one(records, dataset=ds, method=m, Q=q) for q in [0, 160, 400]]
            means = [r["mean_ACC_percent"] if not r["final_nonconverged"] else np.nan for r in rr]
            sds = [r["sample_sd_percent"] if not r["final_nonconverged"] else np.nan for r in rr]
            ax.errorbar([0, 160, 400], means, yerr=sds, color=COLORS[m], marker=MARKERS[m],
                        ls=STYLES[m], ms=3.2, lw=1.05, capsize=2, elinewidth=0.65)
            for r in rr:
                if r["final_nonconverged"]:
                    ax.errorbar(r["Q"], r["mean_ACC_percent"], yerr=r["sample_sd_percent"],
                                marker="x", color="#555555", ms=5.5, capsize=2, elinewidth=0.6,
                                ls="none", markeredgewidth=1)
        ax.set_title(f"({chr(97+index)}) {name(ds)}", loc="left", fontsize=8)
        ax.set(xlim=(-20, 420), ylim=(0, 105), xticks=[0, 160, 400], yticks=[0, 25, 50, 75, 100])
        style_axes(ax)
        if index >= 3:
            ax.set_xlabel("Paid queries, Q")
        if index % 3 == 0:
            ax.set_ylabel("ACC (%)")
    handles = [Line2D([], [], color=COLORS[m], marker=MARKERS[m], ls=STYLES[m], ms=3.5, label=m) for m in METHODS]
    handles.append(Line2D([], [], color="#555555", marker="x", ls="none", label="Terminal only (failed fit)"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.51, 0.005), fontsize=7.1)
    fig.subplots_adjust(left=.085, right=.985, bottom=.19, top=.96, wspace=.18, hspace=.35)
    save(fig, "fig01_annotation_budget_six", [dev_name, con_name, "confirmation_frozen_v1/evaluation.json"], records,
         "Clustering accuracy under paid-query budgets Q=0, 160 and 400. Points and error bars show mean and sample SD over three prespecified seeds (23–25 for the five development corpora; 41–43 for the frozen 20NG Subject/Body confirmation). Published comparators are the disclosed adapters. A cross denotes a terminal-only summary containing at least one final fit that did not satisfy its numerical stopping rule; the corresponding eligible-mean line is broken. Such cells do not support superiority claims. Every positive budget performs selection using only its paid answers; changes therefore describe the whole protocol rather than an isolated information effect. The 20NG confirmation is not pooled with development runs.",
         "Six panels show budget-dependent accuracy for SAPN-MV, native SAPN-EQ, CMVNMF and MVCSC. The relative benefit varies across corpora; native concatenation remains strong. Solver failures are explicitly marked.",
         "Main text: supervision-budget experiment.",
         ["Use audited mean/sample SD on the percentage scale.", "Mask incomplete eligible means and show their all-terminal mean/SD separately as crosses.", "No interpolation beyond straight connections between the three measured budgets."])


def native_difference_figure():
    primary_name, con_name = "primary_comparison_table_v1.json", "confirmation_frozen_v1/evaluation.json"
    primary, con = read(primary_name), read(con_name)
    assert primary["passed"]
    records = []
    for ds in SIX:
        seeds = [41, 42, 43] if ds == "20NG" else SEEDS
        for metric in ["ACC", "NMI", "ARI"]:
            deltas = []
            for seed in seeds:
                if ds == "20NG":
                    mv = one(con, method="SAPN-original-MV", seed=seed, total_queries=400)
                    eq = one(con, method="SAPN-original-EQ", seed=seed, total_queries=400)
                    assert mv["converged"] and eq["converged"]
                else:
                    mv = one(primary["rows"], dataset=ds, method="SAPN-MV", seed=seed, Q=400)
                    eq = one(primary["rows"], dataset=ds, method="SAPN-EQ", seed=seed, Q=400)
                    assert mv["eligible"] and eq["eligible"]
                deltas.append(100 * (mv[metric] - eq[metric]))
            records.append(dict(dataset=ds, metric=metric, seeds=seeds, paired_deltas_pp=deltas,
                                mean_pp=float(np.mean(deltas)), sample_sd_pp=float(np.std(deltas, ddof=1))))
    limit = 5 * np.ceil(max(abs(v) for r in records for v in r["paired_deltas_pp"]) / 5)
    fig, axes = plt.subplots(1, 3, figsize=(178 / 25.4, 94 / 25.4), sharey=True, sharex=True)
    for j, (ax, metric) in enumerate(zip(axes, ["ACC", "NMI", "ARI"])):
        for i, ds in enumerate(SIX):
            r = one(records, dataset=ds, metric=metric)
            for offset, value in zip([-.11, 0, .11], r["paired_deltas_pp"]):
                ax.plot(value, i + offset, marker="o", ms=3, color=COLORS["SAPN-MV"], alpha=.6, mec="none")
            ax.plot(r["mean_pp"], i, marker="D", ms=4.3, color="#202020", mec="white", mew=.5)
        ax.axvline(0, color="#666666", lw=.8)
        ax.axhline(4.5, color="#AAAAAA", lw=.6, ls="--")
        ax.set_title(f"({chr(97+j)}) {metric}", loc="left")
        ax.set(xlim=(-limit-1, limit+1), xlabel="SAPN-MV − SAPN-EQ (pp)")
        ax.xaxis.set_major_locator(MaxNLocator(5))
        style_axes(ax, "x")
    axes[0].set_yticks(range(6), [name(ds) for ds in SIX])
    axes[0].set_ylim(5.5, -.5)
    fig.legend(handles=[Line2D([], [], marker="o", color=COLORS["SAPN-MV"], ls="none", ms=3.5, label="Paired seed"),
                        Line2D([], [], marker="D", color="#202020", ls="none", ms=4.3, label="Mean of three seeds")],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.56, .005), fontsize=7.4)
    fig.subplots_adjust(left=.19, right=.985, bottom=.20, top=.92, wspace=.22)
    save(fig, "fig02_native_concatenation_differences", [primary_name, con_name], records,
         "Paired SAPN-MV minus native SAPN-EQ differences at Q=400 in ACC, NMI and ARI, expressed in percentage points. Small points retain each of the three prespecified seed differences; black diamonds show their arithmetic mean. Positive values favor the multiview pipeline and negative values favor its native equal-energy concatenation control. The dashed horizontal separator distinguishes the previously frozen 20NG confirmation from the five development corpora. Each native pipeline uses its own initialization and graph. Differences are therefore pipeline comparisons, not isolated effects of retaining private factors. Seeds are descriptive repeats within a corpus; no across-corpus significance test is implied.",
         "Three aligned dot plots show mixed signs for multiview versus concatenated SAPN. The 20NG confirmation favors the multiview pipeline, whereas Cora favors concatenation across all three metrics.",
         "Main text: concise native-control evidence, or supplement if a full three-metric table is retained.",
         ["Pair methods by corpus and prescribed seed before subtracting metrics.", "Multiply differences in [0,1]-scale metrics by 100.", "Plot all three paired values and arithmetic mean; no significance inference."])


def fair_reference_figure():
    prefix = "reference_fair_selection_expanded_v2/"
    evaluation = read(prefix + "evaluation.json")
    summary, audit = read(prefix + "summary.json"), read(prefix + "INDEPENDENT_FULL_AUDIT.json")
    assert audit["status"] == "PASS" and audit["evaluation_checked"] and audit["summary_checked"]
    assert summary["evaluation_sha256"] == SOURCES[prefix + "evaluation.json"]["sha256"]
    read(prefix + "PROTOCOL.json")
    assert summary["protocol_sha256"] == SOURCES[prefix + "PROTOCOL.json"]["sha256"]
    records = []
    for q in [160, 400]:
        for ds in DATA:
            deltas = []
            for seed in SEEDS:
                c = one(evaluation, dataset=ds, seed=seed, total_queries=q, method="coverage")
                f = one(evaluation, dataset=ds, seed=seed, total_queries=q, method="fixed-CV")
                assert c["converged"] and f["converged"]
                deltas.append(100 * (c["ACC"] - f["ACC"]))
            r = one(summary["paired"], dataset=ds, Q=q)
            assert r["eligible"]
            same(deltas, r["deltas_pp"])
            same(np.mean(deltas), r["mean_pp"])
            records.append(dict(dataset=ds, Q=q, seeds=SEEDS, paired_deltas_pp=deltas, mean_pp=float(np.mean(deltas))))
    lim = 5 * np.ceil(max(abs(x) for r in records for x in r["paired_deltas_pp"]) / 5)
    fig, axes = plt.subplots(1, 2, figsize=(178 / 25.4, 91 / 25.4), sharex=True, sharey=True)
    for j, (ax, q) in enumerate(zip(axes, [160, 400])):
        for i, ds in enumerate(DATA):
            r = one(records, dataset=ds, Q=q)
            for offset, marker, value in zip([-.14, 0, .14], ["o", "s", "^"], r["paired_deltas_pp"]):
                ax.plot(value, i + offset, marker=marker, ms=4, mfc="white", mec=COLORS["SAPN-MV"], mew=.9)
            ax.plot(r["mean_pp"], i, marker="D", ms=4.5, color="#202020", mec="white", mew=.5)
        ax.axvline(0, color="#555555", lw=.9)
        ax.set_title(f"({chr(97+j)}) Q = {q}", loc="left")
        ax.set(xlim=(-lim-1, lim+1), xlabel="Coverage − tuned fixed reference (ACC, pp)")
        ax.xaxis.set_major_locator(MaxNLocator(6))
        style_axes(ax, "x")
    axes[0].set_yticks(range(5), [name(ds) for ds in DATA]); axes[0].set_ylim(4.5, -.5)
    handles = [Line2D([], [], marker=m, ms=4, mfc="white", mec=COLORS["SAPN-MV"], ls="none", label=f"Seed {s}") for m, s in zip(["o", "s", "^"], SEEDS)]
    handles.append(Line2D([], [], marker="D", ms=4.5, color="#202020", ls="none", label="Mean"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(.55, .005), fontsize=7.4)
    fig.subplots_adjust(left=.14, right=.985, bottom=.20, top=.92, wspace=.16)
    save(fig, "fig03_fair_reference_paired", [prefix + n for n in ["evaluation.json", "summary.json", "INDEPENDENT_FULL_AUDIT.json", "PROTOCOL.json"]], records,
         "Coverage-based reference strength versus independently selected fixed reference on the five development corpora at Q=160 and Q=400. Each open marker is a paired seed difference in evaluation-object ACC; black diamonds denote their mean. Both pipelines use the same paid queries, node-disjoint validation folds, initialization, graph and stopping rules. Fixed reference jointly selects one of nine strengths and three relation coefficients (27 candidates), while coverage searches three relation coefficients; annotation budgets match, search-computation budgets do not. All 30 paired final comparisons satisfy the numerical stopping checks. These are descriptive development comparisons, not an untouched confirmation or a proof of uniform superiority over continuously optimized reference strengths.",
         "Ten corpus-budget comparisons expose all three paired outcomes rather than only average gains. Effects differ by corpus, budget and seed, including reversals.",
         "Main text: the decisive reference-rule comparison.",
         ["Use completed nine-strength / 27-candidate experiment only.", "Pair coverage and independently tuned fixed reference by dataset, budget and seed.", "Plot all individual differences and their mean on one common symmetric scale."])


def ablation_figure():
    source = "sensitivity_summary.json"
    d = read(source); assert not d["pending"] and len(d["blocks"]) == 15
    rows = d["results"]
    configs = [("direct-only", "Consensus-only supervision"), ("private-only", "Private-only supervision"), ("geometry-off", "All geometry removed")]
    records = []
    for profile, label in configs:
        for ds in DATA:
            values = []
            for seed in SEEDS:
                base = one(rows, dataset=ds, seed=seed, profile="baseline")
                alt = one(rows, dataset=ds, seed=seed, profile=profile)
                assert base["converged"] and alt["converged"]
                values.append(100 * (alt["ACC"] - base["ACC"]))
            records.append(dict(dataset=ds, profile=profile, seeds=SEEDS, paired_deltas_pp=values, mean_pp=float(np.mean(values))))
    low = 5 * np.floor(min(v for r in records for v in r["paired_deltas_pp"]) / 5)
    high = max(0, 5 * np.ceil(max(v for r in records for v in r["paired_deltas_pp"]) / 5))
    fig, axes = plt.subplots(1, 3, figsize=(178 / 25.4, 90 / 25.4), sharex=True, sharey=True)
    for j, (ax, (profile, label)) in enumerate(zip(axes, configs)):
        for i, ds in enumerate(DATA):
            r = one(records, dataset=ds, profile=profile)
            for off, value in zip([-.1, 0, .1], r["paired_deltas_pp"]):
                ax.plot(value, i + off, "o", ms=3, color="#888888", mec="none")
            ax.plot(r["mean_pp"], i, marker="D", ms=4.5, color=COLORS["SAPN-MV"], mec="white", mew=.5)
        ax.axvline(0, color="#555555", lw=.8)
        display_label = label.replace(" supervision", "\nsupervision").replace("All geometry removed", "All geometry\nremoved")
        ax.set_title(f"({chr(97+j)}) " + display_label, loc="left", fontsize=7.7)
        ax.set(xlim=(low-1, high+1), xlabel="ΔACC from full SAPN (pp)")
        ax.xaxis.set_major_locator(MaxNLocator(5)); style_axes(ax, "x")
    axes[0].set_yticks(range(5), [name(ds) for ds in DATA]); axes[0].set_ylim(4.5, -.5)
    fig.legend(handles=[Line2D([], [], marker="o", color="#888888", ms=3, ls="none", label="Paired seed"),
                        Line2D([], [], marker="D", color=COLORS["SAPN-MV"], ms=4.5, ls="none", label="Mean of three seeds")],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.56, .005), fontsize=7.4)
    fig.subplots_adjust(left=.14, right=.985, bottom=.20, top=.87, wspace=.20)
    save(fig, "fig04_linked_configuration_ablations", [source], records,
         "Linked configuration controls at Q=400, relative to full SAPN, with the original query-selected relation coefficient held fixed. Small gray points are all three paired seed changes in ACC; blue diamonds are their means. Consensus-only supervision also removes the private endpoint simplex domain; private-only supervision reallocates the supervision share; removing all geometry deletes both smoothing and reference retention. These are linked interventions, not pure single-term causal ablations. All plotted final fits satisfy their stopping checks.",
         "Three panels show the effects of consensus-only supervision, private-only supervision and removing all geometry on five corpora, with complete paired seed observations. Average losses occur in each corpus, with heterogeneous magnitudes.",
         "Main text: mechanism/configuration evidence; explicitly retain the linked-intervention caveat.",
         ["Select the three preregistered structural configurations, retaining every dataset and prescribed seed.", "Compute each change against the matching full-model run, not independent averages.", "Do not attribute linked interventions to isolated loss removal."])


def convergence_figure():
    sources, records = [], []
    for ds in DATA:
        source = f"balanced_selection/{ds}/seed23/Q400_MV-free_final_diagnostics.json"
        d = read(source); sources.append(source)
        assert d["converged"]
        s2 = d["stage2"]
        J = list(s2["J_hist"])
        kkt = [dict(iteration=r["iteration"], residual=r["maximum"]) for r in s2["kkt_history"]]
        assert len(J) == s2["iterations"] + 1
        boundary = None
        if d["continuation"] is not None:
            c = d["continuation"]
            # Only the continuation under the unchanged second-stage objective.
            same(J[-1], c["J_hist"][0], atol=1e-8)
            boundary = len(J) - 1
            J.extend(c["J_hist"][1:])
            kkt.extend(dict(iteration=boundary+r["iteration"], residual=r["maximum"]) for r in c["kkt_history"] if r["iteration"] > 0)
        arr = np.asarray(J, float)
        assert np.all(np.isfinite(arr)) and np.all(arr > 0)
        assert np.all(np.diff(arr) <= 1.1e-9 * np.maximum(1, np.abs(arr[:-1])))
        assert all(r["residual"] > 0 for r in kkt)
        same(kkt[-1]["residual"], d["kkt"])
        records.append(dict(dataset=ds, seed=23, Q=400, stage="stage2_and_same_objective_continuation_only",
                            iteration=list(range(len(J))), J=J, J_relative_to_stage2_initial=(arr/arr[0]).tolist(),
                            maximum_block_residual=kkt, continuation_boundary=boundary,
                            stopping_tolerance=s2["tol_kkt"], converged=d["converged"]))
    fig, axes = plt.subplots(1, 2, figsize=(178 / 25.4, 91 / 25.4))
    line_patterns = ["-", "--", "-.", ":", (0, (5, 1))]
    for index, (r, color, marker) in enumerate(zip(records, DATA_COLORS, ["o", "s", "^", "D", "v"])):
        x, y = r["iteration"], r["J_relative_to_stage2_initial"]
        axes[0].plot(x, y, color=color, ls=line_patterns[index], lw=1.1, label=name(r["dataset"]))
        axes[0].plot(x[-1], y[-1], color=color, marker=marker, ms=4)
        hist = r["maximum_block_residual"]
        axes[1].semilogy([v["iteration"] for v in hist], [v["residual"] for v in hist],
                         color=color, marker=marker, ls=line_patterns[index], ms=3, lw=1, label=name(r["dataset"]))
        if r["continuation_boundary"] is not None:
            for ax in axes:
                ax.axvline(r["continuation_boundary"], color=color, lw=.6, ls=":")
    axes[0].set_title("(a) Second-stage objective", loc="left")
    axes[0].set_ylabel(r"$J_t / J_0$ (stage 2)")
    axes[0].yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    axes[1].set_title("(b) First-order stopping residual", loc="left")
    axes[1].axhline(1e-3, color="#222222", lw=.9, ls="--")
    axes[1].text(.99, 1.10e-3, r"tolerance $10^{-3}$", transform=axes[1].get_yaxis_transform(), ha="right", va="bottom", fontsize=7)
    axes[1].set_ylabel("Maximum normalized block residual")
    for ax in axes:
        ax.set_xlabel("Iteration within stage 2")
        ax.set_xlim(left=0)
        style_axes(ax)
    handles = [Line2D([], [], color=c, marker=m, ls=ls, ms=3.5, label=name(ds)) for ds, c, m, ls in zip(DATA, DATA_COLORS, ["o", "s", "^", "D", "v"], line_patterns)]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.52, .005), fontsize=7.2)
    fig.subplots_adjust(left=.12, right=.985, bottom=.25, top=.91, wspace=.36)
    save(fig, "fig05_second_stage_numerics", sources, records,
         "Numerical trajectories of the original SAPN at Q=400 for prespecified seed 23 on all five development corpora. Panel (a) divides each second-stage objective value by that stage's initial value; stage-one objectives are excluded because the orthogonality coefficient changes between stages. Panel (b) shows the maximum normalized block first-order residual on a logarithmic axis, with the 10^-3 stopping threshold. Points in (b) are the actual recorded checks; endpoints in (a) mark the recorded terminal iterations. These five runs required no continuation. Stopping additionally requires relative objective change below 10^-5. The curves illustrate finite numerical behavior and do not establish factor convergence or global optimality.",
         "Two panels show small further decreases of the stage-two objective and the recorded first-order residuals for five fixed seed runs. All terminal residuals are below the declared tolerance; Cora already starts stage two below that residual threshold.",
         "Supplement: numerical convergence illustration, or a compact main-text optimization figure if space permits.",
         ["Use seed 23 for every corpus, never select a best seed.", "Exclude stage one and normalize only by J at stage-two iteration zero.", "Append only same-objective continuation, if present, after checking the boundary identity.", "Plot recorded residuals on a base-10 logarithmic scale; no smoothing or zero-flooring."])


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


def relation_generalization_figure():
    prefix = "reference_fair_selection_expanded_v2/"
    rows = read(prefix + "evaluation.json")
    audit = read(prefix + "INDEPENDENT_FULL_AUDIT.json")
    summary = read(prefix + "summary.json")
    assert audit["status"] == "PASS" and audit["evaluation_checked"]
    assert summary["evaluation_sha256"] == SOURCES[prefix + "evaluation.json"]["sha256"]
    # Register the exact metric-generating and independent-check scripts as sources.
    for filename in ["run_fair_reference.py", "verify_fair_reference_expanded.py"]:
        path = MAIN / prefix / filename
        SOURCES[prefix + filename] = {"absolute_path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}
        assert "51000" in path.read_text(encoding="utf-8")
    records = []
    for ds in DATA:
        for method in ["coverage", "fixed-CV"]:
            rr = [one(rows, dataset=ds, seed=seed, total_queries=400, method=method) for seed in SEEDS]
            assert all(r["converged"] for r in rr)
            for seed, r in zip(SEEDS, rr):
                other = one(rows, dataset=ds, seed=seed, total_queries=400,
                            method="fixed-CV" if method == "coverage" else "coverage")
                assert r["evaluation_pair_count"] == other["evaluation_pair_count"]
            for field in ["train_BER", "unqueried_pair_BER"]:
                values = [100 * r[field] for r in rr]
                records.append(dict(dataset=ds, method=method, field=field, Q=400, seeds=SEEDS,
                                    values_percent=values, mean_percent=float(np.mean(values)),
                                    evaluation_pair_counts=[r["evaluation_pair_count"] for r in rr],
                                    postseal_evaluation_pair_seed_offset=51000))
    maximum = max(v for r in records for v in r["values_percent"])
    high = max(40, 5 * np.ceil(maximum / 5))
    fig, axes = plt.subplots(1, 2, figsize=(178 / 25.4, 96 / 25.4), sharex=True, sharey=True)
    configs = [("coverage", "Coverage", COLORS["SAPN-MV"], "o", -.16),
               ("fixed-CV", "Tuned fixed reference", "#8A4D91", "D", .16)]
    for j, (ax, field, title) in enumerate(zip(axes, ["train_BER", "unqueried_pair_BER"],
                                             ["Queried training relations", "Unqueried evaluation relations"])):
        for method, label, color, marker, shift in configs:
            for i, ds in enumerate(DATA):
                r = one(records, dataset=ds, method=method, field=field)
                for off, value in zip([-.055, 0, .055], r["values_percent"]):
                    ax.plot(value, i + shift + off, marker=marker, ms=3.1, mfc="white", mec=color,
                            mew=.75, ls="none", alpha=.75, clip_on=False)
                ax.plot(r["mean_percent"], i + shift, marker=marker, ms=5.1, color=color,
                        mec="white", mew=.5, ls="none", clip_on=False)
        ax.set_title(f"({chr(97+j)}) {title}", loc="left", fontsize=7.8)
        ax.set(xlim=(-1, high+1), xlabel="Balanced pair error (%)")
        ax.xaxis.set_major_locator(MaxNLocator(5)); style_axes(ax, "x")
    axes[0].set_yticks(range(5), [name(ds) for ds in DATA]); axes[0].set_ylim(4.5, -.5)
    handles = [Line2D([], [], marker=marker, ms=5.1, color=color, ls="none", label=label)
               for _, label, color, marker, _ in configs]
    handles.append(Line2D([], [], marker="o", ms=3.1, mfc="white", mec="#777777", ls="none", label="Small markers: individual seeds"))
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.55, .005), fontsize=7.2)
    fig.subplots_adjust(left=.14, right=.985, bottom=.22, top=.92, wspace=.15)
    save(fig, "fig07_relation_generalization_same_pairs",
         [prefix + n for n in ["evaluation.json", "summary.json", "INDEPENDENT_FULL_AUDIT.json", "run_fair_reference.py", "verify_fair_reference_expanded.py"]], records,
         "Training versus unqueried-relation balanced error at Q=400 in the completed expanded fixed-reference comparison. Both panels use the same coverage and independently selected fixed-reference terminal predictions. Small open markers show the three prescribed seed values and larger filled markers their arithmetic mean. The left panel uses the 400 paid training relations. The right panel uses identical evaluation pairs across the two methods within each corpus/seed, sampled only after sealing predictions with random seed equal to the experiment seed plus 51000; the count is min(10000, m(m−1)/2), where m is the number of evaluation objects. These diagnostic pairs are not supplied for training or parameter selection and are not counted as additional paid supervision. Pair observations share objects and are not independent replicates. BER is 0.5 times the must-link disagreement rate plus 0.5 times the cannot-link agreement rate. The contrast demonstrates that satisfying queried relations does not ensure low error on unqueried relations; it does not identify a causal propagation mechanism.",
         "Aligned dot plots show very low training balanced pair error but much higher error on unqueried evaluation pairs, for both coverage and tuned fixed reference. All three seed values and the same post-seal evaluation pairs are retained.",
         "Main text: strong evidence for the training-fit versus unqueried-relation gap; use instead of a redundant accuracy-difference figure if space is limited.",
         ["Use train_BER and unqueried_pair_BER from the same expanded-v2 evaluation rows, not older diagnostic summaries.", "Restrict to Q400, coverage/fixed-CV, all five corpora and seeds 23–25.", "Convert BER to percent; plot every seed and arithmetic means on common axes.", "Independent full audit verifies identical seed+51000 evaluation-pair construction and recomputed BER."])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["all", "budget", "native", "fair", "ablation", "convergence", "noise", "relations"], default="all")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    settings = {"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.2,
                "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7.4,
                "legend.fontsize": 7.2, "axes.linewidth": .6, "lines.linewidth": 1,
                "svg.fonttype": "path", "svg.hashsalt": "SAPN-submission-revision-v3",
                "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
                "axes.unicode_minus": True}
    tasks = {"budget": budget_figure, "native": native_difference_figure, "fair": fair_reference_figure,
             "ablation": ablation_figure, "convergence": convergence_figure, "noise": wrong_answer_figure,
             "relations": relation_generalization_figure}
    with plt.rc_context(settings):
        for key, task in tasks.items():
            if args.only in ("all", key):
                task()
                print("Created", key, flush=True)
    source_fields = {
        "fig01_annotation_budget_six": ["annotation_budget_effects_summary.cells: dataset, method, total_queries, mean_ACC, sd_ACC, final_nonconverged", "confirmation_summary.summaries: method, total_queries, mean_ACC, sd_ACC, nonconverged", "confirmation_frozen_v1/evaluation: method, seed, total_queries, ACC"],
        "fig02_native_concatenation_differences": ["primary_comparison_table_v1.rows: dataset, method, seed, Q, eligible, ACC, NMI, ARI", "confirmation_frozen_v1/evaluation: method, seed, total_queries, converged, ACC, NMI, ARI"],
        "fig03_fair_reference_paired": ["expanded_v2/evaluation: dataset, seed, total_queries, method, converged, ACC", "expanded_v2/summary.paired: eligible, deltas_pp, mean_pp"],
        "fig04_linked_configuration_ablations": ["sensitivity_summary.results: dataset, seed, profile, converged, ACC"],
        "fig05_second_stage_numerics": ["Q400_MV-free_final_diagnostics: converged, kkt", "stage2: J_hist, iterations, kkt_history[].iteration, kkt_history[].maximum, tol_kkt", "continuation (null in all five selected runs): J_hist, kkt_history"],
        "fig06_wrong_answer_sensitivity": ["noisy_query_summary.results: dataset, seed, method, fraction, converged, ACC"],
        "fig07_relation_generalization_same_pairs": ["expanded_v2/evaluation: dataset, seed, total_queries, method, converged, train_BER, unqueried_pair_BER, evaluation_pair_count", "run_fair_reference and independent verifier: seed+51000; sampling from evaluation objects only after sealing predictions"],
    }
    for stem, card in FIGURES.items():
        card["source_fields"] = source_fields[stem]
    report = {
        "scope": "Scientific figures from previously completed and audited evidence. No new model fitting or metric computation from class labels.",
        "target": {"medium": "SCI manuscript and supplement", "journal": None,
                   "journal_specific_requirements_verified": False, "width_mm": 178,
                   "raster_dpi": 600, "vector_font_policy": "SVG glyph outlines; not editable text and not embedded-font claims"},
        "software": {"python": __import__("sys").version, "numpy": np.__version__, "matplotlib": matplotlib.__version__,
                     "script": Path(__file__).name, "script_sha256": sha(Path(__file__)),
                     "procedural_guidance_reference": {
                         "authors": "Timothy Kassis; Vinayak Agarwal; Yuhuan He; Darshil Patel; Aubrey M. Brueckner",
                         "year": 2026, "title": "Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents",
                         "doi": "10.48550/arXiv.2609.00065", "url": "https://arxiv.org/abs/2609.00065",
                         "verified_version": "v2 (2026-09-02)", "verified_on": "2026-10-15",
                         "role": "Procedural software attribution only; not evidence for SAPN novelty or experimental findings."}},
        "procedural_skill": {"name": "scientific-visualization", "version_read": "1.4",
                             "path": "C:/Users/LENOVO/.codex/skills/scientific-visualization/SKILL.md",
                             "scope": "Integrity, explicit provenance, uncertainty and vector/raster export guidance; no added scientific finding."},
        "replication_note": "Three seeds describe within-corpus variation and are not independent corpora or significance evidence.",
        "sources": SOURCES, "figures": FIGURES,
        "render_review": "See the separately saved figure_verification.json, tied to output hashes. The generator alone does not certify appearance or journal acceptance.",
    }
    filename = "figure_sources.json" if args.only == "all" else f"figure_sources_{args.only}.json"
    (OUT / filename).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Manifest:", OUT / filename, flush=True)


if __name__ == "__main__":
    main()
