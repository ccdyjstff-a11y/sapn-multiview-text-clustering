"""Re-export the frozen numerical illustration with a clear threshold label.

Exact reuse of the original two-panel Python figure. No fitting, rescoring,
selection, interpolation, smoothing, or new scientific calculations occur.
The existing normalized objective values and recorded residuals are plotted
unchanged. Only the threshold annotation position and its font floor change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
STEM = "fig05_second_stage_numerics"
DATA = ["3Sources", "BBC", "Reuters", "Cora", "CiteSeer"]
COLORS = ["#0072B2", "#D55E00", "#007A5E", "#8A4D91", "#555555"]
MARKERS = ["o", "s", "^", "D", "v"]
PATTERNS = ["-", "--", "-.", ":", (0, (5, 1))]
SETTINGS = {
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.2,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7.4,
    "legend.fontsize": 7.2, "axes.linewidth": .6, "lines.linewidth": 1,
    "svg.fonttype": "none", "svg.hashsalt": "SAPN-fig05-label-repair",
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
    "axes.unicode_minus": True,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")


def display_name(dataset):
    return "Reuters-local" if dataset == "Reuters" else dataset


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=.45)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=.6)


def detect_v4_root():
    for candidate in [HERE, *HERE.parents]:
        if (candidate / "MANUSCRIPT_BUILD_SPEC.json").exists():
            return candidate
    raise RuntimeError("Supply --v4-root for the original manuscript workspace.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v4-root", type=Path, default=None)
    parser.add_argument("--source-data", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--skill-scripts", type=Path,
                        default=Path.home() / ".codex" / "skills" / "nature-figure" / "scripts")
    args = parser.parse_args()
    args.v4_root = args.v4_root or detect_v4_root()
    args.output_dir = args.output_dir or args.v4_root / "build_logs" / "editorial4_fig05_label_repair"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.skill_scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    if args.source_data is None:
        manifest_path = args.v4_root / "figures" / "figure_sources.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        original = manifest["figures"][STEM]
        bundle = {
            "figure": STEM, "records": original["data"],
            "caption_en": original["caption_en"],
            "transformations": original["transformations"],
            "source_manifest_sha256": sha(manifest_path),
            "source_manifest_figure_key": "figures." + STEM,
            "original_builder_sha256": sha(args.v4_root / "figures" / "build_evidence_figures.py"),
            "original_outputs": original["outputs"],
            "diagnostic_sources": {s: manifest["sources"][s] for s in original["sources"]},
            "figure_size_mm": [178, 91],
        }
        args.source_data = args.output_dir / "fig05_source_data.json"
        write_json(args.source_data, bundle)
    else:
        bundle = json.loads(args.source_data.read_text(encoding="utf-8"))
    records = bundle["records"]
    assert [r["dataset"] for r in records] == DATA
    assert len(records) == 5
    frozen_records_hash = canonical_hash(records)
    checks = []
    for record in records:
        assert record["seed"] == 23 and record["Q"] == 400
        assert record["converged"] and record["continuation_boundary"] is None
        assert record["stopping_tolerance"] == 1e-3
        assert record["iteration"] == list(range(len(record["J"])))
        assert len(record["iteration"]) == len(record["J_relative_to_stage2_initial"])
        assert np.isfinite(record["J"]).all() and (np.asarray(record["J"]) > 0).all()
        assert np.isfinite(record["J_relative_to_stage2_initial"]).all()
        residuals = [item["residual"] for item in record["maximum_block_residual"]]
        assert np.isfinite(residuals).all() and (np.asarray(residuals) > 0).all()
        checks.append({
            "dataset": record["dataset"],
            "objective_points": len(record["iteration"]),
            "residual_checks": len(residuals),
            "record_sha256": canonical_hash(record),
        })

    with plt.rc_context(SETTINGS):
        fig, axes = plt.subplots(1, 2, figsize=(178 / 25.4, 91 / 25.4))
        artist_checks = []
        for i, (record, color, marker) in enumerate(zip(records, COLORS, MARKERS)):
            x, y = record["iteration"], record["J_relative_to_stage2_initial"]
            objective, = axes[0].plot(x, y, color=color, ls=PATTERNS[i], lw=1.1,
                                      label=display_name(record["dataset"]))
            endpoint, = axes[0].plot(x[-1], y[-1], color=color, marker=marker, ms=4)
            history = record["maximum_block_residual"]
            hx = [value["iteration"] for value in history]
            hy = [value["residual"] for value in history]
            residual, = axes[1].semilogy(hx, hy, color=color, marker=marker,
                                         ls=PATTERNS[i], ms=3, lw=1,
                                         label=display_name(record["dataset"]))
            exact = all((
                np.array_equal(objective.get_xdata(), x),
                np.array_equal(objective.get_ydata(), y),
                np.array_equal(endpoint.get_xdata(), [x[-1]]),
                np.array_equal(endpoint.get_ydata(), [y[-1]]),
                np.array_equal(residual.get_xdata(), hx),
                np.array_equal(residual.get_ydata(), hy),
            ))
            assert exact
            artist_checks.append({"dataset": record["dataset"],
                                  "all_plotted_coordinates_exact": exact})
        axes[0].set_title("(a) Second-stage objective", loc="left")
        axes[0].set_ylabel(r"$J_t / J_0$ (stage 2)")
        axes[0].yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
        axes[1].set_title("(b) First-order stopping residual", loc="left")
        axes[1].axhline(1e-3, color="#222222", lw=.9, ls="--")
        # The threshold stays at 1e-3. Its label sits just below the dashed
        # reference in the empty interval between the recorded trajectories.
        label = axes[1].text(.80, 8.50e-4, r"tolerance $10^{-3}$",
                             transform=axes[1].get_yaxis_transform(),
                             ha="right", va="top", fontsize=7.2)
        axes[1].set_ylabel("Maximum normalized block residual")
        for ax in axes:
            ax.set_xlabel("Iteration within stage 2")
            ax.set_xlim(left=0)
            style_axes(ax)
        handles = [
            Line2D([], [], color=c, marker=m, ls=ls, ms=3.5, label=display_name(ds))
            for ds, c, m, ls in zip(DATA, COLORS, MARKERS, PATTERNS)
        ]
        fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
                   bbox_to_anchor=(.52, .005), fontsize=7.2)
        fig.subplots_adjust(left=.12, right=.985, bottom=.25, top=.91, wspace=.36)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bounds = label.get_window_extent(renderer)
        label_bounds_px = [bounds.x0, fig.bbox.height - bounds.y1,
                           bounds.x1, fig.bbox.height - bounds.y0]
        alignment = require_matplotlib_panel_alignment(
            fig, json_out=args.output_dir / (STEM + ".alignment.json"),
            overlay_svg=args.output_dir / (STEM + ".alignment.svg"),
            tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True,
        )
        outputs = {}
        for suffix in [".pdf", ".svg", ".png", ".tiff"]:
            ext = suffix[1:]
            destination = args.output_dir / (STEM + suffix)
            fig.savefig(destination, dpi=600, facecolor="white", transparent=False)
            outputs[ext] = {"file": destination.name, "sha256": sha(destination),
                            "bytes": destination.stat().st_size}
        with Image.open(args.output_dir / f"{STEM}.png") as im:
            outputs["png"]["size_px"] = list(im.size)
            outputs["png"]["dpi"] = list(im.info["dpi"])
            assert im.size == (4204, 2149)
        geometry = [
            {"panel": chr(97 + i), "xlim": list(ax.get_xlim()),
             "ylim": list(ax.get_ylim()), "yscale": ax.get_yscale(),
             "plot_bbox_fraction": list(ax.get_position().bounds)}
            for i, ax in enumerate(axes)
        ]
        plt.close(fig)

    assert canonical_hash(records) == frozen_records_hash
    contract = {
        "figure": STEM, "claim": "Recorded numerical trajectories for five fixed runs.",
        "archetype": "quantitative grid", "backend": "python-matplotlib",
        "reuse": "Exact reuse of original frozen coordinates and plotting layout.",
        "width_mm": 178, "height_mm": 91, "dpi": 600,
        "panels": {
            "a": "All recorded normalized stage-two objective values and terminal markers.",
            "b": "All recorded maximum block residual checks and the unchanged 1e-3 threshold."
        },
        "unit": "Prespecified seed 23, Q400, one trajectory per development corpus.",
        "uncertainty": "Not applicable: individual fixed numerical trajectories, not seed aggregates.",
        "changes": [
            "Tolerance label x anchor: 0.99 -> 0.80 in axes fraction.",
            "Tolerance label y anchor: 1.10e-3 bottom -> 8.50e-4 top, below the unchanged dashed threshold.",
            "Tolerance label font: 7 -> 7.2 pt so its math superscript meets the 5 pt floor.",
            "SVG text editable; added PDF/TIFF exports and render-time QA."
        ],
        "unchanged": [
            "All five source records, plotted coordinates, stage definitions and stopping threshold.",
            "Axes limits, dimensions, plot rectangles, colors, markers, line styles, ticks and legend."
        ],
        "source_manifest_sha256": bundle["source_manifest_sha256"],
        "source_bundle_sha256": sha(args.source_data),
        "records_sha256": frozen_records_hash,
        "source_code_sha256": sha(__file__),
        "source_checks": checks,
        "artist_checks": artist_checks,
        "geometry": geometry,
        "label_bbox_at_canvas_dpi": label_bounds_px,
        "caption_en": bundle["caption_en"],
        "exclusions": 0, "new_fitting_or_rescoring": False,
        "outputs": outputs,
        "visual_review_status": "PENDING_PANEL_BY_PANEL_REVIEW",
    }
    write_json(args.output_dir / f"{STEM}_contract.json", contract)
    print(json.dumps({"figure": STEM, "records_sha256": frozen_records_hash,
                      "outputs": outputs}, ensure_ascii=False))


if __name__ == "__main__":
    main()
