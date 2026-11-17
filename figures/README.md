# Scientific figure sources

The map below covers the 13 preceding manuscript figures and the separately sealed allocation-extension figure. Each data path and plotting-source path is checked to exist. `FIGURE_MAP.json` additionally binds each corresponding manuscript PDF identity. No manuscript full text, raw corpus or third-party optimizer is included.

Three portable plotting entries were actually executed from this release. They read retained numeric data or evaluate the stated analytical formulas; they do not train, rescore class labels, select coefficients, or add observations. All other historical generators are provided for source inspection with their original directory and dependency assumptions. Availability of those generators is not a claim that every historical plot has been rerun in this portable layout.

## Executed portable entries

Run from the release root; use fresh output paths for your own redraw. Matplotlib is a separate plotting dependency.

```text
python -m pip install -r figures/requirements.txt
python -B -I figures/plot_reference_risk.py --output-dir ../sapn-figures
python -B -I figures/plot_allocation_acc.py --output-dir ../sapn-figures
python -B -I figures/plot_extension_acc.py --output-dir ../sapn-figures
python -B -I figures/check_exports.py --directory ../sapn-figures --output ../sapn-figures/export-bounds.json
```

Poppler must be on PATH for the last command. The plotters use the local study-authored geometry checker and require no private skill installation. `--help` lists input overrides. Figure 21 can optionally use an independently installed geometry checker through its documented environment override; this is not required.

Figure 21 retains all 400 paired ACC values and both coefficient-selection policies; its near-zero panels are secondary views. Figure 22 retains all 40 paired ACC values (8 groups, 5 seeds), including negative AG News contrasts. Diamonds show means and bars show sample SD, not confidence intervals. Figure 13 evaluates the isolated reference-filter mathematics and conditional risk, not an empirical ACC prediction.

## Manuscript asset to data and source

| Manuscript asset | Data in release | Plotting source | Execution scope |
|---|---|---|---|
| fig00_model_v4.pdf | Programmatic diagram; no empirical CSV | `figures/historical/build_model_figure.py` | historical source |
| fig05_second_stage_numerics.pdf | `figures/source_data/fig05_objectives.csv`<br>`figures/source_data/fig05_residuals.csv`<br>`figures/source_data/fig05_source_data.json` | `figures/historical/build_numerics_figure.py`<br>`figures/historical/build_evidence_figures.py` | historical source |
| fig06_wrong_answer_sensitivity.png | `figures/source_data/fig06_paired_changes.csv`<br>`evidence/original/noisy_query_summary.json` | `figures/historical/build_evidence_figures.py` | historical source |
| fig08_location_control.png | `figures/source_data/fig08_reference_strengths.csv`<br>`figures/source_data/fig08_paired_ACC.csv`<br>`evidence/original/coverage_location_control_v2/summary.json` | `figures/historical/build_new_figures.py` | historical source |
| fig13_reference_filter_risk.pdf | `tables/figures/fig13_reference_filter_source.csv`<br>`evidence/v8/reference_filter/reference_filter_risk_source.csv` | `figures/plot_reference_risk.py`<br>`figures/historical/build_reference_risk_figure.py` | portable redraw executed |
| fig14_unified_ssa_Q400.pdf | `evidence/v11/ssa/figure/figure_source_Q400.csv`<br>`evidence/v11/ssa/figure/figure_summary_Q400.csv`<br>`evidence/v10/native_pcbg/figure/figure_source_Q400.csv`<br>`evidence/v10/native_pcbg/figure/figure_summary_Q400.csv` | `provenance/original_sources/figures/plot_unified_ssa.py.txt` | historical source |
| fig15_supervision_repetition.pdf | `tables/figures/fig15_supervision_repetition_source.csv`<br>`tables/figures/fig15_supervision_repetition_summary.csv` | `figures/historical/build_extension_figures.py`<br>`provenance/original_sources/figures/build_extension_results.py.txt` | historical source |
| fig16_signed_ablation_repetition.pdf | `tables/figures/fig16_signed_ablation_repetition_source.csv`<br>`tables/figures/fig16_signed_ablation_repetition_summary.csv` | `figures/historical/build_extension_figures.py`<br>`provenance/original_sources/figures/build_extension_results.py.txt` | historical source |
| fig17_reference_repetition.pdf | `tables/figures/fig17_reference_repetition_source.csv`<br>`tables/figures/fig17_reference_repetition_summary.csv` | `figures/historical/build_extension_figures.py`<br>`provenance/original_sources/figures/build_extension_results.py.txt` | historical source |
| fig18_mechanism_controls.pdf | `evidence/v5/mechanisms/per_run.csv`<br>`evidence/v5/mechanisms/summary.json`<br>`evidence/v5/mechanisms/RESULTS_READY.json` | `figures/historical/plot_mechanisms_v5.py`<br>`figures/historical/evidence_gate.py` | historical source |
| fig19_reuters_holdout.pdf | `evidence/v6/multilingual_holdout/figures/reuters_holdout_source_data.csv`<br>`evidence/v6/multilingual_holdout/per_run.csv`<br>`evidence/v6/multilingual_holdout/evaluation.json` | `figures/historical/plot_holdout.py` | historical source |
| fig20_mass_transfer.pdf | `evidence/v6/selection_mass_diagnostic/figure_delta_source.csv`<br>`evidence/v6/selection_mass_diagnostic/figure_source.csv`<br>`evidence/v6/selection_mass_diagnostic/paired_change_summary.csv` | `provenance/original_sources/figures/report_and_plot_mass_transfer.py.txt` | historical source |
| fig21_allocation_new10_acc.pdf | `evidence/final/alloc/acc_paired_differences_400.csv`<br>`evidence/final/alloc/acc_paired_directions_40.csv` | `figures/plot_allocation_acc.py`<br>`provenance/original_sources/figures/plot_allocation_acc.py.txt` | portable redraw executed |
| fig22_allocation_extension.pdf | `evidence/allocation_extension/reporting/paired_values_200.csv`<br>`evidence/allocation_extension/reporting/paired_ACC_8.csv` | `figures/plot_extension_acc.py`<br>`provenance/original_sources/figures/summarize_extension.py.txt` | portable redraw executed |

Paths in the table are relative to the release root. Figure 00 is a programmatic model diagram. For figures 05, 06 and 08, `source_data/` exposes compact numeric CSVs flattened from the existing recorded objects; `DATA_DERIVATION.json` records the source hashes and transformations. Flattening does not introduce new estimates, missing-seed substitutions or training.

## Source identity and limits

`SOURCE_ORIGINS.json` records original archive/local paths, hashes and release derivations. Exact original plotting bytes are retained under `provenance/original_sources/figures/` as `.py.txt`; these records are not executable entry points. Historical English copies are also browsable under `historical/`. Historical sources with Chinese comments or filename literals remain archival text only, so active release source comments and docstrings stay English without silently altering historical bytes. Absolute paths in frozen generators are provenance and may require reconstruction to execute.

For the portable derivatives, only input/output paths, imports and QA integration were adjusted; the original numerical formulas and drawing coordinates are retained. The figure 21 drawing-function AST and figure 13 analytical-function AST were checked against their originals. Figure 22 reuses the post-seal script drawing block and reads its saved CSVs instead of rerunning report generation. See `QA_NOTES.md`, `RENDER_RELOCATION_QA.json`, `qa/EXPORT_BOUNDS.json` and `GENERATED_ARTIFACTS.json` for executed checks and their limits.

Generated SVG/PDF/PNG files are research-owned figures and QA overlays; the packaging allowlist binds each approved artifact by SHA256. No blanket permission for unrelated PDF or image content is used.
