# SAPN figure presentation supplement

This author-owned supplement reproduces seven existing scientific figures with readable lettering and vector exports. It makes **presentation changes only**: no fitting, coefficient selection, class scoring, new observations or hypothesis tests. The original scientific results and figure 14 are unchanged. This package is separate from the previously published state-diagnostic Resource 3.

## Run all seven figures

Use Python 3.10 or newer, the pinned runtime packages, and a lawful system installation of Arial. Arial is needed to match the approved model/SI typography exactly; it is not redistributed. DejaVu Sans, supplied by Matplotlib, is retained for figures 06 and 08. Missing Arial produces a clear error rather than silently substituting a different appearance.

From this directory:

```text
python -m pip install -r requirements.txt
python -B run_figures.py --output ../seven-figure-output
```

The output directory must be new and outside the package. All required numerical data and author-owned helpers are bundled, so replay needs no network access, private manuscript, local skills installation, raw corpus, categories or third-party optimizer. Package file hashes are checked before and after execution. The model generator's external audit options are optional and are not used by this command.

| Output folder | Figures | Retained scientific content |
|---|---|---|
| `model/figures/` | 00 | Model nodes, edges, dependency mapping and formulas |
| `si/` | 13, 15, 16, 17 | 1,503 analytic frequency/strength combinations; 315, 525 and 630 original seed-level observations, with the original complete-cell mean/sample-SD rules |
| `raster/` | 06, 08 | 60 wrong-answer paired changes; 30 reference-strength pairs and 30 location-control ACC differences, with original unselected means |

Each figure is exported as editable PDF/SVG and 600 dpi PNG/TIFF. Figures 06 and 08 are true vector re-exports; their former PNGs are not resampled. Use the PDFs in the manuscript. At 166 mm placement width, body/tick/legend text is at least 8.30 pt. Mathematical scripts retain their smaller type hierarchy. The model keeps its dependencies while wrapping queried-answer text and shortening two redundant labels. Figure 08 keeps the original 1:1.7 panel-width ratio and its original SVG crop; historical SVG and raster tight-crop margins differed slightly.

The program independently checks all seven PDFs against the approved vector geometry, styles, text and exact scientific-artist traces or model dependency contract. It also requires byte-identical PNG appearance. Successful execution writes one `EQUIVALENCE_QA.json` receipt; timestamps may change PDF/SVG file hashes without changing the drawing. Existing source CSVs are checked against the JSON values used for drawing, and no rows are removed for appearance. Figures 06 and 08 originally use individual seed markers and means without error bars; this package preserves that encoding.

## Source binding and public provenance

The immutable public input revision is [`df56e6bcc2c5bb82139170e5759523e5ac3936ee`](https://github.com/ccdyjstff-a11y/sapn-multiview-text-clustering/tree/df56e6bcc2c5bb82139170e5759523e5ac3936ee). `SOURCE_PROVENANCE.json` lists an exact public download URL, original SHA256, packaged SHA256 and transformation for every reused input. No download is needed for execution.

The ten CSV files are byte-identical copies of the public author-generated plot data. Two original JSON files are reduced to the fields consumed by the drawing functions: all 90 wrong-answer result rows and 15 block identities, and all 20 location-control cells and 10 paired records remain. This removes unused private artifact-path strings and unused metrics, without recalculating any retained number. The CSVs independently verify the retained values.

Historical Python files are reduced to the exact drawing/helper function definitions that are used, excluding private path-bound entry points and unrelated plots. Their function ASTs are unchanged and checked against the public Git objects. The model and typography wrappers are the new author-owned presentation code. The compact alignment checker is the study's existing public helper; no third-party skill source or optimizer is included. Python libraries and font software remain external runtime dependencies, not redistributed source.

`MANIFEST.json` binds the package. `EXPECTED_OUTPUTS.json` binds the approved scientific appearance; the approved files themselves need not be downloaded. The independent checker can be rerun on existing output:

```text
python -B verify_outputs.py --output ../seven-figure-output
```

These equivalence checks are not a new general-purpose collision auditor. The approved assets already passed rendered font/collision checks and panel-by-panel review. After inserting them into a journal layout, inspect the actual PDF pages and verify that the intended physical width is retained.
