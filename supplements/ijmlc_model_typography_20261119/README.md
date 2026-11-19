# SAPN IJMLC model typography supplement

This supplement reproduces the bold-matrix typography variant of the SAPN model-dependency schematic (Figure 1). It changes eleven displayed labels through ten exact substitutions. The matrix bases X, U, H, R and L are bold; view indices, scalar quantities, the function name R_u and calligraphic sets retain their original styles. It contains no observations, class labels, model fits or accuracy results.

The model diagram has 11 nodes and 13 edges. Their coordinates, colors, line widths, dependencies and explanations are unchanged from the original public figure program. The diagram shows dependencies and joint coupling; it does not represent a sequential optimizer or evidence of a causal accuracy improvement. The two copies of the inputs improve legibility and do not denote separate fits.

## Reproduce

Use Python 3.10 with the packages in `requirements.txt`:

```text
python -m pip install -r requirements.txt
python -B run_model.py --output model_replay
```

The output directory must be new. The runner validates the package, reverses the eleven label changes to recover the exact original public source hash, checks that no non-label operation changed, and executes the exact reviewed generator. It then compares PDF vector geometry, text, font names and positions, embedded fonts, dependency mapping, and PNG bytes against the inspected variant. Successful completion writes `REPRODUCTION_QA.json`. PDF bytes can differ because Matplotlib writes export metadata; this does not bypass the independent content checks.

The outputs are `figures/fig00_model_v4.{pdf,svg,png,tiff}` and a source contract in `figure_qa/`. PDF and SVG are vector outputs with text; PNG and TIFF use 600 dpi. The canvas is 180 × 130 mm, with 9 pt body text. At 166 mm placement width, body text is 8.30 pt and the smallest math scripts are approximately 5.81 pt.

Exact reproduction was checked with Python 3.10.22, Matplotlib 3.10.9, NumPy 2.2.6, Pillow 12.3.0 and PyMuPDF 1.28.2, using Arial (regular, italic and bold) and Comic Sans MS for calligraphic glyphs. These proprietary fonts are not redistributed. They must be lawfully installed for the exact typography check. Different font installations or package versions may render a readable schematic but fail the strict reference comparison; such a failure is not reported as an exact reproduction. Direct execution of `build_model_figure_ijmlc.py --output NEW_DIRECTORY` draws the figure without the strict package comparison and must not be described as a verified match unless checked.

The external nature-figure audit scripts are optional and are not needed for the portable drawing/content check. If installed, the runner accepts `--audit-tools PATH` and `--qa-dependencies PATH`; these run source, PDF text, SVG and collision audits. The one-axis schematic has no comparable-panel alignment requirement. The inherited 180 mm canvas intentionally differs from generic Nature 89/183 mm recommendations.

## Provenance and scope

`build_model_figure_ijmlc.py` retains the exact reviewed source bytes. `PROVENANCE.json` records the ten reversible substitutions and links to the original program in the public repository at commit `efdd603d7f7d5c01411cd98b7591d277bdfee665`. `EXPECTED_FIGURE.json` binds the inspected typography variant to its PDF/PNG and content fingerprints. `MANIFEST.json` covers all six other package files.

The original program and the earlier [Figure Presentation Supplement](https://github.com/ccdyjstff-a11y/sapn-multiview-text-clustering/tree/efdd603d7f7d5c01411cd98b7591d277bdfee665/supplements/figure_presentation_20261118) remain unchanged. That earlier supplement reproduces the original model glyphs and six other figure revisions; this separate supplement supplies only the IJMLC model-glyph variant. It does not reproduce the journal's manuscript-wide equation typography or alter any scientific analysis. The other thirteen manuscript figures are not changed by this variant.

No manuscript, private filesystem paths, original datasets, account information or third-party paper text is included. Publication of this companion does not by itself indicate that the experimental completion batch or the final manuscript has been accepted or finalized.
