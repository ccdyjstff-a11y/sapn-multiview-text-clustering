# Post-hoc best-CV tie sensitivity: compact evidence

Place this directory at `evidence/tie_sensitivity/` in the accompanying release. All `prediction_member` fields are relative to the **release root**, not this directory. The 386 reused prediction files must remain at their existing `evidence/v9/predictions/<sha256>.npz` paths. This package has not been uploaded or externally published by this export.

This is a post-hoc sensitivity analysis. The original evaluation results were already known before the sensitivity protocol was written. Original default-first selection remains the primary analysis. The added rules select the smallest or largest numeric lambda within the original eligible best mean-CV-BER tie set, with tolerance `1e-12`. They do not use test ACC to choose a rule, parameter, or fit. These data are not an independent confirmation or new-corpus generalization study.

## What is included

- `selection_policy_index_900.csv`: the 300 original selections under each of three fixed rules.
- `candidate_cv_index_900.csv`: all three original lambda candidates per original selection, with original CV scores and source-selection hashes. `historical/` is the logical regenerated-run directory described below.
- `logical_final_targets_1500.csv`: all logical results, including independent-CV controls and controls using that rule's Full-selected lambda. Every row links to one complete-query final prediction; CV predictions are never used as final predictions.
- `NEW_FITS_172.json` and `new_fits/`: 172 new complete-query final prediction files, identities, and diagnostics. NPZ `labels` means **predicted cluster IDs**, not class truth; other fields are final consensus membership `H` and paid positive/negative query pairs `P`/`N`.
- `REUSED_FINALS_386.json`: exact-hash references to 386 original complete-query predictions already in the release. Original input, graph, protocol, paid-query, identity, prediction, and optimizer-state hashes document the original reuse checks; optimizer states are not distributed here.
- `reporting/`: 1,500 logical metric records, 558 unique-fit metrics, 6,000 paired metric differences, 600 summaries, six ACC headline counts, and the compact 40-row three-rule ACC table. Means, sample SDs, medians, and W/T/L refer to the same ten paired seed blocks. ACC/NMI/ARI favor positive Full-minus-control differences; BER favors negative differences. No p-values are reported.
- `reporting/reporting_supplements/`: original primary absolute metrics (50 logical groups by five metrics) and pool-excluded pair counts and source hashes (50 dataset/seed blocks). Evaluation pairs are sampled without replacement with `seed+51000`, up to 10,000 unordered pairs; pair counts are not independent replicate counts.
- `provenance/original_artifacts/`: byte-identical original protocol, selection, final seal, evaluation, verification, and reporting artifacts. Historical absolute paths in these files are retained as provenance, not presented as runnable public paths.
- `provenance/original_sources/*.py.txt`: byte-identical scripts that produced this study. Their historical absolute-path dependencies are intentional provenance and are not the portable execution interface.
- `SEALED_MEMBERS_DISTRIBUTION.csv`: the inclusion status of **every** file named by `ALL_FINALS_SEALED.json`. The byte-identical seal also binds files intentionally omitted from this compact package; it must not be read as a claim that every named file is distributed.
- `SOURCE_EXPORT.json`: byte-identical source-to-release copy ledger. `PUBLIC_MANIFEST.json` additionally hashes all generated public mappings and package members. `EXPORT_VERIFICATION.json` records export checks and the frozen scoring/verification bindings.

Raw feature matrices, raw source archives, complete class-truth vectors, warm-start inputs, full graph files, and complete optimizer states are intentionally excluded. The retained final `H` is an output used to derive predictions; it is not the complete optimizer checkpoint. Existing original prediction references are verified against the local release during export.

## Results and their limits

The number of positive mean ACC contrasts under independent CV is 19/20 for primary, 18/20 for minimum-lambda ties, and 16/20 for maximum-lambda ties. For the policy-specific Full-lambda controls it is 19/20 under each rule. Four independent-CV contrasts change from positive to negative relative to primary; no matched-coefficient ACC mean changes sign. Thus the exact primary 19/20 count depends on tie handling, while a majority of mean ACC contrasts remains positive under each of these three specified rules. This is not evidence of stability under arbitrary tie rules, new candidate grids, new samples, or new corpora. The Full-lambda controls match the nominal coefficient; they do not equalize the data-dependent effective influence of each allocation.

All 172 added fits converged and were eligible. All 558 unique final sources were globally sealed before new evaluation labels were loaded. A separate implementation rescored every prediction and reconstructed all 6,000 paired differences and 600 summaries: maximum ACC/BER errors were zero, NMI error was below `6e-16`, and ARI error was below `2e-16`. No scientific result or primary selection was changed during export. Shared corpora, nested budgets, reused fits, multiple policies, and evaluation-pair counts are not treated as independent experimental units.

## Reproduction route

For a direct replay of the distributed predictions, without any training, run:

```text
python evidence/tie_sensitivity/tools/rescore_tie_compact.py --release-root . --labels-map path/to/labels.json --output ../tie_compact_replay.json
```

The JSON has keys `3Sources`, `BBC`, `Reuters`, `Cora`, and `CiteSeer`, each mapped to an ordered external `.npy` class vector; relative paths resolve against the JSON directory. The command verifies the compact manifest and all prediction hashes, reconstructs the original pool-excluded relations using the published paid-query partitions, and checks all five metrics, 1,500 logical records, 6,000 paired differences, and 600 summaries. Its output must be outside the frozen release. This replay does not require raw features or optimizer states and performs no fitting.

1. Obtain and prepare the five external datasets using the release's documented data preparation route; raw data and class truth are supplied locally by the reader.
2. Complete `tools/run_common_study.py historical` using a single consistent `common/` source tree. This recreates the original seed 504--513 blocks, CV artifacts, inputs, graph, full states, and seals needed for identity-checked reuse. Compact published evidence alone is intentionally insufficient for optimizer continuation.
3. Run the parameterized `tools/run_tie_sensitivity.py` supplied with this release, following its separate README: `plan` replays the eligible original CV choices and constructs all three rules and both control types; `train` supplements only genuinely missing complete-query finals and produces a global seal; `score` accepts external class labels only after the global seal verifies. Relative paths in configuration maps resolve against the maps' own directories. Use the source tree that generated the recreated historical protocols; original and comment-translated source hashes must not be interchanged.
4. Read the resulting 1,500 logical scores, 6,000 paired differences, and 600 summary rows. A fresh historical reproduction may have a different missing-fit count; 172 is the observed count for the preserved author run, not a forced target or a selection criterion.

The adapter and its validation report distinguish executed plan/relocation checks from the training path. Packaging does not imply that the entire numerical experiment was rerun through the new adapter. The real 172 fits and their independent rescoring were executed by the frozen provenance scripts; the adapter provides a source-parameterized route to repeat the same protocol without those historical absolute paths. No extra scientific fits were performed for packaging.
