# SAPN reproducibility materials

This directory provides directly inspectable Python sources, result tables, prediction files, paid-query streams, and source mappings for **Allocating pairwise supervision in nonnegative multi-view text clustering**. This snapshot is prepared for public distribution with the manuscript. No DOI or separate open-source license has been assigned. See [RIGHTS.md](RIGHTS.md) for the distribution and reuse status, and [CITATION.md](CITATION.md) for identifying a specific revision.

## What can be run

| Directory | Purpose and boundary |
|---|---|
| `common/` | Frozen common-candidate solver used by the main ten-seed allocation study. All queried endpoints can compare simplex vertices, including eta=0 view blocks. |
| `benchmark/` | Gated historical SAPN benchmark runner, CMVNMF/MVCHSS paper-based Python implementations, FSSMSC acquisition/adaptation bridge, and source-data preparation. External author code is obtained separately. |
| `baselines/` | Historical adapters plus new portable PCBG condition and SSA single-fit entries in `baselines/portable/`. Exact author code remains external; validation scope is documented for each entry. |
| `tools/run_tie_sensitivity.py` | Posthoc three-rule tie sensitivity with identity-checked reuse, missing-final training and global-seal scoring; executed validation is planning/relocation/dry-run/stored-state replay. |
| `tools/run_common_study.py` | Full five-corpus, ten-seed orchestration and an explicitly separate two-corpus extension preset, with a study seal before scoring. |
| `primary_predictions/` | All 210 original primary predictions, mapping and independent external-label rescoring, covering five methods and seven corpora at two budgets. PCBG and SSA-SNMF extensions are separate. |
| `evidence/`, `tables/` | Historical CSV results and source mappings, 500 logical allocation results with 386 unique predicted states, 50 paid-query streams and 300 allocation selections. |
| `provenance/` | Historical archive identity, per-file origins, original source bytes as `.py.txt`, and documentation-translation checks. |
| `figures/` | Figure-to-data/source map, original research plotting sources, and three executed portable redraw entries. |
| `validation/` | Executed release checks, not additional performance experiments. |

The common and gated branches are different algorithms for reporting purposes. Do not substitute one branch's numerical result for another. Original archive hashes remain historical identities; English runnable copies have new hashes and newly bound provenance manifests. `provenance/TRANSLATION_QA.json` proves equality of executable tokens and AST after removing docstrings. Full ASTs differ where `__doc__` changes. Runtime error messages were retained to preserve executable string constants.

## Environment

Use Python 3.10, preferably the recorded 3.10.22 environment, and install the pinned packages into an isolated environment:

```text
python -m pip install -r requirements.txt
```

The recorded versions are NumPy 2.2.6, SciPy 1.15.3, scikit-learn 1.7.2, threadpoolctl 3.7.0, psutil 7.2.2 and h5py 3.16.0. SAPN needs no MATLAB. Historical third-party baselines used MATLAB separately. See `benchmark/ENVIRONMENT.json`; exact bitwise identity on another numerical stack is not promised. Source preparation and runtime measurements are distinct from model-selection cost.

Commands below run from this directory. Use fresh output paths outside the release directory. On Windows prefer a short extraction path, such as `C:/sapn`, to avoid historical evidence path limits. JSON must be UTF-8 without a BOM. Paths in user configuration files are resolved relative to the configuration file unless documented otherwise.

## Quick checks without external corpora

```text
python -B -I tools/verify_evidence.py --output ../sapn-check/evidence.json
python -B -I tools/smoke_equivalence.py --output ../sapn-check/synthetic.json
python -B -I tools/smoke_baselines.py --output ../sapn-check/python-baselines.json
```

The evidence check verifies provided hashes and recomputes allocation summaries from the supplied paired values. It does not independently rescore labels. The synthetic check reconstructs original and English source trees in a temporary relocated directory, runs both in separate isolated Python processes, fits all three allocations through the actual CV/final entry point, seals predictions, scores, and checks states and numerical diagnostics. It changes no historical result. Its temporary historical Python copies are removed automatically.

A real main-study instance was also reproduced from a fresh public 3Sources download: seed 504, Q=160, rebuilt features, warm start and graph, all 18 validation fits and four unique finals, then sealed scoring. Its 44 prediction/state files (440 arrays), 22 numerical diagnostics, three selections and 25 metric values matched the original instance exactly in the recorded environment. See [the executed real reproduction](validation/reproduction_main/README.md) for the unchanged verification receipt, compact evidence, source bindings, independent replay and actual preparation/training/scoring commands. This reproduces one existing instance; it adds no independent performance observation.

## Prepare the seven corpora

Follow [DATA_INPUTS.md](DATA_INPUTS.md). Original raw archives and class-label vectors are external. Prepared views use nonnegative SciPy CSR matrices `X0.npz`, `X1.npz`, etc., each shaped **n objects by d features**, plus `metadata.json` with integer `n`, `k`, and `views`. Labels enter only oracle generation or scoring. Never put a class vector in paid-query or initialization files.

Do not Frobenius-normalize views before the recorded warm start. The solver applies reconstruction normalization at the specified stage. Rebuilding the fixed graph under a different floating-point stack can change tied neighbor ordering. For exact historical replay, use the corresponding graph and initialization identities; regenerated inputs must be compared numerically before claiming exact reproduction.

## One common-candidate block

Prepare 3Sources in an external directory. Then run the paid stream already retained for seed 504:

```text
python -B -I common/run_allocation.py --prepared-directory C:/sapn-data/3Sources --paid-queries evidence/v9/allocation_repeat/paid_queries/3Sources/seed504.npz --seed 504 --budgets 160,400 --output C:/sapn-runs/common/3Sources/seed504
python -B -I common/score_allocation.py --output C:/sapn-runs/common/3Sources/seed504 --labels C:/sapn-data/3Sources/labels_FOR_ORACLE_AND_SEALED_SCORING_ONLY.npy
```

Only run the second command after all predictions required by the intended study have been sealed. For a single-block smoke study, the block is the entire planned study. Each positive budget performs three independent selections from `[20,2,200]`, then Full, consensus-only and views-only final fits plus two controls at Full's selected coefficient. Identical final fits are reused with explicit logical records. Failures remain recorded; do not replace seeds or drop incomplete conditions.

## Full historical common-candidate study

Create `data_map.json` outside this directory:

```json
{
  "3Sources": "C:/sapn-data/3Sources",
  "BBC": "C:/sapn-data/BBC",
  "Reuters": "C:/sapn-data/Reuters",
  "Cora": "C:/sapn-data/Cora",
  "CiteSeer": "C:/sapn-data/CiteSeer"
}
```

Create a separate `labels_map.json` with the same keys and exact class-vector paths. Training does not accept this map.

```text
python -B -I tools/run_common_study.py train --data-map C:/sapn-data/data_map.json --output C:/sapn-runs/common-full
python -B -I tools/run_common_study.py score --data-map C:/sapn-data/data_map.json --labels-map C:/sapn-data/labels_map.json --output C:/sapn-runs/common-full
```

This executes all 50 corpus/seed blocks, seeds 504 through 513, budgets 160 and 400. It writes `STUDY_PREDICTIONS_SEALED.json` only after every block is sealed, then permits scoring. It retains 500 logical finals. The reported historical run used 1,800 validation fits and 386 unique finals; the new run must report its actual reuse and eligibility. A successful launcher exit alone does not certify scientific equivalence or convergence of every candidate.

## Two-corpus allocation extension

The separate `allocation-extension` preset declares the already used 20NG and AG News field representations, seeds 1101 through 1105, and Q=400. It calls the same English common-candidate solver, with eta 0, 0.5 and 1 and the default-first coefficient grid `[20,2,200]`. These are known benchmarks; this extension is not a never-seen independent dataset test. Its completed results and provenance are retained separately from the historical ten-seed study under `evidence/allocation_extension/`. All 50 logical finals (39 unique predictions), 180 validation fits and 30 selections are accounted for, with seven validation ties and no fallback. Independent original-label rescoring after the global seal matched all 250 metric values exactly. This compact export omits raw features, class vectors and full optimizer states; its omission ledger prevents interpreting original seals as a claim that every named artifact is distributed. Negative as well as positive contrasts are retained.

Prepare external `extension_data.json` with keys `20NG` and `AGNews` pointing to the prepared directories from DATA_INPUTS.md, and `extension_labels.json` with the same keys pointing to the exact external label files. Data-map keys identify the study blocks; each block's result preserves the dataset name in its prepared metadata. Use a fresh external query root and output directory:

```text
python -B -I tools/run_common_study.py oracle --profile allocation-extension --data-map C:/sapn-data/extension_data.json --labels-map C:/sapn-data/extension_labels.json --queries-root C:/sapn-runs/extension-queries --output C:/sapn-runs/extension
python -B -I tools/run_common_study.py plan --profile allocation-extension --data-map C:/sapn-data/extension_data.json --queries-root C:/sapn-runs/extension-queries --output C:/sapn-runs/extension
python -B -I tools/run_common_study.py train --profile allocation-extension --data-map C:/sapn-data/extension_data.json --queries-root C:/sapn-runs/extension-queries --output C:/sapn-runs/extension
python -B -I tools/run_common_study.py score --profile allocation-extension --data-map C:/sapn-data/extension_data.json --labels-map C:/sapn-data/extension_labels.json --queries-root C:/sapn-runs/extension-queries --output C:/sapn-runs/extension
```

The oracle alone reads classes to answer the label-blind 30% pool and unique pair stream (seed offsets 21000 and 31000). Skip oracle when using an already verified paid-query root with `<corpus>/seed<seed>.npz`. Training refuses a label-map argument. It freezes input, query and source hashes before fitting, dispatches sequentially with one BLAS thread, and starts extension blocks only with at least 3 GiB of available RAM. All ten blocks and 50 logical finals must be sealed before the score phase opens the class map. A partially written unsealed block stops the launcher for inspection; no replacement seed or reduced complete-case summary is produced. Numerical failures already recorded by the solver remain in its sealed logical records. This stricter operational gate does not alter solver updates or selection.

The CLI also accepts explicit `--corpora 20NG,AGNews --seeds 1101:1105 --budgets 400` with this preset. Different values require `--profile custom` and all three flags; custom runs must not be reported as the frozen extension. `plan` prints identities and the complete expected roster without training. `tools/smoke_study.py` validates the training/score separation on synthetic data only. No real-corpus extension training was launched by release assembly.

## Posthoc tie-selection sensitivity

The separately sealed sensitivity package is in `evidence/tie_sensitivity/`. It retains default-first selection as primary and compares minimum- and maximum-lambda choices within the original eligible best-CV-BER tie set (`1e-12`). The original results were already known: this is posthoc sensitivity, not preregistration or independent confirmation. Independent-CV positive mean ACC contrasts are 19/20, 18/20 and 16/20 under the three rules; policy-specific Full-lambda controls have 19/20 under each. Unfavorable contrasts and all planned cases remain visible.

The export contains 172 new compact complete-query predictions and references 386 existing historical predictions by exact SHA256. These 558 unique finals supply 1,500 logical records, 6,000 paired differences and 600 summaries. Shared fits and repeated policies are not independent replicates. The original seal and an explicit included/omitted-member ledger distinguish compact evidence from omitted full optimizer states and raw inputs.

The portable adapter is already installed at `tools/run_tie_sensitivity.py`; [its command and protocol guide](docs/TIE_SENSITIVITY.md) explains `plan`, `train`, and `score` after rebuilding the historical blocks with a consistent common source tree. Its source-parameterized training branch is available, but validation executed planning, seven byte-identical relocated plan artifacts, `train --dry-run`, refusal checks and stored-state scoring replay only. It did not rerun an optimizer or recreate the historical blocks from fresh data. Original and English common-source hashes must match their own bound historical protocols and cannot be interchanged merely because their executable logic agrees.

A second entry replays the published compact predictions without raw features or optimizer states:

```text
python -B -I evidence/tie_sensitivity/tools/rescore_tie_compact.py --release-root . --labels-map C:/sapn-data/labels_map.json --output C:/sapn-runs/tie_compact_replay.json
```

This exact compact replay was executed with external original class vectors: all 558 predictions, five metrics, 1,500 logical records, 6,000 paired differences and 600 summaries matched, with zero maximum difference. `validation/TIE_COMPACT_REPLAY.json` binds the script, public manifest and query protocol; `validation/tie_portable_adapter/VALIDATION_REPORT.json` separately records the adapter's executed checks and unexecuted training scope. No label vectors or local maps are distributed.

## Gated SAPN and baseline benchmark

Use the same prepared-directory format and three external JSON files. For example, an oracle configuration contains `data`, `labels`, `seed`, `total_queries`, and `output`. Training contains `data`, `queries`, `output`, `seed`, `budgets`, and `methods`. Scoring contains `output` and `labels`.

```json
{"data":"C:/sapn-data/20ng","labels":"C:/sapn-data/20ng/y.npy","seed":41,"total_queries":400,"output":"C:/sapn-runs/20ng_queries.npz"}
```
```json
{"data":"C:/sapn-data/20ng","queries":"C:/sapn-runs/20ng_queries.npz","output":"C:/sapn-runs/20ng_seed41","seed":41,"budgets":[400],"methods":["SAPN-MV"]}
```
```json
{"output":"C:/sapn-runs/20ng_seed41","labels":"C:/sapn-data/20ng/y.npy"}
```

```text
python -B benchmark/reproduce.py oracle --config C:/sapn-runs/oracle.json
python -B benchmark/reproduce.py train --config C:/sapn-runs/train.json
python -B benchmark/reproduce.py score --config C:/sapn-runs/score.json
```

This historical entry uses sibling imports, so use `-B` rather than `-I`. It is included without algorithm edits. The supplied method names are `SAPN-MV`, `SAPN-EQ`, `CMVNMF`, `MVCHSS`, and, after separate author-code acquisition and MATLAB setup, `FSSMSC`. For both SAPN variants use `"methods":["SAPN-MV","SAPN-EQ"]`; for the Python baseline comparison use `"methods":["CMVNMF","MVCHSS"]`. Primary seeds are 23–25 on the five development corpora, 41–43 on 20NG and 101–103 on AG News. The two demonstrated historical field refits are documented in `reproduction/FIELD_REAL_TRAINING_COMPARISON.json`. See [baseline interfaces](baselines/README.md) for the included adaptations, external dependencies, and the explicitly nonportable historical PCBG/SSA training sources.

## Score all 210 stored primary predictions

Prepare a labels map using the exact keys `3Sources`, `BBC`, `Reuters`, `Cora`, `CiteSeer`, `20NG_SubjectBody_2000`, `AGNews_TitleDescription_2000`, with file paths as values. The rescorer checks the class partition and sample count, then evaluates only stored pool-excluded indices.

```text
python -B -I primary_predictions/rescore.py --labels-map C:/sapn-data/primary_labels_map.json --output C:/sapn-runs/primary_rescore.json
```

This exact entry was executed with the external original class partitions for all 210 stored predictions: five methods, seven corpora, two budgets and three seeds. All 630 ACC/NMI/ARI comparisons passed; the maximum absolute difference from the recorded values was `1.1102230246251565e-16`. The unchanged execution report is `validation/PRIMARY_REAL_RESCORE_630.json`; its source/scorer binding and the seven ordered class-partition hashes are in `validation/PRIMARY_REAL_RESCORE_BINDING.json`. Neither labels nor the local path map are distributed. The 210 already include the 60 field-corpus predictions; they are not an additional set of 60. This verifies scoring of stored predictions, not optimizer rerunning, stationarity or extra independent observations. See [EVIDENCE_MAP.md](EVIDENCE_MAP.md) for other claims and limits, and [DATA_DICTIONARY.md](DATA_DICTIONARY.md) for units, denominators and missing outcomes.

The retained SSA-SNMF compact evidence also has a read-only replay entry:

```text
python -B -I reproduction_ssa/source/replay_ssa_compact.py --root reproduction_ssa --manifest-sha c5b62a0fba4c5d8ae45b1af7fb14a97957d741b70bcf4b59cb97868c98768c06 --output C:/sapn-runs/ssa_replay.json
```

This checks paid-query/CV BER, selection and scheduler evidence. It does not rerun MATLAB, reconstruct all factor states, or rescore evaluation-set ACC/NMI/ARI. The supplied source files are the research project's compact replay tools, not redistributed author MATLAB code.

## Redraw the scientific figures

See [the complete figure-to-data/source map](figures/README.md). The analytical reference-risk plot (figure 13), main allocation plot (figure 21) and two-corpus extension plot (figure 22) have portable entries and executed numeric/export checks. Version 1.0.1 also supplies the four typography redraws documented below. Historical generators for the other figures are preserved with source identities and their original dependencies; their presence is not a new portability certification. No raw data or training is needed for these three redraws.

## Release status

The archive used for extraction has SHA256 `3e0e1c75574678d0c0a2026afe98a7b9ab58bc283eac453544e1d3673b044179`. Keep it as a separate historical attachment. Do not replace its frozen members with English copies. The repository provides a directly inspectable form of the research materials. Public-access verification and the exact repository revision are recorded by the publication workflow after upload; preparing these files alone does not establish online availability. An actual verified repository link and revision must be used in the manuscript Declaration. No additional open-source license is inferred from public hosting. See [PUBLICATION_NOTES.md](PUBLICATION_NOTES.md) for the explicit treatment of historical workstation paths, retained exception strings, and the local packaging inventory.

## Repository and version

Browse the code and evidence at https://github.com/ccdyjstff-a11y/sapn-multiview-text-clustering. The release tag is `v1.0.1`; use its full commit SHA to identify the exact version. See [CITATION.md](CITATION.md) for citation boundaries and [RIGHTS.md](RIGHTS.md) for rights status.

## Version 1.0.1 portability and typography update

The fitted model, selection rules and historical experimental results are unchanged. This patch adds:

- [Portable native PCBG](baselines/portable/pcbg/README.md), covering preparation, query-only CV selection, a selected final fit, sealing and external-label scoring. One real 3Sources condition, seed 23 and Q=160, reproduces all seven historical fit outputs and its selected coefficient and metrics.
- [Portable SSA-SNMF](baselines/portable/ssa/README.md), covering identity-checked reconstruction of the original private runtime, label-free MAT preparation and individual MATLAB fits. The exact executed scope is in its VALIDATION.json; this is not a rerun of the full 18-configuration benchmark.
- [Portable typography redraws](figures/qa/final_typography/README.md) for four supplementary figures. All 849 scientific PDF paths retain their geometry and style at 0.001 pt precision. Changes improve body-text legibility at manuscript width without changing data or results.

All original failures and incomplete comparison cells remain. New launchers use explicit paths and bounded execution; they do not reproduce the historical memory-suspension scheduler or provide new independent performance observations. Obtain external author code and raw corpora under their original terms. No additional software license is granted by this patch. The immutable v1.0.0 tag remains available for the preceding release.


## Additional saved-state diagnostics (18 November 2026)

[Online Resource 3](https://github.com/ccdyjstff-a11y/sapn-multiview-text-clustering/releases/download/v1.0.1/Online_Resource_3_State_Diagnostics.zip) supplies the factor states and programs for the allocation-disagreement and CMVNMF checkpoint/scale analyses. [Browse the analysis programs](supplements/state_diagnostics_20261118/README.md); [archive identity and scope](supplements/state_diagnostics_20261118/ARCHIVE.json) identify the exact download.

The supplement contains 386 unique allocation states for 500 logical records, 42 CMVNMF conditions, and verified reference outputs. It adds evidence without changing the SAPN model, primary predictions, or the immutable v1.0.1 commit. It excludes external feature matrices, full class vectors and third-party baseline source. It contains no SSA/PCBG completion result. Obtain external data separately through the recorded access routes. The existing rights and reuse boundary remains in force.


## Seven-figure presentation supplement (18 November 2026)

[Browse the replay code and source data](supplements/figure_presentation_20261118/README.md) or [download the verified ZIP](https://github.com/ccdyjstff-a11y/sapn-multiview-text-clustering/releases/download/v1.0.1/SAPN_Figure_Presentation_Supplement.zip). This supplement reproduces the revised presentation of seven figures from retained numerical sources. Its manifest and output checks bind the data, scientific geometry and model connections; it adds no fitted model or performance result. External fonts and the recorded Python environment are required. Raw corpora, full class vectors and third-party implementations are excluded. The v1.0.1 tag and original scientific files remain unchanged.


## IJMLC model-figure typography supplement (19 November 2026)

[Browse the drawing code and provenance](supplements/ijmlc_model_typography_20261119/README.md) or [download the separate ZIP](https://github.com/ccdyjstff-a11y/sapn-multiview-text-clustering/releases/download/v1.0.1/SAPN_IJMLC_Model_Typography_Supplement.zip). This companion reproduces the bold matrix glyphs in the IJMLC-style model schematic. It changes eleven text labels through ten reversible substitutions; the model dependencies, node and edge geometry, and scientific content are unchanged. The original seven-figure supplement above remains a separate fixed version. This addition contains no model fit, new empirical result, manuscript, or third-party implementation. The existing rights and reuse boundary remains in force.
