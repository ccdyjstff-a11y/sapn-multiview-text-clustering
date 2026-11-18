# Minimal saved-state diagnostic resource

Online Resource 3 supplies the generated numerical states and analysis programs used in the allocation and CMVNMF diagnostics. The repository provides [browsable programs and manifests](https://github.com/ccdyjstff-a11y/sapn-multiview-text-clustering/tree/main/supplements/state_diagnostics_20261118); the complete archive also contains the numerical state files required to run those programs. The resource adds evidence to the identified v1.0.1 research snapshot without changing its model or historical results. Its authorized-distribution and reuse boundary follows [RIGHTS.md](https://github.com/ccdyjstff-a11y/sapn-multiview-text-clustering/blob/df56e6bcc2c5bb82139170e5759523e5ac3936ee/RIGHTS.md). No separate open-source licence or DOI is assigned.

It supports the new allocation decomposition and CMVNMF checkpoint/scale diagnostics for **Allocating pairwise supervision in nonnegative multi-view text clustering**. It contains generated model states, paid relations, original split indices, condition mappings, input hashes and analysis code. It contains no document text, raw or prepared feature matrix, class vector, author baseline source code, or MATLAB/runtime installation.

## Why an additional resource is needed

The original public [v1.0.1 release](https://github.com/ccdyjstff-a11y/sapn-multiview-text-clustering/tree/v1.0.1), commit `df56e6bcc2c5bb82139170e5759523e5ac3936ee`, includes final partitions, query streams and some consensus H arrays. It does not contain the view-specific factors needed for the new allocation identity, or the MU/final U/Hv factor states needed for the new CMVNMF objective and scale analysis. Re-scoring a final partition is a different reproducibility claim from replaying these state diagnostics.

| Content | Scope | Arrays |
|---|---|---|
| `allocation/states/` | 386 unique saved fits mapped to 500 logical conditions | H, Hv0…Hv(V−1), P, N |
| `cmvnmf/states/` | 42 original selected-coefficient conditions | final U/Hv, MU U/Hv when continuation occurred, P, N, pool, test |
| `ALLOCATION_RUNS.json` | Five corpora × ten seeds × two budgets × five controls | Parameters, unique-state references and original input hashes |
| `CMVNMF_RUNS.json` | Seven corpora × three original seeds × two budgets | Fixed selected coefficients, stage aliases, original objective/residual endpoints and provenance |
| `FEATURE_REQUIREMENTS.json` | Seven external feature inputs | Exact row/feature-order fingerprints, historical file hashes and external ordered-label partition hashes |

Four CMVNMF conditions did not continue beyond MU; their MU stage refers to the same final factors instead of storing a redundant copy. Reused allocation fits are not additional observations. Cluster predictions are derived from the factors, so no class vector or redundant prediction vector is stored here.

## Integrity and environment

`MANIFEST.sha256` binds `MANIFEST.json`; the manifest binds every distributed input file. Analysis verifies that seal before using states or external data. Run without `-O` or `-OO`, which disable required integrity assertions. Use `-B` to avoid adding bytecode caches to the resource.

The executed environment is Python 3.10 with NumPy 2.2.6, SciPy 1.15.3, scikit-learn 1.7.2 and threadpoolctl 3.7.0. All numerical analysis is limited to one BLAS thread. No training or model selection occurs. Small floating-point differences on a different numerical stack must be assessed against the recorded diagnostic tolerances rather than treated as evidence of a new fit.

Write outputs to a fresh directory **outside** this resource. Use explicit external paths in configuration JSON files. Relative feature paths resolve against the running process; relative label paths resolve against the label-map JSON directory. Absolute external paths avoid that distinction.

```text
python -B analyze_allocation.py --output /external/results/allocation
python -B analyze_cmvnmf.py --feature-map /external/config/features.json --output /external/results/cmvnmf
python -B score_checkpoints.py --labels-map /external/config/labels.json --primary-root /external/sapn-v1.0.1/primary_predictions --output /external/results/checkpoint_scores
```

Allocation replay needs no external features or labels. The complete CMVNMF reconstruction/objective replay requires the exact external features. Checkpoint accuracy and SAPN-MV comparisons additionally require external class labels and the original public primary-prediction packet. See [ACCESS_ROUTES.md](ACCESS_ROUTES.md) for preparation and scoring paths; raw corpora are not redistributed here.

`FEATURE_MAP.example.json` illustrates explicit feature-view paths. A prepared n-by-d sparse CSR view is checked through its ordered numerical values independently of ZIP serialization. After validation, it is scaled to squared Frobenius norm n and transposed to d-by-n. Reuters also accepts the exact historical MAT through `{"reuters_mat":"/external/Reuters-1200.mat"}`; only its `X` variable is loaded, and featurewise min-max preprocessing is applied before identity checking. Original view order must be retained.

## Interpretation boundaries

The allocation replay evaluates the stated identity, disagreement measure and bounds at saved states. It does not infer that nonzero disagreement must cause a nonzero energy change; linear and quadratic contributions can cancel. The 500 logical comparisons reuse 386 unique states.

CMVNMF compares the original MU and final checkpoints at each run's originally selected coefficient. Cross-validation is not rerun for an MU-only comparator, and stages are not chosen using test scores. Pure scaling and MU-to-final continuation are separate diagnostics. The scale replay retains three argmax changes across the three tested rescalings of one MU checkpoint, involving subnormal factors that round to zero. The original normalized argmax margin is approximately 1.65e-294 and is not itself subnormal. The case is not removed or repaired. Algebraic product invariance and finite-precision partition equality are different statements.

The scoring script reports all 42 pairs and all 14 corpus–budget conditions using original test indices. ACC uses optimal assignment; NMI uses arithmetic normalization; ARI is retained. No p-values, new fits, retries or primary-result replacements are introduced.

## Verified reference outputs

`reference_results/` contains seven CSVs from the complete portable replay: 500 allocation rows and 50 summaries, 42 CMVNMF checkpoint pairs, 252 scale transformations and 14 objective summaries, plus 42 checkpoint-score rows and 14 score summaries. They are reference outputs, not analysis inputs or additional observations. `REPLAY_REFERENCE.json` records their exact hashes and the verified comparison scope. Only the scoring step reads separately obtained external categories.

The source-to-export check covered all 2,829 numeric arrays by dtype, shape and element bits. The five factor-diagnostic CSVs reproduced all 18,558 numeric cells exactly; checkpoint-score differences from the earlier sealed summaries were below 4e-15. The original v1.0.1 tag remains unchanged and did not already contain these factor states. No SSA or PCBG completion result is included in this diagnostic resource.
