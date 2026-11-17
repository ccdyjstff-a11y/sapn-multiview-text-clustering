# Executed real main-study reproduction: 3Sources, seed 504, Q=160

This is a reproduction of one existing main-study instance, not a new independent performance observation. On 17 November 2026 the coordinating run downloaded the original public `3sources.zip` afresh (425,989 bytes; SHA256 `dea0ab41c48f1a50dbce855f7f9f783dd57bb0a3a3636f53f0c665678fb48659`), rebuilt all three aligned matrices, and used the English release common-candidate solver. No archived initialization or graph was supplied. The warm start and graph were rebuilt, the same paid-query stream was used, and scoring followed the complete one-block prediction seal.

`VERIFICATION.json` records exact equality with the retained original instance: 18 validation fits and four unique final fits, 44 prediction/state NPZ files, 440 arrays, 22 numerical diagnostics after excluding only elapsed seconds and source-derived identity hashes, three selections (Full 2, consensus-only 2, views-only 200), and all 25 metric values for five logical finals. The maximum metric difference is zero. This identity is specific to the recorded numerical environment. It is not evidence that all corpora or arbitrary numerical environments will be bitwise identical.

## Included and omitted evidence

`PROTOCOL.json` freezes the real execution's source hashes and scope; `PUBLIC_SOURCE.json` records the actual download; `PREPARATION_IDENTITY.json` records the three elementwise matrix comparisons. `fit/` preserves all 22 compact prediction NPZs, 22 diagnostics and identities, three selections, query/final records, protocol, global seal and post-seal evaluation records. `historical_reference/evaluation.json` is an unchanged copy of the original seed's evaluation for independent value comparison. The earlier `EXECUTION_COMPLETED.json` describes the pre-comparison stage; the later PASS receipt is `VERIFICATION.json`.

`COMPACT_EXPORT.json` binds 86 exported files to their local scientific sources, and records the identities of 25 omitted fit payloads. The original archive, prepared feature matrices, full class vectors, warm-start arrays, graph, full factor states and evaluation-pair answers are omitted. The historical global seal is retained as evidence; the compact directory is deliberately not a complete input to the original full-state seal validator. The reported 440-array full-state comparison was actually executed by the coordinating verification, but cannot be recomputed from this compact subset alone.

The two exact original execution/comparison scripts are preserved as records in `provenance/original_sources/main_instance/*.py.txt`. They retain their historical directory assumptions and are not advertised as portable commands. `verify_compact.py` is a new portable derivative that verifies this export, its current source bindings, the historical selection records, the five logical prediction mappings and 25 metric values. It runs no training and opens no class vector:

```text
python -B -I validation/reproduction_main/verify_compact.py --output C:/sapn-runs/main-compact-replay.json
```

`COMPACT_REPLAY.json` records execution of this command against the supplied files: 86 export hashes, 40 source hashes, 73 present global-seal members, five historical prediction mappings, three selections and 25 metric values passed. This compact replay is separate from the real training recorded above.

## Repeat the real computation with external data

Obtain the exact original archive from the URL in PUBLIC_SOURCE.json. Use the pinned environment, then save this external JSON as `C:/sapn-runs/prepare-main.json`:

```json
{"dataset":"3Sources","format":"3sources_original_zip","source":"C:/external/3sources.zip","expected_sha256":"dea0ab41c48f1a50dbce855f7f9f783dd57bb0a3a3636f53f0c665678fb48659","output":"C:/sapn-data/main-fresh"}
```

Run from the release root with fresh output paths:

```text
python -B benchmark/prepare_data.py --config C:/sapn-runs/prepare-main.json
python -B -I common/run_allocation.py --prepared-directory C:/sapn-data/main-fresh --paid-queries evidence/v9/allocation_repeat/paid_queries/3Sources/seed504.npz --seed 504 --budgets 160 --output C:/sapn-runs/main-fresh-fit
python -B -I common/score_allocation.py --output C:/sapn-runs/main-fresh-fit --labels C:/sapn-data/main-fresh/labels_FOR_ORACLE_AND_SEALED_SCORING_ONLY.npy
python -B -I validation/reproduction_main/verify_compact.py --fresh-fit C:/sapn-runs/main-fresh-fit --output C:/sapn-runs/main-fresh-comparison.json
```

The fixed one-block plan is complete before scoring. The last command additionally compares all 22 retained compact predictions, diagnostics, selections and evaluation records with the reader's fresh output. It does not claim to repeat the original full-state comparison when those historical full-state payloads are unavailable. No additional real training was started during this export step.
