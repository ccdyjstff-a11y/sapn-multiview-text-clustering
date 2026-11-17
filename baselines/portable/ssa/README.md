# Portable SSA-SNMF preparation and single-fit interface

This interface removes the historical research-directory dependency from **source preparation and individual SSA fits**. It uses the existing, unchanged patch generators and formal MATLAB worker. It does not redistribute the external author's implementation and does not replace the historical full-study resource controller, selection record or outcomes.

## Inputs

1. The public SAPN release directory. Its `baselines/ssa_historical/` contains this study's own adapters, generators and pinned external identities.
2. Eight separately acquired author files at commit `4d37f15621ee7bd9d412ad224408658ad243699c` of [kamalberahmand/SSA-SNMF](https://github.com/kamalberahmand/SSA-SNMF/tree/4d37f15621ee7bd9d412ad224408658ad243699c). `EXTERNAL_AUTHOR_IDENTITY.json` lists exact raw URLs and SHA256 values. The preparer checks every file before executing any generator. No further historical data or runtime receipt is needed to prepare a fit.
3. Python with the SAPN release's pinned NumPy, SciPy and psutil dependencies, plus licensed MATLAB and the toolboxes used by the author implementation. Set the actual MATLAB executable explicitly.
4. For real data, the prepared `metadata.json` and `X0.npz`, `X1.npz`, etc. described by the release's `DATA_INPUTS.md`, plus a paid-query packet containing only `pool`, `test`, `pairs`, `answers`. These fields give anonymous pair answers and object indices, never a complete class vector.

## Prepare the private runtime

Example paths below are illustrative and should be replaced by the local paths. Use a fresh output directory outside the published repository.

```text
python -B ssa_portable.py prepare --release-root C:/sapn --author-root C:/external/SSA-SNMF --output C:/runs/ssa-runtime
```

The preparer builds a small private tree using the existing source generators and author files. It verifies that the formal profile is generated from the exact historical minimal profile, applies the same two algorithm repairs and trace-cost rewrite, and records hashes for all nine worker/runtime files. The original author files and generated derivatives remain private and must not be added to a public release without appropriate permission. `PORTABLE_PREPARATION.json` uses runtime-relative file bindings and contains no claim that MATLAB has run.

## Construct one full-query input

```text
python -B ssa_portable.py make-input --release-root C:/sapn --prepared-directory C:/data/3Sources --paid-queries C:/queries/3Sources_seed23.npz --query-count 400 --fit-seed 23 --case-id 3Sources_s23_Q400_final --output C:/runs/ssa-inputs/final.mat
```

Features remain n-by-d and are not renormalized by this entry. The unchanged author-profile normalization runs inside MATLAB. Invalid query packets, full-label fields, negative/nonfinite features, entirely missing dimensions and zero object rows are rejected. Native zero-row incompatibility is retained, not repaired by adding constants. The fit seed must be the protocol's `fit_seed`, which need not equal the query stream's seed. Exact historical reproduction also requires the original ordered features and paid stream.

For validation, choose `--fold 0` or `--fold 1` and provide the frozen `--fold-seed`. This uses the same `RandomState(seed).permutation(pool)` halves as the published query helper, trains only on pairs wholly outside the held-out node half, and omits crossing pairs. Both relation signs must occur in training and validation. An ineligible fold is rejected rather than silently replaced. `--fold final` uses the complete query prefix.

## Run a declared configuration

```text
python -B ssa_portable.py fit --runtime C:/runs/ssa-runtime --source-mat C:/runs/ssa-inputs/final.mat --beta 0.1 --lam 0.2 --kap 0.95 --case-id 3Sources_s23_Q400_final --matlab C:/MATLAB/bin/matlab.exe --output C:/runs/ssa-fit
```

`--plan-only` writes a bound `JOB.json`, MAT request and launcher without starting MATLAB. Run it later with:

```text
python -B ssa_portable.py execute --job-dir C:/runs/ssa-fit
```

The launcher creates a separate preference directory, invokes MATLAB as a hidden batch process on Windows, and preserves logs. The original worker uses one computational thread and validates actual MATLAB function resolution. Python verifies the input and numerical-code hashes before and after the fit. A finite wall limit defaults to 600 seconds and can be set by `--timeout-seconds`; timeout is recorded as execution failure and only the task-created process tree is eligible for cleanup after PID/creation-time checks. Pre-existing MATLAB processes are not terminated. Existing job directories and results are never overwritten. Preparation, input and fit output paths inside the public release are rejected.

`EXECUTION.json` records process and worker status separately. `worker_summary.json` preserves the original numerical contract, unsupported-input and failure statuses. A process exiting normally is not sufficient for inclusion in a primary comparison. An iteration-cap return is not a stationarity or convergence certificate. `native_outputs.mat` retains the membership, predicted partition and diagnostics; it does not retain all dense factor/graph states. This entry never computes ACC/NMI/ARI.

## Reproduce the original selection protocol

The full grid has 18 configurations. The default is `(beta, lam, kap)=(.1,.2,.95)`, followed in Cartesian order by `beta=[.1,.2,.3]`, `lam=[.2,.3]`, `kap=[.8,.9,.95]`, omitting the already-listed default. Use the same eligible node-disjoint folds for every configuration. Obtain each held-out relation BER from its predicted partition and the already-paid validation pairs; use the public `baselines/ssa_historical/historical_tree/review_revision_v9/baseline_gap/native_formal_v1/query_protocol.py` helpers `node_folds`, `eligible_folds`, `balanced_pair_error` and `select_configuration`.

Only scores from worker-eligible fits enter selection. Every required fold must qualify; otherwise that configuration is ineligible. Mean validation BER is minimized, with the original default-first tie rule at 1e-12. If no candidate qualifies, retain the declared default and record fallback. Refit the selected configuration using the `final` input. Seal all predictions for the planned study before opening external class labels. Keep all planned failures, unsupported inputs and incomplete conditions in denominators, and do not replace seeds. The historical phase/resource accounting is not reconstructed by this single-fit entry; new runs must have their own execution records and must not reuse the historical timing or eligibility verdicts.

## Validation scope

See `PREPARATION_IDENTITY_QA.json` and `VALIDATION.json` for actually executed checks. Generation identities, synthetic fits, refusal checks and feature/query input equivalence are distinct from a complete new benchmark. No new real-corpus performance result is supplied by this engineering update. The public v1.0.0 benchmark results and their implementation remain unchanged.

Run the preparation and refusal checks without starting MATLAB:

```text
python -B smoke_ssa_portable.py --release-root C:/sapn --author-root C:/external/SSA-SNMF --work-root C:/runs/ssa-smoke
```

The work root must be new. These tests generate private derivative files and synthetic data outside the release; do not publish their work directory. The recorded real-data replay reproduces one historical fixed candidate, including 230 retained scientific leaves exactly. The later Python wrapper hardening and its separately executed checks are identified by source hashes in `VALIDATION.json`.
