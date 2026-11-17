# Portable tie-selection sensitivity adapter

Copy `run_tie_sensitivity.py` into the release repository's `tools/` directory. It is a standalone orchestration adapter, with no import from the historical sensitivity scripts. The numerical implementation remains in the user-supplied unchanged `common/` package. Neither the adapter nor this validation changed that package, the historical blocks, or the original results.

This is a posthoc sensitivity analysis: the primary study was already known. It is not preregistration, independent confirmation, an additional independent sample, or a procedure for replacing the primary policy with the most favorable result. The frozen historical `build_selection_index.py`, `run_missing_fits.py`, `score_frozen_sensitivity.py` and `PROTOCOL.md` remain provenance, rather than the portable user entry.

## Inputs and commands

First complete the five-corpus, ten-seed historical profile using the release's existing launcher. Its output must contain all 50 blocks, directly under `<historical-root>/<dataset>/seed<seed>/`:

```text
python -B -I tools/run_common_study.py train --profile historical --data-map inputs/data_map.json --output runs/historical
python -B -I tools/run_tie_sensitivity.py plan --common-dir common --historical-root runs/historical --data-map inputs/data_map.json --output runs/tie_sensitivity
python -B -I tools/run_tie_sensitivity.py train --common-dir common --historical-root runs/historical --data-map inputs/data_map.json --output runs/tie_sensitivity --dry-run
python -B -I tools/run_tie_sensitivity.py train --common-dir common --historical-root runs/historical --data-map inputs/data_map.json --output runs/tie_sensitivity
python -B -I tools/run_tie_sensitivity.py score --common-dir common --historical-root runs/historical --data-map inputs/data_map.json --labels-map inputs/labels_map.json --output runs/tie_sensitivity
```

The historical launcher uses its existing paid-query inputs and external prepared views; see its own documentation. `train` above performs real missing fits and should be run only when those computations are intended. The validation supplied with this adapter did **not** execute that real training command.

`--data-map` is JSON with exactly these keys, mapping to prepared directories containing the original `metadata.json`, `X0.npz`, etc.:

```json
{
  "3Sources": "../prepared/3Sources",
  "BBC": "../prepared/BBC",
  "Reuters": "../prepared/Reuters",
  "Cora": "../prepared/Cora",
  "CiteSeer": "../prepared/CiteSeer"
}
```

`--labels-map` uses the same five keys, mapping to ordered `.npy` class vectors. All JSON map values resolve **relative to the map file's directory**, including after relocation; absolute map values also work. CLI directory arguments resolve relative to the working directory. Only `score` accepts `--labels-map`; `plan` and `train` reject that option at argument parsing. There are no workstation paths or built-in data locations in the adapter.

Use the exact common package and Python dependency versions that created the historical blocks. The release's English `common/` creates compatible fresh blocks, whose protocols bind that source. Original historical blocks bind the original package bytes; use that original package to audit those blocks. Comment/docstring AST equivalence is not used to bypass an old source hash. Do not run with Python `-O`, because integrity assertions in the unchanged core must remain enabled.

Dependencies are those of `common/` (NumPy, SciPy, scikit-learn, threadpoolctl); real training additionally uses psutil for the memory gate. Source and dependency versions must match the protocols. No installation or download is performed by this adapter.

## Fixed design and identity checks

The roster is fixed: 3Sources, BBC, Reuters, Cora and CiteSeer; seeds 504–513; budgets 160 and 400; full, consensus-only and views-only. The candidate order is `[20, 2, 200]`. The adapter first validates every original block seal, source, software identity, prepared input, query/fold reconstruction, CV final state and CV BER against the unchanged `allocation_io.validate_seal` implementation.

There are three policies:

- `primary_default`: retain the original default-first choice.
- `sensitivity_min`: choose the smallest numeric lambda among candidates within absolute `1e-12` of the minimum eligible mean CV BER.
- `sensitivity_max`: choose the largest numeric lambda in the same tie set.

If there is no eligible finite candidate, every policy preserves the original default fallback and explicitly flags the absence of an eligible tie. Failed or nonconverged CV folds cannot become eligible candidates. There is no use of evaluation labels or scores in selection, no new folds, no scaling by Q/Qf, and no change to solver, graph, initialization or objective.

The plan always has **900 choices and 1,500 logical targets**: three independently selected allocations plus two controls using each policy's Full-selected lambda, for every policy/corpus/seed/budget. Physical fits are deduplicated by the complete original identity. Missing counts are determined from supplied blocks; **172 missing / 386 reused / 558 unique are the observed original-data counts, not assumptions about future historical reruns**.

Only a complete-Q final with the correct `_final` stem is eligible for reuse. Identity, source protocol, software, input, graph, complete P/N query arrays, prediction/state hashes, terminal status and convergence are checked. CV artifacts never substitute for final states. Existing final artifacts must belong to the historical seal; partial, corrupt or unsealed artifacts stop processing. Existing failures and nonconverged terminal outcomes are retained rather than replaced.

`PLAN.json`, the reviewable CSV indices and `PLAN_FREEZE.json` freeze the design and identities before fitting. References are relative to the declared roots; the plan binds file hashes rather than workstation paths. The same complete input identities produce byte-identical plan artifacts after relocating the adapter and common package.

## Training and evaluation boundaries

Training uses one worker, an exclusive output lock and one BLAS thread. At least 3 GiB available RAM is required before each block. It copies `PROTOCOL.json`, `inputs.npz` and `used_graph.npz` byte-for-byte to independent `output/fits/` blocks, loads the matching prepared views, and calls the unmodified `run_allocation.fit` factory from `--common-dir`. It never writes to historical blocks, prepared data or common source.

Interrupted or failed artifacts are retained. A complete terminal outcome can be validated and resumed without refitting. An orphan/partial terminal or stale process lock requires inspection; the adapter does not delete or replace it. A sealed study cannot be trained again under the same output. All source, input and historical hashes are rechecked before sealing. `ALL_FINALS_SEALED.json` binds **every unique final**, both reused and newly produced, including failure records, before any label access.

Scoring rechecks the plan and every final identity/hash globally before loading the external labels. It reconstructs pool-excluded evaluation pairs with the original `seed + 51000` rule and maximum 10,000 pairs, verifies historical pairs/manifests if present, and stores evaluation pairs only in the independent output. It also checks the external label vector against n, known k and the paid relationship answers. The original historical evaluation files are never overwritten.

Outputs include:

- `evaluation/logical_metrics_1500.csv`: ACC, arithmetic NMI, ARI, BER_train and BER_unqueried, with terminal status and eligibility.
- `evaluation/paired_five_metrics_6000.csv`: all ten per-seed Full-minus-control differences for every group.
- `evaluation/paired_summary_600.csv`: mean, sample SD, median, range and win/tie/loss counts. The difference is `100 × (Full − control)`; positive ACC/NMI/ARI and negative BER differences favor Full. Ties use `1e-12` percentage points.

All ten paired outcomes must be eligible to produce a group summary. Otherwise the planned denominator and available individual outcomes remain visible; there is no reduced complete-case summary, p-value or significance test. Logical targets sharing a physical final remain shared observations.

## Executed validation and limitations

`validate_adapter.py` performs AST/help/parser checks, selection-only planning, comparison with the original frozen indices, a relocated planning run using a copied common package and relative data map, `train --dry-run`, rejection of scoring without a global seal, and scoring of hard links to the **already completed** 172 sensitivity finals. The validation harness never calls `train` without `--dry-run`, nor the fit factory. It verifies linked source hashes again afterward.

The validation report records the actual results; temporary validation state links and local map files are not release artifacts. The numerical training branch is implemented but **was not run again** for this adapter validation. Planning and stored-state scoring replay do not certify a fresh end-to-end optimizer reproduction or add scientific replications.

To repeat this validation with the private original artifacts, invoke `validate_adapter.py --help` and supply the adapter, frozen sensitivity reference directory, matching common package, historical root, external maps and a separate validation output. That harness is optional audit support; readers reproducing a fresh study need only `run_tie_sensitivity.py`.
