# Data dictionary

## Shared conventions

| Field | Meaning |
|---|---|
| `dataset` | Representation identifier. `Reuters` and manuscript `Reuters-local` refer to ReutersEN sample 0. Field-corpus full identifiers are retained in primary manifests. |
| `seed` | Query and initialization randomization within a fixed corpus, not an independent corpus. |
| `budget`, `Q`, `total_queries` | Number of paid unordered pair answers. The 160-query stream is a prefix of the 400-query stream. |
| `ACC`, `NMI`, `ARI` | Fraction-scale clustering metrics on objects outside the query pool. ACC uses optimal assignment; NMI uses arithmetic entropy normalization. ARI can be negative. |
| `BER_train`, `BER_unqueried` | Balanced positive/negative relation error. Lower is better. The second uses pool-excluded pairs, at most 10,000, fixed by seed+51000 in the allocation study. |
| `*_difference_pp`, `*_mean_pp`, `Full_minus_control_pp` | 100 times the paired metric difference. Full minus control throughout the current allocation tables. Positive ACC/NMI/ARI is better; positive BER is worse. |
| `sample_SD_pp` | Sample standard deviation across paired seeds, with denominator n-1. It is neither a confidence interval nor an independent-corpus standard error. |
| `eligible_pairs`, `planned_pairs` | Included matched pairs and the planned denominator. Failed or missing outcomes remain in the denominator and are not comparator wins. |
| `C-I`, `V-I` | Consensus-only or views-only, each independently selected by query-only CV. |
| `C-F`, `V-F` | Consensus-only or views-only evaluated at Full's selected coefficient. |
| `best_BER_tie` | At least two eligible candidates within 1e-12 of minimum mean validation BER. |
| `default_fallback` | No eligible fold/candidate forces the declared default. Selecting an eligible tied default is not a fallback. |
| `wins`, `ties`, `losses` | Ten seed-level ACC directions, with absolute difference at most 1e-12 percentage points treated as a tie. |
| `source_member`, `source_sha256`, `source_csv_line` | Original archive member, its original hash, and one-based CSV line, where supplied. These do not refer to modified runnable code. |
| `status`, `converged`, `kkt` | Method-specific execution and numerical eligibility, not common cross-method global-optimality certificates. |

`provenance/CSV_CATALOG.json` lists each supplied CSV's columns, row count and SHA256. The historical files retain their original names and bytes. Current six review tables are under `evidence/current/` and are traced in `provenance/CURRENT_EVIDENCE.json`.

## Arrays and records

Prepared `Xv.npz` files are SciPy sparse sample-by-feature matrices with aligned row order. `metadata.json` gives `n`, `k`, `views` and source metadata. Optional `MV_graph.npz` is the fixed n-by-n Laplacian. Optional `initialization.npz` contains `H0` and `Hv0`, `Hv1`, etc.; `metadata.initialization_seed` must equal the requested seed. A saved initialization must contain no class vector.

Paid-query NPZs contain integer `pool`, ordered unordered `pairs` with i<j, signed `answers` in {-1,+1}, and optionally `test`, the complement of the query pool. Crossing node-fold edges remain paid and enter final training even though they are not CV train/validation edges.

`primary_predictions/MANIFEST.json` maps 210 artifacts to dataset, method, budget, seed, eligibility, metric values, input identity and original record. These compressed arrays contain predicted labels and evaluation indices, not raw features or class truth. Their rescorer checks each artifact hash and the external class partition.

`evidence/v9/allocation_repeat/PREDICTION_MAP.json` maps 500 logical configurations to paid queries and 386 unique prediction artifacts. Reused predictions are legitimate shared fits, not additional experimental observations. Selection JSONs retain candidate/fold BER, eligible folds, chosen index, chosen value and deterministic rule. `GLOBAL_PROTOCOL.json` and seal records preserve the historical scientific procedure and source identities.

Stored state hashes in historical maps may refer to full numerical states kept only in the separate original archive or original research storage. A state hash is not a claim that the state is distributed in this compact directory. New portable runs save their own full states and stage diagnostics.

## Posthoc tie sensitivity

`evidence/tie_sensitivity/selection_policy_index_900.csv` indexes 300 historical selections under each of three prespecified sensitivity rules: primary default-first, minimum numeric lambda, and maximum numeric lambda within the same eligible best-CV tie set (absolute tolerance 1e-12). `logical_final_targets_1500.csv` identifies three independently selected allocations and two controls using each rule's Full-selected coefficient. Its `prediction_member` paths resolve relative to the release root.

The 1,500 logical targets map to 558 unique complete-query predictions (386 historical, 172 added); reuse does not create independent observations. `reporting/` retains 6,000 paired five-metric differences and 600 summaries over the same ten seeds per group. Differences are 100 times Full minus control: positive ACC/NMI/ARI and negative BER favor Full. Summary SD uses ddof=1; W/T/L uses 1e-12 percentage-point tolerance. Complete group summaries require all ten eligible pairs, rather than silently reducing the denominator. Counts and metrics are posthoc sensitivity results, not replacement primary results or independent-corpus replication. Exported `labels` arrays are cluster predictions; true category vectors remain external.
