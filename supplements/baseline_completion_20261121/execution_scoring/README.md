# Score a fresh baseline completion after its method is sealed

This companion completes the workflow **prepare → select and fit → seal → verify → score** for `public_baseline_execution_tools`. It supplies the post-seal scorer that the frozen runner README described as a remaining interface. The reviewed runner remains unchanged: its manifest SHA256 is `47989992a4f98fd581fa7e7342a5814eb83829aa325d6d74a8e0de22a809675d`. This separate companion records its own hashes and validation.

It scores the new run's **15 SSA conditions** or **4 PCBG conditions**, using an explicit expected global-seal hash. It does not insert new results into the historical 84-row resource, require agreement with published scores, calculate group averages or make inferential comparisons. Failed and otherwise ineligible conditions remain as rows; all five performance fields are empty. A retained finite wrong-k partition keeps its occupied-cluster count and execution status, but does not receive eligible performance scores.

## What the gate checks

Before any external category array is decoded, the scorer verifies the exact reviewed runner file tree, the supplied method-seal hash, the complete condition set, and the sealed run binding. It invokes **only** that runner's `verify` command. This performs the existing full native-evidence and paid-selection checks; it cannot dispatch a new model fit. The runner verifier necessarily reads its bound native evidence to validate the seal. The scorer's separate final-partition decoding for metric calculation happens after category identity and paid-answer checks.

Each consumed plan is bound directly through `RUN_BINDING.json → INPUT_MANIFEST.json → PLAN.json`. The paid queries, final selection, condition record, native receipt and final partition are also hash-bound. A second full method verification and byte recheck occur before score files are created. The scorer refuses absent seals, missing conditions, changed inputs, modified receipts, changed category files or a different runner version.

Class identities do not come solely from a user's label map. `CLASS_IDENTITIES.json` records n/k, original category-file byte hashes and the canonical ordered class-partition hashes derived from two independently pinned public v1.0.1 metadata files:

- `evidence/v10/native_pcbg/partition_export/evidence/FROZEN_PROTOCOL.json`, SHA256 `4a334d1319a17b3d8a266962267c6e19cd2d7ecd9bfff2a78c0afea2b608b117`;
- `primary_predictions/MANIFEST.json`, SHA256 `81ee5eca0f4f5c67238d560bb2a8eddf43d6a36ed5d10d4d5ad4609b7448f9cd`.

The scorer rechecks those public sources and rederives the identities. The external map specifies a path and byte hash for each NPY. Every file's byte hash is checked before any category array is decoded; the decoded vector must match the independently pinned ordered partition and n/k. An equivalent NPY serialization or bijective renumbering of class codes may change the byte hash, but may not change object order or class membership. Every paid answer is then checked against the category vector.

## Five metrics and reporting scope

`baseline_math.py` is the byte-identical reviewed Resource 4 mathematical implementation (SHA256 `e8e97cc332e9e6ca76a08de9125029ee62461b5b2d3d6f0482b53fdc081aa695`):

| Field | Objects or relationships used |
|---|---|
| ACC | Optimal Hungarian label matching on the fixed test-object indices |
| NMI | Arithmetic normalized mutual information on those test objects |
| ARI | Adjusted Rand index on those test objects |
| training_BER | Balanced pair error on all Q paid relationships used for the final fit |
| unqueried_BER | Up to 10,000 unique unordered test-object pairs from the original `evaluation_pair_seed`; both endpoints exclude the query pool |

Pool and test indices must follow the frozen seed rule and be exact complements. The same dataset/seed uses the same test objects and evaluation-pair stream across budgets. If a pair set lacks one relation sign, BER is undefined and remains empty. No zero is substituted for a missing or ineligible result. There is no three-seed aggregation here: the PCBG extension has only two planned seeds per affected corpus, whereas the SSA completion covers five three-seed cells.

## Commands

Use the runner's README to prepare the external features, author sources and private run, and explicitly execute the chosen method. This scorer includes no fitting command and distributes no features, category vectors, partitions, input MATs or author algorithm code. Install NumPy, SciPy, scikit-learn and psutil in the same Python environment used for verification. Use `python -B`.

The examples below use `C:/sapn-tools` for the frozen runner, `C:/sapn-score-tools` for this companion, `C:/sapn-public` for public v1.0.1, and `C:/sapn-run` for the private fresh run. Replace paths explicitly. First read the global-seal SHA256, for example in PowerShell:

```powershell
(Get-FileHash -Algorithm SHA256 -LiteralPath 'C:/sapn-run/SSA/ledger/GLOBAL_PREDICTIONS_SEALED.json').Hash.ToLower()
```

Paste that hash as `<SSA_SEAL_SHA256>` below. A check performs full verification but opens no category arrays:

```text
python -B C:/sapn-score-tools/score_fresh_run.py check --runner-root C:/sapn-tools --release-root C:/sapn-public --work C:/sapn-run --method SSA --expected-seal-sha256 <SSA_SEAL_SHA256>
```

Create an external label-map JSON, using the exact paths produced by the public BBC/Cora/CiteSeer preparer and the actual SHA256 of each NPY. Relative paths resolve against the map file's parent directory. `labels_map.example.json` supplies the original trusted byte hashes; replace them only if the actual equivalent serialization differs. The independent ordered-partition check remains mandatory.

```json
{
  "BBC": {"path": "C:/sapn-data/BBC/y.npy", "sha256": "<ACTUAL_FILE_SHA256>"},
  "Cora": {"path": "C:/sapn-data/Cora/y.npy", "sha256": "<ACTUAL_FILE_SHA256>"},
  "CiteSeer": {"path": "C:/sapn-data/CiteSeer/y.npy", "sha256": "<ACTUAL_FILE_SHA256>"}
}
```

The map is only opened after full method verification. NPY files must remain outside the public release, runner, scoring tools and private run directory. PCBG needs BBC and CiteSeer; SSA needs all three. To calculate a fresh score report:

```text
python -B C:/sapn-score-tools/score_fresh_run.py score --runner-root C:/sapn-tools --release-root C:/sapn-public --work C:/sapn-run --method SSA --expected-seal-sha256 <SSA_SEAL_SHA256> --labels-map C:/sapn-data/labels_map.json --output C:/sapn-scores/SSA-run1
python -B C:/sapn-score-tools/score_fresh_run.py score --runner-root C:/sapn-tools --release-root C:/sapn-public --work C:/sapn-run --method PCBG --expected-seal-sha256 <PCBG_SEAL_SHA256> --labels-map C:/sapn-data/labels_map.json --output C:/sapn-scores/PCBG-run1
```

Each output directory must be new and outside immutable inputs. It contains `FRESH_RUN_ROWS.json`, `fresh_run_metrics.csv`, `SCORING_RECEIPT.json` and `OUTPUT_SHA256.json`. JSON uses null and CSV uses blank cells for undefined/ineligible metrics. The receipt binds the fresh seal, runner, metric implementation, category identities and software versions. It does not assert equality with historical optimization or scores.

## Validation status

```text
python -B C:/sapn-score-tools/test_fresh_scoring.py --runner-root C:/sapn-tools
```

The tests construct temporary synthetic fresh seals and vectors. They test both full denominators, failure missingness, class-order substitution, paid/test scope, five known small-example metric values, exact gate command construction, absent/changed seals, changed final inputs, changed category files and rejection before output on a second verification failure. Synthetic complete-flow tests substitute a test verifier; they do not pretend that synthetic files are real reviewed model runs. Separate runner audits establish the production verifier behavior.

No actual MATLAB rerun, current experimental native result, real category vector or fresh-run test score was read during preparation of this companion. Real fresh scoring remains an explicit operation after an independent optimizer run has a valid complete method seal. This local preparation is not an upload or a publication.
