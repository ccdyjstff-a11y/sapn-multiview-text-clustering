# Public replay of the exact-k-only sensitivity

This small companion reproduces **qualification from the published minimal metadata and aggregation of already computed metrics**. It does not run PCBG, SSA-SNMF or SAPN-MV, read labels/features/partitions, recompute ACC/NMI/ARI from predictions, or repeat CV. The published projection records the non-k worker checks; this replay does not independently inspect the original workers or establish their execution from those records. Their relative evidence keys and hashes are provenance references, not files this script opens.

The protocol was frozen after model development and original benchmark scores were known, and before disclosure of this round's new completion scores. It is not an original preregistration. The primary comparison and selected candidates remain unchanged. The earlier two PCBG groups overlap this analysis and are not independent evidence.

Python 3.10 or later and its standard library are sufficient. No local research repository, skills, external package, MATLAB or network access is required. `non_k_core.py` and `PROTOCOL.json` are byte-identical copies of the frozen author files. The new wrapper adds public input validation, an independently written mean/sample-SD calculation and coverage reporting; it does not change qualification rules.

## Code-only check now

```text
python -B replay_exact_k.py
python -B test_public_replay.py
```

The default command checks only the tool/protocol seals and writes nothing. The tests generate temporary synthetic metadata and scores, remove them afterwards and never open real results. No actual input bundle exists in this prepared code package.

## Replay after the actual analysis has been released

The publisher must first approve the completed analysis and publish its exact `OUTPUT_SHA256.json` digest through the trusted release record. Copy these **five unchanged files** into a separate `data/` directory:

1. `QUALIFICATION_METADATA.json`
2. `SEED_SENSITIVITY.json`
3. `GROUP_SENSITIVITY.json`
4. `ANALYSIS_RECEIPT.json`
5. `OUTPUT_SHA256.json`

The original output manifest indexes eight files. This minimal companion carries only the four JSON payloads needed for replay, plus that original manifest unchanged. The two CSV and two LaTeX entries remain provenance hashes; their absent bytes are **not** claimed to have been verified here. The frozen protocol travels with the code.

```text
python -B replay_exact_k.py --replay --input-root data --expect-source-output-manifest-sha256 PUBLISHED_SHA256 --output NEW_EXTERNAL_DIRECTORY
```

Use the digest from the trusted publication, not a digest generated from an untrusted download. Hashes detect changed bytes; a self-consistent forged manifest is not proof of original execution. The script rejects extra files/subdirectories, escaped or symlinked input files, wrong hashes, duplicate JSON keys, NaN/Infinity, unknown projection fields, absolute/traversing evidence keys and nested containers in scalar metadata. It never follows evidence keys or opens the original worker files. Unknown/missing required qualification fields are handled by the frozen core, with the published exclusion reasons required to match its recomputation. Unknown public container schemas are rejected.

All 84 planned method/run slots must be present. Both methods retain 42 slots, including 18 unsupported slots; the secondary comparison covers four supported corpora, two budgets and three seeds: 24 runs and eight groups per method. The public replay checks every published per-run qualification flag, count, exclusion reason and paired difference. It then recomputes all 16 supported groups using explicit three-seed means and sample SDs and compares them with both the frozen core and the published group table (absolute tolerance `1e-12`). No available-case mean is substituted: each metric needs 3/3 qualified and available pairs. Missing values remain JSON null / empty CSV cells, never zero.

Outputs are all 84 recomputed seed rows, the 16 groups in JSON/CSV, `COVERAGE.json`, a replay receipt and their hash manifest. Coverage distinguishes original outcome categories, strict-gate eligibility, non-k baseline qualification, paired eligibility and complete groups per metric. Category counts are disjoint; exclusion-reason counts can overlap within a run. "Primary eligible" here denotes the strict eligibility flag in the completion inventory; the original primary benchmark files are not changed. Historical-overlap exclusions must agree with the source analysis receipt.

The output directory must be new and outside both the input and code directories. A failed run is not automatically resumed or overwritten. Nothing is published or inserted into a manuscript by this command.

## Author acceptance before release

The exact remaining sequence is in `PUBLICATION_PLAN.json`: wait for the genuine local analysis, copy only the five whitelisted files unchanged, run this public replay in a fresh directory, inspect all 84/16 checks and coverage against the approved analysis, verify that no private path/raw array entered the candidate bundle, and bind the accepted bundle with a public release checksum. This preparation performed only synthetic tests; actual evidence acceptance remains pending.
