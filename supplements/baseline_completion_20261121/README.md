# Online Resource 4: baseline results, fitting and scoring

This resource provides two distinct ways to reproduce the baseline completion. `sealed_results/` recalculates the reported results from the original generated partitions. `execution_tools/` and `execution_scoring/` support a new optimizer run, with external features and separately obtained author code, followed by scoring after that run is sealed. Reproducing partition metrics alone does not reproduce fitting.

| Directory | Operation | Scope and required external inputs |
|---|---|---|
| `sealed_results/` | Recalculate the original five metrics, coverage and complete paired groups | All 42 planned conditions per method, 84 rows total; external category NPY files are required only for finite retained baseline partitions |
| `execution_tools/` | Reconstruct inputs, run paid-only selection, fit and seal an independent completion | 15 SSA conditions, up to 540 CV fits plus 15 finals; four PCBG conditions with 23 retained original CV records, one new CV fit and four new final fits; external features, author source and MATLAB are required |
| `execution_scoring/` | Verify and score the newly sealed optimizer run | Exactly 15 SSA or four PCBG final outcomes; external category NPY files with verified object-order identities are required |

The archive contains generated baseline/SAPN partitions, paid relationships and answers, pool/test indices, execution and selection metadata, original metric tables, provenance and our analysis/orchestration programs. It contains no raw X, complete category vector y, request/native MAT container, private runtime, third-party algorithm source or MATLAB installation. Obtain external inputs and author source through the unmodified public v1.0.1 documentation and the linked inner READMEs. No new licence or DOI is assigned by this archive.

## Recalculate the original results

Follow [sealed_results/README.md](sealed_results/README.md). Use the **inner** `sealed_results/MANIFEST.json` hash from its adjacent `MANIFEST.sha256`; the outer publication manifest is a different identity.

From this extracted resource directory:

```text
python -B sealed_results/rescore.py --root sealed_results --expect-manifest-sha256 <INNER_RESULT_MANIFEST_SHA256> --labels-map C:/external/labels_map.json --output C:/sapn-replay/original-results
```

The replay checks all 84 rows and all 14 corpus–budget groups per method against the original references within absolute tolerance 1e-12, including exact missingness. A paired group requires all three planned seeds to be eligible for both the baseline and original SAPN. Unsupported, missing, runtime-failure and finite wrong-k outcomes remain visible and do not become wins. The separate `verification/REPLAY_QA.json` is the actual complete replay receipt used to admit this publication layer; it binds the unmodified inner result manifest and completed pipeline.

## Run the optimizers independently, then score

Follow [execution_tools/README.md](execution_tools/README.md) for the full setup, author-source identities, scientific input checks, launch policies and resume rules. The companion [execution_scoring/README.md](execution_scoring/README.md) now supplies the post-seal scoring interface mentioned as a remaining step in that frozen runner README. Both directories remain byte-identical to their separately reviewed versions.

The workflow is:

1. Obtain the public v1.0.1 release, the exact external prepared features and the separately acquired author sources.
2. Use `execution_tools/prepare_inputs.py` to verify metadata, then explicitly generate new private inputs. Use `completion_portable.py plan` to inspect the full schedule.
3. Explicitly run the desired method. SSA retains the 600-second whole-process cap; the PCBG extension uses 300-second startup, 1800-second worker and 60-second shutdown limits. These are independent replays, not claims of identical historical scheduling.
4. After the complete method is sealed, invoke `execution_scoring/score_fresh_run.py check`, then `score`, supplying that new method-seal hash and an external category map.

For example, after a successful SSA seal:

```text
python -B execution_scoring/score_fresh_run.py score --runner-root execution_tools --release-root C:/sapn-public --work C:/sapn-run --method SSA --expected-seal-sha256 <NEW_SSA_METHOD_SEAL_SHA256> --labels-map C:/external/labels_map.json --output C:/sapn-scores/SSA-run1
```

New optimizer results can differ across machines or MATLAB versions. Their scorer reports the five metrics and execution status for every planned new condition, leaves all ineligible metric fields empty, and neither replaces the original 84-row results nor demands numerical agreement with them. It computes no group means or inferential claims. In particular, the four-condition PCBG extension is not a complete new 42-condition benchmark.

## Identities and immutable layers

`MANIFEST.json` and `MANIFEST.sha256` at the archive root bind the complete publication layer. `PUBLICATION_SOURCE_BINDINGS.json` links the admitted completed pipeline, original result export, actual replay receipts and both reviewed tool manifests. The original result directory includes its original manifest and sidecar. The tool manifests retain their reviewed hashes:

- execution tools, 12 payload files: `47989992a4f98fd581fa7e7342a5814eb83829aa325d6d74a8e0de22a809675d`;
- fresh-run scoring, eight payload files: `05a9a1e4411b2672dc0c03ebd845d604c2b303dcc3eb489e1c63ffd1b5e19b54`.

Each tool directory also retains its two manifest control files. Private validation paths mentioned in the original QA metadata are provenance pointers only; those files and private workspaces are not in this archive. Use `python -B` so that tool verification is not affected by added bytecode caches. This resource does not modify the original public v1.0.1 snapshot.
