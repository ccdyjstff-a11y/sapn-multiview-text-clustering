# SSA-SNMF: this study's historical adapter sources

This directory preserves the study-authored source needed to inspect the SSA adaptation and its historical batch controller. It is **not a verified portable SSA retraining entry point**. The copied `.py` and `.m` files retain their exact original bytes, including historical path bindings and admission checks. `OWN_SOURCE_MANIFEST.json` records every source path, byte length and SHA256. It contains 33 own-source files: 27 Python files and 6 MATLAB files. Python syntax was parsed without importing or running the modules; all copied code passed a CJK-character scan and has English comments/docstrings. No MATLAB process, training run or network request was started for this packaging step.

## What is included

The `historical_tree/` layout preserves the relative branches used by the original code:

| Branch | Role |
| --- | --- |
| `review_revision_v10/ssa_grouped_v2/` | Complete top-level own Python controller, selection/eligibility policy, query checks, launcher and tests, plus the own MATLAB worker and phase bridge. No generated optimizer is included. |
| `review_revision_v10/ssa_formal_v1/` | Earlier finite-grid preparation and worker patch generator, own launcher, query checks, and own MATLAB worker/bridge inherited by the grouped batch. |
| `review_revision_v10/ssa_minimal_qualification/` | `prepare_profile.py`, which extracts verified spans from separately acquired author code and applies the declared repair/instrumentation edits; and the own synthetic qualification driver whose identity the preparer records. |
| `review_revision_v10/runtime_transition/retirement_review/` | The two source-pinned predecessor-closure readers required by `successor_gate.py`. They inspect operational evidence; they do not generate scientific results or authorize scoring. |
| `review_revision_v10/runtime_transition/future_controller_core/` | The original controller source used by the preparation-time identity check. |
| `review_revision_v9/baseline_gap/native_formal_v1/` | The own query/phase/process helper sources and launcher/clock-bridge templates consumed by the SSA preparation chain. The PCBG-named bridge is a generation template here, not a supplied PCBG fitting entry. |
| `review_revision_v6/literature_followup/SSA_download_manifest.json` | The exact historical download-identity record. It contains metadata, not author source. |

The own numerical adaptation chain is:

```text
externally acquired, SHA-verified SSA author files
  -> ssa_minimal_qualification/prepare_profile.py
  -> local private_generated/ssa_minimal_profile.m and helpers
  -> ssa_formal_v1/prepare_worker.py
  -> local private_generated/ssa_formal_profile.m and helpers
  -> ssa_formal_worker.m + ssa_clock_bridge.m
  -> ssa_grouped_v2 launcher, eligibility and selection controller
```

The minimal patch generator supplies the missing graph Laplacian and changes Cartesian cannot-link indexing to individual symmetric pair indexing. It bypasses the author's demonstration label generation and accepts anonymous queried must-link/cannot-link pairs and a known cluster count. Its additional diagnostics observe the updates. The formal generator exposes the frozen `beta`, `lam`, and `kap` grid and rewrites the scalar trace evaluation without copying a complete author optimizer into this package. These are disclosed adaptations; the wrapper does not claim unchanged-author execution or full feasibility/stationarity of the paper formulation.

## External author dependency

Obtain the author implementation separately from [kamalberahmand/SSA-SNMF](https://github.com/kamalberahmand/SSA-SNMF) at commit `4d37f15621ee7bd9d412ad224408658ad243699c`. `EXTERNAL_AUTHOR_IDENTITY.json` gives the recorded original URLs, byte counts, blob identities and SHA256 values for the eight required source files. These are historical source identities, not a fresh access check. The required author entry `SSA_SNMF.m` has SHA256 `fd0fa1e25ba7c4b1f5e6ccd08321e6ce2626205ed69b06408341220737b89883`.

Neither the author `.m` files nor any `private_generated/` directory is distributed. The tiny source-match strings in the patch generators describe the study's exact edits; the generated full optimizer remains dependent on separately acquired author files. Source availability does not establish redistribution permission for those external files.

## Execution boundary and remaining dependencies

The historical scripts are supplied for audit and adaptation-source recovery. Their preparation sequence is not a one-command reconstruction on another machine. In particular:

1. `prepare_profile.py` expects author files beneath `review_revision_v6/literature_followup/author_source_private_analysis/` and validates their fixed download hashes. It performs preparation at module top level; do not import it as a library.
2. `prepare_worker.py` expects the exact qualified generated profile and the retained own bridge template. It refuses to overwrite an existing generated worker directory.
3. `prepare_formal.py` reads the prior PCBG batch protocol, 72 source MAT inputs, paid-query streams, the SSA design record, and trace-bridge qualification receipts. Those runtime/data artifacts are not synthesized by these sources.
4. `prepare_grouped.py` copies the prior small request files, verifies inherited numerical code and controller bytes, and binds predecessor-closure evidence. `run_grouped.py` requires the exact frozen runtime SHA before dispatch and checks qualified closure and source identities again.
5. Frozen protocols contain historical absolute input, MATLAB and preference-directory paths. They require explicit environment reconstruction and reviewed rebinding. A historical receipt cannot certify a new machine's MATLAB environment or process cleanup.
6. The runtime requires Python scientific packages and process inspection (`numpy`, `scipy`, `psutil`), licensed MATLAB, separately obtained author sources, feature matrices and paid-query records. It intentionally rejects complete class labels from fitting inputs. Missing prerequisites are not replaced with archived final factors or predictions.

The included sources close the **own adapter/patch-generator code chain**, with the runtime and third-party boundaries named above. They do not close the data/environment execution chain. No all-corpus portable retraining equivalence, new model score, or new convergence result is claimed. The release's retained selections, statuses and predictions remain the evidence for the historical execution; re-running it needs a separately reviewed reconstruction of these dependencies.
