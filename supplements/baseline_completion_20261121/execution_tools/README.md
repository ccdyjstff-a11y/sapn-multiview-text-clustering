# Portable fitting of the revision's baseline completion

This companion supplies the missing **selection and execution orchestration** for the revision, using the public v1.0.1 single-fit interfaces. It reconstructs 15 SSA conditions and the five-fit PCBG extension. It does not run the complete original 42-condition study again. The separate Resource 4 replays already sealed predictions and scores; that is a different operation.

The distributed files contain our Python orchestration, protocol fields, content identities, examples and tests. They contain no feature arrays, full category vectors, third-party MATLAB source, numeric input MAT files, optimizer results or MATLAB installation. The generated private work directory contains external features and must not be added to a public result archive.

## Scope and preserved decisions

| Method | Conditions | Selection | New fitting budget | Eligibility and failures |
|---|---|---|---|---|
| SSA | BBC Q160 × seeds 23–25; Cora and CiteSeer Q160/Q400 × seeds 23–25: **15** conditions | 18 default-first candidates, the same two node-disjoint paid folds, complete-fold mean paid BER, global-minimum tie set with tolerance 1e-12 | At most **540 CV fits + 15 final fits**; 45 c0 input carriers are reused by requests for the actual candidates | A terminal ineligible fold skips only that candidate's later fold. High BER never prunes. An unresolved preworker failure stops the controller. All-invalid selection still attempts the declared default final and retains its status. |
| PCBG | BBC seed 24/25 Q400, CiteSeer seed 23/25 Q400: **4** conditions | Reverify **23** original public CV records and recompute their eligible paid BER; replace only BBC_s24_Q400_c0_f0 | **1 CV + 4 final fits = 5**, in condition order | Original finite wrong-k outcomes remain ineligible and are not rerun. A changed selection outside candidate 0 stops the five-fit plan; it does not silently add fits. |

SSA uses the public whole-process **600 s** cap. PCBG uses exactly the reviewed two-line public-launcher transformation: **300 s startup / 1800 s worker / 60 s shutdown**, without memory suspension. The historical `new_fits.timeout_active_seconds=600` describes the older attempt and does not set the extension cap. Both methods use one task at a time, refuse a pre-existing MATLAB process, and wait up to 1800 s for at least 3 GiB available memory before dispatch. These wall-clock policies do not reproduce the original active-time scheduler.

All 15×18×2 SSA decisions and all 24 PCBG CV positions are recorded, including skipped and ineligible entries. This tool does not pool missing entries, replace failures with means or interpret a failure as a win. The original 42 rows per method remain in Resource 4; this execution companion adds no rows to the paper by itself.

## External inputs and software

Use the public release at commit/tag v1.0.1, `df56e6bcc2c5bb82139170e5759523e5ac3936ee`. `PORTABLE_COMPLETION_SPEC.json` pins the public files actually consumed. Obtain BBC, Cora and CiteSeer through that release's `DATA_INPUTS.md` and `data_tools/BBC_CORA_CITESEER_README.md`; use `data_tools/prepare_bbc_citation.py` for the three exact prepared representations. Each prepared directory must have ordered `X0.npz`, `X1.npz`. This orchestration opens those feature files only, not `y.npy` or class-containing metadata. The known n/k and paid relations come from the frozen protocol and public projections. Feature content hashes, not dimensions or compressed NPZ container hashes, establish object order and representation identity.

Install Python with NumPy, SciPy and psutil, and provide an existing compatible MATLAB installation and its required toolboxes as described by the public SSA/PCBG entry documentation. No installer or model training is triggered by the default commands below. Use `python -B` to avoid bytecode caches in exact-whitelist resources. Use a **short private runtime path on Windows**: the existing public SSA preparer reconstructs a nested historical source layout and can exceed Windows path-length limits under a deeply nested manuscript directory.

Acquire the author sources separately; no license or redistribution permission is inferred here:

```text
git -c core.autocrlf=false clone https://github.com/kamalberahmand/SSA-SNMF.git C:/sapn-author/SSA
git -C C:/sapn-author/SSA checkout --detach 4d37f15621ee7bd9d412ad224408658ad243699c
git -c core.autocrlf=false clone https://github.com/YWJ-Best/PCBG.git C:/sapn-author/PCBG
git -C C:/sapn-author/PCBG checkout --detach 8e1f97e21848dc8d9c9440750ff7a1526142d2ab
```

The public launchers check the actual author-file bytes. The SSA preparer makes its documented private compatibility profile; `SSA_RUNTIME_IDENTITY.json` requires all nine generated runtime files to match the reviewed bytes, plus complete author/preparer identities. Relocation changes receipt paths and its serialized hash; it does not change these nine file hashes. PCBG's private launcher and all three support files are checked against fixed public/reviewed content **before import**.

## Metadata check, input preparation and plan

Examples assume the release is `C:/sapn-public`, these companion files are `C:/sapn-tools`, and external features are `C:/sapn-data`. Replace those paths explicitly. Relative values in `prepared_map.json` resolve against that JSON file's parent directory.

```json
{"BBC":"C:/sapn-data/BBC","Cora":"C:/sapn-data/Cora","CiteSeer":"C:/sapn-data/CiteSeer"}
```

The default command only verifies the public source/protocol metadata:

```text
python -B C:/sapn-tools/prepare_inputs.py --release-root C:/sapn-public
```

The following explicit preparation writes private label-free numeric MATs and paid projections, but launches no optimizer:

```text
python -B C:/sapn-tools/prepare_inputs.py --release-root C:/sapn-public --write-inputs --prepared-map C:/sapn-data/prepared_map.json --output C:/sapn-run
python -B C:/sapn-tools/completion_portable.py plan --release-root C:/sapn-public --work C:/sapn-run
python -B C:/sapn-public/baselines/portable/ssa/ssa_portable.py prepare --release-root C:/sapn-public --author-root C:/sapn-author/SSA --output C:/sapn-ssa
```

`plan` validates all 50 regenerated carriers again against canonical feature hashes and frozen paid-fold relationships, checks the 23 public PCBG CV records, and writes the complete conditional execution plan. Its `portable_plan_identity` binds the new input and own-source manifest. MAT container hashes are new local identities; they are never represented as old container identities. SSA carriers deliberately preserve the old PCBG `config` and c0 `case_id` for numerical comparison. The unchanged SSA worker does not load those two carrier fields: each hash-bound request carries the actual SSA candidate's beta/lam/kap and case ID, which are checked again in the worker receipt.

For maintainers who possess the original frozen input MATs, optional `--verify-frozen-reference-map local.json` adds exact numerical comparison of all 50 carriers. The local map is keyed by `SSA/<case_id>` or `PCBG/<case_id>`, with `path` and original `sha256`. It must contain exactly 50 entries. This verification is not required of an external user and does not grant access to any class vector or new native output. Only the seven input fields X/c/ML/CL/seed/config/case_id are loaded.

## Explicit execution and verification

These commands **do start MATLAB**. Run them only when deliberately allocating resources to an independent replay. They have not been run as part of the companion's dry-run validation.

```text
python -B C:/sapn-tools/completion_portable.py run --method SSA --release-root C:/sapn-public --work C:/sapn-run --ssa-runtime C:/sapn-ssa --matlab C:/MATLAB/bin/matlab.exe
python -B C:/sapn-tools/completion_portable.py run --method PCBG --release-root C:/sapn-public --work C:/sapn-run --pcbg-author C:/sapn-author/PCBG --matlab C:/MATLAB/bin/matlab.exe
```

The shared work-directory OS lock prevents simultaneous SSA and PCBG dispatch. `RUN_BINDING.json` fixes runtime/author source, executable, tool sources and input manifest. Existing completed cases are requalified from bound native receipts and paid validation partitions. A partial attempt, missing recorded native evidence, a preworker SSA infrastructure failure, or a changed source/runtime is preserved and stops the run. The tool never erases evidence or automatically retries that attempt. Repair must be separately documented and reviewed, or a fresh replay must use a new work directory and identity. There is no historical `_launchfix` exception in this portable interface.

Create an empty `C:/sapn-run/STOP` to stop before the next dispatch or during the memory wait. A running fit remains governed by its original bounded launcher; STOP does not terminate unrelated processes or rewrite a terminal result. Remove STOP only to explicitly resume. Reissuing `run` after a global seal is a read-only verification, so missing sealed evidence cannot trigger another fit.

```text
python -B C:/sapn-tools/completion_portable.py verify --method SSA --release-root C:/sapn-public --work C:/sapn-run --ssa-runtime C:/sapn-ssa --matlab C:/MATLAB/bin/matlab.exe
python -B C:/sapn-tools/completion_portable.py verify --method PCBG --release-root C:/sapn-public --work C:/sapn-run --pcbg-author C:/sapn-author/PCBG --matlab C:/MATLAB/bin/matlab.exe
```

Each method's global seal is written only after a second full read-only verification of native evidence and all paid-fold selections. The controller accepts no category-file argument and computes no test ACC/NMI/ARI. Fresh optimization can differ numerically or terminate differently across MATLAB versions and machines; a new seal is evidence of that new run, not proof that published test scores were reproduced. Resource 4's original-score replay deliberately checks the original sealed results within 1e-12 and must not be presented as a scorer for arbitrary fresh fits. A separately reviewed post-seal scoring adapter would be needed to report new-run external-label metrics.

## Validation and distribution boundary

```text
python -B C:/sapn-tools/test_execution_tools.py --release-root C:/sapn-public
```

Tests use temporary synthetic files and fake launchers, including all 555 SSA decision/final slots, 23+5 PCBG reuse/dispatch, tie boundaries, wrong-k, failure pruning, incomplete attempts, immutable records, runtime tampering, locks, STOP, memory waits and both clock policies. The validation receipt records the actual test count and source hashes. The separate private dry-run receipt records comparison of all 50 inputs and regeneration of the nine SSA runtime files. No actual optimizer rerun or test-score reading is claimed by these tests.

Distribute only the files enumerated in `TOOLS_SHA256.json`. Do not recursively copy private work, author checkouts, runtime trees, input MATs, original reference maps or validation payload directories. This companion was prepared locally; it is not an upload, publication, DOI assignment or assertion of upstream license terms.
