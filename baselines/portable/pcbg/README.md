# Portable PCBG replay

This entry recreates native PCBG inputs from the public SAPN release and externally prepared corpus features. It calls the byte-preserved historical MATLAB worker and the unchanged external author optimizer. It does not require the original workstation's research directories, admission receipts or intermediate qualification trees.

The historical source and outcome evidence remain under `baselines/pcbg_historical` and `evidence/v10/native_pcbg`. The historical scientific protocol SHA256 is recorded in `IDENTITIES.json`. A new replay produces separate receipts and never replaces paper results.

## Requirements and acquisition

Use the Python environment specified in the root `requirements.txt`, including NumPy, SciPy, scikit-learn and psutil. MATLAB must support `-batch`, sparse matrices, `graph/conncomp` and Java atomic file moves. Validation used the local MATLAB R2025a installation; other releases are not asserted byte-identical. Supply the explicit MATLAB executable path.

Obtain PCBG from <https://github.com/YWJ-Best/PCBG> at commit `8e1f97e21848dc8d9c9440750ff7a1526142d2ab`. Preserve exact repository bytes, including line endings. One acquisition route is:

```text
git -c core.autocrlf=false clone https://github.com/YWJ-Best/PCBG external/PCBG
git -C external/PCBG -c core.autocrlf=false checkout 8e1f97e21848dc8d9c9440750ff7a1526142d2ab
```

The runner verifies all 28 recorded text-file SHA256 identities before and after every fit, not merely a branch name. It does not execute the author demo, create additional label-derived constraints or redistribute any author optimizer. No additional licence to third-party code is granted by this wrapper. The repository's rights and original authors' terms continue to apply.

Prepare data from the original public archives using the root `DATA_INPUTS.md` instructions. For example, download the checked `3sources.zip`, run `benchmark/prepare_data.py` with format `3sources_original_zip`, and obtain `X0.npz`, `X1.npz`, `X2.npz`. BBC, Cora and CiteSeer use `data_tools/prepare_bbc_citation.py`. Preserve all object and view ordering. `IDENTITIES.json` checks canonical sparse matrix values, shapes and order independently of NPZ container timestamps or compression. No old absolute path is used to obtain inputs.

## Prepare one original benchmark condition

Commands below assume this directory is installed as `baselines/portable/pcbg` within the public release. Replace the example absolute paths with your paths. Run in a fresh external output directory.

```text
python -B baselines/portable/pcbg/pcbg_portable.py prepare --release-root C:/sapn-release --prepared C:/sapn-data/3Sources --dataset 3Sources --seed 23 --budget 160 --output C:/sapn-runs/pcbg-3Sources-s23-Q160
```

This reads the public `evidence/v10/native_pcbg/partition_export/PAID_STREAMS.json` and its checked `paid/*.npz` projections. Historical absolute paths in those records are provenance only and are never opened. Inputs contain the original paid pairs and answers; no new oracle queries are made. Budgets 160/400 are prefixes. Folds use seed `s+41000`, node-disjoint endpoint halves, and the same two-sign eligibility requirement as the paper.

For supported inputs, three anchor rates `[0.7, 0.1, 0.3]` and the eligible folds produce candidate MAT files plus final-fit MAT files. Known cluster count and the original native options are fixed. The preparation stage opens only `X*.npz`, never class-label files or metadata with class counts. It checks exact feature identity against the historical benchmark values. No zero-row repair is performed.

The original supported datasets are 3Sources, BBC, Cora and CiteSeer, seeds 23–25. Reuters, 20NG and AGNews remain explicitly unsupported by the native zero-row preprocessing route. Their recorded original seeds are retained in the public paid-stream index. Preparing such an outcome writes `INPUT_UNSUPPORTED`; it does not attempt to repair, fit or score it.

## Fit and select without evaluation labels

```text
python -B baselines/portable/pcbg/pcbg_portable.py run --output C:/sapn-runs/pcbg-3Sources-s23-Q160 --author-root C:/external/PCBG --matlab C:/ProgramFiles/MATLAB/R2025a/bin/matlab.exe
```

The complete condition runs the three rates on the eligible folds, evaluates only queried validation-pair BER, chooses the declared default-first best candidate with the `1e-12` tolerance, and fits the selected candidate on all paid relations. A candidate with a failed required fold cannot win selection. If none is eligible, the declared default is retained with an explicit fallback record. The native strict-cluster/finite-output checks determine final eligibility; a non-eligible outcome remains recorded rather than becoming a primary score.

To inspect one prepared case before running a complete condition, append, for example:

```text
--case 3Sources_s23_Q160_c1_final
```

A single-case run does not constitute full coefficient selection or a sealed scoreable condition. Resuming the complete command reuses only receipts whose plan and artifact hashes still match. It never selects by previously observed clustering accuracy. A partially launched case without a receipt is preserved and requires a separate fresh replay output.

All MATLAB processes are launched without a visible console, with one computation thread. The launcher uses a 3 GiB available-memory precheck and separate 300 s startup, 600 s worker and 60 s shutdown wall-clock limits. It terminates only its own launched process tree when its cap is reached. **This portable launcher does not reproduce the historical controller's suspend/resume memory scheduling or 48-hour global batch order.** It therefore tests scientific input/worker equivalence and produces a new execution record, not an identical performance-timing experiment. Larger corpora may still fail within resource caps. Do not pool such reruns into the paper tables.

## Seal, then score

After all required CV outcomes and the selected final outcome are recorded, the complete command writes `SELECTION.json` and `CONDITION_SEALED.json`. This seals one condition; the original paper's 42-outcome global seal remains separate. To repeat a whole comparison with a strict global boundary, finish and seal every planned condition before scoring any of them.

```text
python -B baselines/portable/pcbg/pcbg_portable.py score --output C:/sapn-runs/pcbg-3Sources-s23-Q160 --labels C:/sapn-data/3Sources/labels_FOR_ORACLE_AND_SEALED_SCORING_ONLY.npy
```

The scorer checks every sealed artifact and the ordered labels against the paid relations before computing pool-excluded ACC, arithmetic-normalized NMI and ARI. These scores are replay diagnostics. Ineligible finals receive no primary mean or replacement performance value. Labels are first read by this separate post-seal command.

## Validation scope

The validation receipt supplied with this entry distinguishes source-hash checking, exact MATLAB input-payload comparison, actual MATLAB execution and any completed end-to-end selection. A code import or input comparison is not reported as a fitted benchmark. The validation archive's local feature matrices, raw author sources, MATLAB preference caches and generated worker outputs are not part of this public entry.

On 17 November 2026, this entry completed 3Sources seed 23 at 160 queries from the corpus prepared from the public archive downloaded earlier that day. All nine prepared MAT payloads matched the historical inputs exactly. Six validation fits and the selected final fit completed with eligible native outputs; all 16 native output fields matched their corresponding historical arrays exactly in each of the seven fits. Query-only selection recovered anchor rate 0.1. After sealing, ACC/NMI/ARI also matched the historical condition. `VALIDATION.json` records the checks and source identities. This is one end-to-end condition, not a fresh seven-corpus benchmark or a reproduction of the original resource scheduler.
