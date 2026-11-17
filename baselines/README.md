# Baseline implementation and adaptation interfaces

The release includes the research project's own implementations and adapters. Obtaining an external author repository does not replace these adaptation sources. No unlicensed author optimizer or generated derivative optimizer is redistributed here.

## CMVNMF and MVCHSS

Use `benchmark/reproduce.py` with `methods` set to `["CMVNMF","MVCHSS"]`, the same explicit paid queries, and budgets 160 and 400. Oracle, training and sealed scoring remain separate commands as shown in the main README.

CMVNMF uses the paper-based implementation `cmvnmf_paper.py`, explicitly not author code, plus the same-objective coordinate continuation in `cmvnmf_accelerated.py`. It retains the declared aligned-data output adapter and known object identity correspondences. The coefficient grid is `[1,.01,100]`.

MVCHSS uses `mvchss_reimplementation.py`, `mvchss_sparse_graph_backend_v1.py`, `mvchss_cached_author.py` and `domain_extension.py`. These are research Python implementations of the documented author-normalized update map and graph recipe, with declared deterministic handling of zero/duplicate observations. The grid is `[150,50,200]`; five starts of 1,000 updates are used per fit. This is not an untouched author MATLAB implementation or a common stationarity certificate. The referenced author graph recipe is https://github.com/Psy680267/MVCHSS at commit `e02341fe1ff0b713d5cc40fffee95d6b4de1b6d7`, also stated in the module documentation.

`tools/smoke_baselines.py` exercises both supplied interfaces through synthetic preparation, oracle generation, CV/final fitting and sealed scoring. Its output under `validation/` records actual execution. It does not rerun the real-data benchmark or establish equality to third-party author code.

## FSSMSC

Included own-source files are `benchmark/fssmsc_bridge.py`, `benchmark/acquire_fssmsc.py`, the patch generator `benchmark/fssmsc/build_adaptation.py`, and the five research worker/diagnostic `.m` files in that folder. The generator reads a separately obtained author checkout and writes locally adapted diagnostic copies. The author's optimization source itself is not included.

Obtain the exact checkout from https://github.com/linghm/FSSMSC at commit `064f9cba06b30d7b152eb046e909a028e386ded7`, then in a working copy of the release run:

```text
python -B benchmark/acquire_fssmsc.py --author-checkout C:/external/FSSMSC
```

The acquisition script validates each required SHA256 in `benchmark/THIRD_PARTY_INPUTS.json` before copying files locally and generating patches. This requires separate permission under the author's terms and does not grant redistribution rights. Generated author copies and derivatives are excluded by `.gitignore`.

Set `methods` to `["FSSMSC"]`, use positive budgets only, and set `matlab_executable` to the actual MATLAB executable in the external training JSON. Then use the same `benchmark/reproduce.py train --config ...` and `score` interface. The adapter receives features, known k and paid pairs, not full classes. It uses the declared nine-configuration grid, native 30-round outer cap, common anchor preparation and clustering adapter. Operational eligibility does not certify paper-level orthogonality or convergence. No new MATLAB execution was performed by this release assembly; historical qualification is preserved separately in the original archive.

## PCBG historical native adapter

`pcbg_historical/` preserves the original own-source formal preparation, clock/worker bridge, process handling, numerical checks, selection and sealed-scoring scripts. `provenance/ARCHIVE_ORIGINS.json` gives their unchanged archive-member identities. The author optimizer is obtained separately at the recorded PCBG commit and file hashes in `provenance/third_party/PCBG_SOURCE_HASHES.json`.

These scripts preserve their historical research-directory assumptions and protocol-bound absolute paths. They are inspectable implementation evidence, not a validated standalone training command for this relocated directory. `prepare_formal.py` expects the prior qualification/resource-policy inputs; `formal_common.py` resolves the historical GAP/MAIN trees; execution verifies frozen protocol/source hashes. Reproduction outside that tree therefore requires explicit path mapping, the original paid inputs and protocol records, author files, and MATLAB. This unresolved portability step is separate from source-code availability and must not be described as solved by downloading PCBG alone. The compact output/selection evidence in `evidence/v10/native_pcbg/partition_export/` can be reviewed directly without retraining.

## SSA-SNMF historical grouped adapter

The own-source grouped training adapter and patch-generation closure are included and documented in `ssa_historical/README.md`. Its manifest binds 33 unchanged own-source files, including the minimal and formal patch generators, grouped controller and own MATLAB workers. Historical-path and runtime-input limitations are explicit. Generated author derivatives are excluded. The directly runnable compact evidence replay remains `reproduction_ssa/source/replay_ssa_compact.py` and has an executed verification record under `validation/`. It checks paid-query BER, selections, scheduler and retained outputs, not full-factor qualification or evaluation-set metrics.
