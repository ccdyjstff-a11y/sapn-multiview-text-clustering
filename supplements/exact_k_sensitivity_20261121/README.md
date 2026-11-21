# SAPN exact-k sensitivity supplement

This supplement accompanies the completed baseline execution supplement. It reproduces qualification from reduced published metadata and aggregation of already computed metrics. The analysis removes only the final exact-cluster-count restriction while retaining the other output requirements. Candidate selection and the primary benchmark are unchanged.

Run from the extracted supplement directory with Python 3.10 or later; only the standard library is required. Choose a new output directory outside this extracted directory.

```text
python -B code/replay_exact_k.py --replay --input-root data --expect-source-output-manifest-sha256 222df1115704543f5752dbd27072bbd9da9210a7b0855670f25e18f432bdb048 --output ../exact_k_replay
```

The replay checks all 84 planned method-condition slots and 16 supported corpus-budget group summaries. Each metric summary requires all three planned pairs; missing values remain missing. It does not fit models, read labels or feature arrays, rescore predictions, repeat cross-validation, or independently verify the original workers. The earlier two PCBG groups overlap this analysis and do not add independent evidence.

The protocol was frozen after the original model development and benchmark results were known, before disclosure of this execution supplement's new scores. It is a supplementary sensitivity analysis, not the original study preregistration.

`code/` contains nine unchanged preparation-time files; its README and publication plan describe that earlier preparation stage. This root README provides the current runnable entry. `data/` contains four JSON payloads and the unchanged original eight-entry output manifest. The four CSV/LaTeX files indexed in that original manifest are not included in this minimal replay bundle; their absent bytes are not verified by the public replay.

The local public replay completed before this release assembly. Its receipt SHA256 is `3144c461a908d3aca4c1643278cea016d2ea39dd8aa9f7022770ef795a9b82c7`. `MANIFEST.json` binds every delivered file. The repository commit and release archive checksum identify the published copy separately. No manuscript, corpus feature matrix, full class vector or third-party algorithm code is included.
