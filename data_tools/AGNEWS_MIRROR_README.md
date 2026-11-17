# Verified AG News mirror preparation

`prepare_agnews_mirror.py` accepts the fastai-hosted AG News archive whose compressed bytes differ from the historical Crepe archive. Before calling the original construction function, it verifies the entire mirror archive and every regular member name, byte length and SHA256. Unexpected, duplicate, linked or special members are rejected. All four regular members were compared byte for byte with the verified historical author archive. Both archive identities remain explicit in output metadata.

The public mirror is [AG News on the fastai S3 host](https://s3.amazonaws.com/fast-ai-nlp/ag_news_csv.tgz), identified by [fastai's own dataset definitions](https://raw.githubusercontent.com/fastai/fastai/master/fastai/data/external.py). Its SHA256 is `9a8c300eabb45750237fcc669f61cb8a3448f3ef6f6098e1ce340e444f6872be` (11,784,419 bytes). The equivalent original author container has SHA256 `4e2fc37d369e6e317a8b62d0d2f11d379f8c3dde60c07c35a46b069a246498ad` (11,784,327 bytes). These are different containers, not two names for identical archive bytes. No archive or raw text is distributed in this release.

Place the wrapper and `agnews_mirror_identity.json` in `data_tools/`, beside the release's unchanged `benchmark/prepare_field_corpora.py` and `benchmark/FIELD_CORPUS_IDENTITIES.json` directory. The adapter pins both files by SHA256. It imports the original `prepare_agnews` function directly and does not rewrite, copy or monkey-patch its construction algorithm. The original preparer's CLI continues to accept only its original pinned archive.

From the release root, using Python 3.10.22, NumPy 2.2.6, SciPy 1.15.3 and scikit-learn 1.7.2:

```shell
python -I data_tools/prepare_agnews_mirror.py --archive /path/to/ag_news_csv.tgz --output /path/to/new_prepared_agnews
```

The output directory must not exist. `--base-preparer` and `--identities` allow explicit relocated paths. `--verify-against /path/to/original_prepared_directory` optionally checks all ordered values, CSR arrays, labels, provenance/source IDs, vocabularies, IDF values and deduplication exclusions against an author-held reference; ordinary preparation requires no historical directory.

The wrapper creates `X0.npz` (title TF-IDF, 2000 by 1795), `X1.npz` (description TF-IDF, 2000 by 4592), ordered `y.npy`, provenance, vocabulary, IDF and exclusion JSON files, plus metadata and a verification report. Labels are reserved for the query oracle and sealed scoring. Graphs, initialization and model training belong to the later benchmark runner. This wrapper trains and evaluates no clustering model.

An actual fresh-directory run with Python `-I`, only the wrapper, its identity JSON, the unchanged base preparer and its frozen identity JSON, regenerated the inputs from the newly downloaded mirror. Every scientific identity matched, and an additional full elementwise comparison against the historical prepared inputs passed. The prepared fingerprint is `c08f8fbacd42a0f3e883d6b39842e4277e2ae0699641d475ff3b7f001e99208a`. See `AGNEWS_MIRROR_PREPARATION_VERIFICATION.json` for exact checks. The raw archive and reference directory were supplied only through explicit CLI arguments. Availability findings are dated 2026-11-17 and do not guarantee future service or imply a redistribution licence.
