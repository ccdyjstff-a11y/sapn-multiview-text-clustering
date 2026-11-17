# Seven-corpus inputs and preparation

Raw third-party archives, extracted source text, feature arrays and class vectors are intentionally external. The preparation commands below are actual interfaces, not algorithm templates. Use new output directories. The original scientific ordering, view construction and numeric preprocessing must be preserved; equal dimensions alone are insufficient.

| Corpus | n / k / views | Prepared feature dimensions | Source and entry |
|---|---|---|---|
| 3Sources | 169 / 6 / 3 | 3560, 3631, 3068 | UCD original archive; `benchmark/prepare_data.py` with `3sources_original_zip` |
| BBC | 2225 / 5 / 2 | 4817, 4818 | UCD four original files; `data_tools/prepare_bbc_citation.py --dataset BBC` |
| Reuters-local | 1200 / 6 / 5 | 2000 each | Grimal ReutersEN sample 0; `benchmark/prepare_data.py` with `reuters_grimal_tar` |
| Cora | 2708 / 7 / 2 | 1433, 2708 | Grimal Cora archive with verified LINQS provenance; `data_tools/prepare_bbc_citation.py --dataset Cora` |
| CiteSeer | 3312 / 6 / 2 | 3703, 3312 | Grimal CiteSeer archive with verified LINQS provenance; corresponding `--dataset CiteSeer` |
| 20NG | 2000 / 20 / 2 | 1265, 10000 | Original by-date training archive; `benchmark/prepare_field_corpora.py --dataset 20ng` |
| AG News | 2000 / 4 / 2 | 1795, 4592 | Released training CSV archive; corresponding `--dataset agnews` |

Historical file hashes and source identities are in `benchmark/INPUT_CATALOG.json`, `FIELD_CORPUS_IDENTITIES.json`, `ORIGINAL_SOURCE_AUDIT.json`, `LINQS_SOURCE_AUDIT.json`, and `data_tools/bbc_citation_identity.json`. File serialization/metadata changes can alter file hashes even when arrays are equal. Preparation verification reports compare array values, ordered class partitions, identifiers and relevant metadata rather than infer identity from shape.

## Exact archives

| Archive | Publisher route | SHA256 |
|---|---|---|
| 3sources.zip | http://mlg.ucd.ie/files/datasets/3sources.zip | `dea0ab41c48f1a50dbce855f7f9f783dd57bb0a3a3636f53f0c665678fb48659` |
| bbc.zip | http://mlg.ucd.ie/files/datasets/bbc.zip | `976a091f7579f65c20b2738829e63a6cbae988ca7af1d7a4dfadefcefd460ca6` |
| ReutersEN.tar.gz | https://lig-membres.imag.fr/grimal/data/ReutersEN.tar.gz | `829e06d95b537a5140b5ff7121ca5d02893c930a78ea3784ad9159b17af12736` |
| Cora.tar.gz | https://lig-membres.imag.fr/grimal/data/Cora.tar.gz | `52e28f37d39751366b46b859cb2bc0ec3757a222552cc24423763fb38509001c` |
| CiteSeer.tar.gz | https://lig-membres.imag.fr/grimal/data/CiteSeer.tar.gz | `a8a73135322e9269d760ede375c4d7ffab8a5f26f88b278381c55d6ef393738a` |
| 20news-bydate.tar.gz | http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz | `8f1b2514ca22a5ade8fbb9cfa5727df95fa587f4c87b786e15c759fa66d95610` |
| ag_news_csv.tar.gz | Author dataset links at https://github.com/zhangxiangxiao/Crepe | `4e2fc37d369e6e317a8b62d0d2f11d379f8c3dde60c07c35a46b069a246498ad` |
| ag_news_csv.tgz (verified member-equivalent mirror) | https://s3.amazonaws.com/fast-ai-nlp/ag_news_csv.tgz | `9a8c300eabb45750237fcc669f61cb8a3448f3ef6f6098e1ce340e444f6872be` |

The coordinating preparation task downloaded BBC (884,338 bytes), Cora (237,550 bytes) and CiteSeer (413,805 bytes) on 17 November 2026 and matched the identities above. BBC's original HTTP route worked; a separate HTTPS attempt failed. The HTTPS failure is not evidence that the original HTTP data are unavailable. Records are in `provenance/access/`. These records are dated access checks, not a guarantee of future availability. No archive is included in this release.

The original 20NG HTTP archive was also freshly downloaded (14,464,277 bytes) and matched its recorded SHA256; its HTTPS attempt timed out. For AG News, the author Drive route returned a raw-file stream through the connected service, but a new local original-container hash was not completed. The fastai S3 mirror was freshly downloaded (11,784,419 bytes) and verified at both container and member level: all four regular members have exactly the same names, lengths and bytes as the retained verified original author container (11,784,327 bytes). The two compressed archive hashes differ. `provenance/access/field_public_access/` contains the dated access and member-comparison records. Do not describe the mirror as an identical original archive or claim a fresh original AG container hash check.

## 3Sources and Reuters

Create an external UTF-8 JSON configuration, for example `C:/sapn-runs/prepare_3sources.json`:

```json
{"dataset":"3Sources","format":"3sources_original_zip","source":"C:/external/3sources.zip","expected_sha256":"dea0ab41c48f1a50dbce855f7f9f783dd57bb0a3a3636f53f0c665678fb48659","output":"C:/sapn-data/3Sources"}
```

```text
python -B benchmark/prepare_data.py --config C:/sapn-runs/prepare_3sources.json
```

The script intersects and lexicographically sorts story IDs across BBC, Guardian and Reuters views, retaining disjoint labels. It does not construct views from later benchmark labels.

For Reuters create a separate configuration:

```json
{"dataset":"Reuters","format":"reuters_grimal_tar","source":"C:/external/ReutersEN.tar.gz","expected_sha256":"829e06d95b537a5140b5ff7121ca5d02893c930a78ea3784ad9159b17af12736","sample":0,"output":"C:/sapn-data/Reuters"}
```

```text
python -B benchmark/prepare_data.py --config C:/sapn-runs/prepare_reuters.json
```

It preserves the released row mapping and view order EN, FR, GR, IT, SP, then applies featurewise min-max transformation. Samples 1 through 5 use the same archive with the corresponding `sample` field and separate output directories. They remain same-source samples. The labels produced by both generic preparation routes are named `labels_FOR_ORACLE_AND_SEALED_SCORING_ONLY.npy`; use this exact file in oracle/scoring configurations rather than assuming `y.npy`.

## BBC, Cora and CiteSeer

The companion [preparation README](data_tools/BBC_CORA_CITESEER_README.md) specifies exact input members and checks. BBC accepts the verified `bbc.zip` directly, or a directory containing `bbc.mtx`, `bbc.classes`, `bbc.docs`, `bbc.terms`. The preparer checks archive and member identities. Cora and CiteSeer accept the original Grimal tar.gz directly.

```text
python -B -I data_tools/prepare_bbc_citation.py --dataset BBC --source C:/external/bbc.zip --output C:/sapn-data/BBC
python -B -I data_tools/prepare_bbc_citation.py --dataset Cora --source C:/external/Cora.tar.gz --output C:/sapn-data/Cora
python -B -I data_tools/prepare_bbc_citation.py --dataset CiteSeer --source C:/external/CiteSeer.tar.gz --output C:/sapn-data/CiteSeer
```

BBC splits the original 9,635-feature representation with seed 42 and preserves the historical columnwise min-max preprocessing. Citation corpora retain the verified content and symmetrized citation views in the recorded object order. Their prepared zero rows are retained. The preparer emits `y.npy` and an identically valued explicit oracle/scoring alias. Read `data_tools/PREPARATION_VERIFICATION_PORTABLE.json` for the actual independent relocation and elementwise checks; those checks are data preparation, not retraining.

## Field corpora

```text
python -B -I benchmark/prepare_field_corpora.py --dataset 20ng --archive C:/external/20news-bydate.tar.gz --output C:/sapn-data/20ng
python -B -I benchmark/prepare_field_corpora.py --dataset agnews --archive C:/external/ag_news_csv.tar.gz --output C:/sapn-data/agnews
```

For the verified fastai mirror use the separate strict wrapper:

```text
python -B -I data_tools/prepare_agnews_mirror.py --archive C:/external/ag_news_csv.tgz --output C:/sapn-data/agnews-mirror
```

It checks the mirror SHA256 and every member before invoking the unchanged original construction function. The original preparer's CLI and bytes remain unchanged. An actual isolated fresh-directory preparation from the new mirror reproduced every ordered matrix value, CSR array, label, source ID, vocabulary, IDF value and deduplication exclusion. See [the mirror preparation instructions](data_tools/AGNEWS_MIRROR_README.md) and `data_tools/AGNEWS_MIRROR_PREPARATION_VERIFICATION.json`. Output metadata records the different container; scientific representation equality does not require equal metadata serialization. No model training was part of this verification.

Both use the frozen sampling seeds, cleaning, duplicate handling, per-class sampling and TF-IDF settings from the manuscript. 20NG uses subject/body and AG News title/description. These are constructed transductive representations, not official multiview dataset releases. Labels support benchmark construction and the simulated oracle, but are not supplied to fitting/selection. The official held-out test collections are not used. Do not change zero rows or vocabulary fitting to improve a baseline's input coverage.

## Strict information boundary

Training receives nonnegative views, known k, optional unlabeled initialization and graph, and paid relation answers. The separate oracle constructs paid answers; the separate scorer reads external class labels only after sealing. Query files contain no class vector. Fitting all features transductively does not permit using evaluation labels for coefficient selection, tie-breaking, or retry decisions.
