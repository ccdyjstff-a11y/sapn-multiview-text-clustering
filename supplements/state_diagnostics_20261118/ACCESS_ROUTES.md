# External data access and fixed-checkpoint scoring

This is a local diagnostic-resource draft. It has not been uploaded or assigned a DOI. No raw text, feature matrices, class arrays, private MATLAB implementation or third-party author source is included. No additional licence is asserted here. The resource supplies the retained factors needed for the new diagnostics; obtain original features and external labels separately through the existing public preparation workflow.

The public v1.0.1 tree already documents exact archive hashes, ordered preprocessing and dated access checks in `DATA_INPUTS.md`, with `benchmark/INPUT_CATALOG.json`, `FIELD_CORPUS_IDENTITIES.json`, `ORIGINAL_SOURCE_AUDIT.json`, `LINQS_SOURCE_AUDIT.json` and `data_tools/bbc_citation_identity.json`. Use these records together with `FEATURE_REQUIREMENTS.json`. Matching dimensions alone is insufficient: representation values and object/feature order must also match. The following routes repeat the recorded access routes; this document does not claim a new availability/download check.

| Corpus | Recorded upstream route | Existing public preparation entry |
|---|---|---|
| 3Sources | [UCD original archive](http://mlg.ucd.ie/files/datasets/3sources.zip) | `benchmark/prepare_data.py`, format `3sources_original_zip` |
| BBC | [UCD original archive](http://mlg.ucd.ie/files/datasets/bbc.zip) | `data_tools/prepare_bbc_citation.py --dataset BBC` |
| Reuters | [Grimal ReutersEN sample](https://lig-membres.imag.fr/grimal/data/ReutersEN.tar.gz) | `benchmark/prepare_data.py`, format `reuters_grimal_tar`, original sample 0 |
| Cora | [Grimal archive](https://lig-membres.imag.fr/grimal/data/Cora.tar.gz), with LINQS provenance documented in public release | `data_tools/prepare_bbc_citation.py --dataset Cora` |
| CiteSeer | [Grimal archive](https://lig-membres.imag.fr/grimal/data/CiteSeer.tar.gz), with LINQS provenance documented in public release | `data_tools/prepare_bbc_citation.py --dataset CiteSeer` |
| 20NG | [Original by-date archive](http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz) | `benchmark/prepare_field_corpora.py --dataset 20ng` |
| AG News | [Author dataset links](https://github.com/zhangxiangxiao/Crepe); [verified member-equivalent mirror](https://s3.amazonaws.com/fast-ai-nlp/ag_news_csv.tgz) | Original: `benchmark/prepare_field_corpora.py --dataset agnews`; mirror: `data_tools/prepare_agnews_mirror.py` |

The AG News original and mirror compressed containers have different hashes; the public preparation verification matched all member bytes and the resulting ordered arrays. Do not treat the mirror container as byte-identical to the original. Both 20NG and AG News use the frozen training-subset construction described in the manuscript, not the official held-out test collections.

Feature diagnostics use `resource_common.load_features`: supplied n-by-d sparse matrices must match the canonical CSR hashes in `FEATURE_REQUIREMENTS.json`, then receive the original Frobenius normalization and transpose. The optional Reuters original MAT input is read with `variable_names=['X']`; the feature diagnostic does not import its class vector. Prepared external `X0.npz`, `X1.npz`, etc. remain outside this resource.

`score_checkpoints.py` uses only external class arrays and retained checkpoint factors; it does not need feature matrices or run a model. Supply a JSON file with seven dataset names mapped to external NPY paths, for example:

```json
{
  "3Sources": "C:/external-prepared/3Sources/labels_FOR_ORACLE_AND_SEALED_SCORING_ONLY.npy",
  "BBC": "C:/external-prepared/BBC/y.npy",
  "Reuters": "C:/external-prepared/Reuters/labels_FOR_ORACLE_AND_SEALED_SCORING_ONLY.npy",
  "Cora": "C:/external-prepared/Cora/y.npy",
  "CiteSeer": "C:/external-prepared/CiteSeer/y.npy",
  "20NG": "C:/external-prepared/20NG/y.npy",
  "AGNews": "C:/external-prepared/AGNews/y.npy"
}
```

Paths are illustrative, not supplied data. Relative paths resolve against this JSON file. The original primary dataset aliases `20NG_SubjectBody_2000` and `AGNews_TitleDescription_2000` are also accepted. External class vectors may use different class codes, but must have the identical ordered partition hash in `FEATURE_REQUIREMENTS.json`.

```text
python -B score_checkpoints.py --labels-map C:/external-prepared/labels-map.json --primary-root C:/public-v1.0.1/primary_predictions --output C:/diagnostic-results/checkpoint-scores
```

The resource must first have its real `MANIFEST.json` and `MANIFEST.sha256` sealed. The command verifies the resource, original primary manifest, all 42 CMVNMF and all 42 SAPN-MV prediction hashes, and original coefficient identities before loading external classes. It decodes every view using the original `1e-30` column-normalization floor, averages the normalized memberships, and applies argmax. This order matters for extremely small MU entries. Four cases without continuation explicitly alias MU to final factors. It checks the query pool, its evaluation complement, paid answers and exact final partition against the primary release.

Outputs contain 42 runs and 14 corpus–budget groups for Hungarian ACC, arithmetic-normalized NMI and ARI; paired changes are in percentage points with sample standard deviations across the original three seeds. They also compare SAPN-MV against both checkpoints and list any cell-mean direction changes. It rechecks all 252 original final/SAPN metric values within absolute tolerance `1e-12`. The output directory must be new and outside the sealed resource; existing results are never overwritten.

This evaluates sensitivity to the stopping checkpoint at the originally selected coefficients. It does not repeat cross-validation, tune MU separately, select a stage using test metrics, perform significance tests or establish accuracy invariance under all stopping rules. Feature-only objective/residual diagnostics and label-based checkpoint scoring have separate inputs and claims.
