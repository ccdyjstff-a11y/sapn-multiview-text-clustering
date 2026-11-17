# Portable preparation of BBC, Cora, and CiteSeer

`prepare_bbc_citation.py` reconstructs the three frozen experimental inputs from externally supplied original source files. Copy it together with `bbc_citation_identity.json` into any directory. It imports only the Python standard library, NumPy, and SciPy. There are no imports from the historical research workspace and no fixed local input paths. The identity JSON records the exact locally measured SHA256 identities, original source URLs, expected prepared identities, and configuration.

## Original inputs and root paths

Obtain the inputs from their original distributors. This code package contains no original datasets or prepared matrices. On 2026-11-17, the original BBC HTTP ZIP and both Grimal HTTPS archives were freshly downloaded and their identities matched the historical inputs. `PUBLIC_SOURCE_STATUS.json` records these checks. BBC's HTTPS endpoint failed during that check; use the original HTTP download URL shown below with the required SHA256 identity.

| Dataset | Original source | Value required for `--source` |
| --- | --- | --- |
| BBC | [Dataset page](http://mlg.ucd.ie/datasets/bbc.html), [original ZIP](http://mlg.ucd.ie/files/datasets/bbc.zip) | Either the exact original `bbc.zip` file, or the directory directly containing **all four** `bbc.mtx`, `bbc.classes`, `bbc.docs`, and `bbc.terms` files. Do not pass the parent of this four-file directory. |
| Cora | [Grimal collection](https://lig-membres.imag.fr/grimal/data.html), [original archive](https://lig-membres.imag.fr/grimal/data/Cora.tar.gz) | The actual `Cora.tar.gz` file, not its parent directory or an extracted folder. |
| CiteSeer | [Grimal collection](https://lig-membres.imag.fr/grimal/data.html), [original archive](https://lig-membres.imag.fr/grimal/data/CiteSeer.tar.gz) | The actual `CiteSeer.tar.gz` file, not its parent directory or an extracted folder. |

BBC source-file SHA256 values are checked individually; when a ZIP is supplied its whole-file identity is checked as well. The citation archives are checked as complete files and their five consumed members are checked individually. Archive members are read in memory, without extracting files to disk. Full identities are in `bbc_citation_identity.json`.

| Source file | Measured SHA256 |
| --- | --- |
| `bbc.zip` (884,338 bytes) | `976a091f7579f65c20b2738829e63a6cbae988ca7af1d7a4dfadefcefd460ca6` |
| `bbc.mtx` | `5d4e5355bdd03b6c6a1eebb8223b5bd864e0ae651c496c135265432c9c471efb` |
| `bbc.classes` | `b477f844381858f5b180d63a08410992e64e919b05f6405ffef778b77f201b5e` |
| `bbc.docs` | `602095cdbaf75e5e38216ff5e46e5add909c85d5dbdfa5d853453c360ba42cea` |
| `bbc.terms` | `6cb78478185343a0c07500dc270e3a4546b538a3b007c83e128c09ff3197d458` |
| `Cora.tar.gz` | `52e28f37d39751366b46b859cb2bc0ec3757a222552cc24423763fb38509001c` |
| `CiteSeer.tar.gz` | `a8a73135322e9269d760ede375c4d7ffab8a5f26f88b278381c55d6ef393738a` |

## Commands

Install NumPy and SciPy in the Python environment used for execution. The verified environment was Python 3.10.22, NumPy 2.2.6, and SciPy 1.15.3. No model framework is needed for this step. The following commands assume that the two portable files have been placed in `data_tools/` under the current directory. Replace the example source and output paths with your chosen absolute paths. Each output directory must be new; the program refuses to overwrite an existing directory.

```text
python -I data_tools/prepare_bbc_citation.py --dataset BBC --source /absolute/raw/bbc --output /absolute/prepared/BBC
python -I data_tools/prepare_bbc_citation.py --dataset Cora --source /absolute/raw/Cora.tar.gz --output /absolute/prepared/Cora
python -I data_tools/prepare_bbc_citation.py --dataset CiteSeer --source /absolute/raw/CiteSeer.tar.gz --output /absolute/prepared/CiteSeer
```

For the BBC ZIP alternative, use `--source /absolute/raw/bbc.zip` in the first command; no manual extraction is needed. The output is identical to preparation from the four original files.

On Windows, paths such as `C:/data/raw/bbc` and `C:/data/prepared/BBC` are accepted. Quote paths containing spaces. The identity file is located relative to the script, so changing the working directory does not change the input identity. Use `--identity /absolute/path/bbc_citation_identity.json` only when locating that same identity file elsewhere.

For an optional direct comparison with an existing frozen preparation, append `--verify-against /absolute/reference/prepared_BBC` (or the corresponding dataset). This checks every ordered matrix value, including implicit zeros, exact CSR index and value arrays, and every label. A normal preparation does not need a reference directory: exact source identities, ordered document identities, feature indices for BBC, shapes, and the frozen prepared-data identity are always checked.

## Dataset definitions and transformations

Every output matrix has documents on rows. All matrices use SciPy CSR format with float64 data, sorted indices, duplicate coordinates summed, and stored zero entries removed. Labels are int64, mapped from sorted unique original labels to `0, ..., k-1`; class IDs are not used to reorder rows.

| Dataset | Documents | Classes | `X0` shape and meaning | `X1` shape and meaning |
| --- | ---: | ---: | --- | --- |
| BBC | 2,225 | 5 | 2,225 x 4,817, first vocabulary partition | 2,225 x 4,818, second vocabulary partition |
| Cora | 2,708 | 7 | 2,708 x 1,433, released content features | 2,708 x 2,708, released citation counts |
| CiteSeer | 3,312 | 6 | 3,312 x 3,703, released content features | 3,312 x 3,312, released citation counts |

BBC starts with the original 9,635 x 2,225 term-by-document Matrix Market matrix. `numpy.random.RandomState(42).permutation(9635)` supplies an ordered permutation: the first 4,817 terms form view 0 and the remaining 4,818 form view 1. These partitions are disjoint and cover the full vocabulary. Within each view, transpose to document-by-feature, convert to float64, and apply per-feature min-max scaling. For a feature with `span = maximum - minimum`, use `(value - minimum) / span` when `span > 1e-12`; otherwise set the whole feature to zero. There is no row normalization. Document order follows `bbc.docs`; original class rows are indexed by the zero-based document indices in `bbc.classes`. Feature names and indices retain permutation order.

Cora and CiteSeer use the exact Grimal archive members `<dataset>/<tag>_content.mtx`, `<dataset>/<tag>_cites.mtx`, `<dataset>/<tag>_act.txt`, `<dataset>/documents-mapping.txt`, and `<dataset>/labels.txt`. The matrices are already document-by-feature and are preserved without min-max scaling, row normalization, feature selection, or row dropping. `documents-mapping.txt` supplies consecutive zero-based row indices and their source document IDs. The citation view contains the released inbound-plus-outbound counts; values greater than one are retained. The script does not binarize or symmetrize it again.

The earlier LINQS provenance audit is summarized in the identity JSON; it is not a runtime dependency and this tool does not redo LINQS alignment. That audit accounted for 5,429 directed Cora citation lines and 4,715 represented CiteSeer lines; the other 17 CiteSeer lines referenced absent content IDs. The present tool reproduces the exact archived Grimal matrices, including this historical source boundary.

## Output data dictionary

| File | Type, layout, and intended use |
| --- | --- |
| `X0.npz`, `X1.npz` | SciPy CSR sparse matrices. Row `i` refers to `metadata.json` -> `sample_ids[i]`. Columns retain the released order or the ordered BBC split defined above. Load with `scipy.sparse.load_npz`. These are the feature inputs to the downstream runner. |
| `y.npy` | NumPy int64 vector of shape `(n,)`, in exactly the same document order. The compatibility name for oracle analysis and sealed scoring. Labels must not enter label-free model training or label-free parameter selection. |
| `labels_FOR_ORACLE_AND_SEALED_SCORING_ONLY.npy` | An identical copy of `y.npy` with an explicit role in its filename. |
| `feature_indices.npz` | BBC only: int64 arrays `view0` (4,817 entries) and `view1` (4,818 entries), giving zero-based rows of the original term-by-document matrix in prepared column order. |
| `feature_names.json` | BBC only: two ordered lists of source vocabulary terms, aligned with the columns of `X0` and `X1`. Generated locally from externally supplied data. |
| `metadata.json` | Dataset identity, dimensions, counts, ordered source document IDs, transformation, label mapping, source identities, actual CLI source path, code/identity hashes, and observed runtime. |
| `INPUT_SHA256.json` | SHA256 of every generated file present before this checksum file is written. It intentionally excludes itself. These file hashes are distinct from the canonical prepared-data hash. |
| `ELEMENTWISE_VERIFICATION.json` | Present only with `--verify-against`. Records label equality, shapes, exact ordered value and CSR array equality, and maximum absolute error for each view. |

Key `metadata.json` fields:

| Key | Meaning |
| --- | --- |
| `dataset`, `n`, `k`, `views`, `dimensions` | Dataset name, document count, class count, view count, and ordered feature dimensions. |
| `class_counts`, `zero_rows` | Counts by zero-based label and number of completely zero rows in each feature view. |
| `sample_ids` | Ordered original document identifiers; `sample_ids[i]` describes row `i` in both views and both label files. |
| `label_original_values` | Original label corresponding to each new zero-based class ID. |
| `label_names` | Citation datasets only: class names in zero-based output label order, read from the archive. |
| `view_meaning`, `preprocessing`, `preparation_kind` | Ordered view definitions and exact preparation rule. |
| `prepared_sha256` | Original frozen canonical identity over ordered int64 labels and each sorted CSR matrix's shape, indptr, indices, and float64 data bytes. It is not the hash of a ZIP or NPZ container. |
| `source_input`, `source_urls`, `input_identity` | Explicit runtime input location, recorded distributor URLs, and the dataset's validated identity specification. |
| `preparation_source_sha256`, `identity_file_sha256` | Actual script and identity JSON checksums used by this run. |
| `runtime` | Observed Python, NumPy, and SciPy versions and whether Python isolated mode was enabled. |
| `training_labels` | Label-use notice for oracle analysis and sealed scoring. |

The canonical prepared-data identity includes native NumPy CSR index bytes from the verified environment. A change of representation or numeric environment that changes this identity causes a failure rather than silently accepting a different preparation.

## Completed verification

On 2026-11-17, all three datasets were actually prepared with Python `-I` in a newly created relocated directory initially containing only the portable script and identity JSON. BBC was prepared twice, from the original four files and from the freshly downloaded ZIP. Cora and CiteSeer were prepared from the freshly downloaded, identity-matching Grimal archives. Inputs and frozen reference directories were supplied through explicit CLI arguments. Neither the script nor identity JSON needs any historical workspace module or fixed historical path. `PREPARATION_VERIFICATION_PORTABLE.json` preserves the release-facing results and executed artifact hashes; the full local log is `PREPARATION_VERIFICATION.json` in the integration workspace.

All matrices in all four preparation runs matched the frozen preparations at every ordered value, with maximum absolute error **0**. All CSR `indptr`, `indices`, and `data` arrays matched. All labels matched exactly. Saved outputs were loaded again and compared independently, including the explicit oracle-label copy. Both BBC feature-index arrays also matched the independently frozen indices exactly for both source modes.

This step prepares input matrices only. Downstream graph construction, warm starts or other initialization, fitting, and evaluation remain the responsibility of the experimental runner. No model training was performed during this preparation verification.

## Integration boundary

Copy `prepare_bbc_citation.py`, `bbc_citation_identity.json`, this README (rename it `BBC_CORA_CITESEER_README.md` if placed in a shared tools directory), `PREPARATION_VERIFICATION_PORTABLE.json`, and `PUBLIC_SOURCE_STATUS.json` into the release. `RELEASE_FILES_SHA256.json` records their exact bytes. Keep the script and identity JSON beside one another unless `--identity` is supplied. Do not package `_verification/`, `_build_identity_local.py`, `_verify_local.py`, or the full local verification log: those are local execution artifacts, and `_verification/` contains generated dataset matrices. Original datasets must be obtained separately by each user.
