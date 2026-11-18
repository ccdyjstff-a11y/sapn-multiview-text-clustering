"""Rescore 42 fixed CMVNMF MU/final checkpoints and original SAPN partitions.

No fit, stage selection, coefficient search or significance test is performed.
All resource/primary prediction bytes are checked before any external classes
are loaded. External labels must match the original ordered partition identity.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from resource_common import verify_resource, read_json, write_csv, sha256

METRICS = ("ACC", "NMI", "ARI")
DATASETS = ("3Sources", "BBC", "Reuters", "Cora", "CiteSeer", "20NG", "AGNews")
PRIMARY_MANIFEST = "81ee5eca0f4f5c67238d560bb2a8eddf43d6a36ed5d10d4d5ad4609b7448f9cd"


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def inside(root, relative):
    root = Path(root).resolve()
    target = (root / relative).resolve()
    need(target.is_relative_to(root) and target != root, "File path escapes supplied root")
    return target


def partition_hash(labels):
    mapping, codes = {}, []
    for label in np.asarray(labels).reshape(-1).tolist():
        if label not in mapping:
            mapping[label] = len(mapping)
        codes.append(mapping[label])
    return hashlib.sha256(np.asarray(codes, dtype="<i8").tobytes()).hexdigest()


def metrics(truth, predicted):
    _, yy = np.unique(truth, return_inverse=True)
    _, zz = np.unique(predicted, return_inverse=True)
    table = np.zeros((yy.max() + 1, zz.max() + 1), dtype=np.int64)
    np.add.at(table, (yy, zz), 1)
    i, j = linear_sum_assignment(-table)
    return {"ACC": float(table[i, j].sum() / len(truth)),
            "NMI": float(normalized_mutual_info_score(truth, predicted, average_method="arithmetic")),
            "ARI": float(adjusted_rand_score(truth, predicted))}


def decode(arrays, prefix, record):
    normalized = []
    for view in range(record["views"]):
        h = arrays[prefix + "_Hv" + str(view)]
        need(h.shape == (record["k"], record["n"]) and np.isfinite(h).all() and (h >= 0).all(), "Invalid stored factor")
        normalized.append(h / np.maximum(h.sum(axis=0, keepdims=True), 1e-30))
    # Preserve the exact original floor/order; do not replace with softmax or
    # unrestricted scale cancellation, which changes decoding of subnormal MU factors.
    return np.mean(normalized, axis=0).argmax(axis=0)


def direction(value):
    return "positive" if value > 1e-12 else "negative" if value < -1e-12 else "zero"


def run(root, labels_map, primary_root, output):
    need(__debug__, "Do not use python -O: resource integrity helper uses assertions")
    root, primary_root, output = Path(root).resolve(), Path(primary_root).resolve(), Path(output).resolve()
    need(not output.exists() and not output.is_relative_to(root), "Output must be a new directory outside the sealed resource")
    resource = verify_resource(root)
    resource_manifest_hash = sha256(root / "MANIFEST.json")
    records_data = read_json(root / "CMVNMF_RUNS.json")
    records = records_data["records"]
    requirements = read_json(root / "FEATURE_REQUIREMENTS.json")["corpora"]
    need(records_data["primary_predictions_manifest_sha256"] == PRIMARY_MANIFEST, "Wrong original primary identity")
    need(sha256(primary_root / "MANIFEST.json") == PRIMARY_MANIFEST, "Original primary manifest changed")
    primary_data = read_json(primary_root / "MANIFEST.json")
    primary = {(r["dataset"], r["seed"], r["Q"], r["method"]): r for r in primary_data["records"]
               if r["method"] in ("SAPN-MV", "CMVNMF")}
    need(len(records) == 42 and len({(r["dataset"], r["seed"], r["budget"]) for r in records}) == 42, "Incomplete/duplicate checkpoint cases")
    need(len(primary) == 84, "Expected 42 original predictions per method")
    seen_paths = {}
    # Integrity and exact selected coefficients are established for all cases
    # before opening the user-supplied category arrays.
    for record in records:
        ds, seed, q = record["dataset"], record["seed"], record["budget"]
        need(ds in DATASETS and q in (160, 400), "Unexpected checkpoint scope")
        need(record["n"] == requirements[ds]["n"] and record["k"] == requirements[ds]["k"], "Checkpoint dimensions differ")
        state = inside(root, record["state_file"])
        need(sha256(state) == record["state_sha256"], "Checkpoint factor bytes changed")
        need(record["mu_aliases_final"] is (not record["continued"]), "Invalid MU/final alias flag")
        for method in ("CMVNMF", "SAPN-MV"):
            p = primary[(record["primary_dataset"], seed, q, method)]
            need(p["eligible"] and p["partition_sha256"] == requirements[ds]["labels_partition_sha256"], "Original primary partition identity differs")
            path = inside(primary_root, p["file"])
            need(sha256(path) == p["sha256"], "Original primary prediction bytes changed")
            seen_paths[str(path)] = p["sha256"]
            if method == "CMVNMF":
                need(float(p["selected_configuration"]["parameter"]) == float(record["beta"]), "Originally selected coefficient differs")
                need(p["selected_configuration"]["selection_sha256"] == record["original_selection_sha256"], "Original coefficient selection identity differs")
    for ds in DATASETS:
        for q in (160, 400):
            cell = [r for r in records if r["dataset"] == ds and r["budget"] == q]
            need(len(cell) == len({r["seed"] for r in cell}) == 3, "Incomplete original paired seed triplet")
    # The only class admission point: integrity checks above are complete.
    labels_map = Path(labels_map).resolve()
    mapping = read_json(labels_map)
    mapping = mapping.get("corpora", mapping)
    labels, label_hashes = {}, {}
    for ds in DATASETS:
        spec = mapping.get(ds)
        if spec is None:
            alias = next(r["primary_dataset"] for r in records if r["dataset"] == ds)
            spec = mapping[alias]
        need(isinstance(spec, str), "labels-map values must be external NPY path strings")
        path = Path(spec)
        if not path.is_absolute():
            path = labels_map.parent / path
        path = path.resolve()
        need(not path.is_relative_to(root), "Class arrays must remain external to this resource")
        y = np.load(path, allow_pickle=False).reshape(-1)
        need(len(y) == requirements[ds]["n"] and len(np.unique(y)) == requirements[ds]["k"], "Class count/object count differs")
        need(partition_hash(y) == requirements[ds]["labels_partition_sha256"], "Class partition or object ordering differs")
        labels[ds] = y
        label_hashes[str(path)] = sha256(path)
    rows, max_error = [], 0.0
    for record in records:
        ds, seed, q = record["dataset"], record["seed"], record["budget"]
        truth = labels[ds]
        with np.load(inside(root, record["state_file"]), allow_pickle=False) as arrays:
            final = decode(arrays, "final", record)
            mu = final.copy() if record["mu_aliases_final"] else decode(arrays, "MU", record)
            pool, test = arrays["pool"], arrays["test"]
            positive, negative = arrays["P"], arrays["N"]
            expected_pool = np.sort(np.random.RandomState(seed + 21000).permutation(record["n"])[:round(.3 * record["n"])])
            need(np.issubdtype(pool.dtype, np.integer) and np.issubdtype(test.dtype, np.integer), "Non-integer split indices")
            need(np.array_equal(pool, expected_pool) and np.array_equal(test, np.setdiff1d(np.arange(record["n"]), pool)), "Original evaluation split differs")
            need(positive.ndim == negative.ndim == 2 and positive.shape[1] == negative.shape[1] == 2
                 and np.issubdtype(positive.dtype, np.integer) and np.issubdtype(negative.dtype, np.integer), "Invalid paid pair array")
            need(len(positive) + len(negative) == q and np.isin(positive, pool).all() and np.isin(negative, pool).all(), "Paid budget/query-pool mismatch")
            need(np.all(truth[positive[:, 0]] == truth[positive[:, 1]]) and np.all(truth[negative[:, 0]] != truth[negative[:, 1]]), "External labels contradict paid answers")
        values = {"MU": metrics(truth[test], mu[test]), "final": metrics(truth[test], final[test])}
        for method, stage in (("CMVNMF", "final"), ("SAPN-MV", "SAPN")):
            original = primary[(record["primary_dataset"], seed, q, method)]
            with np.load(inside(primary_root, original["file"]), allow_pickle=False) as saved:
                need(np.array_equal(saved["query_pool"], pool) and np.array_equal(saved["evaluation_indices"], test), "Original comparator split differs")
                need(saved["labels"].shape == (record["n"],) and np.isfinite(saved["labels"]).all(), "Invalid original prediction")
                if method == "CMVNMF":
                    need(np.array_equal(final, saved["labels"]), "Final factor decoding does not match original primary labels")
                else:
                    values[stage] = metrics(truth[test], saved["labels"][test])
            error = max(abs(values[stage][m] - original["metrics"][m]) for m in METRICS)
            need(error < 1e-12, "Recomputed primary score differs from sealed original")
            max_error = max(max_error, error)
        row = {"dataset": ds, "seed": seed, "Q": q, "selected_beta": record["beta"], "continued": record["continued"], "evaluated_n": len(test)}
        for metric in METRICS:
            for stage in ("MU", "final", "SAPN"):
                row[stage + "_" + metric] = values[stage][metric]
            row["final_minus_MU_pp_" + metric] = 100 * (values["final"][metric] - values["MU"][metric])
            for stage in ("MU", "final"):
                row["SAPN_minus_" + stage + "_pp_" + metric] = 100 * (values["SAPN"][metric] - values[stage][metric])
        rows.append(row)
    groups = []
    for ds in DATASETS:
        for q in (160, 400):
            cell = [r for r in rows if r["dataset"] == ds and r["Q"] == q]
            group = {"dataset": ds, "Q": q, "pairs": 3, "continued_runs": sum(r["continued"] for r in cell)}
            for metric in METRICS:
                for prefix in ("final_minus_MU_pp_", "SAPN_minus_MU_pp_", "SAPN_minus_final_pp_"):
                    vals = [r[prefix + metric] for r in cell]
                    group[prefix + metric + "_mean"] = float(np.mean(vals))
                    group[prefix + metric + "_sample_sd"] = float(np.std(vals, ddof=1))
                for stage in ("MU", "final"):
                    group[metric + "_" + stage + "_direction"] = direction(group["SAPN_minus_" + stage + "_pp_" + metric + "_mean"])
                group[metric + "_mean_direction_changed"] = group[metric + "_MU_direction"] != group[metric + "_final_direction"]
            groups.append(group)
    verify_resource(root)
    need(sha256(primary_root / "MANIFEST.json") == PRIMARY_MANIFEST, "Primary manifest changed during scoring")
    for path, digest in dict(seen_paths, **label_hashes).items():
        need(sha256(path) == digest, "Input changed during scoring")
    summary = {"status": "ALL42_FIXED_CHECKPOINTS_RESCORED", "resource_manifest_sha256": resource_manifest_hash,
               "primary_manifest_sha256": PRIMARY_MANIFEST, "pairs": 42, "groups": 14,
               "primary_metric_comparisons": 252, "maximum_primary_metric_error": max_error,
               "checkpoint_change_counts": {m: {s: sum(direction(r["final_minus_MU_pp_" + m]) == s for r in rows)
                                                  for s in ("positive", "zero", "negative")} for m in METRICS},
               "mean_direction_changes": {m: [{"dataset": g["dataset"], "Q": g["Q"]} for g in groups if g[m + "_mean_direction_changed"]] for m in METRICS},
               "labels_partition_hashes": {ds: partition_hash(y) for ds, y in labels.items()},
               "new_fits": 0, "CV_rerun": False, "stage_selected_using_scores": False,
               "p_values_computed": False, "original_selected_coefficients_fixed": True,
               "interpretation": "Fixed-coefficient checkpoint sensitivity only; not a separately tuned MU-only comparison.",
               "script_sha256": sha256(__file__)}
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "per_run_all_metrics.csv", rows)
    write_csv(output / "summary_14_groups_all_metrics.csv", groups)
    (output / "SCORING_COMPLETE.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    hashes = {p.name: sha256(p) for p in output.iterdir() if p.is_file()}
    (output / "OUTPUT_SHA256.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--labels-map", type=Path, required=True, help="External JSON mapping dataset names to external NPY class arrays")
    parser.add_argument("--primary-root", type=Path, required=True, help="Public v1.0.1 primary_predictions directory")
    parser.add_argument("--output", type=Path, required=True, help="New output directory outside this resource")
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.root, args.labels_map, args.primary_root, args.output), indent=2))
        return 0
    except (RuntimeError, AssertionError, OSError, KeyError, ValueError, TypeError) as error:
        print("FAIL_CLOSED: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
