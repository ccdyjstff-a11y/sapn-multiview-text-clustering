"""Portable replay of all 84 sealed baseline rows, using explicit external labels."""
from pathlib import Path
import argparse
import csv
import json
import sys
from importlib.metadata import version
sys.dont_write_bytecode = True
import numpy as np
from baseline_math import (need, sha256, read_json, write_json, write_csv, child, integer_vector,
                           partition_hash, metrics, ber, sample_pairs, aggregate, METRICS)


def verify_resource(root, expected):
    root = Path(root).resolve()
    need(isinstance(expected, str) and len(expected) == 64, "Explicit expected resource manifest SHA256 required")
    need(sha256(root / "MANIFEST.json") == expected, "Resource manifest differs from expected identity")
    need((root / "MANIFEST.sha256").read_text(encoding="ascii").strip() == expected, "Resource manifest seal differs")
    manifest = read_json(root / "MANIFEST.json")
    need(manifest["schema"] == "sapn-baseline-partitions-v1" and manifest["edition"] == "major_revision_20261118", "Wrong resource edition/schema")
    need(manifest["conditions"] == 84 and manifest["methods"] == {"SSA": 42, "PCBG": 42}, "Incomplete resource scope")
    actual = {str(p.relative_to(root)).replace("\\", "/") for p in root.rglob("*") if p.is_file()}
    need(actual == set(manifest["files"]) | {"MANIFEST.json", "MANIFEST.sha256"}, "Unexpected/missing resource files")
    for name, digest in manifest["files"].items():
        need(sha256(child(root, name)) == digest, "Resource file changed: " + name)
    return manifest


def compare_scalar(actual, expected, label):
    if actual is None or expected is None:
        need(actual is None and expected is None, "Missingness differs: " + label)
        return 0.0, 0
    need(np.isfinite(actual) and np.isfinite(expected), "Nonfinite score: " + label)
    error = abs(float(actual) - float(expected))
    need(error <= 1e-12, "Score differs beyond 1e-12: " + label)
    return error, 1


def compare_groups(groups, csv_path):
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as stream:
        originals = list(csv.DictReader(stream))
    need(len(groups) == len(originals) == 14, "Complete 14-cell reference required")
    lookup = {(r["dataset"], int(r["Q"])): r for r in originals}
    need(len(lookup) == 14, "Duplicate reference group")
    max_error, count = 0.0, 0
    for group in groups:
        original = lookup[(group["dataset"], group["Q"])]
        need(set(group) == set(original), "Group fields differ")
        for field, value in group.items():
            stored = original[field]
            label = str((group["dataset"], group["Q"], field))
            if value is None:
                need(stored == "", "Incomplete group mean was filled: " + label)
            elif isinstance(value, bool):
                need(stored == str(value), "Group eligibility differs: " + label)
            elif isinstance(value, (float, int)):
                error, n = compare_scalar(value, float(stored), label)
                max_error = max(max_error, error); count += n
            else:
                need(value == stored, "Group identity differs: " + label)
    return max_error, count


def validate_packets(root, records, reference):
    packets, contexts = {}, {}
    need(len(records) == len({(r["method"], r["condition_id"]) for r in records}) == 84, "84 unique resource rows required")
    for record in records:
        identity = (record["method"], record["condition_id"])
        row = reference[identity]
        need(all(row[k] == record[k] for k in ("method", "condition_id", "dataset", "seed", "Q", "n")), "Record identity differs from actual scored row")
        with np.load(child(root, record["packet"]), allow_pickle=False) as a:
            expected_keys = {"pool", "test", "paid_pairs", "paid_answers", "SAPN_partition"}
            if record["baseline_partition_present"]:
                expected_keys.add("baseline_partition")
            need(set(a.files) == expected_keys, "Unapproved array member (X/y/private state is forbidden)")
            values = {k: integer_vector(a[k]) for k in a.files}
            values["paid_pairs"] = values["paid_pairs"].reshape(-1, 2)
        n, q = record["n"], record["Q"]
        pool, test, pairs, answers = (values[k] for k in ("pool", "test", "paid_pairs", "paid_answers"))
        need(np.array_equal(np.sort(np.r_[pool, test]), np.arange(n)) and len(np.unique(pool)) == len(pool)
             and len(np.unique(test)) == len(test), "Pool/test object partition differs")
        expected_pool = np.sort(np.random.RandomState(record["seed"] + 21000).permutation(n)[:round(.3 * n)])
        need(np.array_equal(pool, expected_pool) and np.array_equal(test, np.setdiff1d(np.arange(n), pool)), "Frozen query-pool complement differs")
        need(pairs.shape == (q, 2) and answers.shape == (q,) and np.isin(pairs, pool).all()
             and np.isin(answers, [-1, 1]).all(), "Paid query budget/indices differ")
        need((pairs[:, 0] < pairs[:, 1]).all() and len(np.unique(pairs, axis=0)) == q, "Paid pairs duplicated or reversed")
        need(values["SAPN_partition"].shape == (n,) and (values["SAPN_partition"] >= 0).all()
             and len(np.unique(values["SAPN_partition"])) <= record["k"], "Invalid original SAPN partition")
        if "baseline_partition" in values:
            z = values["baseline_partition"]
            need(z.shape == (n,) and len(np.unique(z)) == row["occupied_clusters"], "Baseline partition shape/occupied k differs")
            need(not row["eligible"] or len(np.unique(z)) == record["k"], "Eligible native partition has wrong k")
        else:
            need(row["occupied_clusters"] is None and not row["eligible"], "Missing native partition marked eligible")
        context_key = (record["dataset"], record["seed"])
        if context_key in contexts:
            previous = contexts[context_key]
            need(np.array_equal(previous["test"], test) and previous["evaluation_pair_seed"] == record["evaluation_pair_seed"], "Cross-budget/method evaluation context differs")
        else:
            contexts[context_key] = {"test": test, "evaluation_pair_seed": record["evaluation_pair_seed"]}
        packets[identity] = values
    return packets, contexts


def run(root, expected_manifest, labels_map, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    need(not output.exists() and not output.is_relative_to(root), "Output must be a new directory outside the sealed resource")
    manifest = verify_resource(root, expected_manifest)
    records = read_json(root / "RECORDS.json")
    reference, original_rows = {}, {}
    for method in ("SSA", "PCBG"):
        rows = read_json(root / "reference" / method / "SEED_ROWS.json")
        aggregate(rows, method)  # Validates all original denominators/category/missing rules.
        original_rows[method] = rows
        reference.update({(method, r["condition_id"]): r for r in rows})
    packets, contexts = validate_packets(root, records, reference)
    requirements = read_json(root / "LABEL_REQUIREMENTS.json")
    required = {r["dataset"] for r in records if r["baseline_partition_present"]}
    need(set(requirements) == required <= {"3Sources", "BBC", "Cora", "CiteSeer"}, "External-class admission scope differs")
    mapping_path = Path(labels_map).resolve(); mapping = read_json(mapping_path)
    mapping = mapping.get("corpora", mapping)
    labels, external_files, serialization = {}, {}, {}
    # Verify all external file hashes before decoding the first category array.
    for ds in sorted(required):
        spec = mapping[ds]
        need(isinstance(spec, dict) and isinstance(spec.get("path"), str) and isinstance(spec.get("sha256"), str), "Label map requires explicit path and SHA256")
        path = Path(spec["path"])
        if not path.is_absolute(): path = mapping_path.parent / path
        path = path.resolve()
        need(path.suffix.lower() == ".npy" and not path.is_relative_to(root), "Category arrays must be external NPY files")
        need(sha256(path) == spec["sha256"], "External category byte hash differs: " + ds)
        external_files[ds] = (path, spec["sha256"])
    for ds, (path, digest) in external_files.items():
        req = requirements[ds]
        y = integer_vector(np.load(path, allow_pickle=False), req["n"])
        need(len(np.unique(y)) == req["k"] and partition_hash(y) == req["ordered_partition_sha256"], "External class partition/object order differs: " + ds)
        labels[ds] = y
        serialization[ds] = digest == req["original_label_file_sha256"]
    # Check every admitted paid answer and cross-budget context before any metric.
    for record in records:
        ds = record["dataset"]
        if ds not in labels: continue
        a = packets[(record["method"], record["condition_id"])]; pairs = a["paid_pairs"]; y = labels[ds]
        need(np.array_equal(np.where(y[pairs[:, 0]] == y[pairs[:, 1]], 1, -1), a["paid_answers"]), "Paid answers contradict supplied object ordering")
    rows_by_method, groups_by_method, coverage_by_method = {}, {}, {}
    max_error, scalar_count, missing_count = 0.0, 0, 0
    evaluation_cache = {}
    for method in ("SSA", "PCBG"):
        rows = []
        for original in original_rows[method]:
            row = dict(original); cid = row["condition_id"]
            record = next(r for r in records if r["method"] == method and r["condition_id"] == cid)
            a = packets[(method, cid)]
            for prefix in ("baseline", "SAPN", "delta"):
                for metric in METRICS: row[prefix + "_" + metric] = None
            if "baseline_partition" in a:
                y = labels[record["dataset"]]; context_key = (record["dataset"], record["seed"])
                if context_key not in evaluation_cache:
                    context = contexts[context_key]
                    ep = sample_pairs(context["test"], context["evaluation_pair_seed"])
                    evaluation_cache[context_key] = (ep, np.where(y[ep[:, 0]] == y[ep[:, 1]], 1, -1))
                ep, ea = evaluation_cache[context_key]
                need(not np.intersect1d(ep, a["pool"]).size, "Evaluation pairs overlap paid query pool")
                for prefix in ("baseline", "SAPN"):
                    z = a[prefix + "_partition"]
                    values = metrics(y[a["test"]], z[a["test"]])
                    values.update(training_BER=ber(z, a["paid_pairs"], a["paid_answers"]), unqueried_BER=ber(z, ep, ea))
                    for metric, value in values.items(): row[prefix + "_" + metric] = None if value is None else float(value)
                if row["eligible"] and row["SAPN_eligible"]:
                    for metric in METRICS:
                        first, second = row["baseline_" + metric], row["SAPN_" + metric]
                        row["delta_" + metric] = None if first is None or second is None else first - second
            for prefix in ("baseline", "SAPN", "delta"):
                for metric in METRICS:
                    field = prefix + "_" + metric
                    error, count = compare_scalar(row[field], original[field], method + "/" + cid + "/" + field)
                    max_error = max(max_error, error); scalar_count += count; missing_count += 1 - count
            rows.append(row)
        groups, coverage = aggregate(rows, method)
        error, count = compare_groups(groups, root / "reference" / method / "paired_complete_groups.csv")
        max_error = max(max_error, error); scalar_count += count
        need(coverage == read_json(root / "reference" / method / "COVERAGE.json"), "Complete coverage report differs")
        rows_by_method[method], groups_by_method[method], coverage_by_method[method] = rows, groups, coverage
    verify_resource(root, expected_manifest)
    for path, digest in external_files.values(): need(sha256(path) == digest, "External labels changed during replay")
    result = {"status": "ALL84_BASELINE_ROWS_REPLAYED", "resource_manifest_sha256": expected_manifest,
              "pipeline_complete_sha256": manifest["pipeline_complete_sha256"], "rows_per_method": {"SSA": 42, "PCBG": 42},
              "groups_per_method": {"SSA": 14, "PCBG": 14}, "numeric_comparisons": scalar_count,
              "missing_row_metric_cells_preserved": missing_count, "maximum_absolute_error": max_error,
              "external_label_hashes": {ds: d for ds, (_, d) in external_files.items()},
              "external_serialization_matches_original": serialization,
              "ordered_class_partitions_verified": True, "paid_answers_verified_before_metrics": True,
              "available_case_means_computed": False, "missing_counted_as_win": False, "new_fits": 0,
              "private_algorithm_imported": False, "p_values_computed": False, "original_results_changed": False,
              "runtime_versions": {"python": sys.version.split()[0], **{p: version(p) for p in ("numpy", "scipy", "scikit-learn")}},
              "rescorer_sha256": sha256(__file__)}
    output.mkdir(parents=True, exist_ok=False)
    for method in ("SSA", "PCBG"):
        directory = output / method; directory.mkdir()
        write_json(directory / "SEED_ROWS.json", rows_by_method[method]); write_json(directory / "COVERAGE.json", coverage_by_method[method])
        write_csv(directory / "paired_seed_rows.csv", rows_by_method[method]); write_csv(directory / "paired_complete_groups.csv", groups_by_method[method])
    write_json(output / "REPLAY_QA.json", result)
    write_json(output / "OUTPUT_SHA256.json", {str(p.relative_to(output)).replace("\\", "/"): sha256(p) for p in output.rglob("*") if p.is_file()})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--expect-manifest-sha256", required=True)
    parser.add_argument("--labels-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.root, args.expect_manifest_sha256, args.labels_map, args.output), indent=2)); return 0
    except (RuntimeError, ValueError, OSError, KeyError, TypeError) as error:
        print("FAIL_CLOSED: " + str(error), file=sys.stderr); return 2


if __name__ == "__main__": sys.exit(main())
