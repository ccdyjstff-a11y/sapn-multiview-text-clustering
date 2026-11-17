# Independent grouped S2 compact evidence — preparation only

No actual export is created during preparation. The original S1 compact exporter, S1 results and V10 remain unchanged. This packet may be built only from an actual S2 seal, actual S2 POST_SCORE audit and its bound comparison report. It never reads historical S1 CV/final results, full class truth or the evaluation/summary scores.

Original-workspace export requires both exact hashes:

```text
python export_ssa_compact.py --execute-report-sha <ssa_grouped_reporting/result_packet/REPORT_MANIFEST.json SHA> --audit-sha <its actual POST_SCORE audit SHA>
```

The explicit audit must certify all 42 final rows, 14 groups and recollection of all CV/selection records. Missing or mismatched bindings refuse export. Scoring is not performed by this entry. SciPy is used only to project one native MAT at a time: retained labels for CV/final, and readable final V_star arrays. Nonfinite/ineligible finite partitions remain diagnostic. Original MATs, dense states, raw features, full truth, private author sources, MATLAB preferences and retirement snapshot files are excluded.

Portable replay has the existing packager interface:

```text
python source/replay_ssa_compact.py --root <packet> --manifest-sha <EXPORT_MANIFEST.json SHA>
```

`verify(export_root, expected_manifest_sha=None)` needs only Python and NumPy. It opens packet-relative members, never the original absolute paths. It reconstructs paid node-disjoint folds, CV BER and all-decided candidate selection/default/ties; verifies local skip triggers, group guards, dispatch order, receipt dispositions and consecutive resource-cap stopping; and verifies complete-condition payloads and boundary-stop acknowledgements. Raw native numerical qualification and scheduler/commit eligibility remain separate. An uncommitted finite final cannot become primary. All 864 CV slots, 42 final outcomes, 24 selection records, 21 original query streams and 14 three-seed group denominators remain visible.

The returned flags are `budget_BER_recomputed=True`, `metrics_recomputed=False`, `fitting_performed=False`, `process_exit_rechecked=False`. Replay does not independently reconstruct dense model states, rerun physical memory monitoring/process cleanup, or recompute ACC/NMI/ARI or test-pool BER. Original MAT projection equality is checked source-side at export; after relocation only recorded identities and exported bytes can be verified.

`result_packet/EXPORT_MANIFEST.json` retains kind `SSA_COMPACT_EXPORT`, adds `schema=2`, `batch=SSA_GROUPED_V2`, `prior_S1_results_imported=False`, and uses relative `members_sha256`. Its `public_source` lists the seven `source/` members. Arrays/paid files use the previous compact schema. New `SCHEDULER_MEMBERS.json` maps original S2 paths/hashes to exactly one packet copy under `evidence/`, `receipts/` or `scheduler/`. Historical absolute paths are inert provenance. A future packager must copy manifest-listed members once and allow these S2 scheduler members; the old S1 fixed member whitelist is insufficient.

Remaining actual bindings are the future frozen S2 protocol/runtime, terminal collector-approved ledgers and receipts, global seal, score manifest, passed POST_SCORE audit and comparison report. The local `ledger_validation.py` must byte-match the actual sealed collector source, and `receipt_policy.py` must byte-match frozen S2 `adapter_policy.py`. A later change to either requires an explicit preparation update; no silent policy substitution is allowed. Full training and closure/process replay remain external operations, not portable package claims.

The three representative synthetic tests cover a complete condition (including tampered BER and uncommitted diagnostics), local pruning/all-invalid fallback, and persistent-resource global stopping. They are small in-memory checks, not a full exported-batch replay or native-run qualification.
