# Third-party acquisition identities

These are historical acquisition identities from the research archive. No network availability was rechecked by this release build. The manifests provide file-level SHA256 values and, where recorded, byte counts. Obtain author code separately and verify its applicable terms. Do not run demos that generate additional class-derived constraints outside the declared budget.

| Method | Author repository | Recorded commit | Hash manifest |
|---|---|---|---|
| FSSMSC | https://github.com/linghm/FSSMSC | `064f9cba06b30d7b152eb046e909a028e386ded7` | `benchmark/THIRD_PARTY_INPUTS.json` |
| PCBG | https://github.com/YWJ-Best/PCBG | `8e1f97e21848dc8d9c9440750ff7a1526142d2ab` | `provenance/third_party/PCBG_SOURCE_HASHES.json` |
| SSA-SNMF | https://github.com/kamalberahmand/SSA-SNMF | `4d37f15621ee7bd9d412ad224408658ad243699c` | `provenance/third_party/SSA_SOURCE_HASHES.json` |

For example, the recorded PCBG.m SHA256 is `e7c4c9847b43807ee86099f20b8ac140f652b7d5bbc48370766f5753ef791358`. The recorded FSSMSC_bound.m SHA256 is `821aabc40ef95dcc059cdb797c7858f63206057bbce47e9262e8c26729992905`. File-level records, rather than branch names, define the historical checkout.

The manuscript reports adaptations and eligibility qualifications. FSSMSC uses a pair-input clustering adapter and an author iteration cap, which is not a stationarity certificate. PCBG retains the native partition and unsupported zero-row cases. SSA-SNMF repairs a missing Laplacian and pair indexing, retains other paper/code differences, and has incomplete attempted coverage. Downloading the original code does not by itself reproduce these adapted studies. This compact release provides their retained result evidence and compact partitions, not a newly validated end-to-end baseline retraining wrapper.

CMVNMF and MVCHSS use the supplied research Python implementations through `benchmark/reproduce.py`; their interface qualifications are in `baselines/README.md`. FSSMSC's own Python/MATLAB bridge and acquisition/patch generator are also supplied, while the author optimizer remains external. Historical PCBG/SSA training adapters are provided as traceable source with explicit path-mapping limitations. The 210 primary stored predictions include CMVNMF/MVCHSS/FSSMSC outputs and support independent scoring with external labels.
