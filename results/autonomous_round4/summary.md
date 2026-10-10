# Round4 evidence

Report: [AUTONOMOUS_RESEARCH_ROUND4.md](../../docs/AUTONOMOUS_RESEARCH_ROUND4.md).
Frozen protocol: [BASELINE_AND_OPPORTUNITY_ROUND4.md](../../docs/BASELINE_AND_OPPORTUNITY_ROUND4.md).

- 12 gate screens: 2/4 GiB, three matched K2/off pairs each. No >=10% session improvement.
- Observer OFF/ON repaired pair: separate from algorithm evaluation.
- Physical retention: one prerequisite invariant smoke and one OFF/order/pin triplet. Both candidates failed; conditional reverse triplet was not triggered.
- Failed resource-off retained, including the erroneous native source, partial artifacts and cost.
- 19 GPU ledger entries: 18 VALID (17 performance +1 correctness), 1 FAILED; 0.5462201082031889 GPU-hours; maximum111.9593 seconds; GPU0 only.

Start Git: d403714ea3ed91e585a2b9f80beb363023272de8 plus recorded working changes.
No new algorithm speedup, pristine upstream baseline, statistical significance or full quality/liveness claim.

`metrics.csv`, `round_summary.json`, `control_summary.json`, `resource-on-repaired/diagnosis.json`,
`integration_audit.json`, `budget_summary.json`, `validation.json`, `retention-smoke/smoke.json`
locate all main conclusions. Per-run execution/config/input/output/logs preserve the underlying evidence.

`SHA256SUMS.json` covers271 exported logical files; decompress `.gz` before comparing a logical raw hash.
The manifest predates these explanatory files. Source snapshots separately describe actual server bytes
(Windows Git source may use different line endings); `provenance/source_manifest.json` links them.
Raw server originals remain. Independent audit verifies17 complete performance workloads, exact FG/BG
hashes/counts, per-session timing and175 pool-partition observations. Numerical outputs are saved;
40-input content/quality equivalence remains UNKNOWN.
