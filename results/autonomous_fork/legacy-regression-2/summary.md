# legacy-regression-2

Objective: Historical fixed-workload Compiled OnDemand regression.

See [protocol](../protocol.md), [phase summary](../summary.md), and execution.json for exact environment/source/input controls.

Status: VALID; GPU hours including startup/cleanup: 0.0304816700.

32 resumes, P95 4853.24 ms; session median 23878.57 ms; BG wall 67764.82 ms. Actual inputs match Round2. No algorithm speedup claim.

Anomalies: early source_delta fields (if present) are INVALID; use actual raw source/counters. All phase failures documented in analysis_failures.txt.

Conclusion: diagnostic/regression complete; no coalescing candidate. Round stopped B; next problem needs host-miss/readiness opportunity proof.
