# fork-control-4

Objective: Synthetic CPU-prefix fork restoration diagnostic; no serving candidate.

See [protocol](../protocol.md), [phase summary](../summary.md), and execution.json for exact environment/source/input controls.

Status: FAILED; GPU hours including startup/cleanup: 0.0150874492.

Aggregation helper failed after episode zero; outputs unavailable, exclude from complete performance. failure.txt and original logs/inputs retained.

Anomalies: early source_delta fields (if present) are INVALID; use actual raw source/counters. All phase failures documented in analysis_failures.txt.

Conclusion: diagnostic/regression complete; no coalescing candidate. Round stopped B; next problem needs host-miss/readiness opportunity proof.
