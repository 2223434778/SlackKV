# fork-decode-k2-4

Objective: Synthetic CPU-prefix fork restoration diagnostic; no serving candidate.

See [protocol](../protocol.md), [phase summary](../summary.md), and execution.json for exact environment/source/input controls.

Status: COMPLETE; GPU hours including startup/cleanup: 0.0209584580.

Workflow median 4154.34 ms; child-call TTFT P95 474.15 ms; BG median 8615.56 ms.
Actual H2D 2016 MiB, D2H 2310 MiB; external first calls/episode [1, 1, 1]; preemptions 0.0. Native in-flight sharing already effective. Owned memory-time UNKNOWN.
Single control run, three non-independent episodes; no statistical-significance or quality claim.

Anomalies: early source_delta fields (if present) are INVALID; use actual raw source/counters. All phase failures documented in analysis_failures.txt.

Conclusion: diagnostic/regression complete; no coalescing candidate. Round stopped B; next problem needs host-miss/readiness opportunity proof.
