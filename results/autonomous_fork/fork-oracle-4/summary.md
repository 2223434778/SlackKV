# fork-oracle-4

Objective: Synthetic CPU-prefix fork restoration diagnostic; no serving candidate.

See [protocol](../protocol.md), [phase summary](../summary.md), and execution.json for exact environment/source/input controls.

Status: COMPLETE; GPU hours including startup/cleanup: 0.0723131804.

Workflow median 11125.48 ms; child-call TTFT P95 546.63 ms; BG median 32230.16 ms.
Actual H2D 2016 MiB, D2H 2310 MiB; external first calls/episode [1, 1, 1]; preemptions 0.0. Native in-flight sharing already effective. Owned memory-time UNKNOWN.
Invariant timing excluded from performance; sixteen verified cold exact-output queries pass.

Anomalies: early source_delta fields (if present) are INVALID; use actual raw source/counters. All phase failures documented in analysis_failures.txt.

Conclusion: diagnostic/regression complete; no coalescing candidate. Round stopped B; next problem needs host-miss/readiness opportunity proof.
