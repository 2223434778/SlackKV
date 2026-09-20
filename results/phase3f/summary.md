# Phase 3F

Phase 3B connector-first implementation reused unchanged. FirstTokenGate is a research admission gate: at most one marked resumed request waits for its first token; with max_tokens=1 this approximately serializes measured completions.

| Policy | promotion misses | ABC overlap ms | ABC P95 ms | overall P95 ms | corrected memory-time MiB·s |
|---|---:|---:|---:|---:|---:|
| OnDemand-Trace | 0.0 | 231.10 | 106.55 | 105.83 | 0.00 |
| ResumeAware | 0.0 | 290.77 | 129.71 | 126.46 | 24.39 |
| FirstTokenGate-ResumeAware | 0.0 | 52.79 | 122.75 | 119.88 | 37.67 |

Gate changed ABC pairwise overlap 290.77 -> 52.79 ms, max pre-first-token concurrency 3.0 -> 2.0, area 430.43 -> 287.48 ms. Gate blocked time is in session_metrics.csv. Phase 3E deadline-based residency undercounted delayed admission; Phase 3F uses actual request admission after promotion completion.

OnDemand scheduler-internal WAITING_FOR_REMOTE_KVS transitions are not exposed by this runtime; scheduler_trace.csv records client-visible arrival/admission/first-token transitions without fabricating internal states.

**Result B.** Phase 3F execution status: COMPLETE
