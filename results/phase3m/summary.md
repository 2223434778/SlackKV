# Phase 3M — native promotion latency decomposition

Valid runs: 12/12.

| n | Client E2E | Wait-to-Connector | Connector Phase | Copy | Previous Step | Remaining Step |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 18.98 | 2.70 | 10.15 | 5.33 | 149.28 | 0.00 |
| 1 | 54.28 | 14.74 | 9.92 | 5.28 | 18.19 | 16.64 |
| 2 | 54.23 | 10.99 | 10.67 | 5.38 | 19.13 | 12.90 |
| 3 | 55.20 | 13.51 | 10.02 | 5.28 | 19.00 | 15.42 |

Busy wait fraction median: 0.29.
Normal step in flight at promotion: 9/9 busy runs.
Client first-token delivery lag: median 10.69 ms, max 17.81 ms.
Phase 3L phase bias: YES.
Extra busy latency is concentrated before connector-first starts and tracks the remaining in-flight step time; connector phase and CPU→GPU copy remain stable.
Connector-only execution itself becomes slower: NO. CPU→GPU copy becomes slower: NO.
Remaining/in-flight step relation: STRONG at the per-run diagnostic level.
Dominant latency source: INFLIGHT_MODEL_STEP.

Result A.
Next direction: STEP_TIME_ESTIMATOR_E2E.

Timestamps use one monotonic clock domain. Connector-only semantics and native correctness were unchanged; no scheduling policy was modified.
