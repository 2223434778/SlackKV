# Phase 4F-A — PSA-1 liveness restoration

Result: **RESULT_A_PSA1_RESTORES_LIVENESS**. This is a correctness/liveness experiment, not a performance comparison.

The admission rule is `m <= F` and `F - m >= H`, with `H = max(0, P_h - max(L_h, m if candidate==protected else 0))`. Here `F` is the scheduler's exact free-block count, `m` is candidate prefix blocks, and `L_h` is the engine's actual matching protected lease. The protected job is the earliest-deadline request without a first token, including already-sent requests. `P_h = ceil((Turn1 ISL + 1)/16)` includes new input and first-token progress; cached-prefix blocks alone would omit that footprint.

In U3, old physical admission can accept E02 because its prefix fits at that instant, leaving less than E04's 1278-block progress footprint. The observed PSA micro state was free=2073, required=1323, after=750, reserve=1278; its E02 decision was `PROGRESS_REJECT`. The `progress_rejected` path performs no transfer, lease, model forward, or generation and falls back to OnDemand.

| Gate | Valid / expected | E04 waiting-allocation failures |
|---|---:|---:|
| P0 micro | 3/3 | NA |
| P1 baseline no X | 1/1 nonprogress reproduced | 13 |
| P1 PSA no X | 2/2, 16/16 completed | 0 |
| P2 PSA + X | 3/3, 24/24 completed | 0 |
| P3 protected E04 | 2/2, 16/16 completed | 0 |
| P4 real JIT | 8/8, 64/64 completed | 0 |
| P5 FullSlackKV | 4/4, 32/32 completed | 0 |

P1 E02 progress-rejects: 2/2. P2 E02 progress-rejects: 3/3. P3 candidate E04 admitted: 2/2 with 7-block incremental reserve.

P4 promotion admits per run: [2, 2, 2, 2, 2, 2, 2, 2]; progress rejects per run: [3, 3, 3, 3, 4, 3, 3, 3]. Lease-acquired-before-arrival promotions without remote wait, a conservative confirmed-useful count, were 16 in P4. P5 reservation skips: 12; progress rejects: 0; capacity rejects: 12; confirmed useful promotions: 8. If PSA never triggers in P5, the fixed 3510-block logical reservation may overlap with the progress-safe condition on this workload; that does not establish equivalence.

PSA trace invariant violations: 0. Server crashes: 0; OOM: 0; final physical lease leaks: 0; logical leaks: 0; unsafe logical releases: 0.

Limitation: only the frozen eight first-boundary hybrid jobs, oracle TraceToolProxy timing, pinned vLLM scheduler, and GPU0 were tested. The retained compact artifacts establish usefulness via lease acquisition before arrival; they do not retain the endpoint's exact T6 timestamp for each promotion, so this count is conservative. No latency winner, performance matrix, predictor, or 32-boundary replay is claimed.
