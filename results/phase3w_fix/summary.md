# Phase 3W-FIX — closed-loop prediction feedback

All frozen workload and policy parameters were preserved. The Phase 3S physical lease implementation was reused unchanged.

| Policy | Issued/run | Stale cancel/run | Useful issued/run | Internal misses/run | Local | P95 resume (ms) | Total H2D (MiB) | Duplicate H2D (MiB) | Runtime releases/run | Promotion memory-time (MiB·s) | X slowdown | Combined tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OracleDynamicBlock512-K2 | 6 | 0 | 6 | 0 | 50.0% | 82.18 | 1344.00 | 0.00 | 6 | 92.77 | 1.068 | 162.44 |
| RawPred-ClosedLoop-DynamicBlock512-K2 | 4 | 6 | 4 | 0 | 33.3% | 112.11 | 1344.00 | 0.00 | 2 | 51.83 | 1.065 | 163.01 |
| SafePred200-ClosedLoop-DynamicBlock512-K2 | 6 | 0 | 6 | 0 | 50.0% | 86.09 | 1344.00 | 0.00 | 6 | 167.60 | 1.077 | 161.09 |

| Policy | Stale prefetch after resume | Unsafe reclaims/run | Lease leaks | Reservation leaks | Request crashes |
|---|---:|---:|---:|---:|---:|
| OracleDynamicBlock512-K2 | 0 | 0 | 0 | 0 | 0 |
| RawPred-ClosedLoop-DynamicBlock512-K2 | 0 | 0 | 0 | 0 | 0 |
| SafePred200-ClosedLoop-DynamicBlock512-K2 | 0 | 0 | 0 | 0 | 0 |

## Validation

- Tests: 23 passed in 3.42s; return code 0.
- Zero-error behavior remained unchanged: **PASS**; Oracle/Raw decision match = True and stale cancellations = 0.
- Integration gate: **PASS**, 3/3.
- Measured runs: 9/9; real resume token correctness 108/108.
- Raw crash reproduced: **NO**. Raw safely cancelled 17 stale predicted promotions across measured runs and issued no native prefetch after `RESUME_STARTED`.
- Safe restored the expected 6 promotion / 6 capacity-skip split in every run; all 18 issued promotions were useful.
- A/B runtime release reclaimed logical capacity before E/F admission: **PASS**.
- Controller runtime release followed physical lease release: **PASS**. Unsafe reclaim count was 0 and peak physical leases remained at most 512 blocks.
- Raw cancellations created 0.00 MiB duplicate H2D: cancelled work performed no native transfer and used normal OnDemand fallback.
- Raw versus Safe P95 difference (Raw−Safe): +26.02 ms. Local-fraction difference: -16.7%. Promotion-memory-time difference: -115.77 MiB·s. X-slowdown difference: -0.013. Combined-throughput difference: +1.93 tok/s.

## Direct answers

1. Zero-error behavior remained unchanged: **YES**.
2. The former Raw request crash was not reproduced: **YES, fixed by cancellation/fallback**.
3. Raw stale cancellations: 6/run median (17 total).
4. Native prefetch after `RESUME_STARTED`: **0**.
5. Safe returned from 5/7 to 6/6: **YES, 3/3 runs**.
6. A/B runtime release freed capacity before E/F admission: **YES**.
7. Runtime releases followed physical lease release: **YES**.
8. Runtime-release physical overcommit: **NO**.
9. Safe issued promotions useful: **18/18**.
10. Raw cancellation duplicate H2D: **0 MiB**.
11. Raw P95 112.11 ms versus Safe 86.09 ms.
12. Safe promotion-memory-time cost versus Raw: +115.77 MiB·s.
13. X slowdown: Raw 1.065, Safe 1.077.
14. Combined throughput: Raw 163.01, Safe 161.09 tok/s.
15. Result: **RESULT_A_CLOSED_LOOP_ROBUSTNESS**.
16. Ready for real Agent trace replay: **YES**; next direction `REAL_AGENT_TRACE_REPLAY`. Phase 3W-FIX itself did not start that work.
