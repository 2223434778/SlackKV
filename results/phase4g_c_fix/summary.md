# Phase 4G-C-FIX: Native Absolute-Deadline Timing Repair

Final result(s): RESULT_D_GUARD_ALLOCATOR_MISMATCH.
Start commit: c10a6c7ac69e41b444d00497c5d258da6bfc71b0; workload SHA256: 9b381336a70e73b0efbddbb79d2c88608c19663247e819c1e7a364efd2971105.

Phase4G-C original matrix was invalidated solely by pause-fidelity failure. Phase4G-C-FIX changes only the client pause waiter and timing diagnostics: the Python sleep-plus-busy-spin was replaced by CLOCK_MONOTONIC absolute clock_nanosleep. No SlackKV mechanism, pause value, deadline, or timing threshold was changed.

## Timing calibration

Native backend self-test: True; P95/max absolute wake overshoot=0.21354/0.219442 ms.
Stage T: 6/6 complete valid runs, 192/192 boundaries; aggregate P99 |pause error|=4.122891 ms (gate 10 ms).
Timing classification: RESOLVED.

## Fresh measured matrix

Measured runs: 8/8; requests: 320/320; boundaries: 256/256.
C0 pause |error| P95/max=0.506927/2.533131 ms; sleep overshoot P95=0.442583 ms; dispatch P95=0.075834 ms.
C1 pause |error| P95/max=1.043547/6.242016 ms; sleep overshoot P95=0.972932 ms; dispatch P95=0.073204 ms.

## Phase 4G-C mechanism analysis

The inherited RCA section below uses only the eight fresh measured runs. Stage T and prior Phase 4G-C observations are excluded from causal aggregation.

# Phase 4G-C: Source Recheck × PSA × Allocator-Truth RCA

Final result(s): RESULT_D_GUARD_ALLOCATOR_MISMATCH.
Start commit: c10a6c7ac69e41b444d00497c5d258da6bfc71b0. Workload SHA256: 9b381336a70e73b0efbddbb79d2c88608c19663247e819c1e7a364efd2971105.

This is a mechanism RCA, not a latency-performance comparison. C0 retains one-shot source visibility; C1 consumes only a periodic exact full-ready transition. Candidate ranking, deadlines, latest-feasible-start, reservation, PSA-1, physical guard, leases, K2, BGv2, and workload are frozen.

## Paired mechanism summary

| Metric | C0 Current | C1 Recheck-only |
|---|---:|---:|
| Promotion selections | 40 | 127 |
| Stale protected visibility | 6 (15.0%) | 0 (0.0%) |
| Candidate = protected | 1 (2.5%) | 4 (3.1%) |
| PSA reserve median / P95 / max (blocks) | 1336.0 / 1583.0 / 1584 | 1359 / 1568.0 / 1596 |
| Progress rejects | 4 | 17 |
| Capacity rejects | 20 | 82 |
| Exact allocator no-fit attempts | 20 | 20 |
| Successful / useful promotions | 0 / 0 | 0 / 0 |

## Allocator truth

The pinned vLLM 0.23.0 allocate_slots path calls the coordinator's get_num_blocks_to_allocate() and compares that actual required count to pool free minus in-flight reservations. Its separate full-sequence admission gate uses the pool-free count. This phase records those exact runtime inputs and outcomes; it does not substitute the cached-prefix full footprint for incremental allocator demand. No allocator decision branch was changed.

| Quantity | Median | P95 |
|---|---:|---:|
| Legacy full-prefix required blocks | 1358.0 | 1573.0 |
| Allocator required blocks | 1358.0 | 1456.0 |
| Pool free blocks | 1142.0 | 1453.0 |
| Allocator available blocks | 1209.0 | 1453.0 |
| PSA-1 protected reserve | 1359.0 | 1568.0 |
| PSA-1 minus allocator need | 0.0 | 1430.0 |

Conservatism by identity: candidate=protected gap {'median': -1330.0, 'p90': -1330.0, 'p95': -1330.0} and ratio {'median': 0.005235602094240838, 'p90': 0.005235602094240838, 'p95': 0.005235602094240838}; candidate differs gap {'median': 0.0, 'p90': 1333.0, 'p95': 1430.0} and ratio {'median': 1.0, 'p90': 149.55555555555554, 'p95': 191.42857142857142}.

Protected local-residency strata and median PSA/allocator/gap: [{'local_residency': 'none', 'count': 135, 'median_local_blocks': 0, 'median_PSA1_need': 1363.0, 'median_allocator_need_now': 1358.0, 'median_need_gap': 0.0}, {'local_residency': 'partial', 'count': 9, 'median_local_blocks': 972, 'median_PSA1_need': 7.0, 'median_allocator_need_now': 1337.0, 'median_need_gap': -1330.0}, {'local_residency': 'full', 'count': 23, 'median_local_blocks': 1321, 'median_PSA1_need': 1346.0, 'median_allocator_need_now': 18.0, 'median_need_gap': 1321.0}]. Exact ownership medians: {'local_cached_blocks': {'available_count': 167, 'median_blocks': 0}, 'leased_blocks': {'available_count': 167, 'median_blocks': 0}, 'owned_by_running_blocks': {'available_count': 167, 'median_blocks': 0}, 'owned_by_waiting_blocks': {'available_count': 167, 'median_blocks': 0}, 'evictable_cached_blocks': {'available_count': 167, 'median_blocks': 0}, 'other_refcounted_blocks': {'available_count': 167, 'median_blocks': 0}}.

Instantaneous allocator requirement is not itself a future liveness guarantee: unleased cache blocks can be evicted by a candidate allocation. These are conservatism comparisons, not permission to replace PSA-1.

## C0 progress-reject counterfactual

Among 4 C0 progress rejects, protected source was exact full-ready in 2; protected-as-candidate passed the existing PSA test in 0, exact allocator preflight in 1, and all recorded timing/logical/physical tests in 0. See progress_reject_counterfactual.csv for per-case facts.

Legacy guard and exact allocator agreed on 61/123 (49.6%); false-admit=0, false-reject=62. Capacity rejects split into structural=39 and selection-to-attempt race=1.

Where a guarded prefetch proceeded into allocate_slots, shadow-vs-observed preflight fit agreed on 0 observations at fraction None.

## Run validity and interpretation

Fresh measured runs valid: 8/8; turns complete: 320/320; boundaries: 256/256. Crashes=0, OOM=0, logical leaks=0, lease leaks=0, K2 violations=0.
Source-hit coverage is diagnostic, not a Phase 4G-C run gate: 8/8 runs had fewer local+external cache-hit tokens than the full target (6144 tokens short); 6144 are explained by the pinned OffloadingConnector default offload_prompt_only=true, and 0 remain unexplained. Per-boundary evidence is in source_coverage.csv.
Instrumentation overhead: PASS; maximum source-probe CPU duty C0/C1=3.91%/3.96%; maximum total diagnostic CPU duty C0/C1=4.14%/4.73%.

The causal funnel is recorded per boundary in promotion_funnel.csv. Readiness requires every required block; no partial or percentage threshold is used. Protected KV ownership is split into running, waiting, leased, evictable, and other-refcounted blocks only where the pinned runtime exposes exact block-ID evidence; otherwise the artifact keeps NA.

## Frozen questions

1. C0 stale protected visibility: 6/40 selections.
2. Protected-self recoveries: 0/4 C0 progress-reject cases; exact readiness count 2.
3. PSA-1 median reserve: C0=1336.0, C1=1359 blocks.
4. Exact allocator fit and legacy mismatch directions are tabulated in allocator_guard_comparison.csv.
5. Exact full-target cache-hit equality is diagnostic, not a request-correctness gate; the source-hit shortfall is separately attributed in source_coverage.csv.
6. Runtime instrumentation overhead gate: PASS.
7. Result taxonomy: RESULT_D_GUARD_ALLOCATOR_MISMATCH.
8. Source recheck sufficiency: not established by the frozen thresholds.
9. PSA-2: not supported as the primary explanation by this result alone. Allocator-aligned admission: indicated for separate follow-up. Partial promotion was not tested or implemented.

This experiment distinguishes mechanism evidence within the frozen hybrid trace-driven workload; it does not establish a unique root cause outside this workload/runtime and does not compare performance.
