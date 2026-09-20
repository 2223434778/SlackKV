# Phase 3C — safety-budgeted native JIT

Phase 3B connector-first vLLM code was reused unchanged. The frozen scheduling budget is `p_safe=30 ms`; native service times were recorded but not used to update the budget.

## Correctness

Native sanity: PASS; external tokens 4096; transfer bytes 117440512; model forwards/generated 0/0; post-prefetch local hit 4096; token correctness PASS.

## Median results across three restarts

| Policy | Misses | Long prefetches (>50 ms) | Max client prefetch (ms) | P95 effective (ms) | Idle memory-time (MiB·s) |
|---|---:|---:|---:|---:|---:|
| OnDemand | 0.00 | 0.00 | 0.00 | 104.78 | 0.00 |
| NaiveJIT-Safe30 | 2.00 | 1.00 | 103.89 | 127.67 | 5.04 |
| QueueJIT-Safe30 | 1.00 | 1.00 | 111.63 | 111.06 | 9.62 |

## Interpretation

The Phase 3B ASAP historical reference is 383.61 MiB·s. QueueJIT-Safe30 uses 9.62 MiB·s (2.5% of that reference). The ~100-ms cascade diagnostic is NO. QueueJIT versus NaiveJIT: P95 111.06 vs 127.67 ms and misses 1.00 vs 2.00. QueueJIT versus OnDemand: P95 111.06 vs 104.78 ms.

All cache, generation, timeline, offload, zero-forward and mixed-step invariants passed in all nine runs. No metrics scrape occurs during the measured timeline. This phase uses oracle deadlines and the request-based native promotion endpoint. The Phase 3B connector-first implementation was reused unchanged.

**Result A.** Phase 3C execution status: COMPLETE
