# Phase 4F-B — PSA-safe frozen hybrid performance matrix

## Objective

Compare latency, KV H2D traffic, leased GPU-KV cost, and background interference of four preregistered PSA-safe policies; isolate the incremental value of logical reservation once PSA-1 is present.

## Setup and compared methods

**RESULT_B_VALID_MATRIX_MIXED_TRADEOFF.** The eight-session first-boundary workload has real coding-agent token/KV structure and empirical TraceToolProxy pauses; it is not request-timestamp-faithful real Agent execution.

All runs use Qwen2.5-1.5B-Instruct FP16 on GPU0, vLLM 0.23.0, 4,681 GPU KV blocks (2 GiB), 8-GiB CPU external KV, K2, P_SAFE=102 ms, 110-ms resume guard, X9500→512, and the immutable Phase 4C workload. The four methods are A OnDemand, B guarded evictable JIT, C guarded JIT+lease+PSA-1, and D C plus 3,510-block logical reservation. The unsafe lease-without-PSA variant is excluded.

X-only 8/8: baseline decode 7496.11 ms; first-four median 7451.11 ms, last-four median 7629.01 ms, drift +2.39%. Stage A 4/4; Stage B 32/32; Agent completions and exact token counts 256/256.

## Results

## Policy outcomes (median of eight run-level metrics)

| Policy | P95 Resume ms | TTFT P95 ms | Local % | Useful Promo % | Total H2D MiB | Duplicate H2D MiB | Lease Mem-Time MiB·s | X Slowdown | Combined tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OnDemand-K2 | 2232.04 | 2193.20 | 0.00 | NA | 4594.19 | 0.00 | NA | 1.079 | 134.92 |
| GuardedJIT-K2 | 2232.27 | 2204.70 | 30.64 | 33.3 | 4869.59 | 275.41 | NA | 1.101 | 132.19 |
| GuardedJIT+Lease+PSA1-K2 | 2228.33 | 2189.45 | 24.26 | 83.3 | 4594.19 | 0.00 | 89.41 | 1.095 | 132.94 |
| FullSlackKV+PSA1-K2 | 2196.48 | 2163.93 | 24.26 | 100.0 | 4594.19 | 0.00 | 100.25 | 1.094 | 133.05 |

## Admission and resource costs

| Policy | Promo Admit | Capacity Reject | PSA Reject | Reservation Skip | Peak Lease Blocks | Peak Reserved Blocks | Reservation block·s |
|---|---:|---:|---:|---:|---:|---:|---:|
| OnDemand-K2 | 0.00 | 0.00 | 0.00 | 0.00 | NA | NA | NA |
| GuardedJIT-K2 | 3.00 | 5.00 | 0.00 | 0.00 | NA | NA | NA |
| GuardedJIT+Lease+PSA1-K2 | 2.00 | 3.00 | 3.00 | 0.00 | 2548.00 | NA | NA |
| FullSlackKV+PSA1-K2 | 2.00 | 3.00 | 0.00 | 3.00 | 2548.00 | 2548.00 | 1218.34 |

## Paired component comparisons

| Comparison | Median P95 Δ % | Median H2D Δ MiB | Median duplicate Δ MiB | Median X slowdown Δ | Interpretation |
|---|---:|---:|---:|---:|---|
| B_vs_A | +1.30 | +275.41 | +275.41 | +0.030 | guarded JIT before lease |
| C_vs_B | -1.34 | -275.41 | -275.41 | -0.002 | lease plus PSA (joint change) |
| D_vs_C | -1.06 | +0.00 | +0.00 | +0.000 | incremental logical reservation |
| D_vs_A | -1.18 | +0.00 | +0.00 | +0.025 | full system vs OnDemand |

H1 FAIL; H2 PASS; H3 PASS. D-vs-A P95 paired-median bootstrap 95% interval: -4.06% to +1.94%; descriptive uncertainty only, not a significance test.
B/C/D service-envelope overruns (>102 ms): 22/14/1. B remote waits 7; C 0; D 0. PSA interventions (old physical guard would admit): C 18, D 0.
Output identity mismatches (diagnostic, not validity gate): {'OnDemand-K2': 8, 'GuardedJIT-K2': 8, 'FullSlackKV+PSA1-K2': 8, 'GuardedJIT+Lease+PSA1-K2': 8}. No crashes, OOMs, leaks, unsafe releases, or PSA invariant violations were accepted into the matrix.
Lease occupancy uses LEASE_ACQUIRED→LEASE_RELEASED; unlike evictable JIT lead time, it is an exact leased-residency cost. Logical reservation incremental effects are the D-vs-C row; no winner is assumed. Unsafe lease-without-PSA is excluded from performance comparisons and retained only as historical liveness-negative evidence.
No predictor, 32-boundary replay, PSA-2, capacity sweep, SOTA comparison, or statistical-significance claim was made.

## Interpretation and limitations

D versus A changes the median of run-level P95 values by -1.59% (paired median -1.18%). This is below the preregistered 10% improvement threshold; H1 therefore fails even though H2D stays at the one-load minimum and H3 passes.
B's evictable prefetch produced 7 post-T6 remote waits and 275.41 MiB median duplicate traffic. C adds lease **and** PSA together: those counts fall to 0 and 0.00 MiB, but this matrix cannot assign the entire difference to the lease alone.
C versus D has the same median local fraction and total H2D; paired P95 changes -1.06%. D makes a median 3.0 fewer attempts by skipping 3.0 jobs and avoids C's 18 PSA rejects across eight runs. D's median lease memory-time is +10.84 MiB·s higher. Thus reservation shows promotion selectivity but limited incremental latency/traffic value on this frozen workload, and it is not cost-free.
PSA's observable selection cost is 18 physically feasible promotions rejected in C (versus 0 in D, where logical reservation filters candidates first). These are safe fallbacks, not request failures. This matrix does not isolate PSA's latency cost from lease effects.
The output-identity mismatches are confined to E04: one E04 mismatch in every measured run under every policy. All 256 requested output lengths and source-accounting checks passed. E04's batch-sensitive token identity remains a diagnostic limitation, not evidence of semantic trace replay or KV corruption.
Eight run-level repetitions support directional description only; paired bootstrap intervals are descriptive, not a significance test.

- D_vs_A paired P95 delta: median -1.18% (min -5.25%, max +2.51%).
- B_vs_A paired P95 delta: median +1.30% (min -3.96%, max +7.01%).
- C_vs_B paired P95 delta: median -1.34% (min -4.54%, max +5.21%).
- D_vs_C paired P95 delta: median -1.06% (min -4.07%, max +2.70%).
- C_vs_A paired P95 delta: median -0.06% (min -4.44%, max +3.87%).

## Conclusion and next-step status

The preregistered classification is **RESULT_B_VALID_MATRIX_MIXED_TRADEOFF**. This Phase technically completed with all correctness and liveness gates satisfied; the 10% latency hypothesis failed. No later phase was launched or authorized by this result.
