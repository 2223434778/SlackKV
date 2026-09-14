# Phase 1C

Phase 1B-Fix showed QueueJIT-alone is limited because many real pauses are shorter than migration cost. Phase 1C evaluates oracle, transfer-normalized selective admission with immediate non-contended D2H and one serialized QueueJIT H2D resource.

The retained target-model-compatible TraceLab set contains 283,193 events (`reusable_tokens <= 131072`), mapped primarily at 56 KiB/token; 128 KiB/token uses exactly the same events as stress sensitivity. Closed-loop replay, continuation semantics, token accounting, and conservative `result_at` deadlines are reused from Phase 1B-Fix.

At primary M=32 (median of ten seeds), OffloadAll has miss 67.33%, P95 4855.0 ms, and normalized local memory-time 0.262. SlackRatio gamma 2 has offload fraction 36.37%, miss 5.53%, P95 115.0 ms, and memory-time 0.258. Gamma 3 has offload fraction 31.51%, miss 3.78%, P95 0 ms, and memory-time 0.261. KeepAll remains 0 stall and normalized memory-time 1.0.

FixedTTL results are retained in full. At M=32, TTL 1 s has 32.48% offload, 4.81% miss, 40.2 ms P95, and 0.247 normalized memory-time; TTL 2 s has 22.06% offload, 2.07% miss, 0 ms P95, and 0.250 memory-time. The programmed Pareto scan finds at least one qualifying size-aware comparison.

Criterion C0: PASS. No FeasibilityOracle or SlackRatio gamma>=1 admission violates isolated round-trip feasibility.

Criterion C1: PASS. SlackRatio gamma 2 has >=1000-job M>=16 instances with miss <=10%, P95 <=100 ms, and normalized memory-time <=0.50.

Criterion C2: PASS. The mechanical comparison reports qualifying memory/latency-matched SlackRatio-versus-TTL pairs in `pareto_comparison.csv`.

Remaining misses are dominated by promotion queue collisions and large-KV blocking; D2H contention is intentionally not modeled. This remains an oracle-deadline simulator: no real tensors, model, vLLM, prediction error, or measured post-result request slack.

Phase 1C execution status: COMPLETE
