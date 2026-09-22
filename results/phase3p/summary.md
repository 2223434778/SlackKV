# Phase 3P-FIX

Frozen Phase 3P raw CSVs were postprocessed only. No GPU, vLLM, Qwen, or native prefetch workload was rerun.

Raw validation: 9 runs; 54 sessions; 36 native records; 9 background records; 3 X-only records.

Async internal-ready/local-hit/remote-wait: 18/18, 18/18, 0. Native correctness 18/18; token correctness 54/54; background 9/9; K2 9/9.
Async issue-to-T6 ms: ABC median/max 28.44/37.44; DEF 24.61/30.36; all median/P95/max 24.88/33.72/37.44; p_async=40 sufficient YES.

|Policy|A-F P95|ABC P95|DEF P95|Completion P95|Decode tok/s|X slowdown|Combined tok/s|Makespan|Memory|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|OnDemand-K2|101.32|101.32|79.33|585.44|127.48|1.056|131.69|1545.80|0.00|
|Blocking70-K2|60.80|60.80|50.45|572.63|127.91|1.093|127.19|1515.62|72.30|
|Async40-K2|56.78|56.78|46.96|551.85|128.88|1.079|128.92|1500.09|41.42|

X-only baseline: 192 tokens, 191 post-first tokens, median 2711.43 ms.
Async vs Blocking: P95 -6.62%; DEF -6.93%; memory -42.71%; combined throughput 1.36%.
Result A; next direction: Phase 3Q stable async workload.
