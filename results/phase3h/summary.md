# Phase 3H: two-slot first-token admission

## Objective and frozen setup

- Compared ResumeAware-K3, FirstTokenGate-K1, and FirstTokenGate-K2 on only A/B/C with identical ResumeAware native promotion: p_safe=30 ms and resume_guard=110 ms.
- Runtime: vLLM 0.23.0 source 0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665, PyTorch 2.11.0+cu126, CUDA 12.6, RTX A6000 GPU0, Qwen2.5-1.5B-Instruct FP16.
- Each 4096-token resume generated exactly 32 greedy tokens. Three fresh server restarts were used per policy.

## Median metrics across three runs

| Policy | Max Pre-First | Pre-First Overlap (ms) | Gate Wait (ms) | ABC P95 TTFT (ms) | ABC Completion P95 (ms) | Makespan (ms) | Decode tok/s | Memory-Time (MiB*s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ResumeAware-K3 | 3 | 157.43 | 0.00 | 131.69 | 599.79 | 602.35 | 196.54 | 33.34 |
| FirstTokenGate-K1 | 1 | 0.00 | 140.27 | 159.98 | 619.75 | 655.21 | 180.58 | 49.53 |
| FirstTokenGate-K2 | 2 | 45.12 | 30.31 | 115.23 | 601.62 | 623.64 | 182.99 | 37.22 |

## Comparisons

- K2 vs ResumeAware-K3: ABC P95 TTFT improved 12.50%; decode throughput retained 93.11%; makespan increased 3.53%; pre-first overlap decreased 71.34%.
- K2 vs K1: ABC P95 TTFT improved 27.97%; total gate wait decreased 78.39%; decode throughput increased 1.34%; memory-time decreased 24.85%.

## Mechanism and correctness

- K1 integration: PASS. K2 integration: PASS. K2 reached exactly two pre-first owners in every repetition; C emitted GATE_BLOCKED before admission in every repetition.
- All three policies had zero promotion deadline misses, blackout conflicts, and long native prefetches. All A/B/C resumes generated exactly 32 tokens.
- Native endpoint records directly reported 4096 external tokens loaded and zero model-forward, generated-token, and mixed-step counts for every promotion. Cache transport validation passed.
- All policies retained post-first decode concurrency of three. Later requests were admitted before earlier requests completed: full-request serialization is NO.

## Result

**Result A.** A two-slot pre-first admission bound provides a better latency/batching tradeoff than both unrestricted ResumeAware admission and strict K=1 FirstTokenGate for this isolated A/B/C workload.

## Direct answers

- Does K2 achieve pre-first concurrency exactly 2? YES.
- Does K2 reduce C gate waiting relative to K1? YES.
- Does K2 improve ABC P95 TTFT relative to K1? YES.
- Does K2 beat unrestricted ResumeAware TTFT? YES.
- Is K2 memory-time below K1? YES.
- Does K2 remain non-serial after first tokens? YES.
- Is fixed-K admission worth continuing? PROMISING within the isolated A/B/C workload; Phase 3H does not evaluate D/E/F.
