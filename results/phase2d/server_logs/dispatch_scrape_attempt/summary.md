# Phase 2D — Deadline-Aware JIT Promotion Timing

## Objective and setup

Test whether recovery of an already CPU-offloaded Session A KV can be moved earlier in a real vLLM pause, so the real resume uses GPU-local KV. This uses one RTX A6000, Qwen2.5-1.5B-Instruct, vLLM 0.23.0, PyTorch 2.11.0+cu126, CUDA 12.6, 320 MiB GPU KV, 2 GiB pinned CPU offload, 4096-token A prefix, 2-second resume horizon, and three runs per policy.

The JIT prefetch is an experimental proxy: an ordinary completion request with the exact original prefix triggers CPU→GPU KV recovery; its generated token is discarded. `prefetch_request_ms` includes request and model execution overhead. This is not a production zero-compute promotion API.

## Mechanism checks

- OnDemand external recovery: median external prompt tokens = 4096.0; median local-hit tokens = 0.0.
- JIT prefetch external-hit validation: PASS; across proxy requests, external hit median = 4096.0 tokens.
- Post-prefetch real-resume local-hit validation: PASS; JIT real-resume local-hit median = 4096.0 tokens.
- Local-hit control: TTFT 94.10 ms; local-hit tokens 4096.0; first token ID 1084.
- Token-ID correctness against local-hit control: PASS for every measured real resume.

## Policy medians

| Policy | Prefetch duration (ms) | Prefetch stall (ms) | Early residency (ms) | Actual resume TTFT (ms) | Effective resume latency (ms) | Resume dispatch delay (ms) |
|---|---:|---:|---:|---:|---:|---:|
| OnDemand | NA | NA | NA | 103.46 | 103.46 | 0.14 |
| JIT-100 | 111.63 | 11.85 | 0.00 | 34.68 | 45.15 | 18.93 |
| JIT-250 | 115.02 | 0.00 | 134.86 | 36.96 | 36.96 | 0.12 |
| JIT-500 | 113.20 | 0.00 | 386.57 | 42.52 | 42.52 | 0.21 |

The measured A_first E2E latency median was 198.07 ms across all 12 runs; median GPU→CPU transfer time counter was 5.88 ms, with exactly 117,440,512 offloaded bytes per run. JIT moves resume recovery earlier; it does not remove this original GPU→CPU offload cost. Phase 2C-Fix previously measured about 57.7 ms higher A-first E2E than B-first E2E; that historical first-turn difference is separate from effective resume latency and is not folded into these values.

Among conditions with <=1 ms median prefetch stall, JIT-250 had the least median early residency (134.86 ms); this is a descriptive tradeoff point, not an optimality claim.

**Result A:** Real deadline-aware JIT promotion mechanism validated.

## Limitations and anomalies

- The ordinary-request prefetch proxy includes a generated token and normal model execution. It can overstate a future promotion-only runtime's cost and may change cache state.
- This experiment uses one session, oracle 2-second deadline, one model, and no queue contention or prediction error.
- Pressure is exactly the validated pair of distinct 9500-token requests; no additional pressure was added.
- An initial run failed result aggregation, and a complete follow-up exposed redundant metrics scrapes at dispatch points. Those attempts are archived and excluded; the final runner uses existing snapshots as counters baselines and records dispatch lag explicitly.
- See `raw_runs.csv` and `metric_snapshots.json` for per-run timings and the actual vLLM metric samples.

Phase 2D execution status: COMPLETE
