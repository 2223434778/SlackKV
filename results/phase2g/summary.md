# Phase 2G — Warm-state QueueJIT contention

## Objective and setup

Base a74b0893cbc2a19e982701be0948fa1bbd757de4. Reused vLLM 0.23.0, PyTorch 2.11.0+cu126, CUDA 12.6, GPU0 NVIDIA RTX A6000; Qwen2.5-1.5B-Instruct, float16, TP=1, 320 MiB GPU KV, 2 GiB pinned CPU offload, prefix caching, max model length 10000.

Each of 9 measured restarts sent normal-offload A/B/C first turns (4096 tokens each), waited 1 second, sent the two frozen 9500-token pressure prompts, then promoted C exactly once as an excluded warm-up. Only after the C external-hit validation snapshot was t0 defined. A/B used the frozen 42-ms estimate, deadlines 350/370 ms from t0, three repetitions per ASAP/NaiveJIT/QueueJIT, and one serialized worker.

The request-based promotion proxy includes vLLM request handling, external KV recovery, model execution, and response handling. The C request duration is excluded. No metrics scrape or extra request occurred between measured A/B promotions. This validates scheduling using warm-state serving costs; it does not prove native PCIe contention.

## Mechanism checks

- A/B/C exact first-turn D2H offloads (117,440,512 bytes each): PASS.
- C warm-up recovered 4096 external-hit tokens and 0 local-hit tokens in every run: PASS. Its latency is recorded separately in run_metrics.csv and excluded from policy results.
- C warm-up promotion median was 111.22 ms; measured A/B promotions had median 38.71 ms and range 35.18–43.08 ms, consistent with the expected warm-state range.
- A/B promotion pair recovered 8192 external tokens and 0 local tokens: PASS.
- After promotion, A/B resumes hit 8192 GPU-local tokens and 0 external tokens: PASS.
- Local-hit control and all measured A/B resume first-token IDs match: PASS; controls A=1084, B=1084.
- Warm-state rule (median of 18 A/B proxy durations <=60 ms): PASS.

## Policy medians across three repetitions

| Policy | Misses/run | Max stall (ms) | Total early residency (ms) | A promotion (ms) | B promotion (ms) | First start from t0 (ms) |
|---|---:|---:|---:|---:|---:|---:|
| ASAP | 0.00 | 0.00 | 608.61 | 37.16 | 36.16 | 0.37 |
| NaiveJIT | 1.00 | 19.49 | 1.31 | 40.13 | 40.58 | 308.50 |
| QueueJIT | 0.00 | 0.00 | 24.93 | 42.58 | 36.73 | 286.48 |

NaiveJIT queue-induced B miss/stall: YES by policy medians. QueueJIT reduced max stall from 19.49 ms to 0.00 ms and misses/run from 1.00 to 0.00.
Across the three NaiveJIT repetitions, B missed in 3/3 runs; A also missed once (1/3; 1.50 ms in that run), while B stalled in every run.
QueueJIT began its first promotion 22.02 ms earlier than NaiveJIT. It saved 583.68 ms of total early residency versus ASAP.

Result A: Warm-state real multi-session QueueJIT advantage validated.

## Anomalies and limitations

- This is a request-proxy experiment with oracle deadlines, two measured sessions, one worker, and three repetitions per policy. Promotion duration is not pure H2D and includes normal vLLM execution/handling.
- C is only a promotion-path warm-up. Its service time is never included in A/B policy statistics.
- The isolated Python runtime prints a non-fatal missing optional nvidia_cutlass_dsl .pth warning. In NaiveJIT repetition 2, vLLM also logged EngineDeadError in its async output handler during SIGTERM shutdown; this followed the shutdown trigger, and all 10 HTTP requests in that run returned 200 before teardown. Its cache, resume and token-ID checks passed. No request failure or result row was silently dropped.

Phase 2G execution status: COMPLETE
