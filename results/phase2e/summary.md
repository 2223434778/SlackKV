# Phase 2E — Multi-session real promotion scheduling

## Objective and frozen setup

Test the queueing consequences of treating two paused-session KV restores as deadline-bearing jobs. Base commit: `4055e134c123e1e2f2562e09d46de5387863e280`. Runtime: vLLM 0.23.0, PyTorch 2.11.0+cu126, CUDA 12.6, NVIDIA RTX A6000 (GPU0), Qwen2.5-1.5B-Instruct, float16, TP=1, 320 MiB GPU KV, 2 GiB CPU offload, prefix caching, max model length 10000.

Each measured run used the distinct 4096-token Phase 2C-Fix A/B prefixes, normal offload on both first turns, a 1-second store wait, and exactly the two distinct 9500-token pressure requests with `max_offload_tokens=0`. After pressure, `t0` was recorded; deadlines were A=1500 ms and B=1560 ms, with frozen estimated service p_est=120 ms. Each policy ran in three independent server restarts.

The promotion operation remains an ordinary vLLM completion request carrying the exact cached prefix; its generated token is discarded. Its measured service includes request handling, external KV load, normal model execution, and response handling, not just H2D. A single client-side worker serialized promotions. Real resumes were sent only after both promotions had completed, so this validates scheduling with real runtime service times, not native PCIe contention or a production promotion API.

## Mechanism validation

- A and B initial offloads: PASS; expected 117,440,512 bytes each in every measured run. Expected combined external data is 234,881,024 bytes (~224 MiB).
- Prefix eviction and external promotion: PASS; every run's two serialized promotions observed external-hit delta 8192 tokens and local-hit delta 0.
- Post-promotion resume cache evidence: PASS; every run observed 8192 local-hit tokens and 0 external-hit tokens across A/B resumes.
- Generation correctness against no-pressure controls: PASS; A control token ID=1084, B control token ID=1084.
- Metrics were captured at server ready, after each first turn, after pressure, after both promotions, and after each resume. No metrics request or real resume occurred between promotion requests.

## Policy medians (three repetitions)

| Policy | Misses/run | Max stall (ms) | Total early residency (ms) | A promotion (ms) | B promotion (ms) | Worker span (ms) | First start from t0 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ASAP | 0.00 | 0.00 | 2747.31 | 116.50 | 37.74 | 155.16 | 21.52 |
| NaiveJIT | 0.00 | 0.00 | 33.38 | 113.57 | 38.70 | 152.14 | 1380.52 |
| QueueJIT | 0.00 | 0.00 | 146.92 | 116.17 | 37.59 | 157.32 | 1320.56 |

QueueJIT reduced queue-induced misses/stall versus NaiveJIT: NO (median NaiveJIT vs QueueJIT misses/run 0.00 vs 0.00; max stall 0.00 vs 0.00 ms).
QueueJIT began its first promotion 59.96 ms earlier than NaiveJIT by median first-start time. QueueJIT saved 2600.39 ms (94.7%) median total early residency versus ASAP.
QueueJIT median total early residency was 113.54 ms higher than NaiveJIT, with no measured miss/stall reduction.

**Result B:** Multi-session promotion works, but the configured contention is not sufficient to demonstrate a QueueJIT advantage.

## Limitations and anomalies

- This phase uses oracle deadlines and exactly two sessions; it adds no prediction error, additional session count, model, or GPU.
- Promotion uses a normal completion-request proxy, not pure H2D. Median A/B request service was 116.50/37.74 ms under ASAP despite equal KV sizes; B was much faster than the frozen 120-ms estimate, so this configured contention did not make NaiveJIT miss. Logs report a Triton `_compute_slot_mapping_kernel` JIT during inference; the A/B asymmetry is consistent with first-request warm-up cost, though causality is not established.
- The isolated runtime emits a non-fatal Python startup warning because `nvidia_cutlass_dsl` is absent from a `.pth` hook; CUDA/vLLM starts and all measured requests succeeded. Server shutdown also emitted a multiprocessing semaphore cleanup warning.
- The single promotion worker is an experimental serialized-resource model. This validates the scheduling idea with real vLLM service times; it does not prove native PCIe-level contention behavior.
- No-pressure control prompts and all measured A/B prompts use the same deterministic token construction. The pressure workload was not increased or changed.

Phase 2E uses oracle deadlines and a request-based promotion proxy.

Phase 2E execution status: COMPLETE
