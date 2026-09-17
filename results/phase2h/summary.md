# Phase 2H — Six-session end-to-end real workload

## Objective

Evaluate whether QueueJIT preserves its latency–idle-KV-residency advantage when six real vLLM resume requests execute on their configured oracle timeline alongside one serialized promotion worker.

## Setup and methods

Base commit: 81abb72a5916d0cf21b8f1ff1fba85e54414061d. Runtime: vLLM 0.23.0, PyTorch 2.11.0+cu126, CUDA 12.6, NVIDIA RTX A6000 GPU0, Qwen2.5-1.5B-Instruct, float16, TP=1, 1 GiB GPU KV, 2 GiB pinned CPU offload, prefix caching, max model length 10000.

A–F each use a distinct 4096-token prefix. Oracle deadlines are 300/320/340/650/900/1300 ms from t0. External policies use W plus A–F normal offload, a one-second store wait, four frozen distinct 9500-token pressure requests, and one excluded W promotion before t0. p_est is frozen at 42 ms. KeepAll, OnDemand, ASAP, NaiveJIT and QueueJIT each ran three independent server restarts.

One serialized promotion worker and six separate resume-dispatch tasks share the real timeline. A resume waits for its own in-flight promotion when necessary and never starts a duplicate restore. No metrics scrape occurs from t0 through completion of the final measured resume. Percentiles use linear interpolation at rank (n-1)q over the six sessions.

## Validation

- All W/A–F external first turns offloaded exactly 117,440,512 bytes: PASS.
- W absorbed the cold promotion path with 4096 external and 0 local tokens in every external run: PASS; W median 111.34 ms.
- Measured A–F scheduled promotion service range: 31.79–80.60 ms; median 39.05 ms.
- Real resume timeline: PASS; no resume was sent before its configured deadline.
- Aggregate cache evidence: PASS. KeepAll=24576 local/0 external; OnDemand=0 local/24576 external; scheduled policies=24576 local/24576 external in every valid run.
- First-token IDs against A–F controls: PASS.

## Results — medians across three repetitions

| Policy | Promotion misses | P50 effective latency (ms) | P95 effective latency (ms) | Max effective latency (ms) | Total early residency (ms) | Idle GPU memory-time (MiB·s) |
|---|---:|---:|---:|---:|---:|---:|
| KeepAll | 0.00 | 63.09 | 129.19 | 133.96 | 3810.00 | 426.72 |
| OnDemand | 0.00 | 46.47 | 57.51 | 58.47 | 0.00 | 0.00 |
| ASAP | 0.00 | 31.06 | 53.33 | 54.26 | 3013.09 | 337.47 |
| NaiveJIT | 2.00 | 39.90 | 93.51 | 100.22 | 9.94 | 1.11 |
| QueueJIT | 1.00 | 45.47 | 65.01 | 66.41 | 82.42 | 9.23 |

## Direct answers

- Clustered NaiveJIT A/B/C resumes waited for promotion 6/9 times across three repetitions. Median promotion misses were 2.00; QueueJIT reduced this to 1.00.
- QueueJIT versus NaiveJIT: P95 effective resume latency 65.01 vs 93.51 ms; memory-time 9.23 vs 1.11 MiB·s.
- QueueJIT saved 97.3% idle GPU memory-time versus ASAP (9.23 vs 337.47 MiB·s).
- QueueJIT saved 97.8% idle GPU memory-time versus KeepAll (9.23 vs 426.72 MiB·s).
- QueueJIT versus OnDemand P95 effective resume latency: 65.01 vs 57.51 ms.
- OnDemand therefore dominated QueueJIT on the two primary reported axes in this workload: lower P95 effective latency and zero versus 9.23 MiB·s idle memory-time.

**Result B:** The mechanism works, but QueueJIT is not clearly better than a simpler policy at this six-session scale.

## Anomalies and limitations

- The promotion operation remains a normal completion-request proxy, not pure H2D or a native promotion API. The single client worker is an experimental serialized shared resource; this does not prove native PCIe contention is solved or that SlackKV is production-ready.
- Deadlines are oracle values. This phase adds no prediction error, selective admission, TraceLab replay, second GPU, model/context sweep, or vLLM scheduler patch.
- Idle GPU memory-time covers only measured idle-period session KV residency, not total server GPU utilization.
- Contrary to the expected ordering, KeepAll P95 effective latency was 129.19 ms, higher than OnDemand's 57.51 ms. Cache evidence was valid; clustered concurrent local resumes incurred serving queue delay.
- Real resume concurrency inflated some scheduled promotion proxies above the isolated 42-ms estimate (observed maximum 80.60 ms, notably clustered C). QueueJIT still reduced NaiveJIT's median misses and P95 latency but retained one median promotion miss.
- The isolated Python environment continued to emit the known non-fatal optional nvidia_cutlass_dsl .pth warning. All requests returned HTTP 200 and no vLLM request/model error was found in the compact server logs.

Phase 2H execution status: COMPLETE
