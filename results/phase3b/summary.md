# Phase 3B — Native KV promotion

## Objective

Replace the completion-request promotion proxy with a research-only native CPU-external-to-GPU KV load that executes zero transformer forward and generates zero tokens, then repeat the frozen six-session workload.

## Setup

SlackKV base `22252f3df7366d35bea37911c8471ea88ee68295`; pinned vLLM `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` (0.23.0), PyTorch 2.11.0+cu126, CUDA 12.6, RTX A6000 GPU0, Qwen2.5-1.5B-Instruct, float16, TP=1, 1 GiB GPU KV, 2 GiB CPU offload, prefix caching, max model length 10000. A–F are distinct 4096-token prefixes with oracle deadlines 300/320/340/650/900/1300 ms. Each policy has three independent server restarts.

The local research endpoint routes the normal prompt identity through OffloadingConnector lookup/allocation/load metadata. Its load-only engine step enters `kv_connector_no_forward`; after transfer completion the scheduler caches the loaded blocks, finishes the special request without sampling, and frees its request ownership while leaving prefix-cache blocks reusable.

## Native sanity

- External tokens loaded: 4096
- Transfer bytes: 117,440,512
    - Native endpoint server E2E: 13.89 ms; client E2E: 18.62 ms
- CPU→GPU transfer counter: 5.26 ms
- Model forwards / generated tokens: 0 / 0
- Subsequent resume local/external hit tokens: 4096 / 0
- First-token correctness: PASS
- Sanity gate: PASS
- Connector-only steps: 7; mixed prefetch/model steps: 0

## Calibration

Three warm native client-observed E2E times: 17.10, 17.22, 17.26 ms. Frozen `p_native` is 18 ms (`ceil(max)`), set before the workload. Connector-first steps must contain zero normal model tokens; mixed-step count is recorded in `run_metrics.csv`.

## Six-session results — medians across three repetitions

| Policy | Native deadline misses | P50 effective latency (ms) | P95 effective latency (ms) | Max effective latency (ms) | Idle GPU memory-time (MiB·s) |
|---|---:|---:|---:|---:|---:|
| OnDemand | 0.00 | 74.94 | 111.61 | 112.76 | 0.00 |
| ASAP-ConnectorFirst | 0.00 | 63.55 | 128.58 | 133.10 | 383.61 |
| NaiveJIT-ConnectorFirst | 6.00 | 73.21 | 146.99 | 150.09 | 0.00 |
| QueueJIT-ConnectorFirst | 6.00 | 72.18 | 150.64 | 153.92 | 0.00 |

## Direct comparison

- Phase 2H QueueJIT-Proxy reference: P95 65.01 ms, memory-time 9.23 MiB·s.
- Phase 2H OnDemand reference: P95 57.51 ms, memory-time 0.
- QueueJIT-ConnectorFirst versus current OnDemand: P95 150.64 vs 111.61 ms; memory-time 0.00 vs 0 MiB·s.
- QueueJIT-ConnectorFirst versus old QueueJIT-Proxy: P95 change +85.63 ms; memory-time change -9.23 MiB·s.
- QueueJIT-ConnectorFirst saves 100.0% idle memory-time versus ASAP-ConnectorFirst.
- All workload cache/generation/timeline/offload validations: PASS.

## Anomalies and limitations

The endpoint is research-only and local. It preserves normal OffloadingConnector allocation, bookkeeping, transfer, completion, and prefix-cache lifecycle, but is not a production API. Deadlines remain oracle values. The client-side promotion worker remains serialized. No prediction error, TraceLab replay, second GPU, model/context sweep, or Phase 3B work is included.

The connector-first fix removed the Phase 3A completion-coupling anomaly: in the frozen overlap diagnostic, native completion was approximately 24–27 ms client E2E with CPU→GPU approximately 5 ms, while A's added TTFT was approximately 10–14 ms. In the six-session workload, QueueJIT remained slower than OnDemand at this KV size despite saving essentially all early residency relative to ASAP; this is why the result is B rather than A.

## Conclusion

**Result B:** Native promotion removes proxy overhead, but OnDemand remains competitive at this KV size and hardware regime.

Phase 3B execution status: COMPLETE
