# Phase 3A — Native KV promotion

## Objective

Replace the completion-request promotion proxy with a research-only native CPU-external-to-GPU KV load that executes zero transformer forward and generates zero tokens, then repeat the frozen six-session workload.

## Setup

SlackKV base `e4199eccaf40770ce5df9093b0fdf358831be3f8`; pinned vLLM `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` (0.23.0), PyTorch 2.11.0+cu126, CUDA 12.6, RTX A6000 GPU0, Qwen2.5-1.5B-Instruct, float16, TP=1, 1 GiB GPU KV, 2 GiB CPU offload, prefix caching, max model length 10000. A–F are distinct 4096-token prefixes with oracle deadlines 300/320/340/650/900/1300 ms. Each policy has three independent server restarts.

The local research endpoint routes the normal prompt identity through OffloadingConnector lookup/allocation/load metadata. Its load-only engine step enters `kv_connector_no_forward`; after transfer completion the scheduler caches the loaded blocks, finishes the special request without sampling, and frees its request ownership while leaving prefix-cache blocks reusable.

## Native sanity

- External tokens loaded: 4096
- Transfer bytes: 117,440,512
- Native endpoint E2E: 15.22 ms
- CPU→GPU transfer counter: 7.00 ms
- Model forwards / generated tokens: 0 / 0
- Subsequent resume local/external hit tokens: 4096 / 0
- First-token correctness: PASS
- Sanity gate: PASS

## Calibration

Three warm native endpoint E2E times: 12.30, 13.50, 12.70 ms. Frozen `p_native` is 14 ms (`ceil(max)`), set before the workload.

## Six-session results — medians across three repetitions

| Policy | Native deadline misses | P50 effective latency (ms) | P95 effective latency (ms) | Max effective latency (ms) | Idle GPU memory-time (MiB·s) |
|---|---:|---:|---:|---:|---:|
| OnDemand | 0.00 | 74.53 | 108.86 | 109.89 | 0.00 |
| ASAP-Native | 0.00 | 61.40 | 126.68 | 131.53 | 384.18 |
| NaiveJIT-Native | 6.00 | 71.66 | 132.25 | 135.61 | 0.00 |
| QueueJIT-Native | 6.00 | 71.26 | 129.38 | 131.34 | 0.00 |

## Direct comparison

- Phase 2H QueueJIT-Proxy reference: P95 65.01 ms, memory-time 9.23 MiB·s.
- Phase 2H OnDemand reference: P95 57.51 ms, memory-time 0.
- QueueJIT-Native versus current OnDemand: P95 129.38 vs 108.86 ms; memory-time 0.00 vs 0 MiB·s.
- QueueJIT-Native versus old QueueJIT-Proxy: P95 change +64.37 ms; memory-time change -9.23 MiB·s.
- QueueJIT-Native saves 100.0% idle memory-time versus ASAP-Native.
- All workload cache/generation/timeline/offload validations: PASS.

## Anomalies and limitations

The first workload attempt exposed an overly broad research guard: it rejected a valid step that combined a special connector load with unrelated normal-request model tokens. The guard was corrected to prohibit scheduled model tokens only for the special request ID, focused tests were rerun, and the entire 12-run matrix was restarted from scratch. No failed attempt is included in the CSV results.

Warm native calibration was 12–14 ms, but JIT B promotion reached about 89–93 ms when it overlapped A's real resume, even though the CPU→GPU transfer counter remained about 5 ms. Consequently both JIT policies missed all six frozen deadlines per run and QueueJIT-Native did not improve the old QueueJIT-Proxy P95. This is transparent evidence of scheduler/normal-forward interference around the research endpoint, not a KV copy-time increase or cache correctness failure.

The endpoint is research-only and local. It preserves normal OffloadingConnector allocation, bookkeeping, transfer, completion, and prefix-cache lifecycle, but is not a production API. Deadlines remain oracle values. The client-side promotion worker remains serialized. No prediction error, TraceLab replay, second GPU, model/context sweep, or Phase 3B work is included.

## Conclusion

**Result B:** Native promotion removes the prefetch request's model forward, but OnDemand remains competitive at this KV size and hardware regime. Under co-scheduled real resumes, the native endpoint's completion delay erased the isolated-service advantage and did not improve the old proxy workload P95.

Phase 3A execution status: COMPLETE
