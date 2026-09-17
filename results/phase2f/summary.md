# Phase 2F — Promotion service-time crossover calibration

## Setup

Base commit: 1fc92227a8adfea658354e850a9d7b76ade78b21. Reused vLLM 0.23.0, PyTorch 2.11.0+cu126, CUDA 12.6, RTX A6000 GPU0, Qwen2.5-1.5B-Instruct and unchanged Phase 2E server settings/prompts.
Exactly six independent restarts used frozen AB, BA, AB, BA, AB, BA order. Every run offloaded A_first and B_first normally, waited 1 second, then sent both distinct 9500-token keep-local pressure requests. Promotions were serialized; no scrape or other request occurred between the measured pair.

## Findings

- Every A/B first turn offloaded 117,440,512 bytes: **PASS**.
- Every pair restored 8192 external tokens and 0 local tokens: **PASS** (6/6 valid runs).
- A first median: **112.38 ms**; A second median: **38.26 ms**.
- B first median: **111.68 ms**; B second median: **39.02 ms**.
- A-minus-B service-time difference at first / second position: **0.71 / -0.76 ms**; the identity effect is negligible and changes sign.
- All first median/min/max: **112.03 / 110.18 / 115.56 ms**.
- All second median/min/max: **38.57 / 37.45 / 39.57 ms**.
- Position ratio (first/second): **2.905**.
- Slow request follows: **position 1**.
- Triton/JIT/kernel log evidence: **observed in all six primary runs**; _compute_slot_mapping_kernel was logged **2.16–3.00 s before** the first measured promotion, during the preceding setup/pressure interval. This is consistent with warm-up but does not prove causality; see warmup_log_evidence.txt.
- CPU-to-GPU transfer-only timing: **available** from the observed vLLM counter `vllm:kv_offload_total_time_total` with `transfer_type=CPU_to_GPU`; median pair total **10.893 ms** (range 10.581–13.561 ms across six pairs). The one per-request diagnostic measured A 5.264 ms and B 5.260 ms.
- Warm-state estimate using only second-position samples: **P50 38.57 ms, P90 39.57 ms, max 39.57 ms (n=6)**.
- Is p_est=120 ms reasonable for steady-state scheduling? **NO**. 120 ms is conservative here because it includes cold-path overhead.

## Interpretation

A: Promotion asymmetry is primarily position/warm-up dependent.

Final result: A

Promotion duration is a normal completion-proxy request (request handling, external KV load, model execution, response handling), not pure H2D. Per-request counter attribution from the one diagnostic run is in diagnostic_metrics.json; diagnostic request times are excluded from primary groups. The transfer-time delta excludes Prometheus `_created` timestamp series.

## Anomalies and limitations

- The excluded diagnostic proxy / transfer-counter / residual was A **109.85 / 5.26 / 104.59 ms** and B **35.06 / 5.26 / 29.80 ms**; the measured transfer counter does not explain the position-dependent request-time gap.
- The isolated Python environment emits a non-fatal missing optional nvidia_cutlass_dsl .pth warning; requests and all six cache-evidence checks completed successfully.

- Invalid runs remain visible in promotion_metrics.csv and are excluded from aggregates; no retries or prompt/cache/order tuning were done.
- Compilation-log proximity is observational only.
- Runtime startup/shutdown warnings, if any, are retained in compact server_logs/.

Phase 2F execution status: COMPLETE
