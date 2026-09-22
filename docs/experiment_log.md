# Experiment log

Record each experiment with its purpose, configuration, metrics, result paths, and commit hash.

## 2026-09-14 — Project initialization

- Created the baseline repository structure and GitHub remote.
- No experiments or result artifacts were generated in this setup step.

## 2026-09-14 — Phase 0: hardware and data-movement feasibility

- **Objective:** Measure the actual RTX A6000 topology, direct CUDA peer-memory transfer versus pinned CPU memory, and GPU0 compute/transfer overlap.
- **Hardware:** 2 × NVIDIA RTX A6000 (48 GiB GDDR6 ECC); `nvidia-smi topo -m` reports GPU0↔GPU1 as `SYS`, with no visible NVLink. CUDA peer access is available in both directions.
- **Main command:** `bash scripts/run_phase0.sh`
- **Configuration:** Native CUDA `cudaMemcpyPeerAsync`/`cudaMemcpyAsync`, 16 MiB–8 GiB transfers, 10 warm-ups and 30 measurements; GPU0 FP16 4096×4096 GEMM and a labeled synthetic memory-bandwidth workload, 256 MiB/1 GiB/4 GiB overlap traffic, 30 trials.
- **Main metrics:** At 1 GiB, GPU0→GPU1/GPU1→GPU0/pinned CPU→GPU0/GPU0→pinned CPU P50 bandwidth was 27.08/26.37/26.28/25.88 GB/s. At 4 GiB it was 27.09/26.37/26.12/26.41 GB/s. GEMM compute slowdown at 1 GiB was 2.78% for peer_in and 3.66% for cpu_in.
- **Results:** `results/phase0/` (raw CSV, summaries, environment/topology capture, configuration, commands, and plots).
- **Anomalies:** `SYS` topology despite working CUDA P2P; peer bandwidth was close to pinned-CPU bandwidth for tested GPU0 paths. No failed allocations or failed correctness checks.
- **Execution status:** COMPLETE.

## 2026-09-14 — Phase 1A: queue-aware JIT promotion simulator

- **Objective:** Test oracle-deadline QueueJIT against OnDemand, ASAP-EDF, and NaiveJIT on one serialized inbound resource.
- **Commands:** `python -m pytest -q tests/test_phase1a_simulator.py`; `python experiments/phase1a/run_phase1a.py`; `python scripts/plot_phase1a.py`.
- **Calibration:** Phase 0 commit `0337837b43ddf78821839d86be6a89ae2a7c164a`; CPU→GPU0 P50 service times of 10.330/40.858/164.427 ms at 256 MiB/1 GiB/4 GiB.
- **Workloads:** Controlled 4/8/16/32-job bursts with five seeds; 5,000-job Poisson streams at rho=0.25/0.50/0.75/0.90 with three seeds; rho=0.75 service sensitivity and peer-source subset.
- **Key results:** At N=32, QueueJIT reduced miss rate 76.47% and P95 stall 61.47% versus NaiveJIT, and reduced VRAM-time 98.39% versus ASAP-EDF while lowering P95 stall. At stochastic CPU rho=0.75, it reduced NaiveJIT miss rate 46.46% and P95 stall 26.46%, but had substantially higher P95 stall than ASAP-EDF despite 98.95% lower VRAM-time.
- **Success criteria:** Criterion A PASS; Criterion B PASS in the N=32 controlled burst, but not in the stochastic stream.
- **Anomalies:** The controlled-burst and stochastic tradeoffs differ materially; peer-source and CPU-source results are nearly identical on this topology. The raw audit CSV is 51 MB and compactly preserves all 341,200 job outcomes.
- **Results:** `results/phase1a/`.
- **Execution status:** COMPLETE.

## 2026-09-14 — Phase 1B: TraceLab real-workload simulator validation

- **Objective:** Validate the unchanged oracle-deadline QueueJIT mechanism using public sanitized TraceLab coding-agent characteristics, without real-LLM or vLLM integration.
- **Dataset and calibration:** TraceLab v0.0.2 `syfi_coding_trace.duckdb`, SHA256 `a7bab286bc640844560850965ccf47975cf66407154132abaab90f27ec9be744`; Phase 0 CPU→GPU0 calibration commit `0337837b43ddf78821839d86be6a89ae2a7c164a`.
- **Commands:** `/home/a6000/anaconda3/bin/python -m pytest -q tests/test_phase1a_simulator.py`; `/home/a6000/anaconda3/bin/python experiments/phase1b/run_phase1b.py`; `/home/a6000/anaconda3/bin/python scripts/plot_phase1b.py`.
- **Workloads:** Strict per-round extraction (latest emission to latest result, no continuation latency summing); 563,199 valid opportunities.  Session-preserving M=4/8/16/32/64 replay uses ten seeds and U[0,60] s offsets.  Tuple-preserving Poisson replay has 5,000 jobs, rho=0.25/0.50/0.75/0.90, and ten seeds.  KV footprint sensitivity is 64/128/256 KiB per token.
- **Key results:** At representative M=32, QueueJIT reduces miss rate 71.38%→68.89% and P95 stall 1,114.1→1,100.4 ms versus NaiveJIT, while reducing VRAM-time 579,891.2→181.7 GB*s versus ASAP-EDF with P95 stall +80.5 ms.  Best primary session run is M=32 seed 7: 65.86%→62.81% miss rate.
- **Success criteria:** R1 FAIL and R2 FAIL.  The dominant QueueJIT miss class is service time exceeding the trace pause (`late_arrival`), followed by large-KV domination.
- **Anomalies:** Trace token counts map to very large hypothetical 128-KiB/token KVs (P50 17.10 decimal GB), making many short pauses infeasible on the single-resource model.  The public raw database and derived per-event output remain under gitignored `data/tracelab/`.
- **Results:** `results/phase1b/`; TraceLab reference: Zhu et al., 2026, arXiv:2606.30560.
- **Execution status:** COMPLETE.

## 2026-09-14 — Phase 1B-Fix: closed-loop TraceLab replay audit

- **Objective:** Correct the material open-loop replay defect in Phase 1B, apply target-model context filtering, and audit token/continuation semantics without changing QueueJIT.
- **Commands:** `/home/a6000/anaconda3/bin/python -m pytest -q tests/test_phase1b_fix.py tests/test_phase1a_simulator.py`; `/home/a6000/anaconda3/bin/python experiments/phase1b_fix/run_phase1b_fix.py`; `/home/a6000/anaconda3/bin/python scripts/plot_phase1b_fix.py`.
- **Workloads:** Context-capped subset `reusable_tokens <= 131072`, with 56 KiB/token Qwen-like primary and 128 KiB/token stress sensitivity on exactly the same events.  Session replay causally propagates each session's resume stall.  M=4/8/16/32/64 has ten seeds; tuple-Poisson sensitivity has 5,000 jobs and rho=0.25/0.50/0.75/0.90.
- **Key results:** 283,193/563,199 (50.283%) valid opportunities are retained.  At primary 56 KiB/token M=64, QueueJIT improves median miss rate 60.44%→56.94% and P95 281.5→258.1 ms versus NaiveJIT.  In the 128-KiB M=64 stress case, closed-loop QueueJIT P95 is 2,817.9 ms versus 44,576.1 ms under the identical open-loop audit.
- **Conclusion:** Open-loop replay materially overstated the high-stress tail; after correction, QueueJIT-alone gains remain below 20% in primary session replay.  This does not evaluate selective offload or real-model execution.
- **Results:** `results/phase1b_fix/`.
- **Execution status:** COMPLETE.

## 2026-09-14 — Phase 1C: transfer-normalized selective KV offload

- **Objective:** Evaluate KeepAll, OffloadAll, all seven FixedTTL values, FeasibilityOracle, and all six SlackRatio gamma values using Phase 1B-Fix closed-loop replay.
- **Key result:** At primary 56 KiB/token M=32, gamma 2 reduces OffloadAll miss/P95 from 67.33%/4855.0 ms to 5.53%/115.0 ms at normalized paused-KV memory-time 0.258. C0, C1, and programmed C2 pass.
- **Results:** `results/phase1c/`; D2H is deliberately immediate and non-contended.
- **Execution status:** COMPLETE.

## 2026-09-14 — Phase 1C-Fix

- **Objective:** Correct directional transfer calibration and cross-seed evaluation criteria without changing Phase 1C workloads or policies.
- **Commands:** `/home/a6000/anaconda3/bin/python experiments/phase1c_fix/analyze_phase1c_fix.py`; `/home/a6000/anaconda3/bin/python scripts/plot_phase1c_fix.py`; `/home/a6000/anaconda3/bin/python -m pytest -q tests/test_phase1a_simulator.py tests/test_phase1b_fix.py tests/test_phase1c_fix.py`.
- **Calibration:** D2H is `gpu0_to_pinned_cpu`; H2D is `pinned_cpu_to_gpu0`, both from Phase 0 commit `0337837b43ddf78821839d86be6a89ae2a7c164a`. At 1 GiB their P50 times are 41.488 and 40.858 ms.
- **Method:** Reused completed ten-seed primary 56-KiB/token closed-loop rows; post-processing did not re-sample sessions or change policies. C1 uses ten-seed medians plus >=0.80 seed-pass fraction. C2 uses the same-M FixedTTL ten-seed-median Pareto frontier and strict three-metric dominance.
- **Key results:** At M=32, OffloadAll is 68.52% miss / 5543.1-ms P95 / 0.2707 normalized local-memory time. SlackRatio-2 is 5.70% / 138.8 ms / 0.2462 and strictly dominates FixedTTL-0.5 (6.75% / 636.1 ms / 0.2515); FixedTTL-2 remains a 2.18% / 0-ms / 0.2526 lower-stall frontier point.
- **Criteria:** C0 PASS; C1 PASS (SlackRatio-4 at M=32 has 8/10 seed passes; SlackRatio-3/-4 at M=64 have 8/10 and 9/10); C2 PASS through M=32 SlackRatio-2, with 78.2% P95 improvement over its dominated TTL point.
- **Outcome:** A = Strong GO under this simulator. This is a frontier extension, not universal SlackRatio superiority over every TTL point.
- **Anomalies:** The prior per-seed pairwise C2 export was not a valid cross-seed Pareto test and is replaced by compact aggregate artifacts. D2H remains immediate and non-contended.
- **Results:** `results/phase1c_fix/`.
- **Execution status:** COMPLETE.

## 2026-09-15 — Phase 1E: JIT safety-margin robustness

- **Objective:** Test whether a fixed margin applied to the Phase 1D predicted resume deadline makes QueueJIT robust to prediction error, without changing admission or introducing a new policy.
- **Base/calibration:** Phase 1D base `69814f4411b433a447dac4419b9d8732e930f9ea`; directional D2H/H2D Phase 0 calibration commit `0337837b43ddf78821839d86be6a89ae2a7c164a`.
- **Frozen workload:** 56 KiB/token, context <=131072, closed-loop TraceLab session replay, M=32, seeds 0–4. Policies are KeepAll, OffloadAll-QueueJIT, and FixedTTL-1 only; admission remains `predicted_pause >= 1.0 s`.
- **Method:** For sigma=0.10/0.25/0.50 and margins 0/100/250/500/1000 ms, QueueJIT uses `max(predicted_deadline - margin, release + D2H + H2D)`; actual deadline remains the sole stall and memory-accounting reference. Prediction draws are deterministically shared across margins for each seed/event/sigma.
- **Validation:** `/home/a6000/anaconda3/bin/python -m pytest -q tests/test_phase1a_simulator.py tests/test_phase1b_fix.py tests/test_phase1c_fix.py tests/test_phase1c_fix2.py tests/test_phase1d.py tests/test_phase1e.py` (28 passed); `/home/a6000/anaconda3/bin/python experiments/phase1e/run_phase1e.py`; `/home/a6000/anaconda3/bin/python scripts/plot_phase1e.py`.
- **Key results:** At sigma=0.10, 250 ms reduces median miss rate 17.03%→8.26% at normalized local-memory time 0.0562. At sigma=0.25, 1000 ms reduces miss rate 16.84%→5.89% at 0.1287. At sigma=0.50, 1000 ms reduces miss rate 16.99%→8.71%, but P95 remains 2294.9 ms and memory-time rises to 0.2570.
- **Result:** B — fixed margins help, but medium/high prediction error requires a large margin and makes the GPU-memory tradeoff more costly. No extra margin or policy was tuned.
- **Results:** `results/phase1e/`.
- **Execution status:** COMPLETE.

## 2026-09-15 — Phase 1F: adaptive safety margins

- **Objective:** Compare a fixed 500-ms JIT safety margin against two uncertainty-scaled margins without changing FixedTTL-1 admission.
- **Base/configuration:** Phase 1E base df931dfe238d221ff07800049385fb1c475dd883; 56 KiB/token, context <=131072, closed-loop TraceLab replay, M=32, seeds 0–4, sigma=0.10/0.25/0.50. Strategies are NoMargin, Fixed500ms, Adaptive-0.5, and Adaptive-1.0.
- **Method:** Adaptive margin is k * sigma * predicted_pause; safe deadline retains the Phase 1E physical D2H+H2D lower bound. The exact deterministic Phase 1D/1E event prediction is shared across all strategies. Actual deadlines alone determine stall and memory accounting.
- **Validation:** pytest tests/test_phase1f.py (4 passed); full run produced 60 compact seed rows and 12 five-seed medians. Row count, finite/nonnegative metrics, shared-noise admission fractions, and adaptive-margin monotonicity were asserted. Plot command: python scripts/plot_phase1f.py.
- **Key results:** At sigma=0.10, Fixed500ms is 6.03% miss / 193.9-ms P95 / 0.0614 normalized memory-time; Adaptive-0.5 is worse in P95 and memory. At sigma=0.50, Adaptive-0.5 improves 11.56% / 2794.9 ms / 0.2477 to 8.26% / 370.9 ms / 0.3328, requiring 34.4% more memory-time. Adaptive-1.0 costs 93.6–132.6% more memory-time than Fixed500ms across sigmas.
- **Result:** C — neither specified adaptive rule consistently dominates Fixed500ms; high-noise latency gains require substantially more local memory, and the low-noise Adaptive-0.5 tradeoff is worse. No additional k values were tested.
- **Results:** results/phase1f/.
- **Execution status:** COMPLETE.

## 2026-09-15 — Phase 2A: real GPU tensor offload/prefetch runtime

- **Objective:** Validate the real CUDA tensor D2H -> pinned CPU -> allocator reuse -> JIT H2D path without vLLM or a language model.
- **Base/configuration:** Base 7f3c4966e7e1d6657cedd1a1a6bf73a3afe1a5e1; GPU0 RTX A6000 (48 GiB), PyTorch 2.2.0/CUDA 12.1, float16 contiguous tensors. Sizes are 256 MiB/1 GiB/4 GiB with three warm-ups and ten CUDA-event-timed runs. JIT is 1 GiB, pauses 1/2/5 s, margins 0/250/500 ms, five repetitions per cell. CPU buffers are pinned; empty_cache is not used in the primary run.
- **Validation/commands:** python -m pytest -q tests/test_phase2a.py (3 passed); python experiments/phase2a/run_phase2a.py; python scripts/plot_phase2a.py.
- **Key results:** Runtime D2H/H2D are 43.74/44.40 ms for 1 GiB and 169.84/170.52 ms for 4 GiB, versus Phase 0 P50 41.49/40.86 and 162.65/164.43 ms. Relative errors are 5.4%/8.7% and 4.4%/3.7%; 256 MiB is 26.3%/20.7% slower. Allocated memory decreased after every offload deletion and replacement allocation succeeded; all sampled round trips passed.
- **JIT result:** At 1 GiB and a 2-s pause, 0-ms margin gives 2.59-ms median stall; 250/500-ms margins yield 0-ms median stall with 247.79/497.65-ms median early residency.
- **Conclusion:** A — real tensor transfer, memory reuse, correctness, and the expected margin tradeoff are validated. This remains one-tensor, one-GPU runtime validation only.
- **Anomaly:** An initial O(numel) random-permutation sampling implementation was terminated before result writing and replaced with fixed-seed direct index sampling; reported metrics come from the complete rerun.
- **Results:** results/phase2a/.
- **Execution status:** COMPLETE.

## 2026-09-15 — Phase 2B: real Transformer KV offload/restore

- **Objective:** Validate a real Qwen past_key_values GPU -> pinned CPU -> GPU restore path, allocator reuse, and deterministic continuation correctness without vLLM.
- **Base/configuration:** Base 11c26582e86d20c7edfa67b775523033b1f69734; Qwen/Qwen2.5-1.5B-Instruct, Transformers 4.45.2, PyTorch 2.2.0/CUDA 12.1, GPU0, float16, 4096-token deterministic text prefix, 16 greedy continuation tokens, one warm-up and three measured full runs.
- **Compatibility:** This Transformers/model pair returns a public legacy cache tuple rather than DynamicCache. The final implementation snapshots that tuple and restores through DynamicCache.from_legacy_cache(); no private model cache fields are used.
- **Validation/commands:** python -m pytest -q tests/test_phase2b.py (3 passed); python experiments/phase2b/run_phase2b.py.
- **Key results:** The 28-layer cache is 112 MiB (K/V shape [1,2,4096,128] per layer). Median D2H/H2D are 7.61/6.29 ms, 112 MiB of allocated GPU memory is released, an 89.6-MiB replacement allocation succeeds, and all 16 continuation tokens match with zero first-step logit difference. Baseline/restored first decode is 22.84/22.79 ms; cold resume latency is 29.08 ms.
- **Comparison:** D2H is 39.1% slower than the Phase 2A contiguous-tensor scaled estimate (75.6% versus the Phase 0 piecewise estimate); H2D is 15.4% slower than the Phase 2A scaled estimate. The 56 separate cache tensors add real dispatch overhead.
- **Conclusion:** B — the real-model path and correctness are validated, but fragmented-cache transfer calibration requires adjustment.
- **Results:** results/phase2b/.
- **Execution status:** COMPLETE.

## 2026-09-15 — Phase 2C: vLLM selective-offload capability gate

- **Objective:** Validate selective KV offload in native vLLM serving with a
  normal-offload Session A, a `max_offload_tokens=0` keep-local Session B, and
  observable GPU-pressure/cache evidence.
- **Capability check:** An isolated Python 3.11/CUDA 12.4 runtime imports vLLM
  0.11.2 and provides `OffloadingConnector`, but has no
  `max_offload_tokens` source implementation.  vLLM 0.29.0 contains that
  per-request cap but fails to import on this host because it requires
  `libcudart.so.13`.
- **Validation:** `/home/a6000/anaconda3/bin/python -m pytest -q
  tests/test_phase2c.py` (4 passed).  The capability gate deliberately aborts
  before starting a server, so no unvalidated A/B comparison, TTFT, cache hit,
  or eviction metric is reported.
- **Result:** INCOMPLETE; the current deployment cannot enforce the required
  Session B keep-local control.  This is an integration-environment blocker,
  not a mechanism result.
- **Results:** `results/phase2c/`.

## 2026-09-17 — Phase 2C-Fix: real vLLM selective-offload A/B

- **Objective:** Establish an isolated CUDA-compatible vLLM runtime that supports both `OffloadingConnector` and per-request `max_offload_tokens`, then rerun the original two-session selective-offload experiment.
- **Runtime:** vLLM 0.23.0 source build (`0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`) in a dedicated CUDA 12.6/PyTorch 2.11.0+cu126 environment. FlashInfer 0.6.12 was installed without dependency resolution that would replace Torch. The capability gate passed import, CUDA, connector, and request-level `kv_transfer_params` checks.
- **Design:** GPU0 RTX A6000, Qwen/Qwen2.5-1.5B-Instruct, 320 MiB GPU KV capacity, 2 GiB CPU offload, distinct 4096-token A/B prefixes, two distinct 9500-token pressure requests, and three independent restarts in A,B; B,A; A,B order. A used normal offload; B sent `max_offload_tokens=0`.
- **Validation:** `pytest -q tests/test_phase2c.py tests/test_phase2c_fix.py` (7 passed). A first-turn offloaded 117,440,512 bytes in every repetition; B offloaded 0 bytes. After pressure, A recovered 4096 external KV tokens; B recovered 0 and recomputed a median 4106 prompt tokens. The no-pressure A control hit 4096 local tokens.
- **Key result:** Median resume TTFT was 94.82 ms for A and 148.51 ms for B, a 36.15% reduction (1.57x). All six streamed resumes were valid. This validates selective offload in real vLLM serving, not deadline-aware JIT promotion.
- **Anomaly:** One 9500-token pressure request did not consistently evict B's local prefix, so the frozen pressure workload was increased to two distinct 9500-token requests; no admission, model, or KV-capacity parameter was changed.
- **Results:** `results/phase2c_fix/`.
- **Execution status:** COMPLETE.

## 2026-09-17 — Phase 2D: deadline-aware JIT promotion timing

- **Objective/base:** Test whether a real vLLM prefetch request can move recovery of an already offloaded Session A prefix ahead of its 2-second resume deadline. Base commit: `a8688fd62fed53e58c95c65b8f2efd1d85c1932f`.
- **Frozen setup:** Reused the isolated vLLM 0.23.0 / PyTorch 2.11.0+cu126 / CUDA 12.6 runtime on GPU0 RTX A6000; Qwen2.5-1.5B-Instruct, 320 MiB GPU KV, 2 GiB CPU offload, 4096-token prefix, two distinct 9500-token pressure prompts, and OnDemand/JIT-100/JIT-250/JIT-500 with three independent server restarts each. `A_first` offloaded exactly 117,440,512 bytes in all 12 runs.
- **Method:** After pressure, the deadline is 2 seconds later. JIT uses an ordinary completion with the exact prefix and `max_offload_tokens=0` as a promotion proxy; its generated token is discarded. No metric scrape is inserted between that synchronous prefetch and the real resume. Post-resume source counters are attributed from the serialized request order: external=4096 and local=4096 means the proxy restored the external KV, then the real resume consumed the GPU-local copy. The proxy duration includes ordinary request/model execution and is not a zero-compute production API.
- **Validation/commands:** `bash results/phase2d/commands.sh` (which runs `pytest -q tests/test_phase2d.py`, the fixed 12-run experiment, and the counter reconciliation step); 5 tests passed and both runner/reconciliation modules compiled. The capability gate and server smoke test passed.
- **Key results:** Local-hit control TTFT was 93.24 ms. OnDemand median resume TTFT was 98.81 ms (local=0, external=4096). JIT-100 median prefetch duration/stall/early residency/resume TTFT/effective latency was 114.37/14.49/0/42.17/56.30 ms. JIT-250 was 114.51/0/135.27/42.59/42.59 ms; JIT-500 was 110.71/0/389.15/42.20/42.20 ms. Every JIT proxy recovered 4096 external tokens; each following real resume hit 4096 local tokens and 0 external tokens. All first-token IDs matched the local-hit control (1084). Median `A_first` E2E latency was 196.94 ms and median measured D2H transfer-counter time was 5.96 ms; this original offload cost is separate from effective resume latency and is not removed by JIT.
- **Conclusion:** Result A — the external→local promotion chain and expected stall-versus-early-residency behavior were observed. This validates the timing placement concept using the request proxy, not a native promotion-only control path.
- **Anomalies:** An initial aggregation bug and an instrumentation scrape that delayed dispatch were found and excluded. The final matrix was rerun without a scrape between prefetch and resume; dispatch lag medians were 0.05/0.12/0.14 ms for JIT-100/250/500. The final after-resume counter attribution is reproducible with `experiments/phase2d/reconcile_results.py`. Excluded attempts remain archived under `results/phase2d/server_logs/` and are not included in the reported matrix. The runtime repeatedly logged a missing optional `nvidia_cutlass_dsl` `.pth` dependency warning and vLLM shutdown reported one semaphore-cleanup warning per server; CUDA/runtime/server capability gates and all requests passed, with no vLLM request/model error.
- **Results:** `results/phase2d/`.
- **Execution status:** COMPLETE.

## 2026-09-17 — Phase 2E: two-session real promotion scheduling

- **Objective/base:** Test whether per-session NaiveJIT creates queue misses for two paused real vLLM sessions and whether cumulative QueueJIT slack avoids them. Base commit: `4055e134c123e1e2f2562e09d46de5387863e280`.
- **Frozen setup:** Reused vLLM 0.23.0, PyTorch 2.11.0+cu126, CUDA 12.6, GPU0 RTX A6000, Qwen2.5-1.5B-Instruct, 320 MiB GPU KV, 2 GiB CPU offload and max model length 10000. Distinct 4096-token A/B prefixes each offloaded exactly 117,440,512 bytes, followed by the two unchanged distinct 9500-token pressure requests. Deadlines were t0+1.500 s and t0+1.560 s; estimated service remained fixed at 120 ms. ASAP, NaiveJIT and QueueJIT each ran three independent server restarts with one serialized client-side promotion worker.
- **Validation/commands:** `python -m py_compile experiments/phase2e/*.py tests/test_phase2e.py`; `python -m pytest -q tests/test_phase2e.py` (6 passed); `bash -n results/phase2e/commands.sh`; full run via `python -m experiments.phase2e.run_phase2e` under the isolated-runtime environment (reproduction wrapper: `bash results/phase2e/commands.sh`). The runner passed the server smoke gate. Final checks found 9 run rows/18 job rows, all initial offloads valid, all 9 promotion intervals totaling 8192 external-hit tokens and 0 local-hit tokens each, then 8192 local-hit and 0 external-hit tokens per run across resumes. All 18 generated first-token IDs matched their session control. No negative stall/residency and no server process remained.
- **Key results (three-run medians):** ASAP = 0 misses, 0 ms max stall, 2747.31 ms total early residency; NaiveJIT = 0, 0 ms, 33.38 ms; QueueJIT = 0, 0 ms, 146.92 ms. QueueJIT began 59.96 ms before NaiveJIT and saved 2600.39 ms (94.7%) versus ASAP, but used 113.54 ms more early residency than NaiveJIT without reducing measured misses or stall.
- **Anomaly/interpretation:** Although `p_est` remained the prescribed 120 ms for both jobs, median request-proxy service was asymmetric: A=116.50 ms and B=37.74 ms under ASAP, despite equal KV sizes and external-source evidence for both. Logs record Triton `_compute_slot_mapping_kernel` JIT during inference; the gap is consistent with first-request warm-up cost, but causality is not established. The short B service meant the configured contention did not produce a NaiveJIT miss. This phase did not retune the deadlines, p_est, or workload. A non-fatal missing `nvidia_cutlass_dsl` `.pth` startup warning and one semaphore-cleanup warning on server shutdown recurred; all runtime gates and requests passed.
- **Result:** B — multi-session promotion works, but the configured contention is insufficient to demonstrate a QueueJIT advantage over NaiveJIT.
- **Limitations:** Deadlines are oracle values. Promotion is an ordinary completion-request proxy, whose duration includes request handling, external KV load, model execution and response handling, not pure H2D. The single client worker is an experimental serialized resource model and does not establish native PCIe contention behavior. Real resumes occurred only after both promotions completed.
- **Results:** `results/phase2e/`.
- **Execution status:** COMPLETE.
## 2026-09-17 — Phase 2F: promotion-position crossover calibration

- **Objective/base:** Determine whether the Phase 2E equal-size promotion asymmetry follows session identity or first/second request position. Base commit: 1fc92227a8adfea658354e850a9d7b76ade78b21.
- **Frozen setup:** Reused vLLM 0.23.0, PyTorch 2.11.0+cu126, CUDA 12.6, RTX A6000 GPU0, Qwen2.5-1.5B-Instruct and exact Phase 2E 4096-token A/B prefixes and two 9500-token pressure requests. Exactly six independent restarts used AB, BA, AB, BA, AB, BA. Each A/B first turn used normal offload, followed by a 1-second wait and pressure with max_offload_tokens=0. The serialized measured promotions used max_offload_tokens=0; there was no metrics scrape between the pair.
- **Commands/validation:** bash -n results/phase2f/commands.sh; python -m py_compile experiments/phase2f/*.py tests/test_phase2f.py; python -m pytest -q tests/test_phase2f.py (6 passed); full primary matrix via python -m experiments.phase2f.run_phase2f under the isolated Phase 2C-Fix runtime. Six restarts and twelve promotion rows completed. All 12 first-turn offloads were exactly 117,440,512 bytes. All six promotion pairs showed 8192 external-hit tokens and zero local-hit tokens.
- **Key results:** A-first/A-second medians were 112.38/38.26 ms; B-first/B-second were 111.68/39.02 ms. All first/all second medians were 112.03/38.57 ms (ratio 2.905). A-minus-B differences were +0.71 ms first and -0.76 ms second, so the slow request followed position rather than identity. Result A: promotion asymmetry is primarily position/warm-up dependent.
- **Transfer diagnostic:** The exposed CPU_to_GPU transfer-time counter was vllm:kv_offload_total_time_total. Six primary pair deltas had median 10.893 ms (range 10.581–13.561 ms). One allowed diagnostic run attributed 5.264 ms and 5.260 ms to the individual transfers; corresponding whole proxy requests were 109.85 ms and 35.06 ms. Thus transfer time did not explain the position-dependent request-duration difference. Prometheus *_created timestamp series were excluded from the counter calculation; the existing metric snapshots were reanalyzed without rerunning the six primary services.
- **Warm-up evidence:** _compute_slot_mapping_kernel Triton JIT warnings appeared in all six primary logs, 2.16–3.00 seconds before each first measured promotion, in the preceding setup/pressure interval. This is consistent with warm-up but is not causal proof. Warmed second-position service estimate: P50 38.57 ms, P90/max 39.57 ms. The Phase 2E p_est=120 ms is conservative rather than representative of steady-state proxy service.
- **Anomalies:** The isolated Python environment emitted a non-fatal missing optional nvidia_cutlass_dsl .pth warning; experiment requests and all cache-evidence checks passed. Server logs also report torch.compile/CUDA graphs disabled by the frozen enforce-eager runtime; this setting was unchanged.
- **Results:** results/phase2f/.
- **Execution status:** COMPLETE.

## 2026-09-17 — Phase 2G: warm-state QueueJIT contention

- **Objective/base:** Repeat the two-session real-serving QueueJIT contention test after explicitly warming the promotion path, using calibrated warm-state service time. Base commit: a74b0893cbc2a19e982701be0948fa1bbd757de4.
- **Frozen setup:** Reused vLLM 0.23.0 / PyTorch 2.11.0+cu126 / CUDA 12.6 on GPU0 RTX A6000, Qwen2.5-1.5B-Instruct, 320 MiB GPU KV, 2 GiB CPU offload, max model length 10000. Every restart created exact 4096-token A/B/C sessions, verified each first-turn D2H as 117,440,512 bytes, waited one second, and issued the unchanged two 9500-token pressure prompts. C was then promoted once as warm-up and excluded from measured policy timings. After its external-hit validation snapshot, t0 was set; p_est=42 ms, deadlines A=350 ms/B=370 ms, with ASAP/NaiveJIT/QueueJIT and three restarts each.
- **Commands/validation:** bash results/phase2g/commands.sh (compile, four focused tests, then full experiment); python -m pytest -q tests/test_phase2g.py (4 passed); bash -n results/phase2g/commands.sh; git diff --check. Exactly nine measured runs and 18 A/B job rows completed, plus the single local-hit control. A/B/C offloads, C's 4096 external-hit warm-up, each measured pair's 8192 external hits, and post-promotion 8192 local hits/zero external hits all validated. All resume token IDs matched local controls (1084); no vLLM request errors or server processes remained.
- **Warm-state evidence:** C warm-up proxy median was 111.22 ms. The 18 measured A/B promotions had 38.71-ms median (range 35.18–43.08 ms), satisfying the frozen <=60-ms warm-state check.
- **Key results (three-run medians):** ASAP = 0 misses/run, 0-ms max stall, 608.61-ms total early residency. NaiveJIT = 1 miss/run, 19.49-ms max stall, 1.31-ms total early residency; B missed in all three runs, and A had one 1.50-ms miss. QueueJIT = 0 misses/run, 0-ms max stall, 24.93-ms total early residency. QueueJIT began 22.02 ms earlier than NaiveJIT and saved 583.68 ms (95.9%) of early residency versus ASAP.
- **Result:** A — warm-state real multi-session QueueJIT advantage validated with this request-proxy schedule. The result is not evidence of native PCIe contention.
- **Anomalies:** The isolated Python environment continued to print a non-fatal optional nvidia_cutlass_dsl .pth warning. NaiveJIT repetition 2 also logged vLLM EngineDeadError from the async output handler during SIGTERM teardown, after the server's shutdown trigger; all 10 requests had returned HTTP 200 and that run's cache, resume and token-ID checks passed. No measurement was removed or retuned.
- **Results:** results/phase2g/.
- **Execution status:** COMPLETE.

## 2026-09-17 — Phase 2H: six-session end-to-end real workload

- **Objective/base:** Evaluate six real vLLM sessions with oracle resume deadlines on one end-to-end timeline, comparing KeepAll, OnDemand, ASAP, NaiveJIT, and QueueJIT. Base commit: 81abb72a5916d0cf21b8f1ff1fba85e54414061d.
- **Frozen setup:** Reused vLLM 0.23.0, PyTorch 2.11.0+cu126, CUDA 12.6, RTX A6000 GPU0, and Qwen2.5-1.5B-Instruct. GPU KV capacity was the prescribed 1 GiB; CPU offload was 2 GiB. W and A–F used distinct exact 4096-token prefixes. A–F deadlines were 300/320/340/650/900/1300 ms, with p_est=42 ms. External policies used four fixed 9500-token pressure requests and a W warm-up promotion before t0. Each policy ran three independent server restarts.
- **Timeline methodology:** One serialized promotion worker ran beside six independent resume-dispatch tasks. A resume waited for its own in-flight promotion and did not launch a duplicate restore. No metrics scrape occurred between t0 and the last resume. Effective latency was measured from the configured deadline to the first streamed token. Idle memory-time was 112 MiB multiplied by summed early residency; KeepAll used each session's deadline interval and OnDemand used zero.
- **Commands/validation:** bash results/phase2h/commands.sh; python -m pytest -q tests/test_phase2d.py tests/test_phase2e.py tests/test_phase2g.py tests/test_phase2h.py; python -m py_compile experiments/phase2h/*.py tests/test_phase2h.py; bash -n results/phase2h/commands.sh; git diff --check. All focused tests passed. W/A–F offloads were exactly 117,440,512 bytes, W recovered 4096 external tokens in each external run, all 90 measured sessions completed, cache-source evidence reconciled, and every first-token ID matched its session control.
- **Warm-state evidence:** W promotion median was 111.34 ms. Scheduled A–F promotion proxies had 39.05-ms median and 31.79–80.60-ms range.
- **Key results (three-run medians):** KeepAll = 129.19-ms P95 effective latency and 426.72 MiB·s idle memory-time. OnDemand = 57.51 ms and 0. ASAP = 53.33 ms and 337.47 MiB·s. NaiveJIT = 2 promotion misses, 93.51 ms and 1.11 MiB·s. QueueJIT = 1 promotion miss, 65.01 ms and 9.23 MiB·s. QueueJIT reduced NaiveJIT P95 by 28.50 ms and median misses by one, while saving 97.3% memory-time versus ASAP and 97.8% versus KeepAll.
- **Interpretation:** Result B. The promotion/cache mechanism worked and QueueJIT improved over NaiveJIT, but OnDemand dominated QueueJIT on the primary P95-latency and idle-memory-time axes (57.51 vs 65.01 ms; 0 vs 9.23 MiB·s). QueueJIT therefore was not clearly better than a simpler policy at this six-session scale.
- **Anomalies/limitations:** KeepAll P95 was unexpectedly worse than OnDemand despite valid cache evidence because clustered local resumes experienced serving queue delay. Resume concurrency inflated some request-proxy promotion durations above p_est, with a maximum of 80.60 ms. The known optional nvidia_cutlass_dsl .pth warning recurred but all requests returned HTTP 200 and no vLLM request/model error was found. Promotion remains a normal request proxy and the serialized client worker is not proof of native PCIe contention behavior; deadlines remain oracle values.
- **Results:** results/phase2h/.
- **Execution status:** COMPLETE.

## 2026-09-17 — Phase 3A: native no-forward KV promotion

- **Objective/base:** Replace the normal-completion promotion proxy with a research-only native CPU-external-to-GPU KV path that performs no transformer forward and generates no tokens, then rerun the frozen Phase 2H six-session workload. SlackKV base: `e4199eccaf40770ce5df9093b0fdf358831be3f8`; pinned vLLM source: `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` (v0.23.0).
- **Implementation:** Added local `POST /slackkv/prefetch`, routed through normal prefix hashing, scheduler GPU-block allocation, OffloadingConnector lookup/load jobs, unchanged `OffloadingWorker.transfer_async`, and `kv_connector_no_forward`. Completion marks the loaded blocks computed in the ordinary prefix cache, terminates the research request without sampling, and releases request ownership. The exact 11-file vLLM research patch (332 additions/20 deletions) is in `results/phase3a/vllm_patch.diff`.
- **Validation commands:** `python -m pytest -q tests/test_phase3a.py` (5 passed); vLLM focused tests with plugin autoload disabled, `python -m pytest -q tests/v1/kv_connector/unit/test_slackkv_native_prefetch.py tests/v1/kv_connector/unit/offloading_connector/test_worker_metadata.py` (7 passed); staged sanity/calibration and full workload via `results/phase3a/commands.sh`. Normal local serving, OnDemand connector restore, `max_offload_tokens=0`, native prefetch, cache identity, and greedy token correctness all passed.
- **Native sanity/calibration:** One 4096-token prefix loaded 117,440,512 bytes from CPU, with 15.22-ms endpoint E2E and 7.00-ms CPU→GPU counter time; the subsequent resume hit 4096 local/0 external tokens. Model-forward and generated-token counts were both zero. Three warm E2E times were 12.30/13.50/12.70 ms, freezing `p_native=14 ms` before Part B.
- **Six-session results (three-run medians):** OnDemand P95=108.86 ms, memory-time=0. ASAP-Native P95=126.68 ms, memory-time=384.18 MiB·s. NaiveJIT-Native had 6 misses, P95=132.25 ms, memory-time=0. QueueJIT-Native had 6 misses, P95=129.38 ms, memory-time=0. All 54 scheduled native prefetches loaded the expected external state, all 72 session rows passed generation checks, and all 12 runs passed cache/timeline/offload validation.
- **Interpretation/anomaly:** Result B. The isolated native path removed the fake forward and reduced promotion service to about 12–15 ms, but a prefetch endpoint overlapping a real resume expanded to about 89–93 ms while the copy counter remained about 5 ms. Both JIT policies therefore missed the frozen deadlines, and QueueJIT-Native was worse than current OnDemand and the old Phase 2H proxy P95. The first workload attempt found an overly broad mixed-step guard; after narrowing it to reject only model tokens belonging to the special request ID, all focused tests passed and the complete matrix was rerun from scratch. The recurring optional `nvidia_cutlass_dsl` startup warning was non-fatal.
- **Results:** `results/phase3a/`.
- **Execution status:** COMPLETE.
\n## 2026-09-20 — Phase 3B: connector-first native promotion\n\n- Objective/base: fix native completion coupling at SlackKV base `22252f3df7366d35bea37911c8471ea88ee68295`; pinned vLLM `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`.\n- Validation: SlackKV tests 5 passed; vLLM focused tests 14 passed. Isolated native sanity: 4096 external tokens, 117440512 bytes, CPU→GPU 5.26 ms, client E2E 18.62 ms, zero forward/generated, 4096 local-hit resume, mixed steps 0.\n- Overlap: B at 300 ms and A at 302 ms produced 24.4–27.4 ms client E2E, CPU→GPU 5.23–5.47 ms, mixed steps 0, A added TTFT 10.2–13.7 ms.\n- Frozen workload: A–F deadlines 300/320/340/650/900/1300 ms, 3 repetitions per policy, client warm E2E 17.10/17.22/17.26 ms, `p_native=18 ms`.\n- Results: OnDemand P95 111.61 ms / 0 MiB·s; ASAP-ConnectorFirst 128.58 / 383.61; NaiveJIT-ConnectorFirst 6 misses / 146.99 / 0; QueueJIT-ConnectorFirst 6 misses / 150.64 / 0. All 24576 native external and 24576 local resume tokens, zero special forwards/generated, and cache/generation/timeline checks passed.\n- Interpretation: Result B; connector-first fixed the ~90 ms completion inflation, but OnDemand remains competitive at this KV size and hardware regime. Results: `results/phase3b/`. Execution status: COMPLETE.\n\n## 2026-09-20 — Phase 3C: fixed 30 ms safety service budget\n\n- Objective/base: evaluate whether `p_safe=30 ms` removes the Phase 3B native promotion cascade without changing connector-first vLLM code. Base `07d9f97b49857054fc1b595ccedf6011b0009373`; same vLLM source commit and six-session workload.\n- Validation: focused Phase 3C tests 6 passed; native sanity retained zero model forwards/generated tokens and mixed-step count 0.\n- Matrix: OnDemand, NaiveJIT-Safe30, QueueJIT-Safe30; three independent restarts each; no p_native recalibration.\n- Results (medians): OnDemand P95 104.78 ms, memory-time 0; NaiveJIT-Safe30 2 misses, one >50 ms prefetch, max client prefetch 103.89 ms, P95 127.67 ms, memory-time 5.04 MiB·s; QueueJIT-Safe30 1 miss, one >50 ms prefetch, max 111.63 ms, P95 111.06 ms, memory-time 9.62 MiB·s.\n- Interpretation: one 100-ms-scale event remained but the repeated Phase 3B cascade did not recur. QueueJIT improved over NaiveJIT, but OnDemand remained faster end-to-end. QueueJIT used 2.5% of the Phase 3B ASAP historical memory-time reference (383.61 MiB·s). Result A under the predeclared cascade/stability rule. Results: `results/phase3c/`.\n\n## 2026-09-20 — Phase 3D: resume-aware native promotion\n\n- Objective/base: add a frozen 110 ms future resume blackout calendar to QueueJIT-Safe30 without changing vLLM connector code. Base `a979425d759e16e68082820ffb10a45b0e83fea4`; Phase 3B connector-first implementation reused unchanged.\n- Tests: 8 focused tests passed; sanity native correctness and zero-forward invariants passed.\n- Matrix: OnDemand, QueueJIT-Safe30, ResumeAware-QueueJIT-Safe30; three restarts each; p_safe=30 ms; resume_guard=110 ms.\n- Results (medians): OnDemand P95 109.03 ms, memory-time 0; QueueJIT miss/conflict/long counts 1/1/1, C E2E 108.93–112.52 ms, P95 109.52 ms, memory-time 9.65 MiB·s; ResumeAware counts 0/0/0, C E2E 17.10–18.28 ms, P95 124.47 ms, memory-time 24.43 MiB·s.\n- Interpretation: ResumeAware removed the modeled blackout conflict and long-promotion inflation at 6.4% of historical ASAP memory-time, but its end-to-end P95 remained above OnDemand. Result A for mechanism validation. Results: `results/phase3d/`.\n\n## 2026-09-20 — Phase 3E: gated clustered resume admissions\n\n- Objective/base: test one frozen 40 ms resume spacing with unchanged Phase 3B connector-first runtime. Base `139cda11bd4b2ef1a48b60731f9096e67d0f9695`; p_safe=30 ms; resume_guard=110 ms.\n- Tests: 9 focused tests passed; sanity and native correctness passed.\n- Matrix: OnDemand, ResumeAware-QueueJIT-Safe30, Gated-ResumeAware-QueueJIT-Safe30; three restarts each.\n- Results (medians): OnDemand P95 105.69 ms; ResumeAware ABC overlap 2, ABC P95 125.78 ms, overall P95 122.72 ms, memory-time 24.51 MiB·s; Gated ABC overlap 2, ABC P95 130.97 ms, overall P95 127.82 ms, memory-time 24.39 MiB·s. Promotion misses/conflicts/long prefetches remained zero.\n- Interpretation: planned sends changed A/B/C from 300/320/340 to 300/340/380 ms, but measured in-flight overlap did not decrease and original-deadline charging made tail latency worse. Result B. Results: `results/phase3e/`.\n
## Phase 3F — first-token-aware resume admission

- Objective: test runtime FirstTokenGate against ResumeAware native promotion on the frozen A–F workload.
- Base: `c78b42433efc3dec36645523c12d7e7707c860c7`; vLLM `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`.
- Configuration: `p_safe=30 ms`, `resume_guard=110 ms`, three policies (`OnDemand-Trace`, `ResumeAware`, `FirstTokenGate-ResumeAware`), three server restarts each.
- Tests: `python -m pytest -q tests/test_phase3f.py` (3 passed).
- Result: FirstTokenGate reduced median ABC pairwise overlap from 290.77 ms to 52.79 ms and max pre-first-token concurrency from 3 to 2, but ABC P95 improved only from 129.71 ms to 122.75 ms (<10%); classified Result B.
- Output: `results/phase3f/`.
- Implementation caveat: the recorded gate was applied at the client admission boundary; no vLLM-internal scheduler patch or internal WAITING_FOR_REMOTE_KVS/first-token hook was added, so the artifact is a partial runtime measurement rather than a complete in-engine Phase 3F validation.

## 2026-09-20 — Phase 3F-Fix: in-engine first-token resume gate

- Objective/base: correct the incomplete client-side Phase 3F gate at SlackKV base `bc29bf1167f4d6e5b52474a3a60c7bb11b14073b`; pinned vLLM source `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`.
- Implementation: removed client-side Event admission control from measured runs. Added a research-only scheduler owner in vLLM, deterministic waiting-queue skipping for marked resumes, first-token release from real model output, stale-owner cleanup, and structured scheduler traces. Phase 3B connector-first promotion remains unchanged.
- Frozen matrix: OnDemand-Trace, ResumeAware-Trace, and InEngineFirstTokenGate-ResumeAware; three independent server restarts each; A–F, `max_tokens=1`, `p_safe=30 ms`, `resume_guard=110 ms`. The A/B/C integration pre-gate passed before the nine-run matrix.
- Validation: `PYTHONPATH=. $ENV/bin/python -m pytest -q tests/test_phase3f_fix.py`; full workload via `results/phase3f_fix/commands.sh`; postprocessing via `experiments/phase3f_fix/postprocess.py`. Real traces contain scheduler-generated remote-KV, admission, first-model, first-token, and gate-block events.
- Results (medians): OnDemand overall P95 104.63 ms, ABC pairwise overlap 0 ms, memory-time 0. ResumeAware overall P95 124.34 ms, ABC overlap 94.14 ms, max pre-first-token concurrency 3, corrected memory-time 24.85 MiB·s. In-engine gate overall P95 88.10 ms, ABC overlap 0 ms, max concurrency 1, gate wait 48.87 ms, corrected memory-time 24.73 MiB·s. Promotion misses/conflicts/long prefetches were zero; native prefetch forwards/generated tokens and mixed steps were zero.
- Criterion/result: gate invariant PASS, A→B→C admission order in all repetitions, ABC P95 improved 30.1% versus ResumeAware, memory-time was 6.4% of historical ASAP (383.61 MiB·s). OnDemand implicit backpressure is SUPPORTED by real WAITING_FOR_REMOTE_KVS → REMOTE_KV_READY traces and staggered admissions. Result A.
- Historical correction: the previous Phase 3F Event-based client gate is exploratory only and is not an algorithm baseline because Event.set() could wake multiple waiters. Phase 3F-Fix results are in `results/phase3f_fix/`; exact vLLM patch is `results/phase3f_fix/vllm_phase3f_fix.patch`.

## 2026-09-20 — Phase 3G: multi-token decode overlap (stopped at integration gate)

- Objective/base: test whether the Phase 3F-Fix in-engine FirstTokenGate releases after the first token while later decode overlaps. Base `c6ffb80b90b8d363dfa62f15dcbc5a1b145d9de5`; vLLM `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`.
- Frozen setup: A/B/C integration gate first; six 32-token controls and `max_tokens=32`, `ignore_eos=true`; no client-side gate or new vLLM source changes.
- Integration: real scheduler trace showed gate correctness and post-first overlap: A/B/C admitted in order, B before A finish, C before B finish, and max pre-first ownership 1. Every continuation generated exactly 32 tokens.
- Blocking anomaly: C's measured 32-token sequence diverged from its deterministic control sequence after token 24; A and B matched. Per the protocol the full matrix was not run and the result is Result D, not a performance claim.
- Validation: `tests/test_phase3g.py` 7 passed; raw monotonic timestamps are aligned to saved client `t0_monotonic_s` without first-event rebasing. Results: `results/phase3g/`.
## 2026-09-20 — Phase 3G-CorrectnessDiag

- Objective: diagnose the Phase 3G C continuation mismatch without rerunning the full policy matrix.
- Cases: S1 LocalKeep-Solo-C, S2 Prefetch-Solo-C, S3 LocalKeep-Gated-ABC, S4 Prefetch-Gated-ABC; three fresh server restarts per case; max_tokens=32, temperature=0, ignore_eos=true.
- Result: S1 and S2 were deterministic and exactly matched. S3 C was deterministic but diverged from solo C at one-based token 26. S4 matched S3 exactly for A, B, and C. Native transport invariants passed independently (4096 external tokens, zero native model forwards/generated tokens).
- Interpretation: BATCH_SENSITIVITY; the prior mismatch is not reproduced as an isolated KV-prefetch corruption under the matched gated schedule.
- Result path: results/phase3g_diag/

## 2026-09-21 — Phase 3G-Full: six-session multi-token gated-resume performance

- **Objective/base:** evaluate whether the unchanged Phase 3F-Fix in-engine FirstTokenGate preserves its first-token benefit while allowing multi-token decode overlap on A–F. Base `b0ca5fd755f4ec47c2991d4836fecc2d40956d99`; pinned vLLM source `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`.
- **Frozen matrix:** OnDemand-Trace, ResumeAware-Trace, and InEngineFirstTokenGate-ResumeAware; three clean server restarts per policy. A–F used unchanged 4096-token prefixes/deadlines, `p_safe=30 ms`, `resume_guard=110 ms`, and 32-token greedy resumes with `ignore_eos=true`.
- **Validation:** six focused tests passed. The A/B/C pre-run integration gate passed: 32 tokens each, native endpoint runtime records had 4096 loaded external tokens and zero forward/generated/mixed-step counts, pre-first concurrency was one, and A/B/C admission occurred after the preceding first token but before the preceding request finished. All 54 measured resumes had 32 tokens; all cache transport, timeline, trace, and gated schedule validations passed.
- **Results (three-run medians):** OnDemand ABC/overall P95 TTFT = 194.77/191.80 ms, ABC decode throughput = 171.54 tok/s. ResumeAware = 134.08/133.77 ms, 173.23 tok/s, 39.57 MiB*s memory-time. FirstTokenGate = 157.56/156.04 ms, 162.35 tok/s, 53.94 MiB*s. The gate maintained zero ABC pre-first overlap, max pre-first concurrency one, post-first decode concurrency four, and 1410.74 ms ABC post-first overlap. It did not serialize full requests: B entered after A first token but before A finish, and C likewise after B.
- **Interpretation:** Result C. The in-engine gate is mechanically correct and preserves overlapping decode, but its ABC P95 TTFT was 17.51% worse than ResumeAware under this full six-session multi-token workload; its 93.72% decode-throughput retention and 6.08% makespan increase do not offset that TTFT regression. Gated memory-time remained 14.06% of historical ASAP (383.61 MiB*s). OnDemand remained slower than both native-prefetch policies on ABC P95.
- **Correctness interpretation:** Phase 3G-CorrectnessDiag had already shown LocalKeep-Solo == Prefetch-Solo and LocalKeep-Gated == Prefetch-Gated under matched concurrent execution. This phase stores sequence hashes as diagnostics only; cross-policy greedy sequences are not treated as a cache failure because batch composition changes.
- **Time/source-counter note:** server trace timestamps retain raw `time.monotonic()` and align to saved workload t0. Positive server-intake delay can exceed 50 ms while an engine step is busy; only negative offsets would invalidate time alignment. Native prefetch source attribution uses actual endpoint responses because vLLM's official external source counter does not include prefetch-only endpoint loads.
- **Results:** `results/phase3g_full/`.

## 2026-09-21 - Phase 3H: two-slot first-token admission

- Objective/base: K=2 pre-first admission against ResumeAware-K3 and K=1 on A/B/C only. Base eb5b3eb6e13e5f8a682d1ea941e53fd5032c7de1.
- Results: K2 ABC P95 TTFT 115.23 ms, versus 131.69 ms ResumeAware-K3 and 159.98 ms K1. K2 decode throughput 182.99 tok/s, memory-time 37.22 MiB*s, and Result A.
- Results: results/phase3h/.
## Phase 3I - background decode load

- Frozen policies: ResumeAware-K3, FirstTokenGate-K1, FirstTokenGate-K2.
- Added one ordinary 128-token background request X. t0 was defined after X first token; X remained active beyond C first token in all recorded runs.
- X-only baseline median decode duration was 1824.40 ms and median TPOT was 14.37 ms.
- All X requests generated 128 tokens. Native prefetch endpoint responses reported 4096 external tokens, zero model forwards, zero generated tokens, and zero mixed steps.
- Under the frozen busy-load setup, A/B/C promotions had deadline misses, blackout conflicts, and >50 ms promotions. Therefore the required clean-promotion precondition did not hold and the phase is classified Result D; K1/K2 performance numbers are diagnostic only.
- No vLLM scheduler algorithm was changed; the Phase 3H generalized K-capacity gate was reused unchanged. No Phase 3J/dynamic-K work was started.
## Phase 3J - load-aware promotion calibration

- Stage A used five idle L0 promotions and five one-active-decode L1 promotions. L0 client E2E P50/P95/max was 20.57/20.91/21.84 ms; L1 was 53.88/56.58/57.10 ms.
- The frozen rule produced p_busy_ms=70 ms from max(L1)+10 ms rounded upward to 5 ms. No service-budget sweep was added.
- Stage B ran five FirstTokenGate K=2 A/B/C+X validations. All promotion deadline misses, blackout conflicts, and budget overruns were zero; native endpoint invariants and cache/token checks passed.
- K2 reached pre-first concurrency two in all runs, but C was gate-blocked in only one repetition because server intake admitted C after an earlier owner had already produced its first token. The required K2 exercise condition therefore failed and the phase is classified Result D; no K comparison or dynamic-K conclusion is claimed.
- No vLLM scheduler algorithm changes were made and Phase 3K was not started.
## Phase 3L - promotion service curve under decode load

- Objective/base: measure the existing native promotion endpoint under L0/L1/L2/L3 with 0/1/2/3 ordinary 128-token background decodes. Base `35504db723b6928548c734c7806480cff723621f`; pinned vLLM source `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`.
- Frozen matrix: four load states, five fresh-server repetitions per state (20 promotions total), one fixed 4096-token target prefix, no FirstTokenGate and no vLLM source changes.
- Validation: 20/20 promotions passed active-decode, 128-token background, cache transport, zero-forward, zero-generated-token, and zero-mixed-step checks. Endpoint fields were persisted from actual runtime responses.
- Service curve: client E2E P50 was 20.92/57.09/51.39/53.53 ms for L0/L1/L2/L3; CPU→GPU P50 was 5.51/5.36/5.41/5.59 ms. Non-copy overhead is load-dependent, while copy time is effectively flat. P50 deltas were Δ01=36.16 ms, Δ12=-5.70 ms, Δ23=2.14 ms; the curve is therefore non-monotonic after the first active decode, but has a clear idle-to-busy gap.
- Budget table: raw max+10 ms budgets were 31.60/69.00/62.44/70.93 ms; rounded recommendations are 35/70/65/75 ms. Classification: Result C, because all 20 runs were valid but the n=1/2/3 medians are non-monotonic and the active-count budget spread is borderline rather than a stable decode-count curve; a richer runtime signal is required.
- No Phase 3M work was started. Results: `results/phase3l/`.

## Phase 3M - native promotion latency decomposition

- Objective/base: identify where busy promotion latency is spent using T0–T7 monotonic timestamps, ordinary engine-step timing, and the exact Phase 3L L0–L3 workload. Base `8350af11356913676906b111aabab72e69544172`; pinned vLLM source `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`.
- Frozen matrix: four load states, three fresh-server repetitions per state (12 promotions total), no admission/promotion-policy changes, no FirstTokenGate.
- Validation: 12/12 promotions passed active-through-promotion, cache transport, native zero-forward/zero-generated/zero-mixed-step checks. Server FIRST_TOKEN events were recorded independently for X1/X2/X3; client delivery lag median/max was 10.69/17.81 ms, confirming Phase 3L's client barrier had a measurable phase bias.
- Decomposition medians: request-to-connector-start was 2.70/14.74/10.99/13.51 ms for L0/L1/L2/L3; connector phase was 10.15/9.92/10.67/10.02 ms; CPU→GPU was 5.33/5.28/5.38/5.28 ms. A normal model step was in flight at T0 in 9/9 busy runs, with remaining-step medians 16.64/12.90/15.42 ms.
- Result A: busy overhead is primarily pre-connector blocking associated with an in-flight normal model step; connector phase and copy remain stable. Next direction: `STEP_TIME_ESTIMATOR_E2E`. No Phase 3N work was started. Results: `results/phase3m/`.

## Phase 3K - calibrated busy-load admission comparison

- Reused p_busy=70 ms and resume_guard=110 ms from Phase 3J. Ran ResumeAware-K3, FirstTokenGate-K1, and FirstTokenGate-K2 for five fresh restarts each with the same X background decode.
- All 15 runs were promotion-clean: zero deadline misses, blackout conflicts, and budget overruns; native endpoint invariants and cache/token checks passed.
- Busy-load median ABC P95 TTFT was K3 59.53 ms, K2 59.91 ms, K1 80.76 ms. K1 remained exactly pre-first=1 and K2 exactly pre-first=2 in all runs.
- The busy ranking numerically changed from light-load K2 < K3 < K1 to K3 < K2 < K1, but the K3/K2 gap was only 0.38 ms and K1 did not protect X or improve combined throughput. Dynamic K is not justified; report Result C and do not start Phase 3L.

## 2026-09-22 — Phase 3N: internal KV-ready asynchronous resume handoff

- **Objective:** Determine whether A can safely resume after the scheduler-visible native prefetch completion boundary (T6), before the native HTTP endpoint returns, while an unrelated 9500-token X decode is active.
- **Frozen design:** One 4096-token A KV is offloaded and pressure-evicted; X begins a 128-token ordinary decode; A native prefetch is launched in a separate thread after X first token; A resume uses exact offsets 5/20/30/40/70 ms from prefetch-client send, three fresh restarts each (15 total). No extra offsets were added.
- **Instrumentation:** Reused Phase 3M T0–T7 unchanged. Boundary classification uses the real scheduler `RESUME_ARRIVED` monotonic timestamp. Source-token counters around A resume establish local/external cache attribution; traces establish whether A entered `WAITING_FOR_REMOTE_KVS`.
- **Validation:** `python -m pytest -q tests/test_phase3n.py` (3 passed); 15/15 runs kept X active through prefetch and emitted all required timestamps. Native prefetch always reported real 4096 external tokens, zero forward/generation/mixed steps.
- **Key results:** 6/15 requests arrived in `INTERNAL_READY_PRE_CLIENT_RETURN`, all six had 4096 local-hit / zero external-transfer tokens and no remote-KV wait. Critical-window median TTFT was 59.83 ms versus 56.42 ms after client return (within the predeclared 10% condition). Median T6-ready latency was 31.76 ms; the median T6→client-return gap was 25.02 ms. The 5-ms offset did not produce PRE_READY arrivals because scheduling/transport moved actual arrival past T6; no new offset was introduced.
- **Result:** A — the scheduler-visible T6 boundary supports asynchronous promotion handoff; endpoint-return waiting is unnecessary. Next direction is `ASYNC_PROMOTION_E2E`.
- **Results:** `results/phase3n/`.
- **Execution status:** COMPLETE.
