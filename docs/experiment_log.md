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

## 2026-09-22 — Phase 3O: asynchronous native promotion end-to-end

- **Objective:** Test whether controller scheduling against native internal T6 readiness, rather than HTTP completion, preserves A/B/C resume latency while reducing promoted-KV residency during a 128-token ordinary X decode.
- **Frozen setup:** A/B/C deadlines 300/320/340 ms relative to X server first token; K=2, resume guard 110 ms, Blocking70 service budget 70 ms, Async40 T6 budget 40 ms. Three independent restarts for OnDemand-K2, Blocking70-K2, and Async40-K2 (9 runs); no A–F expansion or budget sweep.
- **Validation:** `python -m pytest -q tests/test_phase3o.py` (4 passed), Async40 integration pre-gate PASS, and all full runs retained X through C first token. T0–T7 instrumentation from Phases 3M/3N and Phase 3H K=2 were reused unchanged.
- **Key results:** Async40 reached T6 before all 9 real scheduler arrivals. All 9 resumed from the local path (run-global 12,288 local source tokens plus no per-session remote-KV wait) with zero external source tokens. It had up to two outstanding HTTP prefetch calls but only one overlapping T0→T6 interval. Median ABC P95 effective resume latency was 60.64 ms versus 58.32 ms for Blocking70 and 106.51 ms for OnDemand. Async40 cut promoted-KV memory-time 52.29→30.93 MiB*s (40.85% lower than Blocking70), while combined throughput changed +0.58%. X decode slowdown relative to the compatible Phase 3I X-only baseline remained below 5% for all policies.
- **Result:** A — internal-ready asynchronous promotion works end-to-end; HTTP completion is unnecessary on the controller critical path and p_async=40 ms is sufficient for this A/B/C+X workload.
- **Next direction:** `FULL_ABCDEF_E2E` (not started here).
- **Results:** `results/phase3o/`.
- **Execution status:** COMPLETE.

## 2026-09-22 — Phase 3P: full six-session async promotion

- **Objective:** Validate frozen Async40-K2 promotion across A–F plus a 192-token X decode.
- **Result:** Async reached T6 before all 18 resume arrivals, with zero remote waits and frozen K=2. A–F median P95 effective latency was 56.78 ms (Async), 60.80 ms (Blocking70), and 101.32 ms (OnDemand); Async memory-time was 41.42 MiB*s versus Blocking 72.30 MiB*s.
- **Outcome:** Result A; next direction SCALE_WORKLOAD. Results: `results/phase3p/`.

## Phase 3P-FIX

Frozen Phase 3P CSV postprocessing repair only; no GPU workload rerun. Corrected full A-F schema, 18 Async internal-ready/local-hit checks, real issue-to-T6 latency, and 192-token X-only baseline denominator 191.

## Phase 3Q

12-session Async40-K2 integration gate stopped before full matrix: all T6 deadlines were met, but only 16,384/49,152 local-hit tokens remained and E-L waited remotely. Classified as Result C capacity-retention limit; no tuning or full replay performed.

## Phase 3R

Capacity-guard wiring and one Stage-A execution committed as INCOMPLETE/EXPERIMENT_INVALID: capacity snapshots were not persisted, so no capacity-aware conclusion was claimed; Stage B was not run.
## 2026-09-22 — Phase 3R-FIX

- Objective: audit the frozen 144-MiB instantaneous KV-capacity guard with one instrumented 12-session Stage-A run.
- Configuration: Async40, K=2, 1-GiB GPU KV, 2-GiB external tier, A–L frozen deadlines, X=256; no Stage B.
- Command: `python -m experiments.phase3r_fix.run_phase3r_fix --stage integration`.
- Validation: 12/12 decisions persisted; native 12/12; token 12/12; K2/X PASS; source 23,600 local + 25,552 external = 49,152.
- Observation: all decisions were PROMOTE with measured free capacity 196.13–763.51 MiB, yet E/F/G/I/J/K/L later waited remotely. Capacity-query latency was 21.29 ms median and target-to-T6 was 47.41 ms median.
- Conclusion: Result B. Instantaneous free capacity is a live but insufficient predictor of future residency. Next direction is `RESIDENCY_RESERVATION`; it was not started.
- Results: `results/phase3r_fix/`.

## 2026-09-22 — Phase 3S

- Objective: validate a minimal GPU KV residency lease that survives ordinary vLLM prefix-cache eviction pressure and hands ownership safely to one real resume.
- Configuration: one 4096-token session A, 117,440,512-byte KV, 32-token resume, four fixed 9500-token post-prefetch pressure prompts, 3 Unleased + 3 Leased fresh restarts.
- Command: `PYTHONPATH=.:/media/a6000/D/ZhaoBinHe/slackkv-phase2c-fix-sourceprobe /media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python -m experiments.phase3s.run_phase3s`.
- Validation: tests 7/7; native 6/6; token count 6/6; source conservation 6/6; lease acquire/release 3/3; no lease leaks.
- Results: unleased controls were evicted 3/3 (0 local, 4096 external, remote wait); leased prefixes remained fully local 3/3 (4096 local, 0 external, no remote wait). All leased runs used 256 blocks. Acquire overhead was 17.701 us median / 17.941 us max; release was 13.081 us median / 15.790 us max.
- Conclusion: Result A. Retaining one block-pool reference protects native-prefetched KV until the real resume safely takes ownership. Next direction `RESERVATION_LEDGER_12_SESSION` was not started.
- Results: `results/phase3s/`.

## 2026-09-22 — Phase 3T

- Objective: integrate the Phase 3S physical KV lease into the frozen 12-session Async40/K2 workload through a conservative two-prefix reservation ledger.
- Configuration: A–L fixed deadlines, X=256, 1-GiB GPU KV, 2-GiB external tier, 256 blocks/session, 512-block ledger, 100-ms handoff guard, Async40, resume guard 110 ms, K=2.
- Commands: `python -m experiments.phase3t.run_phase3t --stage integration`, then `--stage full`, followed by `python -m experiments.phase3t.postprocess`.
- Validation: Stage A PASS; Stage B 3/3; tests 5/5; real resumes 36/36; K2 3/3; background 3/3; source conservation 3/3.
- Results: the controller naturally promoted A/B, E/F, and I/J and skipped the remaining six sessions per run. Across Stage B, 18/18 promotions were local and useful, with zero remote waits, zero T6 misses, zero lease leaks, and zero wasted promotion MiB. Maximum reserved blocks were 512. Aggregate source was 73,728 local + 73,728 external tokens.
- Timing: target-to-issue 0.079/0.174/0.189 ms median/P95/max; issue-to-T6 25.200/31.105/33.195 ms; target-to-T6 25.279/31.286/33.334 ms. Median-run overall/promoted/skipped P95 resume latency was 83.48/58.06/83.96 ms.
- Conclusion: Result A. Finite reservation plus physical leases eliminated the Phase 3Q/3R wasted-promotion failure by intentionally routing unreserved sessions to OnDemand. Next direction `RESERVATION_FULL_BASELINE_MATRIX` was not started.
- Results: `results/phase3t/`.

## 2026-09-23 — Phase 3V

- Objective: compare count-based two-session reservations with a 512-block dynamic reservation ledger under a frozen heterogeneous A–L workload; Phase 3S leases and K=2 are reused unchanged.
- Validation sequence: Phase 3V tests 6/6 (13/13 including archived TraceLab tests); X-only controls 3/3 (median decode 3683.740 ms, TPOT 14.446 ms); uniform FixedCount2/DynamicBlock regression PASS; heterogeneous integration 4/4 PASS; measured workload 12/12 runs and 144/144 exact 32-token responses.
- Uniform regression: both policies promoted/skipped 6/6 with 100% useful promotions, zero promoted remote waits, zero lease leaks, and 512-block peak.
- Heterogeneous measurements: FixedCount2 median 6 promotions / 6 skips, 128 peak blocks, 100% useful; DynamicBlock median 12 promotions / 0 skips, 448 peak blocks, 100% useful. Both had zero promoted remote waits, internal-ready misses, lease leaks, duplicate H2D, and block-prediction errors.
- Traffic and service: all four policies measured 588 MiB total H2D / 0 duplicate MiB. DynamicBlock issue-to-T6 was below the frozen 40-ms budget in 36/36 promotions (max 34.02 ms). NaiveAsync did not reproduce a residency/wasted-traffic failure in this workload.
- Median AL P95 resume latency: OnDemand 107.75 ms; NaiveAsync 56.38 ms; FixedCount2 79.99 ms; DynamicBlock 56.49 ms. DynamicBlock versus FixedCount2 changed P50/P95 by -24.00%/-29.38%, combined throughput by -3.01%, and X slowdown by +0.032.
- Result: `RESULT_A`; next direction `DYNAMIC_BLOCK_RESERVATION_VALIDATED`. The arithmetic `448 + 64` equals 512 and is admitted under the explicitly frozen `<=512` rule; no hidden threshold was applied.
- Results: `results/phase3v/`. Earlier TraceLab-selection artifacts were preserved separately as `results/phase3v_tracelab_incomplete/`; no Phase 3W work was started.
- Setup instrumentation: one measured attempt stopped on delayed per-request GPU→CPU counter attribution, with aggregate expected bytes conserved; after adding bounded counter-settling polls, the repetition reran successfully. No failed attempt was included in the measured dataset.

## 2026-09-23 — Phase 3W

- Objective: evaluate frozen prediction errors for OracleDynamicBlock512, RawPredDynamicBlock512, and SafePred200DynamicBlock512 using the Phase 3V reservation/lease mechanism; no prediction robustness sweep or measured matrix was completed.
- Validation: focused Phase 3W + Phase 3V tests 12/12; X-only controls 3/3 (median X decode 3663.63 ms, median TPOT 14.37 ms); zero-error Oracle/Raw regression passed with matching 6-promote/6-skip decisions, 100% useful promotions, no remote waits or lease leaks, and 0 duplicate H2D.
- Mixed-error integration: Oracle passed the strict check (6 promotions, 6 skips, all useful, no T6 misses/remote waits/internal misses/lease leaks, peak 512 blocks). Safe completed correctly but made 5 promotions and 7 skips, so it failed the frozen 6/6 integration requirement. Raw client stream ended without a generated token; its server log was overwritten by the subsequent Safe run, so the precise Raw server-side root cause is not independently verifiable.
- Stage gate: failed (only Oracle is strictly valid; Safe reservation split failed and Raw request failed). As required, the 9-run measured matrix was not started. No parameters were tuned and no vLLM source was modified.
- Results: `results/phase3w/`. Phase 3X and later work were not started.

## 2026-09-24 — Phase 4A offline eligibility gate

- Verified TraceLab v0.0.2 `syfi_coding_trace.duckdb` against the frozen SHA256 `a7bab286bc640844560850965ccf47975cf66407154132abaab90f27ec9be744` and inspected the released schema.
- Applied the frozen next-round `prefix_tokens`, 300–5000-ms tool-pause, 1024–8192-token prefix, at-most-9000-token input, complete-timestamp, adjacent-round, and session-hash split rules. Context shrink was permitted per the request.
- The replay split contained 19 eligible boundaries and only 4 valid three-boundary episodes from 2 sessions, below the predeclared minimum of 24. Stopped with `TRACE_ELIGIBILITY_INSUFFICIENT` before workload selection, capacity-profile calculation, GPU calibration, integration, or measured policy runs. No filters were relaxed.
- Focused offline tests passed 5/5. Diagnostic artifacts: `results/phase4a/`. This is an incomplete experiment requiring manual review, not a serving result.

## 2026-09-24 — Phase 4A revised real-trace semantics audit

- Objective: audit real inter-turn slack, KV opportunity, causal predictability, and structural replay readiness in the pinned TraceLab v0.0.2 public DuckDB, without GPU serving or vLLM changes.
- Command: `PYTHONPATH=/tmp/slackkv-phase4a-deps /media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python experiments/phase4a/run_phase4a.py external/tracelab_v002/syfi_coding_trace.duckdb results/phase4a`.
- Dataset SHA256 matched the pinned `a7bab286bc640844560850965ccf47975cf66407154132abaab90f27ec9be744`; schema contained 665,453 rounds, 743,819 tool calls, 2,688,829 timing events, and 8,058 sessions. Focused tests passed 10/10.
- Stage A found no explicit next-LLM request/start timestamp. Tool completion, input-ready, and output-side events are not interchangeable with request start. Result: `RESULT_C_NO_VALID_RESUME_SEMANTICS`. Opportunity, predictors, candidate pool, and service-model audit were not run after this mandatory gate; no Phase 4B work began.
- Results: `results/phase4a/`. This revised protocol supersedes the earlier uncommitted eligibility-only Phase 4A attempt above without changing historical phase directories.

## 2026-09-24 — Phase 4B public timing/replayability audit

- Objective: determine whether `netpreme/coding_agent_traces` exposes a request-level timeline sufficient for timed trace replay; CPU/offline only.
- Pinned inputs: `netpreme/coding_agents@83d8d1280b406b0484693017900763a0b0ad8d19`; HF page revision `d56ae0d0bc80f3663a7a61a38c4ee60c26ce7b67`.
- Source result: proxy `request_time` is receive time; scraper `e2e_ms` is a delta average across completions in a Prometheus polling window when `n>1`; `tool_exec_ms` is current request arrival minus prior request arrival, not response-end slack.
- Dataset limitation: public viewer row/session shape and aggregate token statistics were visible, but Hub API timed out and nested Parquet schema was not retrieved. No raw text or parquet downloaded; request/e2e coverage and slack distribution remain unmeasured.
- Result: `RESULT_C_TIMING_NOT_RECONSTRUCTABLE` on verified public timing evidence. Focused tests passed 12/12. No serving, GPU, or predictor stage was run.
- Results: `results/phase4b/`. Commit/push status recorded after repository sync.

## 2026-09-25 — Phase 4B live HF metadata/row-schema recheck

- Objective: repeat the previously blocked public HF schema check after identifying that the A6000 interactive shell, but not non-login remote commands, inherited its HTTPS proxy.
- Command: `bash -lic 'cd /media/a6000/D/ZhaoBinHe/SlackKV && python experiments/phase4b/audit_hf_live.py'`; a separate bounded console query inspected the first 100 rows from each GPT-OSS config.
- Findings: the Hub API verified revision `d56ae0d0bc80f3663a7a61a38c4ee60c26ce7b67`; the sampled 200 sessions / 5,550 turns exposed token-count and text fields only, with no tool or timing-named fields. The repeatable script uses 10 rows/config to limit text-bearing transfer volume.
- Privacy/setup: no row IDs, prompts, completions, or Parquet shards were persisted; no GPU or serving workload was run. The sample is not an exhaustive scan.
- Result: `RESULT_C_TIMING_NOT_RECONSTRUCTABLE` remains; `tool_exec_ms` is inter-arrival time and scraper E2E is not request-aligned. Stage-C slack/replayability counts remain uncomputed. Recheck suite passed 14/14.
- Results: `results/phase4b/`; commit/push recorded in Git history after this recheck.

## 2026-09-25 — Phase 4C real-structure hybrid replay correctness

- Objective: validate that real coding-agent ISL/ISL_new/OSL structure can be reconstructed as deterministic synthetic token sessions with exact local vLLM prefix-cache blocks and controlled TraceToolProxy pauses. This was a replay-fidelity experiment only, not a SlackKV performance comparison.
- Sources: `netpreme/coding_agent_traces` revision `d56ae0d0bc80f3663a7a61a38c4ee60c26ce7b67`, bounded to offsets 0–99 for Pro and Verified; TraceLab v0.0.2 DuckDB SHA256 `a7bab286bc640844560850965ccf47975cf66407154132abaab90f27ec9be744`.
- Coverage/selection: Pro 87/100 eligible and Verified 71/100 eligible; deterministic selection produced 4 sessions/config, 40 requests and 32 boundaries. Raw text and raw instance IDs were not persisted.
- Runtime: vLLM 0.23.0, pinned source `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`, Qwen2.5-1.5B-Instruct local snapshot, max model length 32768, 4-GiB local prefix cache, actual cache block size 16; no external connector, promotion, lease, reservation, K2, pressure, or background request.
- Validation: Stage A0 passed 768-token and 770→768 block-alignment calibration; Stage A1 passed 10/10 prompts/outputs and 8/8 boundaries; Stage B passed 40/40 prompt lengths, 40/40 OSL lengths, 32/32 cached blocks, 8/8 isolated priming turns, 0 external hits, and pause error P50/P95/max 5.10/7.36/7.96 ms. Final unit tests passed 25/25.
- Result: `RESULT_A_HYBRID_REPLAY_CORRECT`. This establishes a correct hybrid replay harness only; it is not timestamp-faithful replay of a real Agent session and makes no performance-superiority claim. Phase 4D was not started.
- Results: `results/phase4c/`.

## 2026-09-26 — Phase 4D-A large-prefix runtime integration (stopped at integration gate)

- Objective: connect the frozen Phase 4C boundary-0 hybrid workload to native promotion, T6 readiness, K2, dynamic reservation, and physical leases. No workload or vLLM algorithm changes were made.
- Frozen Phase 4C workload SHA256: `9b381336a70e73b0efbddbb79d2c88608c19663247e819c1e7a364efd2971105`; 8 jobs, 168,016 cached tokens, 10,501 blocks, 4,594.1875 MiB minimum H2D.
- Runtime-derived capacity: 4,681 total GPU KV blocks; the frozen 75% reservation budget was 3,510 blocks. Real-size promotion calibration passed 9/9 and froze P_SAFE=102 ms. Stage A0 prefix calibration passed.
- X-only baseline passed 3/3. OnDemand and Reservation integration passed 8/8 resumes with source conservation; Reservation promoted two jobs with 2,548-block logical/physical peak and zero lease leaks.
- NaiveJIT-Async-K2 did not complete its eight resumes in repeated fresh integrations (latest: 3 first tokens, 0 finishes after the 120-s client timeout, 53,935 connector-only steps). The strict 3/3 policy integration gate therefore failed, so Stage B was not started and no performance result is assigned.
- Interpretation: this is an incomplete mechanism baseline, not a SlackKV lease failure or an invalid environment. Naive non-progress is retained as diagnostic evidence; it is not reported as a latency result.
- Results: `results/phase4d_a/`; Phase 4D-A did not proceed to later boundaries or prediction work.


## 2026-09-26 — Phase 4D-A-FIX runtime release and Naive liveness audit

- Frozen start commit `86ac84a94b671514c78b69aeb3dd294e137a03a8`, Phase 4C workload SHA256 `9b381336a70e73b0efbddbb79d2c88608c19663247e819c1e7a364efd2971105`, 4,681 GPU KV blocks, 3,510-block reservation budget, P_SAFE=102 ms, and K2=2 were preserved. No vLLM serving algorithm was changed.
- Implemented explicit `episode_id -> lease_key` mapping and structured exact JSON lease parsing. Runtime logs verified E04/E00 physical releases before logical releases and later E06/E07 reservation promotion; 35/35 tests passed.
- Three fresh Reservation-only diagnostic attempts reproduced incomplete progress: all 8 resume arrivals were observed, 3 reached first token, 0 emitted request-finished trace events; the retry with immediate native response persistence returned successful endpoint data for E04/E00, while E06/E07 did not return within the client’s 120-s read timeout. The server remained alive, no OOM/crash was observed, and connector-only scheduling continued. At abort, exact-key release had returned 2,548 logical blocks and later reservations had been admitted.
- Completed T6 records were 98.50 ms and 107.24 ms; one exceeds the frozen 102-ms budget. The service envelope is therefore `P_SAFE_102_INTEGRATION_ENVELOPE_FAIL`; P_SAFE was not recalibrated. The overall Reservation integration acceptance gate did not pass.
- By the frozen prerequisite, Naive L0/L1 and the original Phase 4D-A Stage B matrix were not started. The observed Reservation-path non-progress does not establish that unleased Naive capacity crossing is the cause. Final conservative taxonomy: `RESULT_D_RELEASE_FIX_FAILED` for the failed end-to-end acceptance gate, with the targeted identity mapping itself passing runtime event/order checks.
- Artifacts: `results/phase4d_a_fix/`.

## 2026-09-26 — Phase 4D-B runtime physical-headroom gate

- Added a research-only `/slackkv/prefetch` capacity guard backed by `KVCacheManager.block_pool.get_num_free_blocks()`. It validates required blocks against 16-token prompt alignment and fast-rejects insufficient physical headroom before connector load/allocation.
- G1 passed both sides twice: E06 admitted at 2,135 free / 1,276 required blocks; E00 rejected at 859 free / 1,277 required blocks, returning in 21.3–22.9 ms with zero transfer and no rejected-request lease.
- G2 smoke and three fresh-server repetitions each completed 8/8 resumes. In each valid run E04/E00 were admitted, while E06/E07/E03 were rejected before connector work and used OnDemand fallback. No overcommit, lease/logical leak, crash, OOM, or 5-second prefetch non-progress was observed.
- P_SAFE=102 ms remained frozen and was not exceeded by any of the eight successful promotions in the four valid runs. No performance matrix or superiority claim was run. Final classification: `RESULT_A_PHYSICAL_HEADROOM_RESTORES_LIVENESS`.
- Prior incomplete Phase 4D-A/A-FIX artifacts and failed harness attempts remain preserved; results are in `results/phase4d_b/`.

## 2026-09-27 — Phase 4E-A frozen hybrid policy matrix (stopped at Stage B gate)

- Objective: compare OnDemand-K2, GuardedJIT-Oracle-K2, GuardedJIT+Lease-Oracle-K2, and FullSlackKV-Oracle-K2 on the unchanged Phase 4C first-boundary hybrid workload, using the frozen Phase 4D-B vLLM mechanism and 4-run counterbalanced order. No new vLLM source change was made.
- Frozen inputs: Phase 4C workload SHA256 `9b381336a70e73b0efbddbb79d2c88608c19663247e819c1e7a364efd2971105`; 8 jobs, 168,016 reusable tokens, 4,681 GPU blocks, 3,510-block reservation budget, P_SAFE=102 ms, K2. Command: `PYTHONPATH=/media/a6000/D/ZhaoBinHe/slackkv-phase2c-fix-sourceprobe:$PWD /media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python -m experiments.phase4e_a.run_phase4e_a`.
- Four fresh-server X-only runs were valid (median decode 3,633.51 ms; median TPOT 14.249 ms/token). All four fresh Stage A semantic smokes passed, including 8/8 requests, token counts, source conservation, K2, and background activity.
- Stage B repetition 1 positions 1 and 2 passed. Position 3 (GuardedJIT+Lease) completed all 8 resumes with exact token counts, source conservation, K2 maximum 2, no lease leak, no OOM/crash, but failed the frozen background gate: E03's last Agent FIRST_TOKEN was t0+4,664.73 ms, after X finished at t0+4,521.29 ms (143.44-ms gap). The driver stopped; position 4 and repetitions 2–4 were not run. No partial policy medians, paired deltas, or H1/H2/H3 claims were computed.
- Final classification: `EXPERIMENT_INVALID_MATRIX` (`RUN_INVALID_BACKGROUND`). The failure is preserved under `results/phase4e_a/failures/`; no X-length, workload, deadline, policy, or capacity tuning was applied. The incomplete matrix is not evidence of performance superiority or a component-ablation result.

## 2026-09-27 — Phase 4E-A-FIX background-safe hybrid matrix (stopped at Stage B)

- The only protocol/harness changes were a frozen 512-token background X (9500-token prompt unchanged) and X-dependent TPOT/combined-throughput formulas based on actual generated-token counts. The Phase 4C workload SHA, Phase 4D-B vLLM source diff, K2, P_SAFE=102 ms, deadlines, capacity, and four policy definitions were unchanged. Old Phase 4E-A samples were not reused.
- Four fresh X-only controls passed (median decode 7,447.455 ms, median TPOT 14.574 ms). All four fresh semantic smokes passed; their X-tail margins were 4,117–4,351 ms, above the frozen 1,000-ms protocol threshold.
- In the counterbalanced Stage B sequence, the first 15 runs passed, with 120/120 Agent requests completing and exact output-token counts, 15/15 source-conservation checks, and a minimum measured X-tail margin of 3,935.807 ms. The 16th run (repetition 4, position 4, GuardedJIT+Lease-Oracle-K2) reached eight Agent arrivals but zero scheduler admissions/first tokens/finishes before the 30-second client timeout. Three guarded promotions acquired physical leases and five attempts were capacity-rejected. The server remained alive without OOM or ERROR until forced shutdown.
- The matrix stopped at the first invalid run. **Final result: `EXPERIMENT_INVALID_MATRIX`**, for a request-progress failure in the leased, no-logical-reservation ablation, not for X duration. No policy medians, paired deltas, H1/H2/H3 verdicts, or performance-superiority claim were computed from the incomplete matrix. Compact diagnostics and repetition-level evidence are in `results/phase4e_a_fix/`. No later phase was run.

## 2026-09-27 — Phase 4E-A-RCA lease-induced waiting-head circular wait

- Frozen start commit `e829bee1fb6d45dcdfb7bdce395c5a10154cc892`, Phase 4C workload SHA256 `9b381336a70e73b0efbddbb79d2c88608c19663247e819c1e7a364efd2971105`, 4,681 GPU KV blocks, and K2 were preserved. A trace-only pinned-vLLM patch recorded the real `allocate_slots() -> None -> break` branch; it made no scheduler control-flow change.
- Sequential deterministic preconditioning reproduced the failed lease structure: E00/E01/E02 held 3,930 unique blocks, while earliest E04 (1,271 required blocks) was unleased. No-X U3 failed to progress in 2/2 fresh runs: E04 first allocator failure had 750 free blocks, `will_break_waiting_loop=true`, and 0 admissions/first tokens/completions or lease releases through 12 seconds.
- Three orthogonal no-X controls each progressed 2/2: H2 retained only E00/E01 leases (2,607 blocks), N3 prefetched the same E00/E01/E02 set without leases, and S3 leased the earliest E04 with E00/E05. Every control run completed 8/8 requests.
- With frozen 512-token X, U3 reproduced non-progress 3/3 (E04 first failure at 154 free blocks; server alive, no OOM/error, no lease release), whereas S3 progressed 3/3 and completed 24/24 requests. In all S3 runs, E04's exact lease released before its scheduler admission. Rate-limited U3 traces still showed E04 failing with 750 free blocks after t0+8 s; exact X finish timestamps were not independently persisted.
- Final classification: `RESULT_A_LEASE_HOL_CIRCULAR_WAIT_CONFIRMED` for this controlled large-prefix hybrid workload and pinned waiting-loop semantics. Instantaneous physical promotion fit does not by itself preserve a future resume progress path. No performance matrix, policy winner, or progress-reserve fix was run. Artifacts: `results/phase4e_a_rca/`.

## 2026-09-27 — Phase 4F-A progress-safe leased-promotion admission

- Added research-only PSA-1 to guarded leased native promotion: after exact physical fit, the pinned scheduler requires candidate allocation to leave the incremental blocks needed for the earliest unfinished resume, accounting for an existing or hypothetical matching lease. Ordinary capacity-guard requests retain their prior behavior; waiting-queue `allocate_slots() -> None -> break`, connector, K2, and lease handoff were not changed.
- Frozen Phase 4C workload SHA256 `9b381336a70e73b0efbddbb79d2c88608c19663247e819c1e7a364efd2971105`, GPU 4,681 blocks, logical budget 3,510, P_SAFE=102 ms, K2, and X512 were preserved. P0 micro gates passed 3/3: physical reject, E02 progress reject at free=2,073 / required=1,323 / post-candidate=750 < E04 reserve=1,278, and candidate-aware E04 admission with a 7-block incremental reserve.
- Deterministic P1 no-X baseline reproduced lease HOL non-progress (0/8 admissions); PSA recovered 2/2 runs and 16/16 completions. With X, PSA recovered 3/3 and 24/24 completions. E04-own-promotion control passed 2/2 and 16/16 completions.
- Real concurrent-JIT P4 without logical reservation passed 8/8 fresh-server repetitions and 64/64 completions, with two successful promotions every run and zero E04 HOL allocation failures. FullSlackKV+PSA1 P5 passed 4/4 and 32/32, with 12 total reservation skips, 12 capacity rejects, and zero progress rejects. Across completed runs there were zero PSA trace-invariant violations, final lease leaks, OOMs, or server crashes.
- A first harness attempt stopped at P2 because it counted nine `RESUME_ADMITTED` events despite eight distinct requests: E07 was legitimately re-admitted after preemption. The failed attempt is retained compactly; the corrected harness counts unique request IDs and the full gate sequence was rerun fresh. Focused tests passed 29/29.
- Final classification: `RESULT_A_PSA1_RESTORES_LIVENESS`. This is a liveness/correctness result for the frozen first-boundary hybrid workload, not a performance comparison or a claim of universal safety. Artifacts: `results/phase4f_a/`.

## 2026-09-28 — Phase 4F-B PSA-safe frozen hybrid performance matrix

- Starting commit `012dd5f20efabe2d5132540153be7f6991b81145`; frozen Phase 4C workload SHA256 `9b381336a70e73b0efbddbb79d2c88608c19663247e819c1e7a364efd2971105`. The vLLM mechanism diff fingerprint remained `e33055ec22f1d89c67342e71470134fdef3fda9dbbb5b7ac1e056e71676aab14`. Runtime: Qwen2.5-1.5B-Instruct FP16, TP1, GPU0, 4,681 KV blocks / 2-GiB GPU KV, 8-GiB external KV, K2, P_SAFE=102 ms, 3,510-block logical budget, X9500→512.
- Command: `PYTHONPATH=/media/a6000/D/ZhaoBinHe/slackkv-phase2c-fix-sourceprobe:$PWD /media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python -u -m experiments.phase4f_b.run_phase4f_b`. Eight fresh X-only controls (four before/four after) gave median decode 7,496.106 ms; first/last medians differed by +2.39%, below the 5% drift-warning threshold. Semantic smoke passed 4/4; all 32 Williams-ordered fresh-server runs and 256 Agent resumes completed with exact requested token counts, source conservation, X overlap, K2, and zero crash/OOM/lease or logical leak/PSA invariant violation.
- Median run-level P95 resume (A/B/C/D): 2,232.043 / 2,232.273 / 2,228.328 / 2,196.484 ms. Median total H2D: 4,594.188 / 4,869.594 / 4,594.188 / 4,594.188 MiB; B's median duplicate H2D was 275.406 MiB, versus zero for C/D. Median X slowdown: 1.079 / 1.101 / 1.095 / 1.094. D-vs-A paired median P95 change was -1.18% (bootstrap descriptive 95% interval -4.06% to +1.94%). H1 failed; H2 and H3 passed.
- B produced seven promoted remote waits across eight runs. C's joint lease+PSA configuration eliminated remote waits but made 18 PSA progress rejections. D skipped a median three reservations per run, had zero PSA rejections, and showed only a small latency difference from C while holding a slightly greater median lease memory-time. One E04 output SHA mismatch occurred in every run/policy; output lengths and cache source accounting remained exact. This is a diagnostic limitation, not a semantic replay claim.
- Final classification: `RESULT_B_VALID_MATRIX_MIXED_TRADEOFF`. The first complete PSA-safe performance matrix shows no preregistered 10% P95 advantage for FullSlackKV on this frozen hybrid first-boundary workload. No old incomplete-matrix latency samples, unsafe lease-without-PSA policy, predictor, 32-boundary replay, or next phase was used. Artifacts: `results/phase4f_b/`.

## 2026-09-28 — Phase 4G-A causal 32-boundary multi-turn replay (matrix stopped)

- Started from `f4b19bd0ee0b7f3d4fe4b4732b8e8bcc4b2499f4` and the unchanged Phase 4C hybrid workload SHA256 `9b381336a70e73b0efbddbb79d2c88608c19663247e819c1e7a364efd2971105`. The frozen eight sessions supply 40 turns and 32 causal boundaries per run, 705,728 aligned reusable tokens / 44,108 blocks; the 19,297.25-MiB one-full-crossing sum is not an H2D minimum. Only trace-only vLLM source-readiness and per-resume source-accounting logs were added; serving algorithms were not changed.
- The natural lifecycle probe observed both CPU source-ready and source-not-ready boundaries. No resume prefix was pre-materialized. Final fresh Stage L A/D smoke runs completed 40/40 turns and 32/32 boundaries each, with K2 maximum 2, 10/32 ready and 22/32 not-ready source events, and X-tail margins of 5,795.88 / 5,890.16 ms, above the 3,000-ms gate. A D smoke diagnostic first failed because the harness incorrectly assumed one source trace per request; repeated trace events arose after request preemption. The parser was corrected to retain initial source attribution and account for later external restores, and the full smoke was rerun.
- Four fresh pre-matrix X-only controls passed. Rep1 A and D were valid (40/40 turns each). Rep2 D completed 40/40 turns and 32/32 boundaries with K2 maximum 2, no crash, OOM, lease/logical leak, or capacity-invariant failure, but X finished **25,286.80 ms before the last Agent first token**. This failed the frozen measured-run background-overlap gate. The driver stopped at 2/16 valid measured runs; the remaining runs and post-matrix X-only controls were not attempted.
- Final classification: `EXPERIMENT_INVALID_MATRIX` (`RUN_INVALID_BACKGROUND`). No partial A/D medians, paired deltas, bootstrap intervals, or H1/H2/H3 claims were computed. The fixed 2,048-token X, workload, deadlines, K2, PSA-1, and service budget were not changed to force completion. Results and compact diagnostics are retained under `results/phase4g_a/`.

## 2026-09-28 — Phase 4G-A-FIX sustained-background 32-boundary replay (matrix stopped)

- Reused the frozen Phase 4C eight-session workload (SHA256 `9b381336a70e73b0efbddbb79d2c88608c19663247e819c1e7a364efd2971105`) and unchanged Phase 4G-A vLLM mechanism diff (SHA256 `f0f8e327cc7ff4eb1797852a24e56e0e7f4a77b62e0f9439e88405fd5b470ae4`). The fixed-work background was four 2,048-token chunks, 8,192 generated tokens total, with 50-ms maximum inter-chunk idle gap. No scheduler/mechanism change was made.
- Background calibration passed 4/4, pre-matrix background-only controls passed 4/4, and lifecycle smoke passed 4/4 (40/40 turns and 32/32 boundaries per run). The first 14 measured runs also passed these workload/background checks.
- Measured repetition 8 / position 2 (`FullSlackKV+PSA1-K2`) failed progress before the 30-second watchdog: the server remained alive without OOM or engine error, but 2,572,201 `GATE_BLOCKED` events accumulated; only 9 distinct requests reached first token and finish despite 17 resume arrivals. At capture, K2 still reported two pre-first owners whose requests had already finished, with no running, waiting, or scheduled requests. This supports a stuck/stale admission lifecycle state; the exact transition was not proven. The run is `RUN_INVALID_NONPROGRESS`.
- The matrix stopped at 14/16 measured runs; the final OnDemand run and four post-matrix background-only controls were not run. No partial policy medians, paired comparisons, or performance claim was computed. Final result: `EXPERIMENT_INVALID_MATRIX`. This is not evidence of trace-data failure or background under-duration; those gates passed for completed runs. Artifacts: `results/phase4g_a_fix/`.

## 2026-09-28 — K2 post-first re-admission lifecycle repair

- Start commit `992561a661effdb96cd4c5d8e5357002885ae26f`. Direct failed-run evidence shows E03/E06 re-admitted after17/6 outputs, reacquiring gate ownership that first-token-only release could not remove. Four minimal scheduler changes restrict acquisition/checking to zero outputs, permit idempotent pre-first reentry, and clear ownership/bookkeeping at terminal cleanup. No PSA/lease/reservation/connector/workload parameter changed.
- Command: `python -u -m experiments.k2_lifecycle_fix.validate` using the existing pinned environment (full command in results). Actual production-AST tests changed from7fail/1pass on baseline to8/8pass; existing native capacity tests passed3/3. Baseline/fixed scheduler and full vLLM diff hashes and incremental patch are retained.
- Four fresh GPU runs in fixed D/A/D/D order passed:160/160 turns,128/128 causal boundaries, K2 maximum2, zero gate violations/final logical or lease leaks/OOM/crash. Five post-first re-admissions actually occurred (1–12 outputs), all without gate ownership. Frozen8192-token background completed in each run.
- CPU helper timing is diagnostic only, not an end-to-end performance bound. This repair validation is not a replacement performance matrix. Historical invalid results remain unchanged; no samples were combined. Artifacts: `results/k2_lifecycle_fix/`. Technical result: PASS; no next phase started.

## 2026-09-28 — Phase 4G-A-FIX full matrix rerun after K2 lifecycle repair

- Reran from repair commit `84c75c880fddf5e98cefb70d0b99da43cfe9cb5b` into a new output directory. Frozen Phase 4C SHA256 remained `9b381336a70e73b0efbddbb79d2c88608c19663247e819c1e7a364efd2971105`; repaired vLLM diff fingerprint was `4ec03c5f7bcb331485becea471d6d46b28b2b9ec3f8b0a2997f62e7e8bf15618`. Protocol remained BG-CAL4, background-only pre4, measured16 in the original order, and background-only post4. No old measurements were reused.
- All background calibration and controls passed:4/4 calibration,8/8 X-only,16/16 measured background coverage. All measured runs were valid:640/640 requests,512/512 boundaries, K2 valid16/16, zero crash/OOM/logical leak/lease leak/capacity invariant failure. Post-hoc scheduler-trace audit found four post-first-token readmissions; zero reacquired gate ownership and there were zero lifecycle violations.
- Median boundary P95 was6,434.53 ms OnDemand and6,413.48 ms FullSlackKV. Paired FullSlackKV change was+1.38% (range−19.31% to+10.31%); cumulative TTFT changed−1.71%, session E2E changed+1.29%. H1 failed, H2 and H3 passed. Final result: `RESULT_C_MULTITURN_NO_MATERIAL_LATENCY_VALUE`; this is a mixed measured outcome, not a liveness or dataset failure and not evidence of statistical significance.
- Artifacts are under `results/phase4g_a_fix_k2repair/`. Historical Phase 4G-A-FIX invalid data remains intact. This rerun closes that matrix; no subsequent experiment was started.

## 2026-09-29 — Phase 4G-B source-readiness/headroom RCA (matrix stopped)

- Started from `3c887a710cca06142a00c4ca35ca152e7b590bc0` with the frozen Phase 4C workload SHA256 `9b381336a70e73b0efbddbb79d2c88608c19663247e819c1e7a364efd2971105`. Trace-only probes passed their final instrumentation smoke (CPU duty 3.62%). The first measured M0 OnDemand/no-background run passed.
- The next run, M1 FullSlackKV+PSA1/no-background, did not complete. E01's source-ready event and probe showed the full 1,330-block prefix already locally available; the JIT candidate was nevertheless admitted at 1,563 free blocks with a 7-block incremental PSA reserve. The same native prefetch then produced 253,045 connector-only steps and no completion. Only E01/E03 resumes arrived and neither was scheduled. E03's 1,306-block candidate physically fit at 1,563 free blocks, but PSA rejected it to preserve the protected E01 progress footprint of 1,337 blocks.
- The measured run's worker future timed out at 300 seconds; controlled shutdown then emitted EngineDeadError diagnostics. The server log showed no preceding OOM/independent engine crash. The remaining ten runs were not attempted. Final status: `INCOMPLETE_INVALID_RUN`; no complete-matrix aggregation or A–E classification is claimed. The observation is suggestive of a source-readiness/promotion/headroom interaction, but one failed run cannot isolate a sufficient root cause. Compact diagnostics are in `results/phase4g_b/`; the large temporary raw log was not committed.

## 2026-09-29 — Phase 4G-B-FIX guarded-prefetch liveness repair and fresh full RCA matrix

- Started from `0a32b037641449462699cbe1dc7edfb199369a4d`; the frozen Phase 4C workload SHA256 remained `9b381336a70e73b0efbddbb79d2c88608c19663247e819c1e7a364efd2971105`. The previous Phase 4G-B `INCOMPLETE_INVALID_RUN` and its artifacts were preserved, not merged into this run.
- A fresh pre-fix diagnostic found repeated `allocate_slots(None)` for one guarded native prefetch despite an earlier capacity ADMIT: 1,564 free blocks, 759 reserved by in-flight scheduling, and only 805 schedulable blocks. Connector-first then kept ordinary prefills from progressing. The minimal scheduler repair returns a zero-transfer `capacity_rejected` only for guarded native prefetches at this allocation failure, before connector load; real resumes use OnDemand fallback. Normal requests and historical unguarded prefetch semantics remain unchanged. In one new measured run the branch returned after 149.75 ms, with zero transfer, no lease, and 40/40 turns completed.
- The fresh instrumentation smoke passed (probe CPU duty 4.02%). The complete frozen matrix passed 12/12 measured runs: 480/480 turns, 384/384 dynamic boundaries, K2 at most 2, no final lease/logical leaks, OOM, or server crash. BGv2 validation passed in all eight background runs. Final directed tests: 44 repository tests plus 15 vLLM source tests passed.
- The frozen A–E classifier returned `RESULT_D_PSA_HEADROOM_BLOCKER`: in the M2 OnDemand/BGv2 shadow observations, 128/128 source-in-time boundaries had a sampled physical-fit instant, but 128/128 lacked a sampled contiguous 20-ms full-prefix PSA-safe window. This is a sampled-window mechanism classification, not a continuous-time impossibility proof. Per-boundary probe gaps reached 171.64 ms, so shorter missed windows remain possible. No performance winner or latency claim follows. Artifacts: `results/phase4g_b_fix/`.

## 2026-09-29 - Phase 4G-C source recheck / PSA / allocator-truth RCA (matrix stopped)

- Started from `c1761e980b16eec5d5381e8cf0a6edd4d245bf09`; frozen Phase 4C workload SHA256 remained `9b381336a70e73b0efbddbb79d2c88608c19663247e819c1e7a364efd2971105`. Runtime and controller constants stayed frozen. No new vLLM serving-algorithm change was made.
- C0/C1 smoke passed 2/2. The measured plan contained eight fresh runs; six valid runs completed. Rep4/C1 was attempted and completed all 40 turns / 32 boundaries, but failed pause fidelity; Rep4/C0 was not started. Thus seven runs were attempted, with 280 turns / 224 boundaries executed; only 240 turns / 192 boundaries are from valid measured runs. No partial C0/C1 aggregation or A-E causal classification is reported.
- Frozen pause limits were P95 absolute error <=10 ms and maximum <=30 ms. Rep4/C1 measured P95 12.211563 ms and max 18.885849 ms; E01:b1, E04:b2, and E04:b1 were the three P95-threshold violations. The evidence establishes client-side pause-injection jitter, but not a specific OS scheduler cause. The matrix stopped rather than relaxing the gate.
- In the failed run, K2 max pre-first was 2; BGv2 generated 8,192 tokens and outlasted the last Agent first token by 97,588.074 ms; server alive, no OOM/crash, and zero logical/physical lease leaks. A 768-token source-accounting shortfall was explained by pinned prompt-only offload behavior and remains a separate measurement limitation.
- Targeted command: `/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python -m pytest -q tests/test_phase4g_c.py` (49 passed; an optional missing `nvidia_cutlass_dsl` site-package warning did not affect tests).
- Result: `EXPERIMENT_INVALID_MATRIX`. The observed pause gate failure prevents any source visibility / PSA / allocator-truth causal conclusion. Compact records are under `results/phase4g_c/`; bulky attempt JSON is retained on the server workspace but excluded from Git.

## 2026-09-30 — Phase 4G-C-FIX native absolute-deadline timing repair

- Started at `c10a6c7ac69e41b444d00497c5d258da6bfc71b0`. The frozen Phase 4C workload SHA256 remained `9b381336a70e73b0efbddbb79d2c88608c19663247e819c1e7a364efd2971105`; pinned vLLM remained 0.23.0 with unchanged working-diff fingerprint `bfd789e6849715196435f18a57ed9bbefae7b4b9d3a78d69115614f27d1542e8`.
- Replaced the pause waiter's Python sleep-plus-busy-spin with Linux `CLOCK_MONOTONIC` absolute `clock_nanosleep` (`CLOCK_NANOSLEEP_ABS`). No Python busy-spin remains; the 0.5-ms switch interval, workload, pauses, deadlines, thresholds, controller, and SlackKV/vLLM mechanisms stayed frozen. The 100-sleep self-test passed: absolute wake-overshoot P50/P95/max 0.111/0.214/0.219 ms.
- Full-load Stage T passed 6/6 fresh runs (192/192 boundaries); every run met P95 absolute pause error <=10 ms and max <=30 ms, and aggregate P99 absolute error was 4.123 ms. A single Stage T maximum of 12.342 ms was almost entirely sleep overshoot, but remained below the frozen per-run 30-ms maximum; dispatch delay stayed small. The calibration runs were excluded from mechanism aggregation.
- The fresh measured matrix passed 8/8 runs: 320/320 turns and 256/256 boundaries. C0 absolute pause-error P95/max was 0.507/2.533 ms; C1 was 1.044/6.242 ms. C1's observed jitter was somewhat higher descriptively, without a winner claim. Crashes/OOM/K2 violations/logical leaks/lease leaks were all zero; instrumentation duty gates passed.
- Final mechanism classification: `RESULT_D_GUARD_ALLOCATOR_MISMATCH`. C0/C1 had 6/40 vs 0/127 stale-protected-visibility selections and 1/40 vs 4/127 candidate=protected selections; median PSA reserve was 1,336 vs 1,359 blocks. No promotion completed in either arm. The legacy physical guard agreed with exact allocator preflight in 61/123 attempts (49.6%), with 62 false rejects and no false admits; this meets the frozen guard-mismatch classifier. Source-hit coverage was short by 768 tokens/run (6,144 total), fully explained by pinned `offload_prompt_only=true`; unexplained shortfall was zero. No performance conclusion follows.
- Tests: `tests/test_phase4g_c_fix.py` passed 14/14. Artifacts and exact timing decomposition are under `results/phase4g_c_fix/`; the prior invalid Phase 4G-C outputs remain untouched.

## 2026-10-09 — Phase 4G-D protected-head persistence, retry opportunity, and frontier RCA

- Started from `7bada3599076fa4fbf8e0d73120c37dbe6285ca0`; retained the frozen Phase 4C workload SHA256 `9b381336a70e73b0efbddbb79d2c88608c19663247e819c1e7a364efd2971105`, 2-GiB GPU KV / 4,681 blocks, 8-GiB external KV, BGv2, and `CLOCK_NANOSLEEP_ABS`. The four fresh C1 shadow-only runs passed: 160/160 turns and 128/128 boundaries, zero crash/OOM/K2 violation/logical or lease leak/unsafe release. No actual retry, frontier change, or PSA change was made.
- Reanalyzed Phase 4G-C-FIX's 123 guard observations without zero-filling missing allocator fields: 54 exact-known, 69 missing; the 54 known rows agree 54/54 with zero false-admit and zero false-reject. The prior `RESULT_D_GUARD_ALLOCATOR_MISMATCH` is not supported by the corrected analysis; its historical summary remains unchanged.
- Among 126 selections, a terminal head remained protected in 121 (96.0%). There were 128 poison episodes (median/P95 4,403.8/6,849.1 ms). Five long pre-resume intervals received exact retry probes; none contained a continuous >=20-ms actionable window, so retry recoverability was 0 observed episodes, not proof retry can never help. Four of seven exact-known F0-PSA-blocked, allocator-fit selections were shadow-recoverable under F1/F2; F3 also recovered four as an unsafe upper-bound counterfactual, F4 none. Final classification: `RESULT_B_PROTECTED_FRONTIER_POISONING_DOMINANT`.
- Exact allocator truth covered 46/46 eligible selections; 11 fit and 35 did not. On 61 protected-need rows with exact paired allocator fields, median PSA-1 need was 1,340 blocks versus median instantaneous allocator need 1,292; median paired gap was 0. For the 31 exact-known rows with local protected cache, the median allocator need was 50 blocks and median gap 1,307 blocks; local-positive ownership snapshots had median 1,306 owned blocks and zero median evictable blocks. These are instantaneous observations, not a future-liveness guarantee or authorization to replace PSA-1.
- The source-only recheck fixed stale visibility in the prior phase, but did not produce promotion success here. The evidence prioritizes protected-frontier persistence as a shadow-recoverable blocker ahead of PSA-2; it does not establish that F1/F2 are safe, and does not justify PSA-2 or partial promotion. Timing pause-error aggregate P95/max was 0.758/1.084 ms; diagnostic CPU duty stayed 5.30–5.73%. Tests: `tests/test_phase4g_d.py` passed 73/73. Artifacts: `results/phase4g_d/`.

## 2026-10-09 — Independent architecture reassessment (offline)

- Baseline: `e2e904c2ebe344f1d4dd8cd347fb6fabd246a6f8`. New user handoff authorizes independent research recommendations; no serving policy, workload, historical result or GPU environment changed.
- Audited the specified Phase 2/3/4 reports, controller/analysis code, saved runtime deltas and key raw CSV. Commands: `python -m scripts.research_reassessment`; `python -m pytest -q tests/test_research_reassessment.py --basetemp=.pytest_cache/reassessment-validation-1` (14/14 PASS).
- Recomputed 512 boundaries from the complete 16-run Phase 4G-A-FIX-K2repair matrix. OnDemand observed remote wait P50/P95 run-median values are 58.87/83.96 ms. At the original P95 boundary, nonremote pre-admission median is 6316.89 ms. This interval's blocking causes remain unidentified.
- Fixed-observed-path deletion of all pre-first-output remote-wait intervals changes OnDemand P95 by only 0.905% median (0.775–1.097% range); even additionally deleting all post-admission service changes it by 1.393% median. These are conditional opportunity diagnostics, not a rescheduled-system causal upper bound or a new benchmark. Original paired D/A P95 change recomputed as +1.382%.
- Confirmed the prior missing-value correction (123 observations, 54 known agreements, 69 UNKNOWN), exact eligible allocator 11 fit/35 no-fit, and four shadow recoveries among seven exact-known F0-blocked selections. The classifier's Frontier-dominant label is not a proof of dominant overall-latency causality.
- Compared three architectures; recommend validating joint demand admission/restoration opportunity before changing PSA/frontier. Report includes safety invariants, related-work overlap, minimal falsification plan, kill criteria and missing measurements. No new architecture was implemented or GPU trial started.
- Artifacts: `docs/RESEARCH_REASSESSMENT.md`, `scripts/research_reassessment.py`, `tests/test_research_reassessment.py`, `results/research_reassessment/`. Input SHA256s and per-boundary/run/session diagnostics are preserved. The initial direct-script module-path failure and sandbox temporary-directory fixture failure were corrected without changing scientific definitions.

## 2026-10-09 — Autonomous round 1: admission diagnosis, rejected backfill, execution baseline and correctness control

- User granted autonomous research/implementation/GPU iteration within 30 minutes per run and 4 GPU-hours per round; repository authority header updated while preserving historical rules.
- Eight full BGv2 lifecycles valid (40 agent requests / 32 resumes / 8192 background tokens each), plus one pre-model import failure preserved. Conservative cost 0.400327 GPU-hours, longest 389.35 seconds, GPU0 only; owned servers stopped.
- Actual allocator diagnostic: 28/32 resumes had observed pre-admission capacity failure, all 253 failed samples FULL_SEQUENCE with reservation 0. This is sample evidence, not duration attribution or whole-queue causal classification.
- Bounded capacity backfill implemented and rejected at frozen gates: P95 +16.97%, session median +3.17%, BG wall +2.87%, H2D +3.22%; post-first preemption/restoration observed. Default remains disabled; no frontier/PSA deletion.
- Compiled/Graph OnDemand vs eager, exactly matching 40 inputs in the reverse pair: P95 -15.81%, session median -29.94%, BG wall -48.92%, mixed generated throughput +95.77%; all eight sessions improved. First pair has two drifted future inputs and is excluded from exact-input matched aggregation. No statistical-significance or novel-algorithm claim.
- Non-invariant cold requests reproduced output variation with actual local/external hit 0. Invariant control: 10/10 cold serial/concurrent replays matched the corresponding two sensitive resume outputs exactly. This is bounded correctness evidence, not universal cache/promotion safety.
- Compiled Full vs exact-input matched compiled OnDemand: single-run P95 +16.08%, session +0.40%, BG wall +0.55%; 10 promotions all rejected (9 capacity, 1 progress), 0 successes/leases. Do not expand current JIT/Frontier matrix from this screen.
- Checks: 22 analysis/state/archive/reassessment tests and 15 isolated vLLM native/connector-first/gate-capacity tests passed. Raw gzip logs, runtime baseline/deltas, fingerprints, token/oracle records, commands and UNKNOWN preserved in `results/autonomous_admission/`. Reports: `docs/AUTONOMOUS_RESEARCH_LOG.md`, `docs/AUTONOMOUS_RESEARCH_REPORT_20261009.md`. Source/threshold/workload definitions were not changed to select results.

## 2026-10-09 — Autonomous round 2: capacity interventions and two rejected scheduling architectures

- Started at e239088b3891396cde654669d74020e8fce82ecd, no upstream increment; server historical untracked data retained. Seven valid lifecycles (280 agent calls, 224 resumes, 57344 BG output tokens), one pre-workload fixed-pool validator failure preserved.
- Compiled OnDemand / Qwen2.5-1.5B FP16 / K2 / CPU 8 GiB / same actual inputs: 2/4/8 GiB P95 5199.92/1163.51/445.32 ms, capacity-failed resumes 31/9/0 of 32. Resource increases are not algorithm innovation. Single runs, varied initial issue order, no significance claim.
- Continuation credit: same-source 2-GiB matched screen P95 -39.90%, session median +6.08%, BG wall -0.16%, actual H2D -1.44%; fails session gate, stopped.
- Two-call soft epochs (max idle 1 second, no physical pin): matched screen P95 +106.80%, session +14.37%, actual H2D -38.60%, BG wall -0.08%; fails P95/session gates, stopped. Final epoch programs 0, preemption 0. Arrival-to-first allocator at the tail is 9720.85 ms; suppressed capacity checks do not imply eliminated contention.
- Commands: existing venv Python -m experiments.autonomous_capacity.run with recorded --name/--kv-gib/--trace/--runtime-source/--replay-inputs and opt-in candidate flags; exact observed commands in results/autonomous_capacity/commands.sh. Recompute with -m experiments.autonomous_capacity.summary and .audit.
- Budget: 0.2332685893 GPU-hours including failed startup/load/compile/exit, longest 130.45 s, GPU0 only; all owned servers stopped. 40 local tests and two stages of 15 CPU-only native/connector/gate regressions passed.
- Audit: 87 original export hashes and independent P95/session arithmetic verified; actual H2D/source one-block discrepancy and absent 8-GiB counter preserved, all D2H 5172.5625 MiB. Output hashes differ; no new invariant oracle, content/quality and owned memory-time UNKNOWN. All raw gzip evidence and failures retained.
- Stop condition B in bounded tested scope, no broad scheduler impossibility claim. Results: results/autonomous_capacity/; report: docs/AUTONOMOUS_RESEARCH_REPORT_ROUND2_20261009.md. Commit recorded in Git history; final goal unchanged, failed prototypes default disabled.

## 2026-10-09 — Autonomous Round3: fork restoration falsification, exact oracle and legacy regression

- Baseline7519bda64de6f2c48051b17f60f8e21c83be564e; independent runtime copy, default-off allocation snapshots only. Qwen2.5-1.5B FP16/TP1/APC/Compiled OnDemand, CPU8GiB, GPU0 A6000. Synthetic3episodes×8branches×2turns; fixed first/second contexts24832/25024,128outputs/call, BG3×8192prompt/1024outputs. Not production trace/quality.
- Four default controls: cold4GiB/K8 workflow median4360.56ms, TTFTP95709.27ms; decode4GiB/K8 4134.64/480.42; decode8GiB/K8 4068.17/495.85; decode4GiB/K2 4154.34/474.15. All H2D2016MiB,D2H2310MiB, one external root/episode, preemptions0, final running/waiting0. No candidate and no cross-condition algorithm gain.
- Existing native `_blocks_being_loaded` explains single materialization; first cold-batching explanation corrected after decode diagnostic. First run failed after one episode due obsolete helper;54.31seconds retained. Retry source_delta derived zeros INVALID; full raw events/counters prevail.
- Invariant correctness run separate:16/16cache-salted cold-prefill inputs exact128tokens, raw local/external hits0. Legacy40input/32resume/BG8192replay valid, P95Resume4853.24ms, session median23878.57ms, BGwall67764.82ms,28/32observedcapacityfailures. Matched actual Round2 inputs, no speedup attribution across source/order changes.
- Screening/model: unique fork<1GiB; root-only full attention streaming180.39GB/256steps gives6.86sec conditional DMA cost using old Phase0 reference. Peer-in transfer time advantage~0.333%, not overall gain. Larger7B config feasible on paper but not cached;20sHFprobe timeout, no download/environment changes.
- Validation:52localtests,15CPU-only runtime tests after correcting PYTHONPATH (first collection errors preserved);101original hashes, identical51actual inputs across five completeforkruns, source/transfer and critical-branch arithmetic verified. Raw server data intact. Seven execution ledger entries,0.20048675264813934GPUh,max260.33sec,GPU0only,final GPUs19MiB/0%,no compute apps.
- Stop B within scope; no accepted new algorithm. Report docs/AUTONOMOUS_RESEARCH_ROUND3.md; protocol, commands, full gzip raw inputs/outputs/server logs, source snapshots, oracle and audit in results/autonomous_fork/.

## 2026-10-09 — External review of autonomous rounds 1–3 (CPU evidence audit)

- Reviewed commit: 61139e310a5976cf4311e8bbc4badf82f56439af. No new serving implementation or GPU experiment.
- Objective: independently inspect the reported stopping decisions and identify the next measurable opportunity before proposing another algorithm.
- Actual command: `python review/scripts/review_round3_evidence.py --output review/results/research_review_20261009/spotcheck.json` in the review workspace. Repository reproduction: `python scripts/review_round3_evidence.py --output /tmp/slackkv_round3_spotcheck.json`.
- Scope: representative fork-decode-control-4 raw requests, full saved log, episodes and counters, plus two saved runtime source fingerprints; not the complete 101-file audit.
- Checks passed: 48 child/3 background calls and required output counts; three workflow durations match; actual/source H2D both 2113929216 bytes; workflow median 4134.640602 ms; child TTFT P95 480.419642 ms; runtime fingerprints and per-call timestamp decomposition agree.
- Findings: first-turn client-to-engine / engine-first-to-client medians 90.79 / 115.24 ms need attribution. Three own remote-wait pairs do not instrument follower lookup deferral. Existing singleflight negative conclusion remains supported.
- Artifacts: docs/RESEARCH_REVIEW_AND_NEXT_ROUND_20261009.md and results/research_review_20261009/spotcheck.json. No performance improvement, full runtime correctness, or new algorithm claimed. GPU cost zero for this review; next-round execution remains the remote Codex's task. Commit identity is available in Git history for this entry.

## 2026-10-10 — Autonomous Round4: gate baseline, ownership diagnosis, physical retention control

- Baseline main d403714; protocol docs/BASELINE_AND_OPPORTUNITY_ROUND4.md frozen before runs. Compiled Qwen2.5-1.5B FP16TP1/GPU0, CPUKV8GiB, GPUKV2/4GiB, same40actual inputs and8192BG outputs. Extra resource tracing off in main screens; existing patched runtime logs retained.
- Twelve gate screens:3 independent paired processes/budget, alternating order. Median paired session off/K2−0.37%(2GiB),−0.63%(4GiB); P95−1.51/+12.63%, BGchanges small. No overall acceptance or novelty; all pairs/eight-session timing saved.
- Diagnostic failure resource-off: bad external-token branch indentation, EngineDeadError;51.598seconds and failedsource retained. Repaired commonv2 runtime disabled AST equivalence passes, OFF/ON valid. Observer/session+4.89%,P95+2.29% in one pair; not a clean overhead estimate.175 failed fit snapshots all reservation0, median390block deficit,30/32resume failures,173single/2double physical completion witnesses. Full cause-duration and owned memory-time UNKNOWN.
- Invariant safety smoke before performance:2048real local-hit prefix,16/16resumed/cold tokens exact, actual cold0hits, idle expiration before next request, engine cancellation reached, pins0. Not all40 quality/remote-DMA interleavings.
- Fixed1sec TTL/age programorder/pin control at2GiB: matched OFF/order/pin full work; session+6.87/+20.09%,P95+4.42/+22.20%. pinacq32/release32,28HOLfallback/4TTL; actualH2D19276.25/19276.25/19281.5MiB,D2H5172.5625each. Stop both, no reverse triplet or threshold tuning.
- Independent17run audit matches actual FG/BG hashes and output lengths/P95/session arithmetic;175physical pool partition checks pass.44local and15native CPUtests pass after documented CPUsetup/tempfixture failures.271logical original hashes preserved by lossless gzip, actual runtime55source snapshots included.
-19GPU ledgerentries:18VALID/1FAILED,0.5462201082GPUh,max111.96seconds,GPU0only,final GPUs19MiB/0%,no computeapp. Raw failures and all performance negatives preserved. Full reportdocs/AUTONOMOUS_RESEARCH_ROUND4.md; commands and ledgerresults/autonomous_round4/.

## 2026-10-10 — External CPU review of Round4 negative screens

- Reviewed commit: 465c4d2240b34210e93d6d861e3ec0f9fc830774; no new GPU execution.
- Read Round4 report/protocol, retention and resource-witness code, diagnosis/control summaries. Independently recomputed full session E2E from the three control raw run.json.gz files and gate paired effects from metrics.csv.
- Verified control medians: 23746.274183 / 25377.636648 / 28517.682129 ms. Gate median paired effects: -0.367807% at 2 GiB and -0.629483% at 4 GiB. All eight pin sessions slower in this one triplet; no significance/stable effect claim.
- Raw pin log: 32 acquire/release pairs, 28 HOL fallback +4 expiry, no REAL_REQUEST_ALLOCATED release. Hold times min/median/max 1.6292/274.2577/1006.6065 ms; final pins zero. No handoff event does not imply zero ordinary local hits after fallback.
- Reproduction command and source paths: docs/RESEARCH_REVIEW_ROUND4_20261010.md section 5. Scope excludes full archive/GPU correctness revalidation.
- Decision recommendation: pause further same-workload JIT/gate/fixed-TTL/order tuning and first assess one evidence-backed research opportunity. No serving changes or new experiment authorization beyond existing AGENTS scope. Commit identity: Git history for this entry.
## 2026-10-10 — Public workload CPU opportunity audit

- Gitbaseline1c9c7da, no serving changes. Source semianalysisai/cc-traces-weka-062126 at23f152f6f0f9399a85901b89a6458def0ef16729. Bounded first16complete rows/64MiB acquisition yielded15/46,447,908bytes; SHA256d078503145b09533298928438fe186bbbc40c726901f920872510df04cd47994. Producer-filtered/file-prefix sample, not population estimate.
- Commands: `python -m experiments.opportunity_decision.fetch --output .tmp/opportunity_trace_prefix.jsonl`; `.analyze --input .tmp/opportunity_trace_prefix.jsonl --output results/opportunity_decision_20261010`; `.audit --input .tmp/opportunity_trace_prefix.jsonl --results results/opportunity_decision_20261010`; `pytest -q tests/test_opportunity_decision.py`. Local E:/Software/Anaconda/python.exe; source already present hash verified by final fetch.
- 2661requests/47explicit groups/892nested; inputproxy median118208/P95624320, output460/5053; completed-input-history prefix fraction0.97825504, cross-container0.03464016. No physicalcachehit or new speedup claim. Four tests and92deterministic direct LCP sample comparisons/aggregateCSV arithmetic pass.
- Acquisition anomaly: servermetadata urllibtimeout25seconds, localHTTPS succeeded. GPUcost0; no modeldownload/installation/fullcorpusdownload. Original source under ignored.tmp; compactCSV/summary/manifest/script committed. ActualDAG/tool-only latency/hostloss/compute/ownedmemorytimeUNKNOWN. Decision reportdocs/RESEARCH_OPPORTUNITY_DECISION.md.
- 2026-10-10 host_prefix：CPU公开15轨迹2661请求，7容量×4策略74,508records；cost-tail全部REJECT，prefix-LRU写入模型显著减少。CPU差分320native请求/19测试通过；初次慢实现中断与原源码保留。
- host_prefix_cost：calibration1FAILED(reset接口空体)、calibration2COMPLETE(36calls/379248GPUhit核算)；full1/frontier1、frontier2/full2四run严格匹配12inputs，compiled相同pool1,355,712tokens；P95-17.86/-19.88%，BG-8.08/-9.15%，D2H-94.40%、总迁移-67.54%，H2D8.01倍。
- oracle1执行完成但INVALID(salt字段放错，冷参考GPUhit28720)；oracle2修复后9360external恢复与0命中冷prefill16tokens一致。指标标签初始缺失导致首次分析失败，明确freshzero-transfer起点后结束计数/token来源严格重算；失败全部保留。
- GPUledger8run总0.13880706816885827GPUh，含启动/编译/验证/清理，GPU1未用；单run最大90.52秒。CPU/合成pressurediagnostic不冒充生产trace，整体SessionE2E/fairness/SOTA优势UNKNOWN。原始数据、复现命令、源版本见results/host_prefix_cost_20261010/与三份HOST_PREFIX_*_PROTOCOL.md。

- 2026-10-10 review_host_prefix（只读CPU审查，无新GPU实验）：读取18ad6b7四run requests/backgrounds gzip和before/after.prom，独立以Python复算时间戳、12个记录输入哈希配对、输出数量、token source守恒与H2D=external×28672；读取oracle2原始输出/三快照。续接P95改善17.86/19.88%、后台中位改善8.08/9.15%与报告一致。未重新计算inputs.json.gz哈希、未全量审查服务器日志。结果results/research_review_host_prefix_20261010/spotcheck.json，交接docs/RESEARCH_REVIEW_HOST_PREFIX_20261010.md；现有experiments.host_prefix.analyze_gpu可重算完整归档。

- 2026-10-10 host_prefix_followup_20261010：Compiled Qwen1.5B/native CPU connector，16GPUrun：A两个种子forced-cold partial/NoHost；B两个交替会话fair/full/nohost；C四独立闭环、no-reset，GPU-hot/256MiB、2GiB/512MiB、2GiB/8GiB三组各3方法。固定所有输入/输出，30CPU测试，原始payload哈希/来源/H2D/pool独立复算通过。A P95−17.93/−20.09%；B−8.40/−8.90%未过10%；C pressure E2E median−8.88%/BG makespan−5.84% vsNoHost，未过10%。Hot无机制收益，ample等效Full，负对照个别会话+8.52/+5.46%保留。无新多会话GPU内容oracle。16正常完成；本机工具失败/INVALID空摘要保留、空数据分析修复。新增GPU0.3645630389h含启动/清理，累计1.2500769679h；最长117.63s。复现python -m experiments.host_prefix.followup_queue --stage <a|b|hot|pressure|ample> --name-prefix <unique-> --source /media/a6000/D/ZhaoBinHe/slackkv-autonomous-round4-source。完整报告AUTONOMOUS_RESEARCH_HOST_PREFIX_FOLLOWUP_20261010.md，原始results/host_prefix_followup_20261010/，定义commit158778f。

- 2026-10-11 外部审查c62d9f6：独立复算全部16run原始requests与指标快照，确认单会话胜NoHost、多会话前缀保留及受控无reset约8.88%会话改善；未通过历史10%门槛，hot/ample单会话退化警示保留。新增GPU消耗0，未重跑payload哈希或完整GPU测试。下一自主目标为验证合理GPU/pinned-host预算下的部署价值，不追阈值调参、不规定细碎实验步骤。审查与范围docs/RESEARCH_REVIEW_FOLLOWUP_20261011.md，复算results/research_review_followup_20261011/spotcheck.json。


- 2026-10-12 deployment_opportunity（10-11启动）：基线834ed5a。python -m experiments.deployment_opportunity.analyze --input .tmp/opportunity_trace_prefix.jsonl --expected-sha256 d078503145b09533298928438fe186bbbc40c726901f920872510df04cd47994 --output results/deployment_opportunity_20261011；另执行capacity模块和38项pytest。27独立namespace累计/活跃/未来保留union最大2760192/769344/1332032 proxy tokens，不映射Qwen性能。官方32B配置和权重metadata资源估计，未下载权重。pin32未启动：研究盘挂载点消失，环境/账本不可访问；保留环境证据，不改挂载。新增GPU0，累计1.2500769679056248h。报告AUTONOMOUS_RESEARCH_DEPLOYMENT_OPPORTUNITY_20261012.md，结果results/deployment_opportunity_20261011/；部署价值UNKNOWN，等待恢复已有环境后测量host成本。

- 2026-10-12 pin32_20261012：用户恢复研究盘后，python /media/a6000/D/ZhaoBinHe/SlackKV/.tmp/deployment_pin_probe_20261012.py --root /media/a6000/D/ZhaoBinHe/SlackKV --name pin32_20261012 --gpu 1。32GiB torch.zeros(pin_memory=True)，4CPU threads、60s worker watchdog，PASS，allocation27.01465s，生命周期55.67864s，RSS+32.104GiB，del+GC保留pool/退出回收，SwapFree不变。没有模型/GPU tensor/SLO对照；40 tests PASS。新增含清理0.01574406685GPUh，累计1.2658210347572914h。原始结果results/deployment_opportunity_20261011/pin32_20261012/，报告恢复补充AUTONOMOUS_RESEARCH_DEPLOYMENT_OPPORTUNITY_20261012.md。
