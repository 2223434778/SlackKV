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
