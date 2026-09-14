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
