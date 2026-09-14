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
