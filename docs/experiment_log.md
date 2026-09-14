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
