# Phase 0 — Hardware and Data-Movement Feasibility

## Objective

This experiment measures (1) the actual GPU0/GPU1 topology, (2) direct peer GPU memory versus pinned CPU memory as a transfer tier, and (3) transfer interference with work running on GPU0.

## Hardware topology

Both devices are NVIDIA RTX A6000 GPUs with 48 GiB GDDR6 ECC memory. `nvidia-smi topo -m` reports `SYS` in both GPU0/GPU1 directions: the path traverses PCIe and the SMP interconnect. It does **not** report an `NV#` path, so NVLink is not visible on this platform.

CUDA Runtime detection (`cudaDeviceCanAccessPeer`) returned true for GPU0→GPU1 and GPU1→GPU0. The native benchmark enabled peer access in both directions and uses `cudaMemcpyPeerAsync`; peer rows therefore represent direct CUDA peer copies, not CPU staging.

## Transfer results

Each cell below is P50 effective bandwidth in decimal GB/s from 30 CUDA-event measurements. The final column is P95 latency at 4 GiB.

| Direction | 256 MiB | 1 GiB | 4 GiB | 8 GiB | 4 GiB P95 latency (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPU0→GPU1 | 27.08 | 27.08 | 27.09 | 27.06 | 158.57 |
| GPU1→GPU0 | 26.36 | 26.37 | 26.37 | 26.33 | 162.88 |
| GPU0→pinned CPU | 27.08 | 25.88 | 26.41 | 23.00 | 162.66 |
| pinned CPU→GPU0 | 25.99 | 26.28 | 26.12 | 26.26 | 164.46 |
| GPU1→pinned CPU | 26.87 | 27.01 | 22.26 | 21.01 | 193.63 |
| pinned CPU→GPU1 | 26.08 | 26.20 | 20.85 | 26.14 | 205.81 |

For GPU0 promotion at 1 GiB, direct peer GPU1→GPU0 was 26.37 GB/s versus pinned CPU→GPU0 at 26.28 GB/s (1.003×). At 4 GiB it was 26.37 versus 26.12 GB/s (1.010×). For GPU0 eviction at 1 GiB, GPU0→GPU1 was 27.08 GB/s versus GPU0→pinned CPU at 25.88 GB/s (1.046×); at 4 GiB it was 27.09 versus 26.41 GB/s (1.026×).

## Overlap results

Representative P50 results below use the FP16 GEMM workload and 1 GiB transfer traffic. Slowdowns are relative to their corresponding compute-only or transfer-only medians; no values are clamped.

| Transfer case | Compute slowdown | Transfer slowdown | Overlap efficiency |
| --- | ---: | ---: | ---: |
| peer_in (GPU1→GPU0) | 2.78% | 0.56% | 0.968 |
| peer_out (GPU0→GPU1) | 3.48% | 0.71% | 0.959 |
| cpu_in (pinned CPU→GPU0) | 3.66% | 0.43% | 0.954 |
| cpu_out (GPU0→pinned CPU) | 3.86% | 0.32% | 0.952 |

At 4 GiB, GEMM compute slowdown was 2.76% for peer_in and 3.91% for cpu_in; overlap efficiency was 0.961 and 0.945, respectively. The separately labeled synthetic memory-bandwidth workload is preserved in `overlap_metrics.csv` and `overlap_summary_metrics.csv`.

## Correctness

All 1,440 successful transfer measurements passed deterministic beginning/middle/end data validation. No transfer configuration failed or was skipped, including all 8 GiB pinned-memory configurations.

## Anomalies and limitations

The topology is `SYS` despite bidirectional CUDA peer access; CUDA P2P availability does not imply visible NVLink. Large CPU-transfer directions have more variable bandwidth (for example, GPU1→pinned CPU falls to 21.01 GB/s at 8 GiB). The authoritative transfer measurements use native CUDA; PyTorch is used only for the synthetic overlap workload. No allocation failures occurred.

## Conclusion

On this A6000 GDDR6 platform, direct CUDA peer copies are available in both directions but have only a small measured bandwidth advantage over pinned CPU transfers for the GPU0 paths tested (about 0.3–4.6% at 1–4 GiB). Concurrent transfers overlap substantially with the calibrated GPU0 GEMM workload, with the representative 1 GiB cases showing 2.78–3.86% compute slowdown and 0.952–0.968 overlap efficiency.

Phase 0 execution status: COMPLETE
