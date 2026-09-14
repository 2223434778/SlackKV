#!/usr/bin/env bash
set -euo pipefail
PYTHON=/home/a6000/anaconda3/bin/python
NVCC=/usr/local/cuda-12/bin/nvcc
$PYTHON scripts/collect_phase0_env.py
$NVCC -O3 -std=c++17 bench/phase0_transfer.cu -o build/phase0_transfer
./build/phase0_transfer --output results/phase0/smoke_metrics.csv --p2p-output results/phase0/p2p_matrix.json --sizes-mib 16 --warmup 2 --iterations 3
./build/phase0_transfer --output results/phase0/metrics.csv --p2p-output results/phase0/p2p_matrix.json --warmup 10 --iterations 30
$PYTHON bench/phase0_overlap.py --output results/phase0/overlap_metrics.csv --summary-output results/phase0/overlap_summary_metrics.csv --warmup 5 --iterations 30
$PYTHON scripts/plot_phase0.py
