#!/usr/bin/env bash
set -euo pipefail
PYTHON=/home/a6000/anaconda3/bin/python
$PYTHON -m pip install --target data/tracelab/python duckdb==1.1.3
$PYTHON experiments/phase1b/run_phase1b.py
$PYTHON scripts/plot_phase1b.py
