#!/usr/bin/env bash
set -euo pipefail
PYTHON=/home/a6000/anaconda3/bin/python
$PYTHON -m pytest -q tests/test_phase1a_simulator.py
$PYTHON experiments/phase1a/run_phase1a.py
$PYTHON scripts/plot_phase1a.py
