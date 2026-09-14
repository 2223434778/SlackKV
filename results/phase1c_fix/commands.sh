#!/usr/bin/env bash
set -euo pipefail

# Seed-level replay was completed in the prior Phase 1C-Fix run.
/home/a6000/anaconda3/bin/python experiments/phase1c_fix/analyze_phase1c_fix.py
/home/a6000/anaconda3/bin/python scripts/plot_phase1c_fix.py
/home/a6000/anaconda3/bin/python -m pytest -q tests/test_phase1a_simulator.py tests/test_phase1b_fix.py tests/test_phase1c_fix.py
