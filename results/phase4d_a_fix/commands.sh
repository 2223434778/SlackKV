#!/usr/bin/env bash
set -euo pipefail
# Frozen diagnostic only; no Phase 4D-A Stage B policy matrix.
python -m pytest -q tests/test_phase4d_a_fix.py
python -m experiments.phase4d_a_fix.integration
# Naive L0/L1 intentionally not invoked: the required Reservation integration gate failed.
