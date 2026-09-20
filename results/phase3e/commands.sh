#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=/media/a6000/D/ZhaoBinHe/SlackKV:/media/a6000/D/ZhaoBinHe/slackkv-phase2c-fix-sourceprobe
PY=/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python
$PY -m pytest -q tests/test_phase3e.py
$PY experiments/phase3e/run_phase3e.py --stage sanity
$PY experiments/phase3e/run_phase3e.py --stage workload
