#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=/media/a6000/D/ZhaoBinHe/SlackKV:/media/a6000/D/ZhaoBinHe/slackkv-phase2c-fix-sourceprobe
PY=/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python
$PY -m pytest -q tests/test_phase3c.py
$PY experiments/phase3c/run_phase3c.py --stage sanity
$PY experiments/phase3c/run_phase3c.py --stage workload
