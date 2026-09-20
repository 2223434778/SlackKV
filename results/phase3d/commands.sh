#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=/media/a6000/D/ZhaoBinHe/SlackKV:/media/a6000/D/ZhaoBinHe/slackkv-phase2c-fix-sourceprobe
PY=/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python
$PY -m pytest -q tests/test_phase3d.py
$PY experiments/phase3d/run_phase3d.py --stage sanity
$PY experiments/phase3d/run_phase3d.py --stage workload
