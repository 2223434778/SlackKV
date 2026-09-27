#!/usr/bin/env bash
set -euo pipefail
cd /media/a6000/D/ZhaoBinHe/SlackKV
export PYTHONPATH=/media/a6000/D/ZhaoBinHe/slackkv-phase2c-fix-sourceprobe:$PWD
PY=/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python
"$PY" -m pytest -q tests/test_phase4f_a.py
"$PY" -u -m experiments.phase4f_a.run_phase4f_a
"$PY" -m experiments.phase4f_a.finalize_phase4f_a
