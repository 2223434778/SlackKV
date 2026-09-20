#!/usr/bin/env bash
set -euo pipefail
cd /media/a6000/D/ZhaoBinHe/SlackKV
ENV=/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix
export PYTHONPATH=.
$ENV/bin/python experiments/phase3g/run_phase3g.py --stage control
$ENV/bin/python -m pytest -q tests/test_phase3g.py
# Integration gate was intentionally run as A/B/C only; full 3x3 matrix is
# forbidden after a correctness failure.
