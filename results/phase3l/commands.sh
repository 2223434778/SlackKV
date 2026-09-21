#!/usr/bin/env bash
set -euo pipefail
PY=/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python
$PY -m pytest -q tests/test_phase3l.py
for state in 0 1 2 3; do
  $PY experiments/phase3l/run_phase3l.py --state "$state"
done
$PY experiments/phase3l/postprocess.py
