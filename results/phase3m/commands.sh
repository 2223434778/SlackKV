#!/usr/bin/env bash
set -euo pipefail
PY=/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python
PYTHONPATH=. $PY -m pytest -q tests/test_phase3m.py
for state in 0 1 2 3; do
  PYTHONPATH=. $PY experiments/phase3m/run_phase3m.py --state "$state"
done
PYTHONPATH=. $PY experiments/phase3m/postprocess.py
