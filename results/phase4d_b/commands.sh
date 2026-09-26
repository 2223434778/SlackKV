#!/usr/bin/env bash
set -euo pipefail
PY=/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python
export PYTHONPATH=/media/a6000/D/ZhaoBinHe/slackkv-phase2c-fix-sourceprobe:$PWD
# G0/unit/API validation
"$PY" -m pytest -q tests/test_phase4d_b.py
# G1 exact-headroom safe/unsafe micro-experiment
"$PY" -m experiments.phase4d_b.g1 --mode g1
# G2 smoke + three fresh-server correctness repetitions; not a performance matrix
"$PY" -m experiments.phase4d_b.integration --mode g2 --repeats 3
