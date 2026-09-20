#!/usr/bin/env bash
set -euo pipefail
cd /media/a6000/D/ZhaoBinHe/SlackKV
ENV=/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix
export PYTHONPATH=.
$ENV/bin/python -m pytest -q tests/test_phase3f_fix.py
$ENV/bin/python experiments/phase3f_fix/run_phase3f_fix.py --stage workload
$ENV/bin/python experiments/phase3f_fix/postprocess.py
$ENV/bin/python -m pytest -q tests/test_phase3f_fix.py
