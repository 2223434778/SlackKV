#!/usr/bin/env bash
set -euo pipefail
cd /media/a6000/D/ZhaoBinHe/SlackKV
ENV=/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python
PYTHONPATH=. "$ENV" -m pytest -q tests/test_phase3f.py
PYTHONPATH=. "$ENV" experiments/phase3f/run_clean.py --stage sanity
PYTHONPATH=. "$ENV" experiments/phase3f/run_clean.py --stage workload
