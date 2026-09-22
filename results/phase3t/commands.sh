#!/usr/bin/env bash
set -euo pipefail
VENV=/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix
SRC=/media/a6000/D/ZhaoBinHe/slackkv-phase2c-fix-sourceprobe
cd /media/a6000/D/ZhaoBinHe/SlackKV
PYTHONPATH=. "$VENV/bin/python" -m pytest -q tests/test_phase3t.py
PYTHONPATH=.:"$SRC" "$VENV/bin/python" -m experiments.phase3t.run_phase3t --stage integration
PYTHONPATH=.:"$SRC" "$VENV/bin/python" -m experiments.phase3t.run_phase3t --stage full
PYTHONPATH=. "$VENV/bin/python" -m experiments.phase3t.postprocess
