#!/usr/bin/env bash
set -euo pipefail
VENV=/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix
SRC=/media/a6000/D/ZhaoBinHe/slackkv-phase2c-fix-sourceprobe
cd /media/a6000/D/ZhaoBinHe/SlackKV
PYTHONPATH=. "$VENV/bin/python" -m pytest -q tests/test_phase3s.py
PYTHONPATH=.:"$SRC" "$VENV/bin/python" -m experiments.phase3s.run_phase3s
