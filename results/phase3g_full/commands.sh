#!/usr/bin/env bash
set -euo pipefail
PYTHONPATH=. $VENV/bin/python -m pytest -q tests/test_phase3g_full.py
PYTHONPATH=. $VENV/bin/python -m experiments.phase3g_full.run_phase3g_full --stage integration
PYTHONPATH=. $VENV/bin/python -m experiments.phase3g_full.run_phase3g_full --stage full
PYTHONPATH=. $VENV/bin/python -m experiments.phase3g_full.postprocess
