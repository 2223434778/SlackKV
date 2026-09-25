#!/usr/bin/env bash
set -euo pipefail
cd /media/a6000/D/ZhaoBinHe/SlackKV
PYTHONPATH=. python -m pytest -q tests/test_phase4d_a.py
PYTHONPATH=. python -m experiments.phase4d_a.run_policy --stage policy
# Stage B intentionally not run: NaiveJIT integration did not complete.
