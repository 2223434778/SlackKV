#!/usr/bin/env bash
set -euo pipefail
PYTHONPATH=.:/media/a6000/D/ZhaoBinHe/slackkv-phase2c-fix-sourceprobe /media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python -m pytest -q tests/test_phase3h.py /media/a6000/D/ZhaoBinHe/slackkv-phase2c-fix-sourceprobe/tests/v1/kv_connector/unit/test_slackkv_first_token_gate_capacity.py
PYTHONPATH=.:/media/a6000/D/ZhaoBinHe/slackkv-phase2c-fix-sourceprobe /media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python -m experiments.phase3h.run_phase3h --stage integration
PYTHONPATH=.:/media/a6000/D/ZhaoBinHe/slackkv-phase2c-fix-sourceprobe /media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python -m experiments.phase3h.run_phase3h --stage full
