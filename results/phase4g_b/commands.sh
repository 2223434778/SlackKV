#!/usr/bin/env bash
set -euo pipefail
cd /media/a6000/D/ZhaoBinHe/SlackKV
/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python -m pytest -q tests/test_phase4g_b.py
/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python -m py_compile experiments/phase4g_b/*.py
/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python -m experiments.phase4g_b.run_phase4g_b --run
