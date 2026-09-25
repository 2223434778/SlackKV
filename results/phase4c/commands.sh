#!/usr/bin/env bash
set -euo pipefail
cd /media/a6000/D/ZhaoBinHe/SlackKV
# Source revision and status recorded before implementation.
git rev-parse HEAD
git status --short --branch
sha256sum external/tracelab_v002/syfi_coding_trace.duckdb
/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/pip install duckdb==1.5.5
PYTHONPATH=. /media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python -m pytest -q tests/test_phase4c.py
bash -lic '/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python experiments/phase4c/source_structure.py'
/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python experiments/phase4c/timing_proxy.py
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python experiments/phase4c/run_phase4c.py A0
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python experiments/phase4c/run_phase4c.py A1
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python experiments/phase4c/run_phase4c.py B
/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python experiments/phase4c/finalize_phase4c.py
