#!/usr/bin/env bash
set -euo pipefail

REPO=/media/a6000/D/ZhaoBinHe/SlackKV
ENV_ROOT=/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix
cd "$REPO"
export PATH="$ENV_ROOT/bin:$PATH"
export PYTHONPATH="$REPO"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
LD_LIBRARY_PATH="$(printenv LD_LIBRARY_PATH || true)"
export LD_LIBRARY_PATH="$ENV_ROOT/targets/x86_64-linux/lib:$ENV_ROOT/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$ENV_ROOT/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH"

python -m py_compile experiments/phase2g/*.py tests/test_phase2g.py
python -m pytest -q tests/test_phase2g.py
python -m experiments.phase2g.run_phase2g
