#!/usr/bin/env bash
set -euo pipefail
export ENV_ROOT=/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix
export PYTHONPATH=/media/a6000/D/ZhaoBinHe/SlackKV
export PATH="$ENV_ROOT/bin:$PATH"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LD_LIBRARY_PATH="$ENV_ROOT/targets/x86_64-linux/lib:$ENV_ROOT/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$ENV_ROOT/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
pytest -q tests/test_phase2c.py tests/test_phase2c_fix.py
python -m experiments.phase2c_fix.run_phase2c_fix --seeds 0,1,2
