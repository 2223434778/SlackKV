#!/usr/bin/env bash
set -euo pipefail

REPO=/media/a6000/D/ZhaoBinHe/SlackKV
ENV_ROOT=/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix
VLLM_SOURCE=/media/a6000/D/ZhaoBinHe/slackkv-phase2c-fix-sourceprobe
cd "$REPO"
export PATH="$ENV_ROOT/bin:$PATH"
export PYTHONPATH="$VLLM_SOURCE:$REPO"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export VLLM_SERVER_DEV_MODE=1
LD_LIBRARY_PATH="$(printenv LD_LIBRARY_PATH || true)"
export LD_LIBRARY_PATH="$ENV_ROOT/targets/x86_64-linux/lib:$ENV_ROOT/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$ENV_ROOT/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH"

python -m py_compile experiments/phase3a/*.py tests/test_phase3a.py
python -m pytest -q tests/test_phase3a.py
(
  cd "$VLLM_SOURCE"
  export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
  python -m pytest -q \
    tests/v1/kv_connector/unit/test_slackkv_native_prefetch.py \
    tests/v1/kv_connector/unit/offloading_connector/test_worker_metadata.py
)
python -m experiments.phase3a.run_phase3a --stage sanity
python -m experiments.phase3a.run_phase3a --stage workload
