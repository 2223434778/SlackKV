#!/usr/bin/env bash
# CPU-only reproduction. Run from a checkout containing this M1 implementation.
set -eu
ROOT=/media/a6000/D/ZhaoBinHe/SlackKV
PY=/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python
BASE=/media/a6000/D/ZhaoBinHe/slackkv-autonomous-round4-source
# Exact installation used .tmp/system-m1-code modules, uploaded from this checkout.
CODE="$ROOT/.tmp/system-m1-code"
NATIVE=/media/a6000/D/ZhaoBinHe/slackkv-system-m1-native
RESULTS="$ROOT/results/system_m1_20261012"
export CUDA_VISIBLE_DEVICES=''
export PYTHONPATH="$CODE"
# Already executed: target/output must not exist. For repetition choose NEW names.
# "$PY" -m experiments.system_m1.install --base "$BASE" --target "$NATIVE" --output "$RESULTS/installation.json"
export PYTHONPATH="$NATIVE:$CODE"
export LD_LIBRARY_PATH=/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/targets/x86_64-linux/lib:/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib
# Frozen version 1/v2 modules and audit scripts: source_v1 / source_v2 in RESULTS.
# Exact v2 command (already executed; refuse overwrite):
# timeout 50s "$PY" -m experiments.system_m1.native_audit --output "$RESULTS/native_audit_v2.json" > "$RESULTS/native_audit_v2.log" 2>&1
# Reproduce final v2, new output; CODE must have the checkout's slackkv/__init__.py.
timeout 50s "$PY" -m experiments.system_m1.native_audit --output "$RESULTS/native_audit_reproduce.json" > "$RESULTS/native_audit_reproduce.log" 2>&1
# No GPU serving command is executed here. See report for the UNEXECUTED smoke plan.
