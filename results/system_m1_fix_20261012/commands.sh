#!/usr/bin/env bash
set -eu
ROOT=/media/a6000/D/ZhaoBinHe/SlackKV
CODE="$ROOT/.tmp/system-m1-code"  # Uploaded from the final correction checkout.
PY=/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python
NATIVE=/media/a6000/D/ZhaoBinHe/slackkv-system-m1-native
OUT="$ROOT/results/system_m1_fix_20261012"
export CUDA_VISIBLE_DEVICES=''
export PYTHONPATH="$NATIVE:$CODE"
export LD_LIBRARY_PATH=/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/targets/x86_64-linux/lib:/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib
# Executed: output names already exist. For repetition use NEW names.
# timeout 25s "$PY" -m experiments.system_m1.native_audit --output "$OUT/native_before.json"
# Before command used controller_before.py in slackkv/m1/controller.py; restore
# the final controller before after/run. Do not overwrite historical evidence.
# timeout 25s "$PY" -m experiments.system_m1.native_audit --output "$OUT/native_after.json" > "$OUT/native_after.log" 2>&1
# CPU preparation commands executed, v3 is the final pinned runner:
# timeout 20s "$PY" -m experiments.system_m1.smoke --prepare "$OUT/smoke_preparation_v3" > "$OUT/smoke_prepare_v3.log" 2>&1
# Archived inputs.json.gz contains exact inputs.json bytes; server originals exist.
# For a fresh checkout's archived plan: gzip -dk "$OUT/smoke_preparation_v3/inputs.json.gz"

# GPU NOT_STARTED. Commands below are intentionally commented out. Only after
# explicit NEW 0.5 GPUh allowance, count every policy/name in the same ledger.
# Priority A, content: single card, whole run <=600s including cleanup.
# "$PY" -m experiments.system_m1.smoke --execute --root "$ROOT" --plan "$OUT/smoke_preparation_v3" --name m1_content1 --policy m1 --mode content --gpu 0 --authorized-additional-gpu-hours 0.5
# After A PASS, only within remaining allowance:
# "$PY" -m experiments.system_m1.smoke --execute --root "$ROOT" --plan "$OUT/smoke_preparation_v3" --name m1_mechanism1 --policy m1 --mode mechanism --gpu 0 --authorized-additional-gpu-hours 0.5
# "$PY" -m experiments.system_m1.smoke --execute --root "$ROOT" --plan "$OUT/smoke_preparation_v3" --name fair_mechanism1 --policy fair --mode mechanism --gpu 0 --authorized-additional-gpu-hours 0.5
# "$PY" -m experiments.system_m1.compare_smoke --m1 "$ROOT/results/system_m1_smoke/m1_mechanism1" --fair "$ROOT/results/system_m1_smoke/fair_mechanism1" --output "$OUT/mechanism_pair.json"
# Full and true NoHost commands use the same runner, --policy full / nohost;
# they are prepared, NOT executed, and not prerequisites for A's completion.
