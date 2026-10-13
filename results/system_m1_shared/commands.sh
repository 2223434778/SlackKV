#!/bin/sh
# Replay template on established environment. Exact original argv is in the
# launch.json files. Unique names and the SAME continuous ledger are required.
set -eu
ROOT=/media/a6000/D/ZhaoBinHe/SlackKV
CODE="$ROOT/.tmp/system-m1-code"
PY=/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python
OVERLAY=/media/a6000/D/ZhaoBinHe/slackkv-system-m1-native
export CUDA_VISIBLE_DEVICES=''
PYTHONPATH="$OVERLAY:$CODE" "$PY" -m experiments.system_m1.native_audit --output "$ROOT/results/system_m1_shared/native_audit_repeat.json"
export PYTHONPATH="$CODE"
"$PY" -m experiments.system_m1.smoke --config "$CODE/experiments/system_m1/shared_config.json" --prepare "$ROOT/results/system_m1_shared/preparation_repeat"
# The original successful content4 used preparation4 (same input bytes, earlier
# static-fair metadata); its exact runner is archived under source_content4.
"$PY" -m experiments.system_m1.smoke --config "$CODE/experiments/system_m1/shared_config.json" --execute --root "$ROOT" --plan "$ROOT/results/system_m1_shared/preparation_repeat" --name m1_shared_content_repeat --policy m1 --mode content --authorized-additional-gpu-hours 0.5
for POLICY in m1 fair full; do
  "$PY" -m experiments.system_m1.smoke --config "$CODE/experiments/system_m1/shared_config.json" --execute --root "$ROOT" --plan "$ROOT/results/system_m1_shared/preparation_repeat" --name "m1_shared_${POLICY}_repeat" --policy "$POLICY" --mode mechanism --authorized-additional-gpu-hours 0.5
done
"$PY" -m experiments.system_m1.analyze_contention --base "$ROOT/results/system_m1_smoke" --variant shared_v1 --run-suffix _repeat --output "$ROOT/results/system_m1_shared/analysis_repeat.json"
# Preserved launch.json files hold the actual commands for all four content
# attempts. This file does not grant additional GPU time or reset accounting.
