#!/bin/sh
# Reproduction on the established Ubuntu environment; preserve existing names.
set -eu
ROOT=/media/a6000/D/ZhaoBinHe/SlackKV
CODE="$ROOT/.tmp/system-m1-code"
PY=/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python
OVERLAY=/media/a6000/D/ZhaoBinHe/slackkv-system-m1-native
export CUDA_VISIBLE_DEVICES=''
export PYTHONPATH="$OVERLAY:$CODE"
# A unique output path is required; these paths were the original execution.
"$PY" -m experiments.system_m1.native_audit --output "$ROOT/results/system_m1_followup/native_terminal_audit.json"
export PYTHONPATH="$CODE"
"$PY" -m experiments.system_m1.smoke --config "$CODE/experiments/system_m1/contention_config.json" --prepare "$ROOT/results/system_m1_followup/preparation_chat"
"$PY" -m experiments.system_m1.smoke --config "$CODE/experiments/system_m1/contention_config.json" --execute --root "$ROOT" --plan "$ROOT/results/system_m1_followup/preparation_chat" --name m1_followup_content2 --policy m1 --mode content --authorized-additional-gpu-hours 0.5
for POLICY in m1 fair full; do
  "$PY" -m experiments.system_m1.smoke --config "$CODE/experiments/system_m1/contention_config.json" --execute --root "$ROOT" --plan "$ROOT/results/system_m1_followup/preparation_chat" --name "m1_followup_${POLICY}1" --policy "$POLICY" --mode mechanism --authorized-additional-gpu-hours 0.5
done
"$PY" -m experiments.system_m1.analyze_contention --base "$ROOT/results/system_m1_smoke" --output "$ROOT/results/system_m1_followup/analysis.json"
# An independent retry uses unique preparation/run/output names and the SAME
# ledger. No numeric grant is implied by this file; obey remaining allowance.
