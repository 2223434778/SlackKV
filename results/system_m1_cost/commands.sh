#!/bin/sh
# Exact executed GPU argv is in launch.json and each original execution.json.
# Replay only under applicable user budget, same original family ledger, unique names.
set -eu
ROOT=/media/a6000/D/ZhaoBinHe/SlackKV
CODE="$ROOT/.tmp/system-m1-code"
PY=/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python
OVERLAY=/media/a6000/D/ZhaoBinHe/slackkv-system-m1-native
# Current CPU diagnosis compares functions built from the archived unoptimized source.
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$OVERLAY:$CODE" "$PY" -m experiments.system_m1.cost_diagnosis --reference "$ROOT/results/system_m1_cost/reference/native.py" --output "$ROOT/results/system_m1_cost/diagnosis_repeat.json" --repeats 100 --batches 5
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$OVERLAY:$CODE" "$PY" -m experiments.system_m1.native_audit --output "$ROOT/results/system_m1_cost/native_audit_repeat.json"
PYTHONPATH="$CODE" "$PY" -m experiments.system_m1.analyze_cost --base "$ROOT/results/system_m1_smoke" --unused-evidence "$ROOT/results/system_m1_cost/unused_write_evidence.json" --output "$ROOT/results/system_m1_cost/analysis_repeat.json"
# The baseline ran before optimized/native.py was uploaded; reference/native.py
# and optimized/native.py preserve both actual byte sequences. To rerun GPU,
# install each only in the isolated CODE tree with no owned process active;
# retain pinned executed_sources/contention.py and smoke.py and preparation5.
# Use UNIQUE run names and do not rerun launch.py against existing directories.
# --authorized-additional-gpu-hours 0.5 is the original total grant, not new credit.
