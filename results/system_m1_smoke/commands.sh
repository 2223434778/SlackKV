#!/usr/bin/env bash
set -eu
ROOT=/media/a6000/D/ZhaoBinHe/SlackKV
CODE="$ROOT/.tmp/system-m1-code"
PY=/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python
BASE="$ROOT/results/system_m1_smoke"
PLAN="$BASE/preparation_path_fix"
export PYTHONPATH="$CODE"
# All GPU commands below were executed and are commented to avoid duplicate runs.
# The original pre-PATH-fix content1 used smoke_preparation_v3 in the correction directory.
# Ordered names/policies/modes, same model/input/resource configuration:
# m1_content1 m1 content FAILED; m1_content2 m1 content FAILED
# m1_content3 m1 content FAILED; m1_content4 m1 content PASS
# m1_mechanism1 m1 mechanism; fair_mechanism1 fair mechanism
# full_mechanism1 full mechanism; nohost_mechanism1 nohost mechanism
# fair_mechanism2 fair mechanism; m1_mechanism2 m1 mechanism
# nohost_mechanism2 nohost mechanism
# For each name/policy/mode above:
# "$PY" -m experiments.system_m1.smoke --execute --root "$ROOT" --plan "$PLAN" --name NAME --policy POLICY --mode MODE --gpu 0 --authorized-additional-gpu-hours 0.5
# Exact argv, owned pid, start time in *_launch.json. Every execution is in the same family ledger.
# Snapshot runner/native versions listed in provenance_notes.json. A current
# checkout's logging-only wire-param fix changes runner SHA: prepare a NEW plan
# before any authorized future replay; do not overwrite archived plans/results.
# CPU-only independent reanalysis (also works against compressed files in Git):
# python -m experiments.system_m1.analyze_smoke --base results/system_m1_smoke --output .tmp/m1-smoke-reanalysis.json
