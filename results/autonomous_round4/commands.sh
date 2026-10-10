#!/usr/bin/env bash
# Run from the SlackKV repository. Review the report/protocol before any new GPU use.
# Raw execution.json and launch logs give every actual command and observed version.
PY=/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python
REPLAY=results/autonomous_capacity/capacity-2/inputs.jsonl.gz

# Existing retained runtime paths; do not overwrite them or any recorded run folder.
# Baseline: /media/a6000/D/ZhaoBinHe/slackkv-autonomous-round4-source
# Diagnostic: /media/a6000/D/ZhaoBinHe/slackkv-round4-diagnostic-v2-source
# Control: /media/a6000/D/ZhaoBinHe/slackkv-round4-control-source
# To recreate independent copies from the retained Round3 runtime, apply native
# resources.patch to the diagnostic copy; retention.patch THEN retention_hol.patch
# to the control copy with /home/a6000/.local/bin/apply_patch in that copy's cwd.
# Verify their scheduler/allocator hashes against execution.json and actual archived
# source bytes; disabled AST audit is required for diagnostic copies. Historical
# patched foundation is not interchangeable with pip pristine vLLM of the same version.

# Actual original first run (already exists; rerun requires new preregistered names):
# "$PY" -m experiments.autonomous_round4.run --name gate-2-p1-k2 --kv-gib 2 --replay-inputs "$REPLAY"
# "$PY" -m experiments.autonomous_round4.batch  # remaining eleven in plan.json order
# Initial followup failed at resource-off; repaired followup.py then used v2 runtime:
# "$PY" -m experiments.autonomous_round4.followup
# "$PY" -m experiments.autonomous_round4.control_batch
# Neither control passed; reverse triplet was not launched by the frozen rule.

# CPU-only recomputation; scripts rewrite derived JSON, so first audit archived bytes.
"$PY" -m experiments.autonomous_round4.export results/autonomous_round4
"$PY" results/autonomous_round4/integration_audit.py
"$PY" -m experiments.autonomous_round4.analyze results/autonomous_round4
"$PY" -m experiments.autonomous_round4.control_summary results/autonomous_round4
"$PY" -m experiments.autonomous_round4.diagnose results/autonomous_round4/resource-on-repaired
"$PY" -m pytest -q tests/test_autonomous_round4.py tests/test_round4_retention.py tests/test_autonomous_admission.py tests/test_autonomous_capacity.py

# Native CPU regression used the control copy with retention disabled (15 passed):
# PYTHONPATH=/media/a6000/D/ZhaoBinHe/slackkv-round4-control-source:$PWD \
# SLACKKV_RETENTION_CONTROL=0 SLACKKV_RETENTION_PIN=0 "$PY" -m pytest -q \
# /media/a6000/D/ZhaoBinHe/slackkv-round4-control-source/tests/v1/kv_connector/unit/test_slackkv_native_prefetch.py \
# /media/a6000/D/ZhaoBinHe/slackkv-round4-control-source/tests/v1/kv_connector/unit/test_slackkv_first_token_gate_capacity.py \
# /media/a6000/D/ZhaoBinHe/slackkv-round4-control-source/tests/v1/kv_connector/unit/test_slackkv_connector_first.py
