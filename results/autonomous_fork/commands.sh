#!/bin/sh
# Observed commands, in order. Names already exist: do not overwrite.
PY=/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python
cd /media/a6000/D/ZhaoBinHe/SlackKV
$PY -m experiments.autonomous_fork.run --name fork-control-4 --kv-gib 4 --gate 8 --background-phase cold
# Above failed after episode zero; aggregation helper corrected, cost retained.
$PY -m experiments.autonomous_fork.run --name fork-control-4-retry --kv-gib 4 --gate 8 --background-phase cold
$PY -m experiments.autonomous_fork.run --name fork-decode-control-4 --kv-gib 4 --gate 8 --background-phase decode
$PY -m experiments.autonomous_fork.run --name fork-decode-control-8 --kv-gib 8 --gate 8 --background-phase decode
$PY -m experiments.autonomous_fork.run --name fork-decode-k2-4 --kv-gib 4 --gate 2 --background-phase decode
$PY -m experiments.autonomous_fork.run --name fork-oracle-4 --kv-gib 4 --gate 8 --background-phase decode --batch-invariant --oracle
$PY -m experiments.autonomous_admission.run --round-directory autonomous_fork --name legacy-regression-2 --execution default --kv-gib 2 --trace --runtime-source /media/a6000/D/ZhaoBinHe/slackkv-autonomous-round3-source --replay-inputs results/autonomous_capacity/capacity-2/inputs.jsonl
# Actual runs were detached with nohup into owned *.launch.log; watchdog includes cleanup.
# CPU-only runtime regression: first omitted the second PYTHONPATH entry and failed; both records retained.
cd /media/a6000/D/ZhaoBinHe/slackkv-autonomous-round3-source
CUDA_VISIBLE_DEVICES='' PYTHONPATH=/media/a6000/D/ZhaoBinHe/slackkv-autonomous-round3-source:/media/a6000/D/ZhaoBinHe/SlackKV $PY -m pytest -q tests/v1/kv_connector/unit/test_slackkv_native_prefetch.py tests/v1/kv_connector/unit/test_slackkv_connector_first.py tests/v1/kv_connector/unit/test_slackkv_first_token_gate_capacity.py
