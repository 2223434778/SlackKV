#!/bin/sh
# Observed commands; immutable run names already exist. See report for fresh-name reruns.
PY=/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python
cd /media/a6000/D/ZhaoBinHe/SlackKV
# runner SHA256: f462398ee2f36f7a92331a1c4d2a7035c02671d7da9885ca253bcabf98c81d2b
$PY -m experiments.autonomous_capacity.run --name capacity-2 --kv-gib 2 --trace
# runner SHA256: f462398ee2f36f7a92331a1c4d2a7035c02671d7da9885ca253bcabf98c81d2b
$PY -m experiments.autonomous_capacity.run --name capacity-4 --kv-gib 4 --trace --replay-inputs results/autonomous_capacity/capacity-2/inputs.jsonl
# runner SHA256: 00f8ad2e0003baaa38e69ed228d89448d8e3cf5c24cdf5eec57eb66a499368be
$PY -m experiments.autonomous_capacity.run --name capacity-4-retry --kv-gib 4 --trace --replay-inputs results/autonomous_capacity/capacity-2/inputs.jsonl
# runner SHA256: 305c7c6e9d1ce39639f735b683fbbddb8a37e5bf1680e14033adfa1b3c8afaf9
$PY -m experiments.autonomous_capacity.run --name capacity-8 --kv-gib 8 --trace --replay-inputs results/autonomous_capacity/capacity-2/inputs.jsonl
# runner SHA256: 305c7c6e9d1ce39639f735b683fbbddb8a37e5bf1680e14033adfa1b3c8afaf9
$PY -m experiments.autonomous_capacity.run --name credit-control-2 --kv-gib 2 --trace --runtime-source /media/a6000/D/ZhaoBinHe/slackkv-autonomous-round2-source --replay-inputs results/autonomous_capacity/capacity-2/inputs.jsonl
# runner SHA256: 305c7c6e9d1ce39639f735b683fbbddb8a37e5bf1680e14033adfa1b3c8afaf9
$PY -m experiments.autonomous_capacity.run --name credit-candidate-2 --kv-gib 2 --trace --runtime-source /media/a6000/D/ZhaoBinHe/slackkv-autonomous-round2-source --continuation-credit --replay-inputs results/autonomous_capacity/capacity-2/inputs.jsonl
# runner SHA256: dd738b183a83e5658d20058b2cc598f767f592d158e5af6b993f4dd2d87403fc
$PY -m experiments.autonomous_capacity.run --name epoch-control-2 --kv-gib 2 --trace --runtime-source /media/a6000/D/ZhaoBinHe/slackkv-autonomous-round2-source --replay-inputs results/autonomous_capacity/capacity-2/inputs.jsonl
# runner SHA256: dd738b183a83e5658d20058b2cc598f767f592d158e5af6b993f4dd2d87403fc
$PY -m experiments.autonomous_capacity.run --name epoch-candidate-2 --kv-gib 2 --trace --runtime-source /media/a6000/D/ZhaoBinHe/slackkv-autonomous-round2-source --working-set-epochs --replay-inputs results/autonomous_capacity/capacity-2/inputs.jsonl
