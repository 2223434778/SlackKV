#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=/media/a6000/D/ZhaoBinHe/SlackKV:/media/a6000/D/ZhaoBinHe/slackkv-phase2c-fix-sourceprobe
PY=/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python
$PY -m pytest -q tests/test_phase3b.py
(cd /media/a6000/D/ZhaoBinHe/slackkv-phase2c-fix-sourceprobe && $PY -m pytest -q tests/v1/kv_connector/unit/test_slackkv_native_prefetch.py tests/v1/kv_connector/unit/test_slackkv_connector_first.py tests/v1/kv_connector/unit/offloading_connector/test_worker_metadata.py)
$PY experiments/phase3b/run_phase3b.py --stage sanity
$PY experiments/phase3b/run_phase3b.py --stage workload
