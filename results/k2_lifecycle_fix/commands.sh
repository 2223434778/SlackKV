set -euo pipefail
cd /media/a6000/D/ZhaoBinHe/SlackKV
PY=/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python
# Baseline snapshot was captured before the committed scheduler delta was applied.
if SLACKKV_SCHEDULER_SOURCE=/tmp/slackkv-k2-lifecycle-baseline.py "$PY" -m pytest -q tests/test_k2_lifecycle_fix.py; then
    echo 'ERROR: baseline unexpectedly passed regression tests' >&2
    exit 1
fi
"$PY" -m pytest -q tests/test_k2_lifecycle_fix.py
PYTHONPATH=/media/a6000/D/ZhaoBinHe/slackkv-phase2c-fix-sourceprobe "$PY" -m pytest -q --confcutdir=/media/a6000/D/ZhaoBinHe/slackkv-phase2c-fix-sourceprobe/tests/v1/kv_connector/unit /media/a6000/D/ZhaoBinHe/slackkv-phase2c-fix-sourceprobe/tests/v1/kv_connector/unit/test_slackkv_first_token_gate_capacity.py
# Requires a fresh output directory; never overwrite historical validation artifacts.
"$PY" -u -m experiments.k2_lifecycle_fix.validate
