# Dataset must be the SHA256-verified TraceLab v0.0.2 release asset.
PYTHONPATH=/tmp/slackkv-phase4a-deps /media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python experiments/phase4a/run_phase4a.py external/tracelab_v002/syfi_coding_trace.duckdb results/phase4a
PYTHONPATH=/tmp/slackkv-phase4a-deps /media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python -m pytest -q tests/test_phase4a.py
