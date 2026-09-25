#!/usr/bin/env bash
set -euo pipefail

# Run this audit from the A6000 login shell so its configured HTTPS proxy is present.
python experiments/phase4b/audit_hf_live.py

# This fetch is bounded (10 rows/config); text fields are parsed in memory only.
/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python -m pytest -q tests/test_phase4b.py tests/test_phase4b_live.py

# Rebuild machine-readable validation from the compact audit artifacts.
python experiments/phase4b/run_phase4b.py
