#!/usr/bin/env bash
set -euo pipefail

# Pure CPU/offline validation; no HF prompt/parquet data is fetched.
python -m pytest -q tests/test_phase4b.py
python experiments/phase4b/run_phase4b.py
