#!/usr/bin/env bash
set -euo pipefail
/home/a6000/anaconda3/bin/python experiments/phase1c/run_phase1c.py
/home/a6000/anaconda3/bin/python scripts/plot_phase1c.py
