#!/usr/bin/env bash
set -euo pipefail
/home/a6000/anaconda3/bin/python experiments/phase1c_fix2/run_phase1c_fix2.py
/home/a6000/anaconda3/bin/python experiments/phase1c_fix2/analyze_phase1c_fix2.py
/home/a6000/anaconda3/bin/python experiments/phase1c_fix2/finalize_phase1c_fix2.py
