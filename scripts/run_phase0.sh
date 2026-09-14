#!/usr/bin/env bash
# Full Phase 0 execution. Individual unsupported configurations are recorded by the benchmarks.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="results/phase0"; BUILD="build"; PYTHON="${PYTHON:-/home/a6000/anaconda3/bin/python}"; NVCC="${NVCC:-/usr/local/cuda-12/bin/nvcc}"
mkdir -p "$OUT/plots" "$BUILD"
"$PYTHON" scripts/collect_phase0_env.py
"$NVCC" -O3 -std=c++17 bench/phase0_transfer.cu -o "$BUILD/phase0_transfer"
printf '%s\n' '#!/usr/bin/env bash' 'set -euo pipefail' \
  'PYTHON=/home/a6000/anaconda3/bin/python' 'NVCC=/usr/local/cuda-12/bin/nvcc' \
  '$PYTHON scripts/collect_phase0_env.py' \
  '$NVCC -O3 -std=c++17 bench/phase0_transfer.cu -o build/phase0_transfer' \
  './build/phase0_transfer --output results/phase0/smoke_metrics.csv --p2p-output results/phase0/p2p_matrix.json --sizes-mib 16 --warmup 2 --iterations 3' \
  './build/phase0_transfer --output results/phase0/metrics.csv --p2p-output results/phase0/p2p_matrix.json --warmup 10 --iterations 30' \
  '$PYTHON bench/phase0_overlap.py --output results/phase0/overlap_metrics.csv --summary-output results/phase0/overlap_summary_metrics.csv --warmup 5 --iterations 30' \
  '$PYTHON scripts/plot_phase0.py' > "$OUT/commands.sh"
chmod +x "$OUT/commands.sh"
"$BUILD/phase0_transfer" --output "$OUT/smoke_metrics.csv" --p2p-output "$OUT/p2p_matrix.json" --sizes-mib 16 --warmup 2 --iterations 3
"$PYTHON" - <<'PY'
import csv, sys
rows=list(csv.DictReader(open('results/phase0/smoke_metrics.csv')))
valid=[r for r in rows if r['success']=='true']
if len(valid) != 18 or any(r['correctness_pass']!='true' for r in valid):
    raise SystemExit('smoke test failed: expected 18 successful validated measurements')
PY
"$BUILD/phase0_transfer" --output "$OUT/metrics.csv" --p2p-output "$OUT/p2p_matrix.json" --warmup 10 --iterations 30
"$PYTHON" bench/phase0_overlap.py --output "$OUT/overlap_metrics.csv" --summary-output "$OUT/overlap_summary_metrics.csv" --warmup 5 --iterations 30
"$PYTHON" scripts/plot_phase0.py
