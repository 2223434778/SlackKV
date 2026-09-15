#!/usr/bin/env python3
"""Regenerate the compact Phase 1E safety-margin plots from aggregate CSV."""
import csv
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
rows = list(csv.DictReader((ROOT / "results/phase1e/aggregated_metrics.csv").open()))
fixed = [r for r in rows if r["policy"] == "FixedTTL-1"]
plots = ROOT / "results/phase1e/plots"
plots.mkdir(parents=True, exist_ok=True)

for metric, label, filename in [
    ("deadline_miss_rate", "Deadline miss rate", "miss_rate_vs_margin.png"),
    ("stall_ms_p95", "P95 resume stall (ms)", "p95_stall_vs_margin.png"),
    ("normalized_local_memory_time", "Normalized local memory-time", "memory_time_vs_margin.png"),
]:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for sigma in (0.1, 0.25, 0.5):
        group = sorted((r for r in fixed if float(r["sigma"]) == sigma), key=lambda r: float(r["margin_ms"]))
        ax.plot([float(r["margin_ms"]) for r in group], [float(r[metric]) for r in group], marker="o", label=f"σ={sigma:g}")
    ax.set_xlabel("Safety margin (ms)")
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / filename, dpi=180)
    plt.close(fig)
