#!/usr/bin/env python3
"""Regenerate the compact Phase 1F plots from aggregated metrics."""
import csv
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
rows = list(csv.DictReader((ROOT / "results/phase1f/aggregated_metrics.csv").open()))
strategies = ("NoMargin", "Fixed500ms", "Adaptive-0.5", "Adaptive-1.0")
sigmas = (0.1, 0.25, 0.5)
plots = ROOT / "results/phase1f/plots"
plots.mkdir(parents=True, exist_ok=True)

for metric, label, filename in [
    ("deadline_miss_rate", "Deadline miss rate", "miss_rate_vs_sigma.png"),
    ("normalized_local_memory_time", "Normalized local memory-time", "memory_time_vs_sigma.png"),
]:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for strategy in strategies:
        group = [next(r for r in rows if r["strategy"] == strategy and float(r["sigma"]) == sigma) for sigma in sigmas]
        ax.plot(sigmas, [float(r[metric]) for r in group], marker="o", label=strategy)
    ax.set_xlabel("Prediction noise σ")
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / filename, dpi=180)
    plt.close(fig)

fig, ax = plt.subplots(figsize=(6.4, 4.2))
for sigma in sigmas:
    group = [next(r for r in rows if r["strategy"] == strategy and float(r["sigma"]) == sigma) for strategy in strategies]
    ax.plot([float(r["normalized_local_memory_time"]) for r in group], [float(r["stall_ms_p95"]) for r in group], marker="o", label=f"σ={sigma:g}")
    for row in group:
        ax.annotate(row["strategy"].replace("Adaptive-", "A-"), (float(row["normalized_local_memory_time"]), float(row["stall_ms_p95"])), fontsize=7)
ax.set_xlabel("Normalized local memory-time")
ax.set_ylabel("P95 resume stall (ms)")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(plots / "stall_vs_memory_tradeoff.png", dpi=180)
