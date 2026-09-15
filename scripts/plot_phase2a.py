#!/usr/bin/env python3
"""Regenerate Phase 2A plots from committed compact CSV artifacts."""
import csv
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/phase2a/plots"
OUT.mkdir(parents=True, exist_ok=True)
transfer = list(csv.DictReader((ROOT / "results/phase2a/transfer_metrics.csv").open()))
jit = list(csv.DictReader((ROOT / "results/phase2a/jit_metrics.csv").open()))

sizes = [float(row["size_mib"]) for row in transfer]
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for key, label, style in [
    ("d2h_median_ms", "Runtime D2H", "o-"),
    ("h2d_median_ms", "Runtime H2D", "s-"),
    ("phase0_d2h_p50_ms", "Phase 0 D2H P50", "o--"),
    ("phase0_h2d_p50_ms", "Phase 0 H2D P50", "s--"),
]:
    ax.plot(sizes, [float(row[key]) for row in transfer], style, label=label)
ax.set_xscale("log", base=2)
ax.set_xlabel("Tensor size (MiB)")
ax.set_ylabel("Latency (ms)")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "transfer_latency_vs_size.png", dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.4, 4.2))
for pause_s in (1.0, 2.0, 5.0):
    rows = [row for row in jit if float(row["pause_s"]) == pause_s]
    ax.plot(
        [float(row["margin_ms"]) for row in rows],
        [float(row["resume_stall_ms_median"]) for row in rows],
        marker="o",
        label=f"pause={pause_s:g}s",
    )
ax.set_xlabel("Safety margin (ms)")
ax.set_ylabel("Median resume stall (ms)")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "jit_stall_vs_margin.png", dpi=180)
