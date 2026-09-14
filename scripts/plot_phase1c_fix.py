#!/usr/bin/env python3
"""Regenerate compact Phase 1C-Fix Pareto plots from aggregate CSV files."""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "phase1c_fix"
PLOTS = OUT / "plots"


def read(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def f(row, key):
    return float(row[key])


def scatter(ax, rows, y_key, title):
    for kind, color, marker in (("FixedTTL-", "#4c78a8", "o"), ("SlackRatio-", "#f58518", "s"), ("OffloadAll", "#e45756", "^"), ("KeepAll", "#54a24b", "D")):
        selected = [row for row in rows if row["policy"].startswith(kind)]
        if selected:
            ax.scatter([f(row, "normalized_local_memory_time_median") for row in selected], [f(row, y_key) for row in selected],
                       label=kind.rstrip("-"), color=color, marker=marker, alpha=.8)
    ax.set_xlabel("Normalized local-memory time")
    ax.set_ylabel("P95 stall (ms)" if y_key == "stall_ms_p95_median" else "Deadline miss rate")
    ax.set_title(title)
    ax.grid(alpha=.25)


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    aggregate = read(OUT / "aggregated_policy_metrics.csv")
    for y_key, name in (("stall_ms_p95_median", "p95_vs_memory_time.png"), ("deadline_miss_rate_median", "miss_rate_vs_memory_time.png")):
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
        for ax, m in zip(axes, (16, 32, 64)):
            scatter(ax, [row for row in aggregate if int(row["concurrency"]) == m], y_key, f"M={m}")
        axes[0].legend(fontsize=8)
        fig.savefig(PLOTS / name, dpi=170)
        plt.close(fig)
    frontier = read(OUT / "ttl_pareto_frontier.csv")
    for m in (16, 32, 64):
        fig, ax = plt.subplots(figsize=(5.4, 4.1), constrained_layout=True)
        rows = [row for row in aggregate if int(row["concurrency"]) == m]
        scatter(ax, rows, "stall_ms_p95_median", f"TTL frontier and SlackRatio, M={m}")
        front = [row for row in frontier if int(row["concurrency"]) == m and row["on_frontier"] == "True"]
        ax.plot([f(row, "normalized_local_memory_time") for row in front], [f(row, "stall_ms_p95") for row in front], color="#4c78a8", alpha=.45)
        ax.legend(fontsize=8)
        fig.savefig(PLOTS / f"ttl_pareto_vs_slackratio_m{m}.png", dpi=170)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.2, 4.1), constrained_layout=True)
    for m, color in ((16, "#4c78a8"), (32, "#f58518"), (64, "#54a24b")):
        rows = sorted([row for row in aggregate if int(row["concurrency"]) == m and row["policy"].startswith("SlackRatio-")], key=lambda row: float(row["policy"].split("-", 1)[1]))
        gamma = [float(row["policy"].split("-", 1)[1]) for row in rows]
        ax.plot(gamma, [f(row, "deadline_miss_rate_median") for row in rows], marker="o", label=f"M={m}", color=color)
    ax.set_xlabel("SlackRatio gamma")
    ax.set_ylabel("Median deadline miss rate")
    ax.grid(alpha=.25); ax.legend()
    fig.savefig(PLOTS / "slackratio_gamma_sensitivity.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
