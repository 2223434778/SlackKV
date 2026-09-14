#!/usr/bin/env python3
"""Regenerate all Phase 1A plots from committed CSV files only."""
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


OUT = Path("results/phase1a")
PLOTS = OUT / "plots"
POLICIES = ("OnDemand", "ASAP-EDF", "NaiveJIT", "SlackKV-QueueJIT")


def read(name):
    with (OUT / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def grouped(rows, value, extra=lambda row: True):
    values = defaultdict(list)
    for row in rows:
        if extra(row):
            values[(row["policy"], float(row["rho"]))].append(float(row[value]))
    return {key: statistics.median(items) for key, items in values.items()}


def line_plot(rows, metric, path, ylabel, selector):
    values = grouped(rows, metric, selector)
    plt.figure(figsize=(8, 5))
    for policy in POLICIES:
        points = sorted((rho, value) for (candidate, rho), value in values.items() if candidate == policy)
        if points:
            plt.plot([point[0] for point in points], [point[1] for point in points], marker="o", label=policy)
    plt.xlabel("Nominal offered load, ρ = λE[p]"); plt.ylabel(ylabel); plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(PLOTS / path, dpi=160); plt.close()


def residency_series(rows, policy):
    events = []
    for row in rows:
        completion = float(row["completion_time_s"]); deadline = float(row["deadline_s"]); size = int(row["kv_size_bytes"]) / 1e9
        if deadline > completion:
            events.extend(((completion, size), (deadline, -size)))
    level = 0.0; times = []; levels = []
    for time_s, delta in sorted(events):
        times.append(time_s); levels.append(level); level += delta; times.append(time_s); levels.append(level)
    return times, levels


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    aggregate = read("aggregate_metrics.csv")
    base = lambda row: row["experiment"] == "stochastic_stream" and row["source_tier"] == "cpu" and float(row["service_multiplier"]) == 1.0
    line_plot(aggregate, "deadline_miss_rate", "deadline_miss_rate_vs_load.png", "Deadline miss rate", base)
    line_plot(aggregate, "stall_ms_p95", "p95_stall_vs_load.png", "P95 resume stall (ms)", base)
    line_plot(aggregate, "prefetch_vram_time_gbs_total", "vram_time_vs_load.png", "Prefetch VRAM-time (GB·s)", base)
    values = grouped([row for row in aggregate if base(row) and float(row["rho"]) == .75], "stall_ms_p95")
    ratio = grouped([row for row in aggregate if base(row) and float(row["rho"]) == .75], "prefetch_to_keep_ratio")
    plt.figure(figsize=(7,5))
    for policy in POLICIES:
        key=(policy,.75)
        if key in values: plt.scatter(ratio[key], values[key], s=70); plt.annotate(policy, (ratio[key], values[key]), xytext=(5,5), textcoords="offset points")
    plt.scatter(1.0, 0.0, s=70, marker="x", label="KeepReference"); plt.xlabel("Normalized prefetch VRAM-time / KeepReference"); plt.ylabel("P95 resume stall (ms)"); plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(PLOTS / "stall_vs_vram_time_tradeoff.png", dpi=160); plt.close()
    subset=[row for row in aggregate if base(row) and row["policy"] in ("NaiveJIT","SlackKV-QueueJIT")]
    line_plot(subset, "deadline_miss_rate", "naivejit_vs_queuejit.png", "Deadline miss rate", lambda row: True)
    raw=read("metrics.csv")
    example=[row for row in raw if row["run_id"] == "stochastic_stream|cpu|m1|r0.75|n5000|s101"]
    plt.figure(figsize=(9,5))
    for policy in POLICIES:
        times, levels=residency_series([row for row in example if row["policy"]==policy], policy)
        if times: plt.plot(times, levels, label=policy)
    plt.xlabel("Simulation time (s)"); plt.ylabel("Early-prefetched KV resident (GB)"); plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(PLOTS / "prefetched_memory_over_time_example.png", dpi=160); plt.close()
    sensitivity=[row for row in aggregate if row["experiment"]=="service_sensitivity" and row["policy"] in ("NaiveJIT","SlackKV-QueueJIT")]
    plt.figure(figsize=(8,5))
    for policy in ("NaiveJIT","SlackKV-QueueJIT"):
        points=sorted((float(row["service_multiplier"]),float(row["stall_ms_p95"])) for row in sensitivity if row["policy"]==policy)
        plt.plot([x for x,_ in points],[y for _,y in points],marker="o",label=policy)
    plt.xlabel("Service-time multiplier (sensitivity only)"); plt.ylabel("P95 resume stall (ms)"); plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(PLOTS / "service_time_sensitivity.png", dpi=160); plt.close()


if __name__ == "__main__":
    main()
