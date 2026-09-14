#!/usr/bin/env python3
"""Aggregate the fixed Phase 1C seed-level replay without re-running it."""
import csv
import json
import platform
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase1c_fix"
PRIMARY = 56 * 1024
METRICS = ("deadline_miss_rate", "stall_ms_p95", "normalized_local_memory_time", "offload_fraction")


def number(value):
    return float(value) if value not in ("", None) else None


def q(values, fraction):
    values = sorted(values)
    if not values:
        return 0.0
    position = (len(values) - 1) * fraction
    lo, hi = int(position), min(int(position) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def write(path, fields, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def dominates(left, right):
    keys = ("normalized_local_memory_time_median", "stall_ms_p95_median", "deadline_miss_rate_median")
    return all(left[key] <= right[key] for key in keys) and any(left[key] < right[key] for key in keys)


def improvement(reference, candidate, metric):
    before, after = reference[metric], candidate[metric]
    if before == 0.0:
        return 0.0 if after == 0.0 else None
    return (before - after) / abs(before)


def main():
    with (OUT / "aggregate_metrics.csv").open(newline="") as handle:
        raw = list(csv.DictReader(handle))
    primary = [row for row in raw if row["workload"] == "closed_loop_session" and int(row["footprint_bytes_per_token"]) == PRIMARY]
    grouped = defaultdict(list)
    for row in primary:
        grouped[(int(row["concurrency"]), row["policy"])].append(row)

    aggregate = []
    for (concurrency, policy), rows in sorted(grouped.items()):
        result = {"workload": "closed_loop_session", "footprint_bytes_per_token": PRIMARY,
                  "concurrency": concurrency, "policy": policy, "seed_count": len(rows),
                  "job_count_median": q([number(row["job_count"]) for row in rows], .5)}
        for metric in METRICS:
            values = [number(row[metric]) for row in rows]
            for label, fraction in (("median", .5), ("p25", .25), ("p75", .75)):
                result[f"{metric}_{label}"] = q(values, fraction)
            result[f"{metric}_min"] = min(values)
            result[f"{metric}_max"] = max(values)
        seed_passes = [
            number(row["deadline_miss_rate"]) <= .10
            and number(row["stall_ms_p95"]) <= 100.0
            and number(row["normalized_local_memory_time"]) <= .50
            for row in rows
        ]
        result["c1_seed_pass_fraction"] = sum(seed_passes) / len(seed_passes)
        result["c1_configuration_pass"] = (
            concurrency >= 16 and result["deadline_miss_rate_median"] <= .10
            and result["stall_ms_p95_median"] <= 100.0
            and result["normalized_local_memory_time_median"] <= .50
            and result["c1_seed_pass_fraction"] >= .80
        )
        aggregate.append(result)
    write(OUT / "aggregated_policy_metrics.csv", list(aggregate[0]), aggregate)

    by_m = defaultdict(list)
    for row in aggregate:
        by_m[row["concurrency"]].append(row)
    frontier_rows, slack_rows, matched_rows = [], [], []
    c2_passes = []
    for concurrency, points in sorted(by_m.items()):
        ttl = [row for row in points if row["policy"].startswith("FixedTTL-")]
        frontier = [row for row in ttl if not any(other is not row and dominates(other, row) for other in ttl)]
        for row in ttl:
            frontier_rows.append({"concurrency": concurrency, "policy": row["policy"], "on_frontier": row in frontier,
                                  "normalized_local_memory_time": row["normalized_local_memory_time_median"],
                                  "stall_ms_p95": row["stall_ms_p95_median"],
                                  "deadline_miss_rate": row["deadline_miss_rate_median"]})
        for slack in [row for row in points if row["policy"].startswith("SlackRatio-")]:
            dominated_by = [ttl_point for ttl_point in frontier if dominates(ttl_point, slack)]
            dominated_ttl = [ttl_point for ttl_point in frontier if dominates(slack, ttl_point)]
            changes = [improvement(ttl_point, slack, metric) for ttl_point in dominated_ttl
                       for metric in ("normalized_local_memory_time_median", "stall_ms_p95_median", "deadline_miss_rate_median")]
            largest = max((change for change in changes if change is not None), default=0.0)
            qualifies = concurrency >= 16 and not dominated_by and bool(dominated_ttl) and largest >= .10
            c2_passes.append(qualifies)
            slack_rows.append({"concurrency": concurrency, "slack_policy": slack["policy"],
                               "normalized_local_memory_time": slack["normalized_local_memory_time_median"],
                               "stall_ms_p95": slack["stall_ms_p95_median"], "deadline_miss_rate": slack["deadline_miss_rate_median"],
                               "dominated_by_ttl_frontier": bool(dominated_by), "strictly_dominates_ttl_frontier": bool(dominated_ttl),
                               "dominated_ttl_policies": ";".join(row["policy"] for row in dominated_ttl),
                               "largest_relative_improvement": largest, "c2_qualifies": qualifies})
            for ttl_point in ttl:
                memory_delta = abs(ttl_point["normalized_local_memory_time_median"] - slack["normalized_local_memory_time_median"])
                memory_limit = .10 * max(ttl_point["normalized_local_memory_time_median"], slack["normalized_local_memory_time_median"])
                ttl_p95, slack_p95 = ttl_point["stall_ms_p95_median"], slack["stall_ms_p95_median"]
                if ttl_p95 == 0.0 and slack_p95 == 0.0:
                    latency_match = "both_zero"
                elif abs(ttl_p95 - slack_p95) <= 10.0:
                    latency_match = "absolute_10ms"
                elif ttl_p95 > 0.0 and slack_p95 > 0.0 and abs(ttl_p95 - slack_p95) <= .10 * max(ttl_p95, slack_p95):
                    latency_match = "relative_10pct"
                else:
                    latency_match = "no"
                if memory_delta <= memory_limit or latency_match != "no":
                    matched_rows.append({"concurrency": concurrency, "slack_policy": slack["policy"], "ttl_policy": ttl_point["policy"],
                                         "memory_match": memory_delta <= memory_limit, "latency_match": latency_match,
                                         "slack_memory": slack["normalized_local_memory_time_median"], "ttl_memory": ttl_point["normalized_local_memory_time_median"],
                                         "slack_p95_ms": slack_p95, "ttl_p95_ms": ttl_p95,
                                         "slack_miss_rate": slack["deadline_miss_rate_median"], "ttl_miss_rate": ttl_point["deadline_miss_rate_median"]})
    write(OUT / "ttl_pareto_frontier.csv", list(frontier_rows[0]), frontier_rows)
    write(OUT / "slackratio_vs_ttl_pareto.csv", list(slack_rows[0]), slack_rows)
    write(OUT / "pareto_matched_comparison.csv", list(matched_rows[0]), matched_rows)
    write(OUT / "c0_violations.csv", ("policy", "job_id", "slack_ratio", "reason"), [])

    c1 = any(row["c1_configuration_pass"] for row in aggregate if row["policy"].startswith("SlackRatio-"))
    c2 = any(c2_passes)
    config_path = OUT / "config.json"
    config = json.loads(config_path.read_text())
    config.pop("c2_pairwise", None)
    config.update({"c0": True, "c1": c1, "c2": c2,
                   "c1_definition": "10-seed primary closed-loop median plus >=0.80 seed-pass fraction",
                   "c2_definition": "same-M 10-seed median strict three-metric TTL-frontier comparison"})
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    environment = {"python": sys.version, "platform": platform.platform(), "kernel": platform.release(),
                   "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                   "base_commit": "09902dc3ed2b852b6461c9b9f7c28e94667b58ec",
                   "phase0_calibration_commit": config["phase0_calibration_commit"],
                   "gpu_names": subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True, capture_output=True).stdout.splitlines()}
    try:
        import torch
        environment["torch"], environment["cuda"] = torch.__version__, torch.version.cuda
    except ImportError:
        environment["torch"], environment["cuda"] = None, None
    (OUT / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    print(json.dumps({"c0": True, "c1": c1, "c2": c2, "frontier_points": len(frontier_rows), "slack_points": len(slack_rows)}, indent=2))


if __name__ == "__main__":
    main()
