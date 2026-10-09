"""Read-only reassessment of committed metrics; no runtime/model imports."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

BASELINE = "e2e904c2ebe344f1d4dd8cd347fb6fabd246a6f8"


def number(value):
    if value in (None, "", "NA", "UNKNOWN", "None"):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def nearest_rank(values, fraction=0.95):
    if not values or not 0 < fraction <= 1:
        raise ValueError("invalid quantile input")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("non-finite quantile input")
    return sorted(values)[math.ceil(len(values) * fraction) - 1]


def remote_intervals(events):
    """Return observed remote waits before first output, or UNKNOWN (None)."""
    ordered = sorted(events, key=lambda row: float(row["timestamp_monotonic"]))
    first = next((float(row["timestamp_monotonic"]) for row in ordered
                  if row["event"] == "FIRST_TOKEN"), None)
    names = {row["event"] for row in ordered}
    if first is None or not {"RESUME_ARRIVED", "RESUME_ADMITTED", "REQUEST_FINISHED"} <= names:
        return None
    intervals = []
    start = None
    for row in ordered:
        time = float(row["timestamp_monotonic"])
        if time > first:
            break
        if row["event"] == "WAITING_FOR_REMOTE_KVS":
            if start is not None:
                return None
            start = time
        elif row["event"] == "REMOTE_KV_READY":
            if start is None:
                return None
            intervals.append((start, time))
            start = None
    return intervals if start is None else None


def deletion_improvement(rows, field):
    if any(row[field] is None for row in rows):
        return None
    original = nearest_rank([row["deadline_latency_ms"] for row in rows])
    counterfactual = nearest_rank([
        max(0.0, row["deadline_latency_ms"] - row[field]) for row in rows
    ])
    return 100 * (original - counterfactual) / original if original else None


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key: "UNKNOWN" if value is None else value
                         for key, value in row.items()} for row in rows)


def analyze(root, output):
    phase = root / "results/phase4g_a_fix_k2repair"
    inputs = [phase / name for name in ("boundary_metrics.csv", "resume_trace.csv",
                                       "session_metrics.csv", "run_metrics.csv", "prefetch_metrics.csv")]
    inputs += [root / "results/phase4f_b/paired_deltas.csv",
               root / "results/phase4g_d/allocator_truth.csv",
               root / "results/phase4g_d/frontier_recoverability.csv",
               root / "results/phase4g_c_fix/allocator_guard_comparison.csv",
               root / "results/phase2c_fix/resume_metrics.csv",
               root / "results/phase2d/metrics.csv",
               root / "results/phase4e_a_rca/liveness_runs.csv",
               root / "results/phase4g_d/controller_selection.csv",
               root / "results/phase4g_d/frontier_shadow.csv"]
    events = defaultdict(list)
    for row in read_csv(inputs[1]):
        events[row["policy"], row["repetition"], row["session"]].append(row)
    boundaries = []
    groups = defaultdict(list)
    for row in read_csv(inputs[0]):
        session = row["episode_id"] + ":turn:" + row["resume_turn_index"]
        trace = events[row["policy"], row["repetition"], session]
        intervals = remote_intervals(trace)
        admit = next((float(item["timestamp_monotonic"]) for item in sorted(
            trace, key=lambda item: float(item["timestamp_monotonic"]))
                      if item["event"] == "RESUME_ADMITTED"), None)
        remote = None if intervals is None else sum((end - start) * 1000 for start, end in intervals)
        preadmit_remote = None if intervals is None or admit is None else sum(
            max(0.0, min(end, admit) - start) * 1000 for start, end in intervals)
        result = {key: row[key] for key in ("policy", "repetition", "episode_id", "boundary_id")}
        for key in ("deadline_latency_ms", "TTFT_ms", "scheduler_wait_ms", "post_admit_to_first_ms"):
            result[key] = number(row[key])
            if result[key] is None:
                raise ValueError(f"missing measured latency: {row['boundary_id']} {key}")
        if abs(result["TTFT_ms"] - result["scheduler_wait_ms"] - result["post_admit_to_first_ms"]) > 0.001:
            raise ValueError("latency components do not add up")
        if preadmit_remote is not None and preadmit_remote > result["scheduler_wait_ms"] + 0.001:
            raise ValueError("remote wait exceeds enclosing interval")
        result.update(remote_wait_before_first_ms=remote, remote_wait_before_admit_ms=preadmit_remote,
                      nonremote_pre_admit_ms=None if preadmit_remote is None else max(
                          0, result["scheduler_wait_ms"] - preadmit_remote),
                      erase_remote_and_all_post_ms=None if preadmit_remote is None else
                          preadmit_remote + result["post_admit_to_first_ms"],
                      erase_entire_pre_admit_ms=result["scheduler_wait_ms"])
        boundaries.append(result)
        groups[row["policy"], row["repetition"]].append(result)
    runs = []
    for (policy, rep), rows in sorted(groups.items()):
        if len(rows) != 32:
            raise ValueError("incomplete run")
        tail = sorted(rows, key=lambda row: row["deadline_latency_ms"])[30]
        complete = all(row["remote_wait_before_first_ms"] is not None for row in rows)
        runs.append({"policy": policy, "repetition": rep, "boundaries": len(rows),
                     "remote_unknown": sum(row["remote_wait_before_first_ms"] is None for row in rows),
                     "deadline_P95_ms": nearest_rank([row["deadline_latency_ms"] for row in rows]),
                     "remote_P50_ms": statistics.median([row["remote_wait_before_first_ms"] for row in rows]) if complete else None,
                     "remote_P95_ms": nearest_rank([row["remote_wait_before_first_ms"] for row in rows]) if complete else None,
                     "tail_boundary_id": tail["boundary_id"],
                     "tail_nonremote_pre_admit_ms": tail["nonremote_pre_admit_ms"],
                     "tail_remote_ms": tail["remote_wait_before_first_ms"],
                     "tail_post_admit_ms": tail["post_admit_to_first_ms"],
                     "fixed_path_remote_deletion_P95_improvement_percent": deletion_improvement(rows, "remote_wait_before_first_ms"),
                     "fixed_path_remote_plus_all_post_deletion_P95_improvement_percent": deletion_improvement(rows, "erase_remote_and_all_post_ms"),
                     "erase_all_pre_admit_diagnostic_P95_improvement_percent": deletion_improvement(rows, "erase_entire_pre_admit_ms")})
    sessions = []
    for row in read_csv(inputs[2]):
        selected = [item for item in groups[row["policy"], row["repetition"]]
                    if item["episode_id"] == row["episode_id"]]
        total = sum(item["remote_wait_before_first_ms"] for item in selected) if all(
            item["remote_wait_before_first_ms"] is not None for item in selected) else None
        e2e = number(row["session_e2e_ms"])
        sessions.append({"policy": row["policy"], "repetition": row["repetition"],
                         "episode_id": row["episode_id"], "session_e2e_ms": e2e,
                         "four_boundary_remote_ms": total,
                         "fixed_path_E2E_deletion_percent": 100 * total / e2e if total is not None and e2e else None})
    summary = {"baseline_commit": BASELINE, "scope": "Offline fixed-observed-path deletion; NOT a rescheduled-system causal bound",
               "input_sha256": {str(path.relative_to(root)).replace('\\', '/'): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},
               "boundary_count": len(boundaries), "policies": {}}
    for policy in sorted({row["policy"] for row in runs}):
        selected = [row for row in runs if row["policy"] == policy]
        summary["policies"][policy] = {key: {"median": statistics.median(values), "min": min(values), "max": max(values)}
            if (values := [row[key] for row in selected if row[key] is not None]) else None
            for key in selected[0] if key not in {"policy", "repetition", "tail_boundary_id"}}
        summary["policies"][policy]["session_fixed_path_E2E_deletion_percent_median"] = statistics.median(
            row["fixed_path_E2E_deletion_percent"] for row in sessions if row["policy"] == policy)
    summary["promotion_status_counts"] = dict(Counter(row["status"] for row in read_csv(inputs[4])))
    summary["phase4f_paired_P95_percent"] = {comparison: statistics.median(
        100 * float(row["P95_relative_delta"]) for row in read_csv(inputs[5]) if row["comparison"] == comparison)
        for comparison in sorted({row["comparison"] for row in read_csv(inputs[5])})}
    truth = [row for row in read_csv(inputs[6]) if row["context"] == "SELECTION_CANDIDATE" and row["eligible_selection"] == "True"]
    if len(truth) != 46:
        raise ValueError("unexpected eligible allocator cohort")
    summary["phase4g_d_eligible_allocator"] = dict(Counter(row["allocator_fit"] for row in truth))
    frontier = read_csv(inputs[7])
    summary["phase4g_d_frontier"] = {"selections": len(frontier),
        "F0_blocked_including_unknown": sum(row["current_F0_PSA_reject"] == "True" for row in frontier),
        "exact_F0_blocked": sum(row["current_F0_PSA_reject"] == "True" and row["exact_allocator_fit"] != "UNKNOWN" for row in frontier),
        "shadow_recoverable": sum(row["frontier_recoverable_selection"] == "True" for row in frontier)}
    from experiments.phase4g_d.analysis import reanalyze_allocator_csv
    summary["phase4g_c_corrected_allocator"] = reanalyze_allocator_csv(inputs[8])
    summary["phase4g_c_corrected_allocator"]["source_artifact"] = "results/phase4g_c_fix/allocator_guard_comparison.csv"
    measured = read_csv(inputs[3])
    if len(measured) != 16 or any(row["valid"] != "True" for row in measured):
        raise ValueError("incomplete or invalid measured matrix")
    paired = []
    for rep in sorted({row["repetition"] for row in measured}):
        by_policy = {row["policy"]: row for row in measured if row["repetition"] == rep}
        old, new = by_policy["OnDemand-K2"], by_policy["FullSlackKV+PSA1-K2"]
        paired.append(100 * (float(new["deadline_P95_ms"]) / float(old["deadline_P95_ms"]) - 1))
    summary["phase4g_paired_P95_change_percent"] = statistics.median(paired)
    early = read_csv(inputs[9])
    summary["phase2c_resume_TTFT_median_ms"] = {label: statistics.median(float(row["resume_ttft_ms"]) for row in early if row["admission"] == label)
                                              for label in sorted({row["admission"] for row in early})}
    summary["phase2d_effective_resume_median_ms"] = {row["policy"]: float(row["median_effective_resume_latency_ms"]) for row in read_csv(inputs[10])}
    summary["phase4e_liveness_counts"] = dict(Counter(row["classification"] for row in read_csv(inputs[11])))
    selections = read_csv(inputs[12])
    summary["phase4g_d_snapshot_timing_ms"] = {}
    for key in ("selection_to_scheduler_ms", "snapshot_roundtrip_ms", "source_sample_age_ms"):
        values = [value for row in selections if (value := number(row[key])) is not None]
        summary["phase4g_d_snapshot_timing_ms"][key] = {
            "known": len(values), "median": statistics.median(values),
            "P95": nearest_rank(values), "max": max(values)}
    summary["phase4g_d_frontier_unknown_lease_counts"] = {
        label: sum(row.get(label + "_id") not in ("", "NA") and
                   number(row.get(label + "_existing_lease_blocks")) is None
                   for row in read_csv(inputs[13])) for label in ("F0", "F1", "F2", "F3", "F4")}
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "boundary_decomposition.csv", boundaries)
    write_csv(output / "run_opportunity.csv", runs)
    write_csv(output / "session_opportunity.csv", sessions)
    (output / "audit.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "input_sha256"}, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/research_reassessment"))
    args = parser.parse_args()
    analyze(Path(__file__).resolve().parents[1], args.output)
