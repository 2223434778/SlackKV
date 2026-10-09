"""Independent per-run observed-path opportunity and descriptive paired results."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from experiments.autonomous_admission.analyze import nearest_rank
from experiments.autonomous_admission.artifacts import read_json, run_paths
from scripts.research_reassessment import remote_intervals


def observed_opportunity(result: dict) -> dict:
    grouped = defaultdict(list)
    for event in result["resume_trace"]:
        grouped[event["session"]].append(event)
    latencies, removed, waits = [], [], []
    for turn in result["turn_rows"]:
        if turn["turn_index"] == 0:
            continue
        intervals = remote_intervals(grouped[f'{turn["episode_id"]}:turn:{turn["turn_index"]}'])
        if intervals is None:
            return {"status": "UNKNOWN"}
        wait = sum((end - start) * 1000 for start, end in intervals)
        latency = (turn["first_ns"] - turn["issue_ns"]) / 1e6
        if wait > latency + .01:
            raise ValueError("remote intervals exceed TTFT")
        latencies.append(latency)
        removed.append(max(0, latency - wait))
        waits.append(wait)
    original = nearest_rank(latencies, .95)
    return {"status": "OBSERVED", "remote_wait_median_ms": statistics.median(waits),
            "remote_wait_p95_ms": nearest_rank(waits, .95),
            "fixed_path_remote_deletion_p95_improvement_percent":
                100 * (original - nearest_rank(removed, .95)) / original,
            "caveat": "fixed observed timeline subtraction; NOT a rescheduled causal upper bound"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    rows = []
    for execution in sorted(args.directory.glob("*/execution.json")):
        metadata = json.loads(execution.read_text())
        runs = run_paths(execution.parent)
        row = {"name": execution.parent.name, "execution": metadata}
        if runs:
            result = read_json(runs[0])
            row["observed_opportunity"] = observed_opportunity(result)
        rows.append(row)
    summary = {"runs": rows, "gpu_hours_conservative": sum(
        row["execution"].get("gpu_hours_conservative", 0) for row in rows)}
    pairs = [json.loads((args.directory / name).read_text()) for name in
             ("execution_comparison.json", "execution_reverse_comparison.json")]
    matched = [pair for pair in pairs if pair["prompt_hashes_match"]]
    summary["execution_input_matched_pair_count"] = len(matched)
    summary["execution_excluded_input_drift_pair_count"] = len(pairs) - len(matched)
    summary["execution_input_matched_descriptive"] = {
        key: {"median_percent_change": statistics.median(pair["percent_change"][key] for pair in matched),
              "individual_percent_changes": [pair["percent_change"][key] for pair in matched]}
        for key in pairs[0]["percent_change"]} if matched else "UNKNOWN"
    summary["execution_structure_only_observations"] = [pair["percent_change"] for pair in pairs
                                                        if not pair["prompt_hashes_match"]]
    summary["not_claimed"] = ["statistical significance", "production generalization", "novel algorithm",
                              "quality equivalence in non-invariant mode", "complete GPU KV memory-time"]
    summary["not_claimed"].append("repeated acceleration with exactly identical inputs")
    (args.directory / "round_summary.json").write_text(json.dumps(summary, indent=2) + "\n", newline="\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
