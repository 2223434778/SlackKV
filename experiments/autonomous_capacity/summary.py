"""Recompute measured capacity probes and frozen algorithm screen gates."""
from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

from experiments.autonomous_admission.artifacts import read_json, run_paths
from experiments.autonomous_admission.compare import compare
from experiments.autonomous_admission.round_summary import observed_opportunity
from experiments.phase4c.synthetic_tokens import token_hash
from experiments.autonomous_capacity.audit import counter


def input_hashes(path: Path) -> dict[str, str]:
    raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    rows = [json.loads(line) for line in raw.decode().splitlines()]
    hashes = {row["session"]: token_hash(row["prompt"]) for row in rows}
    if len(rows) != 40 or set(hashes) != {f"E{i:02}:turn:{j}" for i in range(8) for j in range(5)}:
        raise ValueError("invalid or duplicate actual input identities")
    return hashes


def screen(baseline: dict, candidate: dict, a_config: dict, b_config: dict) -> dict:
    result = compare(baseline, candidate)
    controls = ("kv_bytes", "gate", "source_sha", "execution_mode", "batch_invariant",
                "allocator_trace", "scheduler_sha256", "allocator_sha256", "bg_tokens")
    matched = all(a_config[key] == b_config[key] for key in controls)
    deltas = result["percent_change"]
    gates = {"config_match": matched,
             "inputs_match": result["prompt_hashes_match"],
             "lifecycle": baseline["valid"] and candidate["valid"],
             "p95": deltas["resume_p95_ms"] <= -10,
             "session_median": deltas["session_e2e_median_ms"] <= 0,
             "background_wall": deltas["BG_wall_ms"] <= 5,
             "background_tpot": deltas["BG_TPOT_ms"] <= 5,
             "h2d": deltas["agent_external_H2D_MiB"] <= 5}
    # Numerical identity is recorded separately; performance gate is not a KV oracle.
    result.update(gates=gates, eligible_for_replication=all(gates.values()),
                  quality_equivalence="UNKNOWN unless separate correctness oracle",
                  purpose="bounded actual-demand continuation priority",
                  conclusion="screen only; no statistical or novelty claim")
    result.pop("pass_screen")
    return result


def summarize(directory: Path) -> dict:
    rows, runs, configs = [], {}, {}
    for execution in sorted(directory.glob("*/execution.json")):
        config = json.loads(execution.read_text())
        row = {"name": execution.parent.name, "execution": config}
        paths = run_paths(execution.parent)
        if len(paths) > 1:
            raise ValueError("multiple runs in one execution")
        if paths:
            run = read_json(paths[0])
            hashes_path = execution.parent / "inputs.jsonl"
            if not hashes_path.exists():
                hashes_path = hashes_path.with_suffix(".jsonl.gz")
            actual = input_hashes(hashes_path)
            reported = {f'{t["episode_id"]}:turn:{t["turn_index"]}': t["prompt_sha256"]
                        for t in run["turn_rows"]}
            if actual != reported:
                raise ValueError("actual inputs differ from reported prompt hashes")
            row["actual_input_hashes"] = actual
            row["analysis"] = read_json(execution.parent / "analysis.json")
            h2d = counter(execution.parent / "metrics_after_timed_workload.prom",
                          "vllm:kv_offload_total_bytes_total", 'transfer_type="CPU_to_GPU"')
            row["actual_KV_H2D_MiB"] = None if h2d is None else h2d / 2**20
            row["observed_opportunity"] = observed_opportunity(run)
            agent_seconds = (max(t["finish_ns"] for t in run["turn_rows"])
                             - min(t["issue_ns"] for t in run["turn_rows"])) / 1e9
            row["agent_window_seconds"] = agent_seconds
            row["agent_generated_tokens_per_second"] = sum(
                t["output_tokens"] for t in run["turn_rows"]) / agent_seconds
            row["allocated_pool_GiB_seconds_BG_window"] = (config["kv_bytes"] / 2**30
                                                          * run["BG_total_wall_ms"] / 1000)
            source_counts = Counter(event["session"] for event in run["source_trace"])
            row["repeated_source_sessions"] = {k: v for k, v in source_counts.items() if v > 1}
            row["initial_issue_order"] = [t["episode_id"] for t in sorted(
                [t for t in run["turn_rows"] if t["turn_index"] == 0], key=lambda t: t["issue_ns"])]
            resumes = [r for r in row["analysis"]["requests"] if r["turn_index"] > 0]
            tail = sorted(resumes, key=lambda r: r["ttft_ms"])[30]
            row["tail_boundary"] = tail
            runs[execution.parent.name], configs[execution.parent.name] = run, config
        rows.append(row)
    pairs = {}
    for label, a, b in (("credit_2", "credit-control-2", "credit-candidate-2"),
                        ("credit_reverse_2", "credit-control-2-reverse", "credit-candidate-2-reverse"),
                        ("credit_4", "credit-control-4", "credit-candidate-4")):
        if a in runs and b in runs:
            pairs[label] = screen(runs[a], runs[b], configs[a], configs[b])
    for label, a, b in (("epoch_2", "epoch-control-2", "epoch-candidate-2"),
                        ("epoch_reverse_2", "epoch-control-2-reverse", "epoch-candidate-2-reverse"),
                        ("epoch_4", "epoch-control-4", "epoch-candidate-4")):
        if a in runs and b in runs:
            pairs[label] = screen(runs[a], runs[b], configs[a], configs[b])
            pairs[label]["purpose"] = "two-call soft working-set epochs"
    by_name = {row["name"]: row for row in rows}
    for label, pair in pairs.items():
        prefix = "epoch" if label.startswith("epoch") else "credit"
        suffix = "2-reverse" if "reverse" in label else label.rsplit("_", 1)[1]
        a = by_name[f"{prefix}-control-{suffix}"]["actual_KV_H2D_MiB"]
        b = by_name[f"{prefix}-candidate-{suffix}"]["actual_KV_H2D_MiB"]
        change = 100 * (b / a - 1) if a not in (None, 0) and b is not None else None
        pair["percent_change"]["actual_KV_H2D_MiB"] = change
        pair["gates"]["h2d"] = change is not None and change <= 5
        pair["eligible_for_replication"] = all(pair["gates"].values())
        pair["H2D_primary_measurement"] = "actual engine transfer counter; source estimate remains separate"
    return {"runs": rows, "algorithm_pairs": pairs,
            "gpu_hours_conservative": sum(row["execution"].get("gpu_hours_conservative", 0) for row in rows),
            "capacity_probes_are_not_algorithm_speedups": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = summarize(args.directory)
    (args.directory / "round_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", newline="\n")
    print(json.dumps({"gpu_hours": result["gpu_hours_conservative"],
                      "runs": [{"name": r["name"], "status": r["execution"]["status"],
                                "p95": r.get("analysis", {}).get("resume_p95_ms")}
                               for r in result["runs"]],
                      "algorithm_pairs": {k: {"deltas": v["percent_change"], "gates": v["gates"]}
                                          for k, v in result["algorithm_pairs"].items()}}, indent=2))


if __name__ == "__main__":
    main()
