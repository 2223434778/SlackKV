"""Recompute paired effects; fail closed on missing or mismatched conditions."""
from __future__ import annotations

import argparse
import gzip
import json
import statistics
from pathlib import Path

from experiments.autonomous_admission.artifacts import read_json, run_paths
from experiments.autonomous_admission.analyze import analyze
from experiments.autonomous_capacity.summary import input_hashes
from experiments.autonomous_capacity.audit import counter


def schedule() -> list[dict]:
    rows = []
    for budget in (2, 4):
        for pair in range(1, 4):
            methods = ["k2", "off"] if (pair + budget // 2) % 2 == 0 else ["off", "k2"]
            for method in methods:
                rows.append({"name": f"gate-{budget}-p{pair}-{method}",
                             "budget": budget, "pair": pair, "method": method})
    return rows


def paired_effect(a: dict, b: dict) -> dict:
    controls = ("kv_bytes", "source_sha", "execution_mode", "batch_invariant",
                "allocator_trace", "scheduler_sha256", "allocator_sha256", "bg_tokens",
                "backfill_limit", "continuation_credit", "working_set_epochs",
                "gate", "source_directory", "runner_sha256", "replay_sha256", "git_head")
    if not all(a["config"][k] == b["config"][k] for k in controls):
        raise ValueError("paired configuration mismatch")
    if a["inputs"] != b["inputs"] or a["outputs"] != b["outputs"]:
        raise ValueError("paired actual workload mismatch")
    if not a["config"]["gate_enabled"] or b["config"]["gate_enabled"]:
        raise ValueError("wrong gate intervention")
    return {k: 100 * (b["metrics"][k] / a["metrics"][k] - 1) for k in
            ("session_e2e_median_ms", "resume_p95_ms", "resume_median_ms",
             "BG_wall_ms", "BG_TPOT_ms", "mixed_generated_tokens_per_second")}


def summarize(root: Path) -> dict:
    runs, pairs, conditions = {}, {}, {}
    ledger = [json.loads(p.read_text()) for p in root.glob("*/execution.json")]
    for row in schedule():
        directory = root / row["name"]
        if not (directory / "execution.json").exists():
            continue
        config = read_json(directory / "execution.json")
        paths = run_paths(directory)
        if config["status"] != "VALID" or len(paths) != 1:
            runs[row["name"]] = {"status": config["status"], "config": config}
            continue
        run = read_json(paths[0])
        if not run["valid"] or len(run["turn_rows"]) != 40 or run["BG_total_generated_tokens"] != 8192:
            raise ValueError("incomplete lifecycle or background workload")
        outputs_path = directory / "output_tokens.jsonl"
        raw = outputs_path.read_bytes() if outputs_path.exists() else gzip.decompress(
            outputs_path.with_suffix(".jsonl.gz").read_bytes())
        outputs = [json.loads(line) for line in raw.decode().splitlines()]
        resumed = [r for r in outputs if r["effective_params"].get("slackkv_trace_resume") is True]
        if len(resumed) != 32 or any(r["effective_params"].get("slackkv_first_token_gate")
                                     is not config["gate_enabled"] for r in resumed):
            raise ValueError("effective gate intervention differs from config")
        inputs = directory / "inputs.jsonl"
        if not inputs.exists():
            inputs = inputs.with_suffix(".jsonl.gz")
        hashes = input_hashes(inputs)
        reported = {f'{t["episode_id"]}:turn:{t["turn_index"]}': t["prompt_sha256"]
                    for t in run["turn_rows"]}
        if hashes != reported:
            raise ValueError("actual and reported inputs differ")
        metrics = analyze(run, [])
        metrics.pop("requests")
        metrics["session_e2e_max_ms"] = max(metrics["session_e2e_ms"].values())
        metrics["resume_p99_ms"] = max((t["first_ns"] - t["issue_ns"]) / 1e6
                                       for t in run["turn_rows"] if t["turn_index"] > 0)
        metrics["actual_preemptions"] = counter(directory / "metrics_after_timed_workload.prom",
                                                "vllm:num_preemptions_total")
        metrics["queue_growth"] = "UNKNOWN: finite closed-loop workload, no open-arrival stability claim"
        for direction, name in (("CPU_to_GPU", "H2D"), ("GPU_to_CPU", "D2H")):
            value = counter(directory / "metrics_after_timed_workload.prom",
                            "vllm:kv_offload_total_bytes_total", f'transfer_type="{direction}"')
            metrics[f"actual_{name}_MiB"] = None if value is None else value / 2**20
        record = {"status": config["status"], "config": config, "inputs": hashes,
                  "outputs": {f'{t["episode_id"]}:turn:{t["turn_index"]}': t["output_tokens"]
                              for t in run["turn_rows"]}, "metrics": metrics,
                  "initial_issue_order": [t["episode_id"] for t in sorted(
                      [t for t in run["turn_rows"] if t["turn_index"] == 0], key=lambda t: t["issue_ns"])],
                  "max_gate_owners": run["max_pre_first"]}
        runs[row["name"]] = record
    for budget in (2, 4):
        effects = []
        for pair in range(1, 4):
            a, b = (runs.get(f"gate-{budget}-p{pair}-{m}") for m in ("k2", "off"))
            if a and b and a["status"] == b["status"] == "VALID":
                effect = paired_effect(a, b)
                pairs[f"{budget}-p{pair}"] = effect
                effects.append(effect)
        complete = len(effects) == 3
        medians = {k: statistics.median(e[k] for e in effects) for k in effects[0]} if effects else {}
        conditions[str(budget)] = {"valid_pairs": len(effects), "complete": complete,
                                  "median_paired_percent_change": medians,
                                  "promising": complete and medians["session_e2e_median_ms"] <= -10
                                  and medians["BG_wall_ms"] <= 5 and medians["BG_TPOT_ms"] <= 5}
    return {"runs": runs, "pairs": pairs, "conditions": conditions,
            "gpu_hours_conservative": sum(r.get("gpu_hours_conservative", 0) for r in ledger),
            "quality_equivalence": "UNKNOWN; performance mode is not a content oracle",
            "novelty": "baseline gate intervention, no algorithm novelty claim"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = summarize(args.directory)
    (args.directory / "round_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", newline="\n")
    print(json.dumps({k: v for k, v in result.items() if k != "runs"}, indent=2))
