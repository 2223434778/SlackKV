"""Control and observer-calibration screens, independently of the original K2 pairs."""
import argparse
import gzip
import json
from pathlib import Path

from experiments.autonomous_admission.artifacts import read_json, run_paths
from experiments.autonomous_admission.analyze import analyze
from experiments.autonomous_capacity.summary import input_hashes
from experiments.autonomous_capacity.audit import counter


def load(directory: Path) -> dict:
    config = read_json(directory / "execution.json")
    if config["status"] != "VALID":
        raise ValueError("control comparison needs valid runs")
    path, = run_paths(directory)
    run = read_json(path)
    if run["max_pre_first"] != 0 or any(e.get("gate_marked") for e in run["resume_trace"]):
        raise ValueError("custom first-token gate was not actually disabled")
    hashes_path = directory / "inputs.jsonl"
    if not hashes_path.exists():
        hashes_path = hashes_path.with_suffix(".jsonl.gz")
    hashes = input_hashes(hashes_path)
    if hashes != {f'{t["episode_id"]}:turn:{t["turn_index"]}': t["prompt_sha256"] for t in run["turn_rows"]}:
        raise ValueError("actual input mismatch")
    outputs = {f'{t["episode_id"]}:turn:{t["turn_index"]}': t["output_tokens"] for t in run["turn_rows"]}
    if len(outputs) != 40 or run["BG_total_generated_tokens"] != 8192 or not run["valid"]:
        raise ValueError("incomplete control workload")
    metrics = analyze(run, [])
    metrics.pop("requests")
    metrics["session_e2e_max_ms"] = max(metrics["session_e2e_ms"].values())
    for direction, name in (("CPU_to_GPU", "H2D"), ("GPU_to_CPU", "D2H")):
        value = counter(directory / "metrics_after_timed_workload.prom", "vllm:kv_offload_total_bytes_total",
                        f'transfer_type="{direction}"')
        metrics[f"actual_{name}_MiB"] = None if value is None else value / 2**20
    metrics["actual_preemptions"] = counter(directory / "metrics_after_timed_workload.prom", "vllm:num_preemptions_total")
    log = path.parent / "server_logs/measured_rep1_a.log"
    raw = log.read_bytes() if log.exists() else gzip.decompress(log.with_suffix(".log.gz").read_bytes())
    events = [json.loads(line.split("SLACKKV_ROUND4_RETENTION ", 1)[1]) for line in raw.decode(errors="replace").splitlines()
              if "SLACKKV_ROUND4_RETENTION " in line]
    entries, acquisitions, releases = {}, 0, 0
    for event in events:
        key = event["program"]
        if event["event"] == "ACQUIRE":
            if key in entries or any(n < 2 for n in event["ref_counts"]):
                raise ValueError("invalid pin acquisition")
            entries[key] = event["block_ids"]
            acquisitions += 1
        elif event["event"] == "RELEASE":
            if entries.pop(key, None) != event["block_ids"] or any(n < 0 for n in event["ref_counts"]):
                raise ValueError("invalid pin release")
            releases += 1
    if entries or (events and events[-1]["retained_programs"] != 0):
        raise ValueError("physical retention leak")
    return {"config": config, "inputs": hashes, "outputs": outputs, "metrics": metrics,
            "physical_pin_acquisitions": acquisitions, "physical_pin_releases": releases,
            "final_pins": len(entries),
            "release_reasons": {reason: sum(e["event"] == "RELEASE" and e["reason"] == reason for e in events)
                                for reason in sorted({e["reason"] for e in events if e["event"] == "RELEASE"})},
            "initial_issue_order": [t["episode_id"] for t in sorted([t for t in run["turn_rows"] if t["turn_index"] == 0],
                                                                   key=lambda t: t["issue_ns"])]}


def compare(a, b, *, observer=False):
    controls = ["kv_bytes", "source_sha", "execution_mode", "batch_invariant", "scheduler_sha256",
                "allocator_sha256", "bg_tokens", "gate_enabled", "backfill_limit",
                "continuation_credit", "working_set_epochs", "gate", "source_directory",
                "runner_sha256", "replay_sha256", "git_head"]
    if not observer:
        controls.append("allocator_trace")
    if not all(a["config"][k] == b["config"][k] for k in controls):
        raise ValueError("control configuration mismatch")
    if a["inputs"] != b["inputs"] or a["outputs"] != b["outputs"]:
        raise ValueError("control actual workload mismatch")
    changes = {k: 100 * (b["metrics"][k] / a["metrics"][k] - 1) for k in
               ("session_e2e_median_ms", "session_e2e_max_ms", "resume_p95_ms", "BG_wall_ms",
                "BG_TPOT_ms", "mixed_generated_tokens_per_second")}
    for k in ("actual_H2D_MiB", "actual_D2H_MiB"):
        x, y = a["metrics"][k], b["metrics"][k]
        changes[k] = 100 * (y / x - 1) if x not in (None, 0) and y is not None else None
    return {"percent_change": changes, "pass_screen": not observer
            and changes["session_e2e_median_ms"] <= -10 and changes["BG_wall_ms"] <= 5
            and changes["BG_TPOT_ms"] <= 5, "novelty": "none claimed", "sample_count": 1}


def summarize(root: Path):
    runs, comparisons = {}, {}
    for name in ("resource-off-repaired", "resource-on-repaired", "control-off", "control-order", "control-pin",
                 "reverse-pin", "reverse-order", "reverse-off"):
        if (root / name / "execution.json").exists() and read_json(root / name / "execution.json")["status"] == "VALID":
            runs[name] = load(root / name)
    for label, a, b, observer in (("observer", "resource-off-repaired", "resource-on-repaired", True),
                                  ("program_order", "control-off", "control-order", False),
                                  ("physical_retention", "control-off", "control-pin", False),
                                  ("pin_increment", "control-order", "control-pin", False),
                                  ("reverse_order", "reverse-off", "reverse-order", False),
                                  ("reverse_pin", "reverse-off", "reverse-pin", False)):
        if a in runs and b in runs:
            comparisons[label] = compare(runs[a], runs[b], observer=observer)
    return {"runs": runs, "comparisons": comparisons,
            "quality": "separate small invariant smoke; forty-input/default-mode equivalence UNKNOWN"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = summarize(args.directory)
    (args.directory / "control_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", newline="\n")
    print(json.dumps({k: v for k, v in result.items() if k != "runs"}, indent=2))
