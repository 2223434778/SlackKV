"""Independent fork accounting. Missing counters remain UNKNOWN (JSON null)."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics


def read(path: Path) -> bytes:
    if path.exists():
        return path.read_bytes()
    return gzip.decompress(Path(str(path) + ".gz").read_bytes())


def events(raw: str, marker: str) -> list[dict]:
    return [json.loads(line.split(marker + " ", 1)[1]) for line in raw.splitlines() if marker + " " in line]


def counter(raw: str, name: str, label: str | None = None) -> float | None:
    rows = [line for line in raw.splitlines() if line.startswith(name + "{") and (label is None or label in line)]
    return sum(float(line.rsplit(" ", 1)[1]) for line in rows) if rows else None


def remote_intervals(traces: list[dict]) -> list[dict]:
    waiting = {}
    rows = []
    for event in traces:
        key = event["request_id"]
        if event["event"] == "WAITING_FOR_REMOTE_KVS":
            if key in waiting:
                raise ValueError("overlapping remote waits for one request")
            waiting[key] = event
        elif event["event"] == "REMOTE_KV_READY":
            if key not in waiting:
                raise ValueError("remote ready lacks start")
            start = waiting.pop(key)
            elapsed = (event["timestamp_monotonic"] - start["timestamp_monotonic"]) * 1000
            if elapsed < 0:
                raise ValueError("negative remote interval")
            rows.append(dict(session=event["session"], elapsed_ms=elapsed))
    if waiting:
        raise ValueError("unfinished remote waits")
    return rows


def validate_inputs(inputs: list[dict], requests: list[dict]) -> None:
    observed = {r["session"]: hashlib.sha256(json.dumps(r["prompt"]).encode()).hexdigest() for r in inputs}
    reported = {r["session"]: r["prompt_sha256"] for r in requests}
    if len(observed) != len(inputs) or len(reported) != len(requests) or observed != reported:
        raise ValueError("raw inputs differ from recorded identities or contain duplicates")


def critical_path(requests: list[dict], episode: int) -> dict:
    rows = [r for r in requests if r["session"].startswith(f"F{episode}:")]
    final = max(rows, key=lambda r: r["done_ns"])
    first_session = final["session"].rsplit(":", 1)[0] + ":0"
    first = next(r for r in rows if r["session"] == first_session)
    earliest = min(r["issue_ns"] for r in rows)
    components = dict(initial_issue_offset=first["issue_ns"] - earliest,
                      first_ttft=first["first_ns"] - first["issue_ns"],
                      first_post_token_service=first["done_ns"] - first["first_ns"],
                      tool_and_client_gap=final["issue_ns"] - first["done_ns"],
                      second_ttft=final["first_ns"] - final["issue_ns"],
                      second_post_token_service=final["done_ns"] - final["first_ns"])
    if any(v < 0 for v in components.values()):
        raise ValueError("inconsistent critical branch timestamps")
    return dict(episode=episode, last_branch=final["session"],
                components_ms={k: v / 1e6 for k, v in components.items()},
                total_ms=(final["done_ns"] - earliest) / 1e6,
                qualification="client-observed critical branch; post-token service includes scheduling and compute")


def analyze(directory: Path) -> dict:
    execution = json.loads(read(directory / "execution.json"))
    if execution["status"] == "RUNNING":
        raise ValueError("analysis requires finalized execution ledger; live cost is not zero")
    if execution.get("run_kind") == "historical_regression":
        from experiments.autonomous_admission.artifacts import read_json, run_paths
        from experiments.autonomous_admission.analyze import analyze as legacy_analyze
        result = dict(name=directory.name, status=execution["status"],
                      run_kind="historical_regression", gpu_hours=execution["gpu_hours_conservative"])
        if execution["status"] == "VALID":
            paths = run_paths(directory)
            if len(paths) != 1:
                raise ValueError("historical regression requires exactly one result")
            from experiments.autonomous_capacity.summary import input_hashes
            run = read_json(paths[0])
            actual = input_hashes(directory / ("inputs.jsonl" if (directory / "inputs.jsonl").exists() else "inputs.jsonl.gz"))
            reported = {f'{t["episode_id"]}:turn:{t["turn_index"]}': t["prompt_sha256"] for t in run["turn_rows"]}
            if actual != reported:
                raise ValueError("historical actual inputs mismatch")
            logs = list(paths[0].parent.rglob("*.log"))
            logs.extend(p for p in paths[0].parent.rglob("*.log.gz") if not Path(str(p)[:-3]).exists())
            allocator = []
            for path in logs:
                if "compact" not in path.name:
                    raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
                    allocator.extend(events(raw.decode(errors="replace"), "SLACKKV_ALLOCATOR_TRUTH"))
            summary = legacy_analyze(run, allocator)
            result.update({k: v for k, v in summary.items() if k != "requests"})
            result["actual_h2d_bytes"] = counter(read(directory / "metrics_after_timed_workload.prom").decode(), "vllm:kv_offload_total_bytes_total", 'transfer_type="CPU_to_GPU"')
            result["actual_inputs_match_round2"] = actual == input_hashes(directory.parent.parent / "autonomous_capacity/capacity-2/inputs.jsonl.gz") if (directory.parent.parent / "autonomous_capacity/capacity-2/inputs.jsonl.gz").exists() else actual == input_hashes(directory.parent.parent / "autonomous_capacity/capacity-2/inputs.jsonl")
        return result
    result = {"name": directory.name, "status": execution["status"],
              "kv_gib": execution["kv_gib"], "gate": execution["gate"],
              "background_phase": execution.get("background_phase", "cold"),
              "coalesce": execution["coalesce"], "batch_invariant": execution["batch_invariant"],
              "gpu_hours": execution["gpu_hours_conservative"]}
    if "oracle_pass" in execution:
        result["oracle_pass"] = execution["oracle_pass"]
    if result["status"] != "COMPLETE":
        return result
    requests = json.loads(read(directory / "requests.json"))
    episodes = json.loads(read(directory / "episodes.json"))
    log = read(directory / f"server_logs/{directory.name}.log").decode(errors="replace")
    source = [e for e in events(log, "SLACKKV_SOURCE_TRACE") if e["session"].startswith("F")]
    alloc = events(log, "SLACKKV_FORK_ALLOC")
    traces = events(log, "SLACKKV_RESUME_TRACE")
    branch_traces = [e for e in traces if e["session"].startswith("F")]
    gate_peak = max((e.get("num_slackkv_pre_first_token", 0) for e in branch_traces), default=None)
    if gate_peak is not None and gate_peak > execution["gate"]:
        raise ValueError("first-token gate exceeded")
    h2d = counter(read(directory / "metrics_after.prom").decode(), "vllm:kv_offload_total_bytes_total", 'transfer_type="CPU_to_GPU"')
    d2h = counter(read(directory / "metrics_after.prom").decode(), "vllm:kv_offload_total_bytes_total", 'transfer_type="GPU_to_CPU"')
    branches = [r for r in requests if r["session"].startswith("F")]
    inputs = list(map(json.loads, read(directory / "inputs.jsonl").decode().splitlines()))
    validate_inputs(inputs, requests)
    required = {f"F{ep}:{i}:{t}" for ep in range(3) for i in range(8) for t in range(2)} | {f"BG{ep}" for ep in range(3)}
    if len(requests) != 51 or {r["session"] for r in requests} != required:
        raise ValueError("incomplete/duplicate workload")
    if any(len(r["tokens"]) != (128 if r["session"].startswith("F") else 1024) for r in requests):
        raise ValueError("model work changed")
    prompts = {r["session"]: r["prompt"] for r in inputs}
    for ep in range(3):
        root = prompts[f"F{ep}:0:0"][:24576]
        for branch in range(8):
            first = prompts[f"F{ep}:{branch}:0"]
            second = prompts[f"F{ep}:{branch}:1"]
            if len(first) != 24832 or first[:24576] != root or len(second) != 25024 or second[:24832] != first:
                raise ValueError("fork structure or context work changed")
        if len(prompts[f"BG{ep}"]) != 8192:
            raise ValueError("background context changed")
    if len(source) != 48 or {e["session"] for e in source} != required - {f"BG{i}" for i in range(3)}:
        raise ValueError("missing or repeated first source identity; inspect preemption")
    finished = [e["session"] for e in branch_traces if e["event"] == "REQUEST_FINISHED"]
    if len(finished) != 48 or set(finished) != required - {f"BG{i}" for i in range(3)}:
        raise ValueError("missing/repeated branch terminal event")
    sources_by_ep = [[e for e in source if e["session"].startswith(f"F{ep}:") and e["session"].endswith(":0")] for ep in range(3)]
    if any(not any(e["external_cached_tokens"] for e in group) for group in sources_by_ep):
        raise ValueError("CPU-fork intervention not realized")
    if "VLLM_COMPILE" not in log or "FULL_AND_PIECEWISE" not in log:
        raise ValueError("strong execution baseline missing")
    rid_session = {e["request_id"]: e["session"] for e in traces if e["event"] == "RESUME_ARRIVED"}
    sampled_duplication = []
    for a in alloc:
        own_roots = [ids[:1536] for rid, ids in a["owned"].items()
                     if rid_session.get(rid, "").startswith(a["session"].rsplit(":", 2)[0] + ":")]
        if own_roots:
            sampled_duplication.append(len(set(x for ids in own_roots for x in ids)) / 1536)
    result.update(workflow_median_ms=statistics.median(e["workflow_ms"] for e in episodes),
                  workflow_each_ms=[e["workflow_ms"] for e in episodes],
                  branch_ttft_p95_ms=sorted(r["ttft_ms"] for r in branches)[math.ceil(.95 * len(branches)) - 1],
                  bg_median_ms=statistics.median(e["bg_ms"] for e in episodes),
                  h2d_bytes_actual=h2d, d2h_bytes_actual=d2h,
                  source_restore_bytes=sum(e["external_cached_tokens"] for e in source) * 28672,
                  external_first_calls_each=[sum(e["external_cached_tokens"] > 0 for e in group) for group in sources_by_ep],
                  allocated_root_copies_peak_sample=max(sampled_duplication, default=None),
                  owned_kv_memory_time="UNKNOWN: allocation snapshots are not a residency trajectory",
                  preemptions=counter(read(directory / "metrics_after.prom").decode(), "vllm:num_preemptions_total"),
                  observed_first_token_gate_peak=gate_peak,
                  final_running_requests=counter(read(directory / "metrics_after.prom").decode(), "vllm:num_requests_running"),
                  final_waiting_requests=counter(read(directory / "metrics_after.prom").decode(), "vllm:num_requests_waiting"),
                  inputs={r["session"]: r["prompt_sha256"] for r in requests},
                  outputs={r["session"]: hashlib.sha256(json.dumps(r["tokens"]).encode()).hexdigest() for r in requests},
                  allocation_events=alloc,
                  source_events=source,
                  coalescing_events=events(log, "SLACKKV_RESTORE_WAIT"))
    result["critical_path_each"] = [critical_path(requests, ep) for ep in range(3)]
    if any(abs(row["total_ms"] - episode["workflow_ms"]) > .001
           for row, episode in zip(result["critical_path_each"], episodes)):
        raise ValueError("independent workflow arithmetic mismatch")
    waits = remote_intervals(traces)
    result["remote_wait_intervals"] = waits
    result["remote_wait_median_ms"] = statistics.median(r["elapsed_ms"] for r in waits) if waits else None
    result["remote_wait_interpretation"] = "connector readiness observation, not pure DMA duration or a global speedup bound"
    if h2d is not None and h2d != result["source_restore_bytes"]:
        result["transfer_accounting"] = "MISMATCH: retain actual counter, not forced equality"
    else:
        result["transfer_accounting"] = "VERIFIED" if h2d is not None else "UNKNOWN"
    return result


def compare(a: dict, b: dict) -> dict:
    if a["inputs"] != b["inputs"]:
        raise ValueError("inputs not matched")
    keys = ("kv_gib", "gate", "background_phase", "batch_invariant")
    if any(a[k] != b[k] for k in keys):
        raise ValueError("resource or execution regimes not matched")
    measures = ("workflow_median_ms", "branch_ttft_p95_ms", "bg_median_ms", "h2d_bytes_actual")
    changes = {k: (None if a[k] is None or b[k] is None else
                   (100 * (b[k] / a[k] - 1) if a[k] != 0 else (0 if b[k] == 0 else None)))
               for k in measures}
    valid = all(changes[k] is not None for k in measures)
    accepted = valid and changes["workflow_median_ms"] <= -10 and changes["branch_ttft_p95_ms"] <= 5 and changes["bg_median_ms"] <= 5 and changes["h2d_bytes_actual"] <= 0
    return {"baseline": a["name"], "candidate": b["name"], "percent_change": changes,
            "performance_screen_pass": accepted, "outputs_exact": a["outputs"] == b["outputs"],
            "content_correctness": "UNKNOWN unless separate exact oracle passes"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, help="write recomputation separately from exported evidence")
    args = parser.parse_args()
    results = [analyze(p.parent) for p in sorted(args.directory.glob("*/execution.json"))]
    output = dict(runs=results, gpu_hours=sum(r["gpu_hours"] for r in results))
    (args.output or args.directory / "analysis.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", newline="\n")
    print(json.dumps({"gpu_hours": output["gpu_hours"], "runs": [{k: v for k, v in r.items() if k not in ("inputs", "outputs", "allocation_events", "source_events", "coalescing_events")} for r in results]}, indent=2))


if __name__ == "__main__":
    main()
