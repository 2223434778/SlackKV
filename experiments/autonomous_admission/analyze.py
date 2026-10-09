"""Actual allocator samples, without extrapolating them to unobserved requests."""
from __future__ import annotations

import argparse
import gzip
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from experiments.autonomous_admission.artifacts import read_json, run_paths


def nearest_rank(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("empty distribution")
    return sorted(values)[math.ceil(quantile * len(values)) - 1]


def analyze(result: dict, allocator: list[dict]) -> dict:
    events = defaultdict(list)
    for event in result["resume_trace"]:
        events[event["session"]].append(event)
    samples = defaultdict(list)
    for sample in allocator:
        if sample.get("session"):
            samples[sample["session"]].append(sample)
    rows = []
    for turn in result["turn_rows"]:
        session = f'{turn["episode_id"]}:turn:{turn["turn_index"]}'
        first = next((round(float(e["timestamp_monotonic"]) * 1e9) for e in events[session]
                      if e["event"] == "RESUME_ADMITTED"), None)
        arrived = next((round(float(e["timestamp_monotonic"]) * 1e9) for e in events[session]
                        if e["event"] == "RESUME_ARRIVED"), None)
        # Priming requests lack resume admission trace; do not infer their interval.
        before = [s for s in samples[session] if arrived is not None and first is not None
                  and arrived <= int(s["timestamp_monotonic_ns"]) <= first]
        failures = [s for s in before if s["allocator_fit"] is False]
        rows.append({
            "session": session, "turn_index": turn["turn_index"],
            "ttft_ms": (turn["first_ns"] - turn["issue_ns"]) / 1e6,
            "pre_admission_ms": None if first is None or arrived is None else (first - arrived) / 1e6,
            "pre_admission_allocator_samples": len(before),
            "pre_admission_failed_samples": len(failures),
            "failed_gates": sorted({s["gate"] for s in failures}),
            "max_observed_deficit_blocks": max((s["allocator_required_blocks"] -
                                                  s["allocator_available_blocks"] for s in failures), default=None),
            "max_observed_inflight_reserved_blocks": max((s["inflight_reserved_blocks"] for s in failures), default=None),
            "max_observed_preemptions": max((s.get("num_preemptions", 0) for s in samples[session]), default=None),
            "first_failure_ns": min((int(s["timestamp_monotonic_ns"]) for s in failures), default=None),
            "last_failure_ns": max((int(s["timestamp_monotonic_ns"]) for s in failures), default=None),
            "observation_status": "OBSERVED" if before else "UNKNOWN",
        })
    sessions = defaultdict(list)
    for turn in result["turn_rows"]:
        sessions[turn["episode_id"]].append(turn)
    session_ms = [(max(t["finish_ns"] for t in turns) - min(t["issue_ns"] for t in turns)) / 1e6
                  for turns in sessions.values()]
    session_metrics = {key: (max(t["finish_ns"] for t in turns) - min(t["issue_ns"] for t in turns)) / 1e6
                       for key, turns in sessions.items()}
    resumes = [r for r in rows if r["turn_index"] > 0]
    return {
        "valid": result["valid"], "resume_count": len(resumes),
        "resume_p95_ms": nearest_rank([r["ttft_ms"] for r in resumes], .95),
        "resume_median_ms": statistics.median(r["ttft_ms"] for r in resumes),
        "session_e2e_median_ms": statistics.median(session_ms),
        "session_e2e_ms": session_metrics,
        "BG_wall_ms": result["BG_total_wall_ms"],
        "BG_TPOT_ms": result["BG_weighted_TPOT_ms"],
        "BG_tail_finish_ms": result["BG_tail_request_finish_ms"],
        "resumes_with_observed_capacity_failure": sum(bool(r["pre_admission_failed_samples"]) for r in resumes),
        "resumes_without_allocator_observation": sum(r["observation_status"] == "UNKNOWN" for r in resumes),
        "allocator_sample_count": len(allocator), "requests": rows,
        "agent_external_H2D_MiB": sum(s["external_cached_tokens"] for s in result["source_trace"]) * 28672 / 1048576,
        "external_restore_source_events": len(result["source_trace"]),
        "mixed_generated_tokens_per_second": (sum(t["output_tokens"] for t in result["turn_rows"]) +
                                               result["BG_total_generated_tokens"]) /
            ((max(result["X_finish_ns"], max(t["finish_ns"] for t in result["turn_rows"])) -
              result["X_first_ns"]) / 1e9),
        "GPU_KV_memory_time": "UNKNOWN in uninstrumented screen",
        "D2H_bytes": "UNKNOWN; source readiness is not transfer accounting",
        "interpretation": "Samples are actual checks; no duration attribution or causality from absent checks.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    paths = run_paths(args.directory)
    if len(paths) != 1:
        raise ValueError("expected exactly one run")
    run = paths[0].parent
    result = read_json(paths[0])
    allocator = []
    logs = list(run.rglob("*.log"))
    logs.extend(p for p in run.rglob("*.log.gz") if not Path(str(p)[:-3]).exists())
    for log in logs:
        if "compact" in log.name:
            continue
        raw = (gzip.decompress(log.read_bytes()).decode(errors="replace") if log.suffix == ".gz"
               else log.read_text(errors="replace"))
        for line in raw.splitlines():
            if "SLACKKV_ALLOCATOR_TRUTH " in line:
                allocator.append(json.loads(line.split("SLACKKV_ALLOCATOR_TRUTH ", 1)[1]))
    summary = analyze(result, allocator)
    (args.directory / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n", newline="\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "requests"}, indent=2))


if __name__ == "__main__":
    main()
