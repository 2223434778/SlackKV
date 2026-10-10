"""Observed resource dependency witnesses, distinct from latency attribution."""
import argparse
import gzip
import json
import statistics
from collections import Counter
from pathlib import Path

from experiments.autonomous_round4.resources import release_witness
from experiments.autonomous_admission.artifacts import read_json, run_paths
from experiments.autonomous_admission.analyze import analyze, nearest_rank


def diagnose(directory: Path) -> dict:
    paths = run_paths(directory)
    if len(paths) != 1:
        raise ValueError("expected one complete diagnostic run")
    run = read_json(paths[0])
    log = paths[0].parent / "server_logs/measured_rep1_a.log"
    raw = log.read_bytes() if log.exists() else gzip.decompress(log.with_suffix(".log.gz").read_bytes())
    resources, allocator = [], []
    for line in raw.decode(errors="replace").splitlines():
        for marker, target in (("SLACKKV_ROUND4_RESOURCES ", resources),
                               ("SLACKKV_ALLOCATOR_TRUTH ", allocator)):
            if marker in line:
                target.append(json.loads(line.split(marker, 1)[1]))
    witnesses = []
    for event in resources:
        check = event["last_allocator_check"]
        if event["event"] != "ALLOCATE_NONE" or not check or check["fit"] is not False:
            continue
        refs = {int(k): v for k, v in event["busy_ref_counts"].items()}
        owners = {r["request_id"]: r["block_ids"] for r in event["requests"]
                  if r["request_id"] in event["running_ids"]}
        chosen = release_witness(check["required"], check["available"], owners, refs)
        by_id = {r["request_id"]: r for r in event["requests"]}
        active_union = set(b for ids in owners.values() for b in ids)
        inflight_union = set(b for r in event["requests"]
                             if r["request_id"] in event["inflight_prefill_ids"]
                             or r["status"] == "WAITING_FOR_REMOTE_KVS" for b in r["block_ids"])
        witnesses.append({"target_session": event["target_session"],
                          "timestamp_monotonic_ns": event["timestamp_monotonic_ns"],
                          "required": check["required"], "available": check["available"],
                          "deficit": check["required"] - check["available"],
                          "inflight_reserved": check["reserved"],
                          "active_unique_blocks": len(active_union),
                          "inflight_unique_blocks": len(inflight_union),
                          "active_or_inflight_union": len(active_union | inflight_union),
                          "busy_unique_blocks": len(refs),
                          "idle_evictable_blocks": len(event["idle_evictable_block_ids"]),
                          "completion_witness": chosen,
                          "witness_sessions": None if chosen is None else [by_id[k]["session"] for k in chosen],
                          "witness_remaining_outputs": None if chosen is None else [
                              by_id[k]["max_tokens"] - by_id[k]["output_tokens"] for k in chosen],
                          "last_check_age_us": (event["timestamp_monotonic_ns"] - check["timestamp_monotonic_ns"]) / 1000})
    stages = []
    for turn in run["turn_rows"]:
        if turn["turn_index"] == 0:
            continue
        session = f'{turn["episode_id"]}:turn:{turn["turn_index"]}'
        events = [e for e in run["resume_trace"] if e["session"] == session]
        firsts = {name: min((e["timestamp_monotonic"] for e in events if e["event"] == name), default=None)
                  for name in ("RESUME_ARRIVED", "RESUME_ADMITTED", "FIRST_TOKEN")}
        if any(v is None for v in firsts.values()):
            raise ValueError("missing trace stage")
        arrive, admit, first = (firsts[k] for k in ("RESUME_ARRIVED", "RESUME_ADMITTED", "FIRST_TOKEN"))
        stages.append({"session": session, "ttft_ms": (turn["first_ns"] - turn["issue_ns"]) / 1e6,
                       "client_to_engine_ms": (arrive * 1e9 - turn["issue_ns"]) / 1e6,
                       "engine_arrival_to_admission_ms": (admit - arrive) * 1000,
                       "admission_to_engine_first_ms": (first - admit) * 1000,
                       "engine_to_client_first_ms": (turn["first_ns"] - first * 1e9) / 1e6})
    counts = Counter(None if w["completion_witness"] is None else len(w["completion_witness"]) for w in witnesses)
    result = {"valid": run["valid"], "metrics": analyze(run, allocator), "stages": stages,
              "stage_medians_ms": {k: statistics.median(r[k] for r in stages) for k in stages[0] if k != "session"},
              "resources_event_count": len(resources), "allocation_failure_snapshots": len(witnesses),
              "completion_witness_counts": {str(k): v for k, v in counts.items()}, "witnesses": witnesses,
              "ownership_cpu_ms": sum(r["instrumentation_cpu_ns"] for r in resources) / 1e6,
              "instrumentation_snapshot_p95_us": nearest_rank([r["instrumentation_cpu_ns"] / 1000 for r in resources], .95) if resources else None,
              "caveats": ["100ms failure snapshots, not complete cause intervals",
                          "release witnesses freeze decode growth, reservations and other references",
                          "client intervals combine serialization/API/IPC/output observation",
                          "ownership CPU measure excludes JSON/logging; on/off calibration required"]}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = diagnose(args.directory)
    (args.directory / "diagnosis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", newline="\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"metrics", "stages", "witnesses"}}, indent=2))
