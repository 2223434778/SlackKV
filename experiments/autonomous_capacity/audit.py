"""Independent arithmetic, transfer accounting and raw per-request gate evidence."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

from experiments.autonomous_admission.artifacts import read_json, run_paths


def raw_bytes(path: Path) -> bytes:
    if path.exists():
        return path.read_bytes()
    return gzip.decompress(Path(str(path) + ".gz").read_bytes())


def counter(path: Path, name: str, label: str | None = None) -> float | None:
    rows = [line for line in path.read_text().splitlines()
            if line.startswith(name + "{") and (label is None or label in line)]
    return sum(float(line.rsplit(" ", 1)[1]) for line in rows) if rows else None


def audit(directory: Path) -> dict:
    verified = []
    for record in json.loads((directory / "SHA256SUMS.json").read_text()):
        path = directory / record["path"]
        if record["path"] == "round_summary.json" and (directory / "exported_round_summary.json").exists():
            path = directory / "exported_round_summary.json"
        raw = raw_bytes(path)
        if len(raw) != record["bytes"] or hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError(f'archive hash mismatch: {record["path"]}')
        verified.append(record["path"])
    rows = []
    for execution in sorted(directory.glob("*/execution.json")):
        paths = run_paths(execution.parent)
        if not paths:
            continue
        run = read_json(paths[0])
        turns = run["turn_rows"]
        if len(turns) != 40 or len([t for t in turns if t["turn_index"] > 0]) != 32:
            raise ValueError("incomplete request set")
        # Fixed n=32: nearest-rank P95 is the 31st sorted observation.
        p95 = sorted((t["first_ns"] - t["issue_ns"]) / 1e6
                     for t in turns if t["turn_index"] > 0)[30]
        session_times = [
            (max(t["finish_ns"] for t in turns if t["episode_id"] == f"E{i:02}")
             - min(t["issue_ns"] for t in turns if t["episode_id"] == f"E{i:02}")) / 1e6
            for i in range(8)]
        session_median = statistics.median(session_times)
        saved = read_json(execution.parent / "analysis.json")
        if abs(p95 - saved["resume_p95_ms"]) > 1e-9 or abs(session_median - saved["session_e2e_median_ms"]) > 1e-9:
            raise ValueError("independent arithmetic differs")
        prom = execution.parent / "metrics_after_timed_workload.prom"
        h2d = counter(prom, "vllm:kv_offload_total_bytes_total", 'transfer_type="CPU_to_GPU"')
        d2h = counter(prom, "vllm:kv_offload_total_bytes_total", 'transfer_type="GPU_to_CPU"')
        source_bytes = sum(s["external_cached_tokens"] for s in run["source_trace"]) * 28672
        log = paths[0].parent / "server_logs/measured_rep1_a.log"
        raw = raw_bytes(log).decode(errors="replace")
        events = defaultdict(list)
        for line in raw.splitlines():
            for marker in ("SLACKKV_EPOCH_BLOCKED", "SLACKKV_EPOCH_ADMITTED",
                           "SLACKKV_EPOCH_FINISHED", "SLACKKV_CONTINUATION_CREDIT",
                           "SLACKKV_ALLOCATOR_TRUTH"):
                if marker + " " in line:
                    events[marker].append(json.loads(line.split(marker + " ", 1)[1]))
        heads = [e["cold_head"] for e in events["SLACKKV_CONTINUATION_CREDIT"]]
        if len(heads) != len(set(heads)):
            raise ValueError("a cold head spent continuation credit twice")
        blocked = defaultdict(list)
        for e in events["SLACKKV_EPOCH_BLOCKED"]:
            blocked[e["session"]].append(e["timestamp_monotonic"])
        allocator = defaultdict(list)
        for e in events["SLACKKV_ALLOCATOR_TRUTH"]:
            allocator[e.get("session")].append(e["timestamp_monotonic_ns"] / 1e9)
        waited = []
        for session, times in blocked.items():
            arrival = next(e["timestamp_monotonic"] for e in run["resume_trace"]
                           if e["session"] == session and e["event"] == "RESUME_ARRIVED") if \
                any(e["session"] == session and e["event"] == "RESUME_ARRIVED" for e in run["resume_trace"]) else None
            waited.append({"session": session, "sample_count": len(times),
                           "blocked_sample_span_ms": (max(times) - min(times)) * 1000,
                           "arrival_to_first_allocator_ms": (min(allocator[session]) - arrival) * 1000
                           if allocator[session] and arrival is not None else None,
                           "continuous_block_duration": "UNKNOWN from powers-of-two samples"})
        epochs_final = events["SLACKKV_EPOCH_FINISHED"][-1]["active_programs"] if events["SLACKKV_EPOCH_FINISHED"] else None
        if epochs_final not in (None, 0):
            raise ValueError("final logical epochs not empty")
        rows.append({"name": execution.parent.name, "p95_independent_ms": p95,
                     "session_median_independent_ms": session_median,
                     "H2D_bytes_actual": h2d, "D2H_bytes_actual": d2h,
                     "source_restore_bytes": source_bytes,
                     "H2D_counter_status": ("UNKNOWN: missing series" if h2d is None else
                                            "VERIFIED against source" if h2d == source_bytes else
                                            "MISMATCH: retain actual counter; source-only accounting insufficient"),
                     "H2D_actual_minus_source_bytes": None if h2d is None else h2d - source_bytes,
                     "preemptions_actual": counter(prom, "vllm:num_preemptions_total"),
                     "credit_successes": len(heads), "final_epoch_programs": epochs_final,
                     "epoch_admissions": len(events["SLACKKV_EPOCH_ADMITTED"]),
                     "epoch_blocked_requests": sorted(waited, key=lambda e: e["blocked_sample_span_ms"], reverse=True),
                     "compiled_execution_log_present": "VLLM_COMPILE" in raw and "FULL_AND_PIECEWISE" in raw})
    return {"archive_verified_files": len(verified), "runs": rows,
            "caveats": ["final counters are not critical-path DMA durations",
                        "epoch skipped allocator checks are unobserved, not capacity-fit evidence"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = audit(args.directory)
    (args.directory / "independent_audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", newline="\n")
    print(json.dumps({"verified_files": result["archive_verified_files"],
                      "runs": [{k: v for k, v in row.items() if k != "epoch_blocked_requests"}
                               for row in result["runs"]]}, indent=2))


if __name__ == "__main__":
    main()
