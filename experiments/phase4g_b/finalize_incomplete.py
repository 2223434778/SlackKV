"""Generate compact diagnostics for the first invalid Phase 4G-B run."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from experiments.phase4g_b.run_phase4g_b import _write_instrumentation_patch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase4g_b"
RAW_LOG = Path("/tmp/slackkv-phase4g-b-server-logs/server_logs/measured_rep2_d.log")
FAILURE = OUT / "failures" / "measured_rep2_d.json"
MARKERS = {
    "capacity": "SLACKKV_CAPACITY_TRACE ",
    "resume": "SLACKKV_RESUME_TRACE ",
    "offload": "SLACKKV_OFFLOAD_TRACE ",
    "probe": "SLACKKV_READY_PROBE ",
}


def read_events(path: Path) -> dict[str, Any]:
    events: dict[str, Any] = {name: [] for name in MARKERS}
    events["connector_only_steps"] = 0
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            for name, marker in MARKERS.items():
                pos = line.find(marker)
                if pos >= 0:
                    try:
                        events[name].append(json.loads(line[pos + len(marker):]))
                    except (ValueError, IndexError):
                        pass
                    break
            else:
                if "SLACKKV_CONNECTOR_ONLY_STEP" in line:
                    events["connector_only_steps"] += 1
    return events


def compact(events: dict[str, Any]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in events["capacity"]:
        groups[(str(row.get("lease_key", "NA")), str(row.get("decision", "NA")))].append(row)
    capacity_summary = []
    for (key, decision), rows in sorted(groups.items()):
        rows.sort(key=lambda item: float(item.get("timestamp_monotonic", 0)))
        first, last = rows[0], rows[-1]
        capacity_summary.append({
            "lease_key": key, "decision": decision, "count": len(rows),
            "first_timestamp_monotonic": first.get("timestamp_monotonic"),
            "last_timestamp_monotonic": last.get("timestamp_monotonic"),
            "request_id": first.get("request_id"), "free_blocks": first.get("free_blocks_before"),
            "required_blocks": first.get("candidate_required_blocks", first.get("required_blocks")),
            "progress_reserve_blocks": first.get("progress_reserve_blocks"),
            "protected_lease_key": first.get("protected_lease_key"),
        })
    probes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in events["probe"]:
        if row.get("event") == "SAMPLE":
            probes[str(row.get("boundary_id", "NA"))].append(row)
    probe_summary = []
    for boundary, rows in sorted(probes.items()):
        rows.sort(key=lambda item: int(item.get("timestamp_monotonic_ns", 0)))
        first, last = rows[0], rows[-1]
        probe_summary.append({
            "boundary_id": boundary, "sample_count": len(rows),
            "first_timestamp_monotonic_ns": first.get("timestamp_monotonic_ns"),
            "last_timestamp_monotonic_ns": last.get("timestamp_monotonic_ns"),
            "first_source_fully_ready": first.get("source_fully_ready"),
            "first_local_cached_blocks": first.get("local_cached_blocks"),
            "first_external_ready_needed_blocks": first.get("external_ready_needed_blocks"),
            "first_free_blocks": first.get("free_blocks"), "required_blocks": first.get("required_blocks"),
            "last_source_fully_ready": last.get("source_fully_ready"),
            "last_local_cached_blocks": last.get("local_cached_blocks"),
            "last_external_ready_needed_blocks": last.get("external_ready_needed_blocks"),
            "last_free_blocks": last.get("free_blocks"),
        })
    return {
        "event_counts": {
            "capacity_guard": len(events["capacity"]),
            "capacity_decisions": dict(Counter(str(row.get("decision", "NA")) for row in events["capacity"])),
            "connector_only_steps": events["connector_only_steps"],
            "resume_events": dict(Counter(str(row.get("event", "NA")) for row in events["resume"])),
            "offload_events": dict(Counter(str(row.get("event", "NA")) for row in events["offload"])),
        },
        "capacity_by_request": capacity_summary,
        "source_probes_by_boundary": probe_summary,
        "resume_trace": [{key: row.get(key) for key in (
            "event", "session", "timestamp_monotonic", "step_id", "num_running", "num_waiting",
            "num_scheduled_requests", "total_num_scheduled_tokens")} for row in events["resume"]],
        "offload_trace": events["offload"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(field for row in rows for field in row)) or ["status"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def finalize() -> dict[str, Any]:
    if not FAILURE.is_file() or not RAW_LOG.is_file():
        raise FileNotFoundError("failed-run JSON and raw diagnostic log are required")
    failure = json.loads(FAILURE.read_text(encoding="utf-8"))
    smoke = json.loads((OUT / "runs/smoke_rep2_a/run.json").read_text(encoding="utf-8"))
    baseline = json.loads((OUT / "runs/measured_rep1_a/run.json").read_text(encoding="utf-8"))
    diagnostic = compact(read_events(RAW_LOG))
    patch_sha = _write_instrumentation_patch(OUT / "instrumentation.patch")
    config = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
    config["instrumentation_patch_sha256"] = patch_sha
    config["classification"] = "INCOMPLETE_INVALID_RUN"
    (OUT / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    source_manifest = json.loads((OUT / "source_manifest.json").read_text(encoding="utf-8"))
    source_manifest["new_trace_only_instrumentation_patch_sha256"] = patch_sha
    (OUT / "source_manifest.json").write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "failures/measured_rep2_d_diagnostic.json").write_text(
        json.dumps(diagnostic, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(OUT / "failures/source_probe_summary.csv", diagnostic["source_probes_by_boundary"])
    write_csv(OUT / "failures/capacity_guard_summary.csv", diagnostic["capacity_by_request"])
    write_csv(OUT / "partial_run_inventory.csv", [
        {"run_id": baseline["run_id"], "condition": "M0", "policy": "A", "valid": baseline["valid"],
         "turns_complete": baseline["turns_complete"], "boundaries": baseline["boundaries"],
         "note": "single valid run; not aggregated"},
        {"run_id": "measured_rep2_M1", "condition": "M1", "policy": "D", "valid": False,
         "turns_complete": "NA", "boundaries": "NA", "note": failure.get("error", "invalid run")},
    ])
    report = {
        "phase": "4G-B", "start_commit": "3c887a710cca06142a00c4ca35ca152e7b590bc0",
        "workload_sha256": "9b381336a70e73b0efbddbb79d2c88608c19663247e819c1e7a364efd2971105",
        "instrumentation_smoke": {"valid": smoke.get("valid"), "run_id": smoke.get("run_id"),
                                  "probe_cpu_duty_fraction": smoke.get("probe_cost_summary", {}).get("probe_cpu_duty_fraction")},
        "measured_expected": 12, "valid_completed_runs": 1,
        "failed_run": {"run_id": "measured_rep2_M1", "condition": "M1", "policy": "D",
                       "error": failure.get("error"), "driver_future_timeout_s": 300,
                       "shutdown_engine_dead_error": any("EngineDeadError" in line for line in failure.get("last_server_lines", []))},
        "failed_run_trace_counts": diagnostic["event_counts"],
        "instrumentation_patch_sha256": patch_sha,
        "full_matrix_aggregated": False, "final_result": "INCOMPLETE_INVALID_RUN",
    }
    (OUT / "validation_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    capacities = {(row["lease_key"], row["decision"]): row for row in diagnostic["capacity_by_request"]}
    probes = {row["boundary_id"]: row for row in diagnostic["source_probes_by_boundary"]}
    e01_probe = probes.get("E01:b0", {})
    e01_cap = capacities.get(("E01:turn:1", "ADMIT"), {})
    e03_reject = capacities.get(("E03:turn:1", "PROGRESS_REJECT"), {})
    res_counts = diagnostic["event_counts"]["resume_events"]
    summary = f"""# Phase 4G-B — incomplete mechanism RCA

Final status: **INCOMPLETE_INVALID_RUN**. The frozen 12-run matrix stopped at its first invalid measured run. One valid OnDemand run is retained as an execution record only. No full-matrix aggregate, policy comparison, or A–E bottleneck classification is claimed.

## Frozen inputs and gates

- Start commit: `3c887a710cca06142a00c4ca35ca152e7b590bc0`
- Phase 4C workload SHA256: `9b381336a70e73b0efbddbb79d2c88608c19663247e819c1e7a364efd2971105`
- Instrumentation smoke: {smoke.get('valid')} (`{smoke.get('run_id')}`), probe CPU duty {smoke.get('probe_cost_summary', {}).get('probe_cpu_duty_fraction')} (<5% gate).
- Measured sequence: M0/A/no-background repetition 1 passed; M1/D/no-background repetition 2 did not complete. Ten later runs were not started.

## First invalid run evidence

The M1 FullSlackKV+PSA1 no-background run's driver timed out waiting for a request future after 300 seconds. The server log recorded {diagnostic['event_counts']['capacity_guard']:,} capacity-guard checks and {diagnostic['event_counts']['connector_only_steps']:,} connector-only steps, with no native-prefetch completion event. Only {res_counts.get('RESUME_ARRIVED', 0)} resumes arrived ({', '.join(row.get('session', 'NA') for row in diagnostic['resume_trace'])}); neither arrival had scheduled tokens. Shutdown then produced EngineDeadError messages; the log does not show an independent OOM or engine crash before controlled teardown.

For E01, a `SOURCE_READY` event preceded its promotion attempt. The readiness probe found {e01_probe.get('first_local_cached_blocks', 'NA')} local blocks out of {e01_probe.get('required_blocks', 'NA')}, with {e01_probe.get('first_external_ready_needed_blocks', 'NA')} external blocks needed. At the first guarded attempt, exact free blocks were {e01_cap.get('free_blocks')} for {e01_cap.get('required_blocks')} required; PSA's incremental reserve was {e01_cap.get('progress_reserve_blocks')}, so the candidate was admitted. The same E01 request then repeated in connector-only scheduling without completing.

E03 shows the contrasting PSA decision: free blocks {e03_reject.get('free_blocks')} were enough for its {e03_reject.get('required_blocks')}-block physical fit, but its protected-progress reserve was {e03_reject.get('progress_reserve_blocks')}; PSA emitted `PROGRESS_REJECT`. This one run exposes an interaction between source readiness, JIT promotion, and protected headroom, but does not establish which factor is sufficient or generalize beyond this observation.

The inherited native-prefetch client uses a 5-second HTTP timeout, while the lifecycle driver awaited worker futures for 300 seconds before teardown. The failed run and compact event summaries are retained under `failures/`; the 315-MiB raw server log remains outside the repository.

## Interpretation and limitations

This is not a completed mechanism matrix: there is one valid run and one invalid run, no repeated M1 condition, and no background-X condition. Do not infer a policy result or attribute the non-progress solely to source readiness, physical headroom, PSA, or JIT timing. No serving policy, workload, scheduler algorithm, or threshold was changed after observing this run, and no further matrix runs were attempted.

Artifacts: `partial_run_inventory.csv`, `validation_report.json`, `failures/measured_rep2_d_diagnostic.json`, `failures/source_probe_summary.csv`, and `failures/capacity_guard_summary.csv`.
"""
    (OUT / "summary.md").write_text(summary, encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(finalize(), indent=2, sort_keys=True))
