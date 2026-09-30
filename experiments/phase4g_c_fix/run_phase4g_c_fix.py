"""Native absolute-deadline timing repair and fresh Phase 4G-C rerun."""
from __future__ import annotations

import csv
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

from experiments.phase4d_a import runtime
from experiments.phase4g_c import run_phase4g_c as base
from experiments.phase4g_c_fix import absolute_sleep, metrics
from experiments.phase4g_a.workload import ROOT, build_manifest, load_source

OUT = ROOT / "results/phase4g_c_fix"
OLD_OUT = ROOT / "results/phase4g_c"
TMP_LOGS = Path("/tmp/slackkv-phase4g-c-fix-server-logs")
EXPECTED_VLLM_DIFF_SHA = metrics.VLLM_DIFF_SHA256
SELFTEST_SLEEPS = 100
SELFTEST_SPACING_NS = 20_000_000
STAGE_T_ORDER = metrics.STAGE_T_ORDER
MEASURED_ORDER = metrics.MEASURED_ORDER
TIMING_FIELDS = [
    "stage", "condition", "repetition", "episode_id", "boundary_id",
    "proxy_pause_ms", "previous_finish_ns", "target_issue_ns", "sleep_enter_ns",
    "sleep_wakeup_ns", "request_issue_ns", "sleep_requested_ms",
    "sleep_overshoot_ms", "dispatch_delay_ms", "pause_error_ms", "backend",
    "clock_nanosleep_return_code", "EINTR_retry_count",
]
RUN_TIMING_FIELDS = [
    "stage", "condition", "repetition", "run_id", "complete", "valid",
    "turns_complete", "boundaries", "timing_boundary_count",
    "pause_error_abs_P95_ms", "pause_error_abs_max_ms",
    "sleep_overshoot_P50_ms", "sleep_overshoot_P95_ms",
    "sleep_overshoot_P99_ms", "sleep_overshoot_max_ms",
    "dispatch_delay_P50_ms", "dispatch_delay_P95_ms",
    "dispatch_delay_P99_ms", "dispatch_delay_max_ms",
    "pause_error_P50_ms", "pause_error_P95_ms",
    "pause_error_P99_ms", "pause_error_max_ms",
    "source_probe_cpu_duty", "diagnostic_cpu_duty",
    "X_tail_finish_ms", "lifecycle_internal_valid", "reason",
]
ALL_TIMING_ROWS: list[dict[str, Any]] = []
RUN_ROWS: list[dict[str, Any]] = []
_RAW_SEND_ROWS: list[tuple[Any, ...]] = []


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _pct(values: list[float], q: float) -> float | None:
    return metrics.nearest_rank(values, q) if values else None


def _summary_rows(rows: list[dict[str, Any]], run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    groups = []
    for stage in ("timing_calibration", "measured"):
        stage_rows = [row for row in rows if row["stage"] == stage]
        if stage_rows:
            groups.extend((stage, condition, [row for row in stage_rows if row["condition"] == condition])
                          for condition in ("C0", "C1"))
            groups.append((stage, "ALL", stage_rows))
    for stage, condition, group in groups:
        represented = [row for row in run_rows if row["stage"] == stage
                      and (condition == "ALL" or row["condition"] == condition)]
        output.append({
            "stage": stage, "condition": condition, "runs": len(represented),
            "boundaries": len(group), **metrics.summarize_timing(group),
            "runs_passing_old_gate": sum(
                bool(row.get("complete"))
                and float(row.get("pause_error_abs_P95_ms") if row.get("pause_error_abs_P95_ms") is not None else math.inf) <= 10
                and float(row.get("pause_error_abs_max_ms") if row.get("pause_error_abs_max_ms") is not None else math.inf) <= 30
                for row in represented
            ),
        })
    return output


def _write_timing_artifacts() -> None:
    write_csv(OUT / "timing_trace.csv", ALL_TIMING_ROWS, TIMING_FIELDS)
    write_csv(OUT / "timing_summary.csv", _summary_rows(ALL_TIMING_ROWS, RUN_ROWS), [
        "stage", "condition", "runs", "boundaries",
        "sleep_overshoot_P50_ms", "sleep_overshoot_P95_ms", "sleep_overshoot_P99_ms",
        "sleep_overshoot_max_ms", "sleep_overshoot_abs_P95_ms",
        "dispatch_delay_P50_ms", "dispatch_delay_P95_ms", "dispatch_delay_P99_ms",
        "dispatch_delay_max_ms", "pause_error_P50_ms", "pause_error_P95_ms",
        "pause_error_P99_ms", "pause_error_max_ms", "pause_error_abs_P95_ms",
        "pause_error_abs_P99_ms", "pause_error_abs_max_ms", "runs_passing_old_gate",
    ])
    write_csv(OUT / "timing_calibration.csv",
              [row for row in RUN_ROWS if row["stage"] == "timing_calibration"],
              RUN_TIMING_FIELDS)


def _write_static_artifacts(fingerprint: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(load_source())
    status = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)
    start_status = [line for line in status.splitlines() if not line[3:].startswith(
        ("experiments/phase4g_c_fix", "results/phase4g_c_fix", "tests/test_phase4g_c_fix.py"))]
    write_json(OUT / "config.json", {
        "phase": "4G-C-FIX", "conditions": ["C0", "C1"],
        "workload_sha256": metrics.WORKLOAD_SHA256,
        "P_SAFE_ms": 102, "resume_guard_ms": 110, "K2": 2,
        "reservation_budget_blocks": 3510, "source_recheck_interval_ms": 20,
        "pause_backend": absolute_sleep.BACKEND, "python_busy_spin": False,
        "python_switch_interval_s": metrics.SWITCH_INTERVAL_S,
        "pause_p95_gate_ms": 10, "pause_max_gate_ms": 30,
        "offload_prompt_only": "pinned default true; not overridden",
        "background": "BGv2: 4 x 2048 output tokens, 9500-token prompt",
        "workload_turns": 40, "boundaries_per_run": 32,
        "stage_t_order": STAGE_T_ORDER, "measured_run_order": MEASURED_ORDER,
        "stage_t_aggregate_p99_abs_pause_gate_ms": 10,
    })
    write_json(OUT / "source_manifest.json", {
        **fingerprint, "phase4c_workload": str(ROOT / "results/phase4c/hybrid_workload.csv"),
        "start_git_status": start_status,
        "runtime_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "vllm_source": str(base.VLLM_SOURCE),
        "vllm_upstream_commit": base.VLLM_COMMIT,
        "vllm_working_diff_sha256": fingerprint["vllm_diff_sha256"],
        "prior_phase4g_c_instrumentation_patch_sha256":
            base.sha256((OLD_OUT / "instrumentation.patch").read_bytes()),
        "manifest_summary": {
            "sessions": 8, "turns": 40, "boundaries": len(manifest),
            "reusable_tokens": sum(int(row["required_prefix_tokens"]) for row in manifest),
            "reusable_blocks": sum(int(row["required_blocks"]) for row in manifest),
        },
        "timing_backend_code": str(ROOT / "experiments/phase4g_c_fix/absolute_sleep.py"),
    })
    write_json(OUT / "environment.json", base._environment())
    (OUT / "commands.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"cd {ROOT}\n"
        f"{runtime.ENV_ROOT}/bin/python -m pytest -q tests/test_phase4g_c_fix.py\n"
        f"{runtime.ENV_ROOT}/bin/python -m experiments.phase4g_c_fix.run_phase4g_c_fix\n",
        encoding="utf-8",
    )
    (OUT / "instrumentation.patch").write_bytes((OLD_OUT / "instrumentation.patch").read_bytes())


def _self_test() -> dict[str, Any]:
    start_ns = time.monotonic_ns()
    rows = []
    for index in range(SELFTEST_SLEEPS):
        target_ns = start_ns + (index + 1) * SELFTEST_SPACING_NS
        enter_ns = time.monotonic_ns()
        result = absolute_sleep.absolute_sleep_until_ns(target_ns)
        wake_ns = time.monotonic_ns()
        rows.append({
            "sleep_index": index + 1, "target_ns": target_ns,
            "sleep_enter_ns": enter_ns, "sleep_wakeup_ns": wake_ns,
            "wake_overshoot_ms": (wake_ns - target_ns) / 1e6,
            "return_code": result.return_code,
            "EINTR_retry_count": result.eintr_retry_count,
            "backend": absolute_sleep.BACKEND,
        })
    values = [abs(float(row["wake_overshoot_ms"])) for row in rows]
    summary = {
        "backend": absolute_sleep.BACKEND, "sleeps": len(rows), "spacing_ms": 20,
        "wake_overshoot_abs_P50_ms": _pct(values, 0.50),
        "wake_overshoot_abs_P95_ms": _pct(values, 0.95),
        "wake_overshoot_abs_max_ms": max(values),
        "pass": len(rows) == 100 and _pct(values, 0.95) <= 3 and max(values) <= 10,
    }
    write_csv(OUT / "timing_selftest.csv", rows, [
        "sleep_index", "target_ns", "sleep_enter_ns", "sleep_wakeup_ns",
        "wake_overshoot_ms", "return_code", "EINTR_retry_count", "backend",
    ])
    write_json(OUT / "timing_selftest_summary.json", summary)
    if not summary["pass"]:
        raise RuntimeError("EXPERIMENT_INVALID_HOST_TIMING")
    return summary


def _recover_completed_lifecycle(stage: str, run_repetition: int, condition: str) -> dict[str, Any] | None:
    encoded = run_repetition * 10 + (condition == "C1")
    label = f"{stage}_rep{run_repetition}_{condition.lower()}"
    path = OUT / "runs" / label / "runs" / f"measured_rep{encoded}_d" / "run.json"
    if not path.is_file():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    controller = base.CURRENT_CONTROLLER
    log = TMP_LOGS / "server_logs" / f"measured_rep{encoded}_d.log"
    probe = base.marker_events(log, "SLACKKV_READY_PROBE ")
    capacity = base.marker_events(log, "SLACKKV_CAPACITY_TRACE ")
    snapshots = base.marker_events(log, "SLACKKV_SELECTION_SNAPSHOT ")
    allocator = base.marker_events(log, "SLACKKV_ALLOCATOR_TRUTH ")
    costs = [row for row in probe if row.get("event") == "PROBE_COST"]
    span_ns = (
        max(int(row["timestamp_monotonic_ns"]) for row in costs)
        - min(int(row["timestamp_monotonic_ns"]) for row in costs) + 20_000_000
        if costs else 0
    )
    probe_cpu_ns = sum(int(row.get("probe_duration_ns", 0)) for row in costs)
    allocator_cpu_ns = sum(int(row.get("capacity_trace_cpu_ns", 0))
                           for row in [*capacity, *snapshots])
    allocator_cpu_ns += sum(int(row.get("instrumentation_cpu_ns", 0)) for row in allocator)
    result.update({
        "run_id": label, "condition": condition, "repetition": run_repetition,
        "ready_probe_events": probe, "capacity_trace": capacity,
        "selection_snapshots": snapshots, "allocator_truth": allocator,
        "controller_selection": controller.selection_rows if controller else [],
        "controller_visible_ready_ids": sorted(controller.offload_events) if controller else [],
        "source_recheck_events": controller.source_rechecks if controller else [],
        "recheck_behavior_updates": controller.recheck_behavior_updates if controller else 0,
        "promotions": ({key: {field: value for field, value in item.items() if field != "future"}
                        for key, item in controller.promotions.items()} if controller else {}),
        "source_probe_cpu_duty": probe_cpu_ns / span_ns if span_ns else None,
        "allocator_truth_cpu_duty": allocator_cpu_ns / span_ns if span_ns else None,
        "diagnostic_cpu_duty": (probe_cpu_ns + allocator_cpu_ns) / span_ns if span_ns else None,
        "artifact_root": str(OUT),
    })
    return result


def _background_tail_finish_ms(result: dict[str, Any]) -> float:
    if result.get("background_tail_finish_ms") is not None:
        return float(result["background_tail_finish_ms"])
    finish_ns = result.get("X_finish_ns")
    turns = result.get("turn_rows", [])
    if finish_ns not in (None, "NA") and turns:
        return (int(finish_ns) - max(int(row["finish_ns"]) for row in turns)) / 1e6
    return float(result.get("X_tail_margin_ms", -1))

def _non_timing_lifecycle_pass(result: dict[str, Any], condition: str,
                               controller: Any) -> tuple[bool, str]:
    turns = result.get("turn_rows", [])
    boundaries = result.get("boundary_rows", [])
    source_events = result.get("source_trace", [])
    trace_counts = result.get("resume_trace_counts", {})
    time_order = all(
        int(row["previous_finish_ns"]) <= int(row["registered_ns"]) <= int(row["actual_issue_ns"])
        and int(row["target_resume_issue_ns"]) - int(row["previous_finish_ns"])
            == round(float(row["proxy_pause_ms"]) * 1e6)
        for row in boundaries
    )
    checks = [
        (len(turns) == 40 and len(boundaries) == 32, "turn_or_boundary_count"),
        (time_order, "deadline_or_timestamp_order"),
        (int(result.get("max_pre_first", 99)) <= 2, "K2"),
        (result.get("server_alive") is True and not result.get("OOM"), "server_or_OOM"),
        (result.get("capacity_valid") is True, "capacity_invariant"),
        (int(result.get("lease_leaks", 99)) == 0, "lease_leak"),
        (trace_counts.get("RESUME_ARRIVED") == 32
         and trace_counts.get("FIRST_TOKEN") == 32
         and trace_counts.get("REQUEST_FINISHED") == 32, "resume_trace"),
        (len({row.get("session") for row in source_events}) == 32, "source_trace_count"),
        (int(result.get("X_tokens", 0)) == base.bgv2.BG_TOTAL_OUTPUT_TOKENS, "BGv2_token_count"),
        (_background_tail_finish_ms(result) > 0, "BGv2_tail"),
        (controller is not None and not controller.ledger.active, "logical_leak"),
        (controller is not None and controller.release_queue.unsafe_release_count == 0,
         "unsafe_release"),
        (controller is not None and (condition != "C0" or controller.recheck_behavior_updates == 0),
         "C0_consumed_recheck"),
        (controller is not None and (condition != "C1" or controller.recheck_behavior_updates > 0),
         "C1_no_recheck_transition"),
        (float(result.get("source_probe_cpu_duty") or 1) < .05, "source_probe_overhead"),
        (float(result.get("diagnostic_cpu_duty") or 1) < .07, "diagnostic_overhead"),
    ]
    failed = [name for passed, name in checks if not passed]
    return not failed, ",".join(failed)


def _capture_run(
    condition: str, run_repetition: int, display_repetition: int, stage: str,
    manifest: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], str | None]:
    original_send = runtime.send
    manifest_by_id = {str(row["boundary_id"]): row for row in manifest}
    before = len(_RAW_SEND_ROWS)

    def capture_send(prompt, max_tokens, *, stream=False, params=None, on_first=None):
        params = params or {}
        if params.get("slackkv_trace_resume") is not True:
            return original_send(prompt, max_tokens, stream=stream, params=params, on_first=on_first)
        observation = absolute_sleep.take_wait_observation()
        boundary_id = str(params.get("slackkv_trace_boundary_id", ""))
        if observation is None or boundary_id not in manifest_by_id:
            raise RuntimeError("TIMING_CAPTURE_MISSING_WAIT_OR_BOUNDARY")
        job = manifest_by_id[boundary_id]
        episode_id = str(params.get("slackkv_session", "")).partition(":turn:")[0]
        issue_ns = time.monotonic_ns()
        try:
            return original_send(prompt, max_tokens, stream=stream, params=params, on_first=on_first)
        finally:
            _RAW_SEND_ROWS.append((
                stage, condition, display_repetition, episode_id, boundary_id,
                float(job["proxy_pause_ms"]), observation.target_ns,
                observation.sleep_enter_ns, observation.sleep_wakeup_ns,
                issue_ns, observation.return_code, observation.eintr_retry_count,
            ))

    runtime.send = capture_send
    result = None
    error = None
    try:
        result = base.run_one(condition, run_repetition, manifest, stage)
    except BaseException as exc:
        error = repr(exc)
        if stage == "timing_calibration":
            result = _recover_completed_lifecycle(stage, run_repetition, condition)
    finally:
        runtime.send = original_send

    raw_rows = _RAW_SEND_ROWS[before:]
    boundary_meta = {
        str(row["boundary_id"]): row for row in result.get("boundary_rows", [])
    } if result else {}
    timing_rows = []
    for raw in raw_rows:
        (row_stage, row_condition, row_rep, episode, boundary_id, pause_ms,
         target_ns, enter_ns, wake_ns, issue_ns, return_code, retry_count) = raw
        life = boundary_meta.get(boundary_id, {})
        previous_finish_ns = int(life.get(
            "previous_finish_ns", target_ns - round(pause_ms * 1e6)
        ))
        timing_rows.append({
            "stage": row_stage, "condition": row_condition, "repetition": row_rep,
            "episode_id": episode, "boundary_id": boundary_id,
            "proxy_pause_ms": pause_ms, "previous_finish_ns": previous_finish_ns,
            "target_issue_ns": target_ns, "sleep_enter_ns": enter_ns,
            "sleep_wakeup_ns": wake_ns, "request_issue_ns": issue_ns,
            "sleep_requested_ms": max(0, target_ns - enter_ns) / 1e6,
            "sleep_overshoot_ms": (wake_ns - target_ns) / 1e6,
            "dispatch_delay_ms": (issue_ns - wake_ns) / 1e6,
            "pause_error_ms": (issue_ns - target_ns) / 1e6,
            "backend": absolute_sleep.BACKEND,
            "clock_nanosleep_return_code": return_code,
            "EINTR_retry_count": retry_count,
        })
    ALL_TIMING_ROWS.extend(timing_rows)
    _write_timing_artifacts()
    if result:
        result["artifact_root"] = str(OUT)
        result["condition"] = condition
        result["repetition"] = display_repetition
    return result, timing_rows, error


def _run_metrics_row(stage: str, condition: str, repetition: int, result: dict[str, Any] | None,
                     rows: list[dict[str, Any]], error: str | None, non_timing_pass: bool) -> dict[str, Any]:
    gate = metrics.timing_gate(rows)
    stats = metrics.summarize_timing(rows)
    complete = bool(result and result.get("turns_complete") == 40
                    and result.get("boundaries") == 32 and len(rows) == 32)
    return {
        "stage": stage, "condition": condition, "repetition": repetition,
        "run_id": result.get("run_id") if result else "",
        "complete": complete, "valid": complete and non_timing_pass and bool(gate["pass"]),
        "turns_complete": result.get("turns_complete", 0) if result else 0,
        "boundaries": result.get("boundaries", 0) if result else 0,
        "timing_boundary_count": len(rows),
        "pause_error_abs_P95_ms": gate["p95_abs_ms"],
        "pause_error_abs_max_ms": gate["max_abs_ms"],
        **stats,
        "source_probe_cpu_duty": result.get("source_probe_cpu_duty") if result else None,
        "diagnostic_cpu_duty": result.get("diagnostic_cpu_duty") if result else None,
        "X_tail_finish_ms": _background_tail_finish_ms(result) if result else None,
        "lifecycle_internal_valid": result.get("valid") if result else False,
        "reason": error or ("" if non_timing_pass else "NON_TIMING_LIFECYCLE_INVALID"),
    }


def _write_failure(result_code: str, stage: str, detail: str,
                   extra: dict[str, Any] | None = None) -> None:
    payload = {
        "phase": "4G-C-FIX", "start_commit": metrics.START_COMMIT,
        "workload_sha256": metrics.WORKLOAD_SHA256, "stage": stage,
        "failure_detail": detail,
        "timing_root_cause": metrics.timing_root_cause(ALL_TIMING_ROWS, False),
        "stage_t_runs": RUN_ROWS,
        "stage_t_timing_boundaries": sum(
            row["stage"] == "timing_calibration" for row in ALL_TIMING_ROWS
        ),
        "measured_runs": sum(row["stage"] == "measured" for row in RUN_ROWS),
        "final_result": result_code,
    }
    payload.update(extra or {})
    write_json(OUT / "validation_report.json", payload)
    (OUT / "summary.md").write_text(
        "# Phase 4G-C-FIX\n\n"
        f"Final status: {result_code}. Stage: {stage}.\n\n"
        "The original Phase 4G-C matrix was invalidated solely by pause-fidelity failure. "
        "This fix replaces the Python sleep-plus-busy-spin waiter with Linux "
        "CLOCK_MONOTONIC absolute clock_nanosleep; SlackKV mechanisms and frozen pause values "
        "were not changed.\n\n"
        f"Failure detail: {detail}\n\n"
        "No causal aggregation uses Stage T or prior Phase 4G-C artifacts.\n",
        encoding="utf-8",
    )


def _timing_gate_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [abs(float(row["pause_error_ms"])) for row in rows]
    return {
        **metrics.summarize_timing(rows), "boundaries": len(rows),
        "pause_error_abs_P95_ms": _pct(errors, .95) if errors else None,
        "pause_error_abs_P99_ms": _pct(errors, .99) if errors else None,
        "pause_error_abs_max_ms": max(errors) if errors else None,
    }


def _final_summary(report: dict[str, Any], timing_report: dict[str, Any]) -> None:
    old_summary = (OUT / "summary.md").read_text(encoding="utf-8") if (OUT / "summary.md").exists() else ""
    lines = [
        "# Phase 4G-C-FIX: Native Absolute-Deadline Timing Repair", "",
        f"Final result(s): {', '.join(report.get('final_results', []))}.",
        f"Start commit: {metrics.START_COMMIT}; workload SHA256: {metrics.WORKLOAD_SHA256}.", "",
        "Phase4G-C original matrix was invalidated solely by pause-fidelity failure. "
        "Phase4G-C-FIX changes only the client pause waiter and timing diagnostics: the Python "
        "sleep-plus-busy-spin was replaced by CLOCK_MONOTONIC absolute clock_nanosleep. "
        "No SlackKV mechanism, pause value, deadline, or timing threshold was changed.", "",
        "## Timing calibration", "",
        f"Native backend self-test: {timing_report['selftest']['pass']}; "
        f"P95/max absolute wake overshoot="
        f"{timing_report['selftest']['wake_overshoot_abs_P95_ms']}/"
        f"{timing_report['selftest']['wake_overshoot_abs_max_ms']} ms.",
        f"Stage T: {timing_report['stage_t_valid_runs']}/6 complete valid runs, "
        f"{timing_report['stage_t_boundaries']}/192 boundaries; aggregate P99 |pause error|="
        f"{timing_report['stage_t_pause_error_abs_P99_ms']} ms (gate 10 ms).",
        f"Timing classification: {timing_report['timing_root_cause']}.", "",
        "## Fresh measured matrix", "",
        f"Measured runs: {report.get('valid_runs', 0)}/8; requests: "
        f"{report.get('requests_completed', 0)}/320; boundaries: "
        f"{report.get('boundaries_completed', 0)}/256.",
        f"C0 pause |error| P95/max={timing_report.get('C0_pause_abs_P95_ms')}/"
        f"{timing_report.get('C0_pause_abs_max_ms')} ms; sleep overshoot P95="
        f"{timing_report.get('C0_sleep_overshoot_P95_ms')} ms; dispatch P95="
        f"{timing_report.get('C0_dispatch_delay_P95_ms')} ms.",
        f"C1 pause |error| P95/max={timing_report.get('C1_pause_abs_P95_ms')}/"
        f"{timing_report.get('C1_pause_abs_max_ms')} ms; sleep overshoot P95="
        f"{timing_report.get('C1_sleep_overshoot_P95_ms')} ms; dispatch P95="
        f"{timing_report.get('C1_dispatch_delay_P95_ms')} ms.", "",
        "## Phase 4G-C mechanism analysis", "",
        "The inherited RCA section below uses only the eight fresh measured runs. Stage T and "
        "prior Phase 4G-C observations are excluded from causal aggregation.", "",
        old_summary,
    ]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty result directory: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    base.OUT = OUT
    base.TMP_LOGS = TMP_LOGS
    base.START_COMMIT = metrics.START_COMMIT
    base.phase4g_c_wait_until = absolute_sleep.phase4g_c_fix_wait_until
    manifest = build_manifest(load_source())
    fingerprint = base.frozen_checks()
    if fingerprint["vllm_diff_sha256"] != EXPECTED_VLLM_DIFF_SHA:
        _write_failure("EXPERIMENT_INVALID_ENVIRONMENT", "source verification",
                       f"vLLM diff SHA mismatch: {fingerprint['vllm_diff_sha256']}")
        raise RuntimeError("EXPERIMENT_INVALID_ENVIRONMENT: vLLM diff fingerprint")
    _write_static_artifacts(fingerprint)
    try:
        selftest = _self_test()
    except BaseException as exc:
        _write_failure("EXPERIMENT_INVALID_HOST_TIMING", "native self-test", repr(exc))
        raise

    for display_rep, condition in STAGE_T_ORDER:
        run_repetition = 100 + display_rep
        result, rows, error = _capture_run(
            condition, run_repetition, display_rep, "timing_calibration", manifest,
        )
        controller = base.CURRENT_CONTROLLER
        non_timing_pass, non_timing_reason = (
            _non_timing_lifecycle_pass(result, condition, controller)
            if result else (False, "run_incomplete")
        )
        row = _run_metrics_row("timing_calibration", condition, display_rep, result, rows,
                               error if error and not result else None, non_timing_pass)
        row["complete"] = bool(result and result.get("turns_complete") == 40
                               and result.get("boundaries") == 32 and len(rows) == 32)
        row["valid"] = bool(row["complete"] and non_timing_pass
                            and metrics.timing_gate(rows)["pass"])
        if not non_timing_pass:
            row["reason"] = non_timing_reason
        RUN_ROWS.append(row)
        _write_timing_artifacts()
        if not row["complete"] or not non_timing_pass:
            _write_failure("EXPERIMENT_INVALID_TIMING_CALIBRATION", "Stage T",
                           row["reason"] or "incomplete full-load calibration run")
            raise RuntimeError("EXPERIMENT_INVALID_TIMING_CALIBRATION")

    stage_rows = [row for row in ALL_TIMING_ROWS if row["stage"] == "timing_calibration"]
    stage_run_rows = [row for row in RUN_ROWS if row["stage"] == "timing_calibration"]
    stage_p99 = _pct([abs(float(row["pause_error_ms"])) for row in stage_rows], .99)
    per_run_pass = all(
        float(row["pause_error_abs_P95_ms"]) <= 10
        and float(row["pause_error_abs_max_ms"]) <= 30
        for row in stage_run_rows
    )
    stage_pass = (
        len(stage_run_rows) == 6 and len(stage_rows) == 192
        and all(row["valid"] for row in stage_run_rows)
        and per_run_pass and stage_p99 <= 10
    )
    timing_report = {
        "selftest": selftest,
        "stage_t_valid_runs": sum(bool(row["valid"]) for row in stage_run_rows),
        "stage_t_boundaries": len(stage_rows),
        "stage_t_pause_error_abs_P99_ms": stage_p99,
        "stage_t_per_run_gates_pass": per_run_pass,
        "timing_root_cause": metrics.timing_root_cause(stage_rows, stage_pass),
    }
    if not stage_pass:
        _write_failure(
            "TIMING_BACKEND_NOT_STABLE_ENOUGH" if per_run_pass
            else "EXPERIMENT_INVALID_TIMING_CALIBRATION",
            "Stage T", "Stage T requires six per-run passes and aggregate P99 <= 10 ms.",
            timing_report,
        )
        raise RuntimeError("TIMING_BACKEND_NOT_STABLE_ENOUGH")

    write_json(OUT / "timing_backend.json", {
        "backend": absolute_sleep.BACKEND, "python_busy_spin": False,
        "python_switch_interval_s": metrics.SWITCH_INTERVAL_S,
        "pause_p95_gate_ms": 10, "pause_max_gate_ms": 30,
        "stage_t_runs": 6, "stage_t_boundaries": 192,
        "stage_t_aggregate_abs_pause_error_P99_ms": stage_p99, "frozen": True,
    })

    measured = []
    for repetition, condition in MEASURED_ORDER:
        result, rows, error = _capture_run(
            condition, repetition, repetition, "measured", manifest,
        )
        non_timing_pass, reason = (
            _non_timing_lifecycle_pass(result, condition, base.CURRENT_CONTROLLER)
            if result else (False, "run_incomplete")
        )
        row = _run_metrics_row("measured", condition, repetition, result, rows,
                               error if error and not result else None, non_timing_pass)
        row["complete"] = bool(result and result.get("turns_complete") == 40
                               and result.get("boundaries") == 32 and len(rows) == 32)
        row["valid"] = bool(row["complete"] and non_timing_pass
                            and metrics.timing_gate(rows)["pass"]
                            and result and result.get("valid"))
        if not row["valid"]:
            row["reason"] = error or reason or "PAUSE_FIDELITY_GATE"
        RUN_ROWS.append(row)
        _write_timing_artifacts()
        if not row["valid"]:
            timing_report["timing_root_cause"] = metrics.timing_root_cause(rows, False)
            _write_failure("EXPERIMENT_INVALID_MATRIX", "measured matrix", row["reason"],
                           timing_report)
            raise RuntimeError(f"EXPERIMENT_INVALID_MATRIX:{row['reason']}")
        result["valid"] = True
        result["artifact_root"] = str(OUT)
        result["repetition"] = repetition
        measured.append(result)
        write_json(OUT / "progress.json", {
            "completed_measured_runs": len(measured), "expected_measured_runs": 8,
            "completed": [{"repetition": int(run["repetition"]),
                           "condition": run["condition"], "valid": run["valid"]}
                          for run in measured],
        })

    if not metrics.validate_measured_matrix(measured, str(OUT)):
        _write_failure("EXPERIMENT_INVALID_MATRIX", "measured matrix",
                       "fresh 8-run order/count/root validation failed", timing_report)
        raise RuntimeError("EXPERIMENT_INVALID_MATRIX: refusing partial/mixed aggregation")
    report = base.finalize(measured, manifest)
    measured_rows = [row for row in ALL_TIMING_ROWS if row["stage"] == "measured"]
    timing_report["measured_timing"] = metrics.summarize_timing(measured_rows)
    for condition in ("C0", "C1"):
        condition_rows = [row for row in measured_rows if row["condition"] == condition]
        timing_report[f"{condition}_pause_abs_P95_ms"] = metrics.nearest_rank(
            [abs(float(row["pause_error_ms"])) for row in condition_rows], .95)
        timing_report[f"{condition}_pause_abs_max_ms"] = max(
            abs(float(row["pause_error_ms"])) for row in condition_rows)
        timing_report[f"{condition}_sleep_overshoot_P95_ms"] = metrics.nearest_rank(
            [float(row["sleep_overshoot_ms"]) for row in condition_rows], .95)
        timing_report[f"{condition}_dispatch_delay_P95_ms"] = metrics.nearest_rank(
            [float(row["dispatch_delay_ms"]) for row in condition_rows], .95)
    timing_report["timing_root_cause"] = "RESOLVED"
    report.update({
        "phase": "4G-C-FIX", "start_commit": metrics.START_COMMIT,
        "timing_backend": absolute_sleep.BACKEND, "python_busy_spin": False,
        "switch_interval_s": metrics.SWITCH_INTERVAL_S,
        "self_test": timing_report["selftest"], "stage_t": timing_report,
        "measured_pause_timing": timing_report["measured_timing"],
        "measured_runs_valid": len(measured),
    })
    write_json(OUT / "validation_report.json", report)
    _final_summary(report, timing_report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
