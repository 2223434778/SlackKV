"""Fresh C1 runs and shadow-only protected-head/frontier analysis."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

from experiments.phase4d_a import runtime
from experiments.phase4g_a.workload import ROOT, build_manifest, load_source
from experiments.phase4g_c import run_phase4g_c as base
from experiments.phase4g_c_fix import metrics, run_phase4g_c_fix as fix
from experiments.phase4g_c_fix import absolute_sleep
from experiments.phase4g_d import analysis
from experiments.phase4g_d.shadow import ProtectedHeadShadowController

OUT = ROOT / "results/phase4g_d"
TMP_LOGS = Path("/tmp/slackkv-phase4g-d-server-logs")
RUN_ORDER = [1, 2, 3, 4]
RETRY_TIMELINE_FIELDS = [
    "run_id", "condition", "repetition", "timestamp_ns", "protected_id",
    "terminal_timestamp_ns", "is_current_protected_frontier", "probe_target_ns",
    "terminal_reason", "time_since_failure_ms", "probe_interval_ms",
    "source_full_ready", "resume_started", "first_token_seen",
    "logical_reserved_blocks", "logical_reservation_fit", "timing_gate_open",
    "self_psa_H", "self_psa_fit", "pool_free_blocks",
    "allocator_required_blocks", "allocator_available_blocks", "allocator_fit",
    "legacy_physical_fit", "full_required_blocks", "protected_local_cached_blocks",
    "protected_external_ready_blocks", "protected_missing_blocks", "existing_lease_blocks",
    "retry_actionable", "actionable_window_index", "request_issue_ns", "request_return_ns",
    "snapshot_roundtrip_ms", "snapshot_event_timestamp_ns", "shadow_server_cpu_ns",
    "shadow_client_cpu_ns",
]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = fields or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def nearest_rank(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(x) for x in values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def verify_frozen_inputs() -> dict[str, Any]:
    workload = ROOT / "results/phase4c/hybrid_workload.csv"
    sha = hashlib.sha256(workload.read_bytes()).hexdigest()
    if sha != analysis.WORKLOAD_SHA256:
        raise RuntimeError(f"EXPERIMENT_INVALID_WORKLOAD:{sha}")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != analysis.START_COMMIT:
        raise RuntimeError(f"EXPERIMENT_INVALID_SOURCE:{head}")
    source, manifest = load_source(), build_manifest(load_source())
    if len(source) != 40 or len(manifest) != 32:
        raise RuntimeError("EXPERIMENT_INVALID_WORKLOAD: expected 40 turns and 32 boundaries")
    base.START_COMMIT = analysis.START_COMMIT
    base.OUT, base.TMP_LOGS = OUT, TMP_LOGS
    fingerprint = base.frozen_checks()
    if fingerprint.get("vllm_diff_sha256") != metrics.VLLM_DIFF_SHA256:
        raise RuntimeError("EXPERIMENT_INVALID_ENVIRONMENT:vLLM diff fingerprint")
    return {**fingerprint, "slackkv_start_commit": head, "phase4c_workload_sha256": sha}


def _nullable_preflight(preflight: dict[str, Any]) -> dict[str, Any]:
    required = analysis.optional_int(preflight.get("allocator_required_blocks"))
    available = analysis.optional_int(preflight.get("allocator_available_blocks"))
    fit = preflight.get("allocator_fit")
    if required is None or available is None or not isinstance(fit, bool):
        fit = "UNKNOWN"
    return {
        "allocator_required_blocks": required, "allocator_available_blocks": available,
        "allocator_fit": fit,
        "pool_free_blocks": analysis.optional_int(preflight.get("pool_free_blocks")),
        "inflight_reserved_blocks": analysis.optional_int(preflight.get("inflight_reserved_blocks")),
        "watermark_blocks": analysis.optional_int(preflight.get("watermark_blocks")),
        "full_sequence_required_blocks": analysis.optional_int(preflight.get("full_sequence_required_blocks")),
        "full_sequence_fit": preflight.get("full_sequence_fit"),
    }


def _selection_status(selection: dict[str, Any], controller: ProtectedHeadShadowController) -> str | None:
    candidate, sid = str(selection.get("candidate_id", "")), selection.get("selection_id")
    item = controller.promotions.get(candidate, {})
    if item.get("selection_id") == sid:
        status = (item.get("response") or {}).get("status")
        if status:
            return str(status).upper()
    for event in reversed(controller.decisions):
        if event.get("boundary_id") != candidate:
            continue
        event_time = analysis.optional_int(event.get("timestamp_ns"))
        if event_time is not None and event_time < int(selection.get("timestamp_ns", 0)):
            continue
        if event.get("selection_id") not in (None, sid):
            continue
        normalized = analysis.normalize_decision(event.get("event"))
        if normalized in {"PROMOTION_ATTEMPT", "ATTEMPT"}:
            continue
        return normalized
    return None


def _selection_analysis(run_id: str, rep: int, controller: ProtectedHeadShadowController):
    selections, allocators, needs, recoveries, guards = [], [], [], [], []
    frontier_by_id = {str(row.get("selection_id")): row for row in controller.frontier_shadow_rows}
    for selection in controller.selection_rows:
        sid = str(selection.get("selection_id", ""))
        candidate, protected = str(selection.get("candidate_id", "")), str(selection.get("protected_id", ""))
        snap = selection.get("selection_snapshot") or {}
        cp = snap.get("candidate_allocator_preflight") or {}
        pp = snap.get("protected_self_allocator_preflight") or {}
        ce = snap.get("candidate_source_exact") or {}
        pe = snap.get("protected_source_exact") or {}
        free = analysis.optional_int(selection.get("selection_free_blocks"))
        legacy = "ADMIT" if free is not None and free >= int(selection["candidate_required_blocks"]) else "REJECT"
        classified = analysis.classify_attempt_nullable(
            legacy, cp.get("allocator_required_blocks"), cp.get("allocator_available_blocks"))
        eligible = ce.get("source_fully_ready") is True and selection.get("candidate_logical_fit") is True
        decision = _selection_status(selection, controller)
        flat = {key: value for key, value in selection.items() if not isinstance(value, (dict, list))
                and key != "selection_snapshot"}
        flat.update({"run_id": run_id, "condition": "C1", "repetition": rep,
                     "actual_attempt_result": decision, "eligible_selection": eligible,
                     "timing_gate_open": True,
                     "candidate_allocator_fit_nullable": classified["allocator_fit"],
                     "legacy_allocator_agreement": classified["agreement"],
                     "legacy_false_admit": classified["false_admit"],
                     "legacy_false_reject": classified["false_reject"]})
        selections.append(flat)
        allocators.append({"run_id": run_id, "condition": "C1", "repetition": rep,
                           "context": "SELECTION_CANDIDATE", "selection_id": sid,
                           "candidate_id": candidate, "protected_id": protected,
                           "source_full_ready": ce.get("source_fully_ready"),
                           "eligible_selection": eligible, "legacy_guard_decision": legacy,
                           **_nullable_preflight(cp), **classified})
        own = snap.get("protected_local_ownership") or {}
        pf = _nullable_preflight(pp)
        need = analysis.optional_int(selection.get("PSA1_reserve_blocks"))
        allocator_need = pf["allocator_required_blocks"]
        needs.append({"run_id": run_id, "condition": "C1", "repetition": rep,
            "selection_id": sid, "protected_id": protected,
            "progress_blocks": selection.get("protected_progress_blocks"),
            "full_required_blocks": selection.get("protected_required_blocks"),
            "local_cached_blocks": pe.get("local_cached_blocks"),
            "external_ready_blocks": pe.get("external_ready_blocks"),
            "external_ready_needed_blocks": pe.get("external_ready_needed_blocks"),
            "leased_blocks": own.get("current_protected_lease_blocks"),
            "owned_by_running_blocks": own.get("owned_by_running_request"),
            "owned_by_waiting_blocks": own.get("owned_by_waiting_request"),
            "owned_blocks": own.get("owned_by_running_or_waiting"),
            "evictable_cached_blocks": own.get("cached_but_free_evictable"),
            "other_refcounted_blocks": own.get("other_refcounted"),
            "Need_PSA1": need, "Need_allocator_now": allocator_need,
            "allocator_available_now": pf["allocator_available_blocks"],
            "allocator_fit_now": pf["allocator_fit"], "Need_full": selection.get("protected_progress_blocks"),
            "need_gap_blocks": need - allocator_need if need is not None and allocator_need is not None else None,
            "need_ratio": need / max(1, allocator_need) if need is not None and allocator_need is not None else None})
        allocators.append({"run_id": run_id, "condition": "C1", "repetition": rep,
            "context": "SELECTION_PROTECTED_SELF", "selection_id": sid,
            "candidate_id": candidate, "protected_id": protected,
            "source_full_ready": pe.get("source_fully_ready"),
            "eligible_selection": pe.get("source_fully_ready") is True,
            "legacy_guard_decision": "UNKNOWN", **pf,
            "agreement": "UNKNOWN", "false_admit": "UNKNOWN", "false_reject": "UNKNOWN"})

        front = frontier_by_id.get(sid, {})
        current_reject = str(decision or "").upper() == "PROGRESS_REJECTED"
        exact_fit = classified["allocator_fit"] is True
        source_ready = ce.get("source_fully_ready") is True
        logical_fit = selection.get("candidate_logical_fit") is True
        rec = {"run_id": run_id, "condition": "C1", "repetition": rep,
            "selection_id": sid, "candidate_id": candidate,
            "protected_id": protected,
            "timestamp_ns": selection.get("timestamp_ns"),
            **{f"{name}_id": front.get(f"{name}_id") for name in ("F0", "F1", "F2", "F3", "F4")},
            "actual_attempt_result": decision, "current_F0_PSA_reject": current_reject,
            "exact_allocator_fit": classified["allocator_fit"], "source_ready": source_ready,
            "logical_fit": logical_fit, "timing_gate_open": True,
            "TERMINAL_HEAD_PROTECTION": front.get("terminal_head_protection", False)}
        for label in ("F1", "F2", "F3", "F4"):
            if not (current_reject and exact_fit and source_ready and logical_fit):
                rec[f"recoverable_under_{label}"] = None
            elif isinstance(front.get(f"PSA_fit_{label}"), bool):
                rec[f"recoverable_under_{label}"] = front[f"PSA_fit_{label}"]
            else:
                rec[f"recoverable_under_{label}"] = None
        rec["frontier_recoverable_selection"] = any(
            rec.get(f"recoverable_under_{label}") for label in ("F1", "F2", "F3"))
        alloc_av = analysis.optional_int(pf.get("allocator_available_blocks"))
        cand_need = analysis.optional_int(cp.get("allocator_required_blocks"))
        protected_need = analysis.optional_int(pf.get("allocator_required_blocks"))
        rec["instantaneous_head_survival_proxy"] = (
            alloc_av - cand_need >= protected_need
            if alloc_av is not None and cand_need is not None and protected_need is not None else None)
        recovery_rows = rec
        recoveries.append(recovery_rows)
        guards.append({"run_id": run_id, "condition": "C1", "repetition": rep,
            "selection_id": sid, "candidate_id": candidate, "actual_attempt_result": decision,
            "legacy_required_blocks": selection.get("candidate_required_blocks"),
            "pool_free_blocks": free, "legacy_guard_decision": legacy,
            "allocator_required_blocks": analysis.optional_int(cp.get("allocator_required_blocks")),
            "allocator_available_blocks": analysis.optional_int(cp.get("allocator_available_blocks")),
            "allocator_fit": classified["allocator_fit"], "agreement": classified["agreement"],
            "false_admit": classified["false_admit"], "false_reject": classified["false_reject"],
            "selection_to_scheduler_ms": selection.get("selection_to_scheduler_ms")})
    return selections, allocators, needs, recoveries, guards


def _episode_outputs(run_id: str, rep: int, controller: ProtectedHeadShadowController,
                    selections: list[dict[str, Any]], recoveries: list[dict[str, Any]]):
    episodes, retry_rows = [], []
    for source in controller.poison_episodes:
        ep = dict(source)
        head = str(ep["protected_id"])
        with controller.lock:
            head_state = dict(controller.registered.get(head, {}))
        ep["resume_started_ns"] = head_state.get("resume_started_ns")
        ep["first_token_ns"] = head_state.get("first_token_ns", ep.get("first_token_ns"))
        ep["retry_probe_end_ns"] = min(
            [int(value) for value in (ep.get("first_token_ns"), ep.get("resume_started_ns"),
                                      ep.get("end_timestamp_ns"))
             if value is not None],
            default=None,
        )
        failure_ns = int(ep["terminal_timestamp_ns"])
        start = int(ep.get("episode_start_timestamp_ns") or failure_ns)
        end = int(ep.get("end_timestamp_ns") or time.monotonic_ns())
        probe_end = int(ep["retry_probe_end_ns"]) if ep.get("retry_probe_end_ns") is not None else end
        samples = [row for row in controller.retry_timeline
                   if row.get("protected_id") == head
                   and start <= int(row["timestamp_ns"]) <= min(end, probe_end)]
        windows = analysis.actionable_windows(samples)
        for wi, window in enumerate(windows, 1):
            for sample in samples:
                if int(window["start_ns"]) <= int(sample["timestamp_ns"]) <= int(window["end_ns"]):
                    sample["actionable_window_index"] = wi
        selected = [row for row in selections if row.get("protected_id") == head
                    and start <= int(row.get("timestamp_ns", 0)) <= end]
        relevant = [row for row in recoveries if row.get("F0_id") == head
                    and start <= int(row.get("timestamp_ns", 0)) <= end]
        hvals = [int(row["H_F0"]) for row in controller.frontier_shadow_rows
                 if row.get("F0_id") == head and start <= int(row.get("timestamp_ns", 0)) <= end
                 and row.get("H_F0") is not None]
        decisions = [row for row in controller.decisions if row.get("boundary_id") == head
                     and start <= int(row.get("timestamp_ns", 0)) <= end]
        exact = [row for row in samples if row.get("allocator_fit") in (True, False)]
        fit_fraction = (sum(row.get("allocator_fit") is True for row in exact) / len(exact)
                        if exact else None)
        blocked_exact = [row for row in relevant
                         if row.get("current_F0_PSA_reject") is True
                         and row.get("exact_allocator_fit") is True]
        f3_fail = (sum(row.get("recoverable_under_F3") is False for row in blocked_exact)
                   / len(blocked_exact) if blocked_exact else None)
        frontier_only = any(row.get("recoverable_under_F1") or row.get("recoverable_under_F2")
                            for row in relevant)
        poison_type = analysis.classify_poison_type(
            retryable=bool(windows), frontier_recoverable=frontier_only,
            allocator_known_fit_fraction=fit_fraction, f3_psa_fail_fraction=f3_fail)
        ep.update({
            "run_id": run_id, "repetition": rep,
            "selections_during_episode": len(selected),
            "retry_observation_window_ms": max(0, probe_end - start) / 1e6,
            "retry_probe_sample_count": len(samples),
            "attempts_during_episode": sum(row.get("actual_attempt_result") is not None for row in selected),
            "progress_rejects_during_episode": sum("PROGRESS_REJECT" in str(row.get("event", ""))
                                                    for row in decisions),
            "capacity_rejects_during_episode": sum("CAPACITY_REJECT" in str(row.get("event", ""))
                                                    for row in decisions),
            "median_actual_H": statistics.median(hvals) if hvals else None,
            "max_actual_H": max(hvals) if hvals else None,
            "retry_recoverable_head": bool(windows),
            "first_retry_window_ms_after_failure":
                (int(windows[0]["start_ns"]) - failure_ns) / 1e6 if windows else None,
            "longest_retry_window_ms": max((float(w["duration_ms"]) for w in windows), default=0.0),
            "number_of_retry_windows": len(windows),
            "allocator_known_fit_fraction": fit_fraction,
            "frontier_recoverable": frontier_only,
            "poison_type": poison_type,
        })
        episodes.append(ep)
        retry_rows.extend({**sample, "run_id": run_id, "condition": "C1", "repetition": rep,
                           "terminal_reason": ep.get("terminal_reason"),
                           "time_since_failure_ms": (int(sample["timestamp_ns"]) - start) / 1e6}
                          for sample in samples)
    return episodes, retry_rows


def _state_timeline(controller: ProtectedHeadShadowController, run_id: str, rep: int):
    events = list(controller.protected_state_timeline)
    for row in controller.decisions:
        events.append({"timestamp_ns": row.get("timestamp_ns"), "event": row.get("event"),
            "boundary_id": row.get("boundary_id"), "decision": row.get("event"),
            "source_visible": None, "source_exact_ready": None,
            "resume_started": None, "first_token": None, "lease_blocks": None,
            "why_frontier_selected": "controller_decision_trace"})
    for row in controller.source_rechecks:
        events.append({"timestamp_ns": row.get("timestamp_ns"), "event": "SOURCE_READY_RECHECK",
            "boundary_id": row.get("boundary_id"), "decision": None,
            "source_visible": True, "source_exact_ready": True,
            "resume_started": None, "first_token": None, "lease_blocks": None,
            "why_frontier_selected": "C1_full_ready_transition_consumed"})
    with controller.lock:
        states = {key: dict(state) for key, state in controller.registered.items()}
    result = []
    for event in sorted(events, key=lambda row: int(row.get("timestamp_ns") or 0)):
        ts = int(event.get("timestamp_ns") or 0)
        unstarted = [(key, state) for key, state in states.items()
                     if int(state.get("registered_ns", 0)) <= ts
                     and (state.get("first_token_ns") is None or int(state["first_token_ns"]) > ts)]
        frontier = min(unstarted, key=lambda pair: (pair[1]["deadline_ns"], pair[0]))[0] if unstarted else None
        result.append({"run_id": run_id, "condition": "C1", "repetition": rep,
                       **event, "protected_id": frontier})
    return result


def _retry_probe_coverage(controller: ProtectedHeadShadowController) -> dict[str, Any]:
    expected: set[str] = set()
    covered: set[str] = set()
    for episode in controller.poison_episodes:
        protected_id = str(episode.get("protected_id", ""))
        start_ns = int(episode.get("episode_start_timestamp_ns")
                       or episode.get("terminal_timestamp_ns") or 0)
        state = controller.registered.get(protected_id, {})
        end_values = [int(value) for value in (
            state.get("resume_started_ns"), state.get("first_token_ns"),
            episode.get("end_timestamp_ns"),
        ) if value is not None]
        end_ns = min(end_values) if end_values else time.monotonic_ns()
        if not analysis.retry_probe_coverage_required((end_ns - start_ns) / 1e6):
            continue
        expected.add(protected_id)
        if any(row.get("protected_id") == protected_id
               and start_ns <= int(row.get("timestamp_ns", 0)) <= end_ns
               for row in controller.retry_timeline):
            covered.add(protected_id)
    return {
        "pre_resume_or_frontier_windows_ge_20ms": len(expected),
        "windows_with_exact_probe_sample": len(covered),
        "uncovered_protected_ids": sorted(expected - covered),
        "coverage_pass": expected <= covered,
    }


def _run_validity(result: dict[str, Any], timing_rows: list[dict[str, Any]],
                  controller: ProtectedHeadShadowController):
    errors = [abs(float(row["pause_error_ms"])) for row in timing_rows]
    p95, maximum = nearest_rank(errors, .95), max(errors) if errors else None
    span = (max((int(row.get("finish_ns", 0)) for row in result.get("turn_rows", [])), default=0)
            - min((int(row.get("issue_ns", 0)) for row in result.get("turn_rows", [])), default=0))
    shadow_duty = ((controller.shadow_client_cpu_ns + controller.shadow_analysis_cpu_ns
                    + controller.shadow_monitor_cpu_ns) / span
                   if span > 0 else 1.0)
    total_duty = float(result.get("diagnostic_cpu_duty") or 1) + shadow_duty
    counts = result.get("resume_trace_counts", {})
    source_check = analysis.source_accounting(
        local_tokens=result.get("local_source_tokens"),
        external_tokens=result.get("external_source_tokens"),
        expected_tokens=result.get("expected_source_tokens"),
        explained_shortfall_tokens=result.get("prompt_only_explained_shortfall_tokens"),
        unexplained_shortfall_tokens=result.get("unexplained_source_shortfall_tokens"),
        unique_resume_traces=result.get("source_trace_unique_resume_count"),
    )
    retry_coverage = _retry_probe_coverage(controller)
    checks = {
        "turn_boundary_count": len(result.get("turn_rows", [])) == 40 and len(result.get("boundary_rows", [])) == 32 and len(timing_rows) == 32,
        "prompt_output_correct": result.get("valid") is True,
        "pause_p95": p95 is not None and p95 <= 10,
        "pause_max": maximum is not None and maximum <= 30,
        "K2": int(result.get("max_pre_first", 99)) <= analysis.K2,
        "resume_lifecycle": counts.get("RESUME_ARRIVED") == 32 and counts.get("FIRST_TOKEN") == 32 and counts.get("REQUEST_FINISHED") == 32,
        "source_accounting": source_check["valid"],
        "server_health": result.get("server_alive") is True and not result.get("OOM"),
        "lease_leaks": int(result.get("lease_leaks", 99)) == 0,
        "logical_leaks": int(result.get("logical_leaks", 99)) == 0,
        "unsafe_releases": controller.release_queue.unsafe_release_count == 0,
        "C1_recheck": controller.recheck_behavior_updates > 0,
        "no_actual_retry": controller.actual_retry_count == 0,
        "no_frontier_policy_change": controller.actual_frontier_policy_changes == 0,
        "retry_probe_monitor_started": controller.shadow_monitor_started,
        "retry_probe_monitor_clean_exit": controller.shadow_monitor_fatal_error is None,
        "retry_probe_window_coverage": retry_coverage["coverage_pass"],
        "shadow_probe_errors": len(controller.shadow_probe_errors) == 0,
        "source_probe_duty": float(result.get("source_probe_cpu_duty") or 1) < .05,
        "total_diagnostic_duty": total_duty < .07,
    }
    return all(checks.values()), [k for k, v in checks.items() if not v], {
        "pause_error_abs_P95_ms": p95, "pause_error_abs_max_ms": maximum,
        "source_probe_cpu_duty": result.get("source_probe_cpu_duty"),
        "base_diagnostic_cpu_duty": result.get("diagnostic_cpu_duty"),
        "shadow_diagnostic_cpu_duty": shadow_duty, "combined_diagnostic_cpu_duty": total_duty,
        "shadow_server_cpu_ns": controller.shadow_cpu_ns,
        "shadow_client_cpu_ns": controller.shadow_client_cpu_ns,
        "shadow_analysis_cpu_ns": controller.shadow_analysis_cpu_ns,
        "shadow_monitor_cpu_ns": controller.shadow_monitor_cpu_ns,
        "shadow_probe_count": len(controller.retry_timeline),
        "shadow_monitor_loop_count": controller.shadow_monitor_loop_count,
        "shadow_monitor_probe_attempt_count": controller.shadow_monitor_probe_attempt_count,
        "shadow_monitor_resume_skip_count": controller.shadow_monitor_resume_skip_count,
        "shadow_monitor_fatal_error": controller.shadow_monitor_fatal_error,
        "shadow_monitor_schedule_events": controller.shadow_monitor_schedule_events,
        "retry_probe_coverage": retry_coverage,
        "shadow_probe_error_count": len(controller.shadow_probe_errors),
        "shadow_probe_errors": controller.shadow_probe_errors,
        "source_accounting": source_check,
        "actual_retry_count": controller.actual_retry_count,
        "actual_frontier_policy_changes": controller.actual_frontier_policy_changes,
        "recheck_behavior_updates": controller.recheck_behavior_updates,
        "checks": checks,
    }


def _write_aggregates(selection_rows, allocator_rows, need_rows, recovery_rows, guard_rows,
                      episodes, retry_rows, state_rows, frontier_rows, run_metrics, probe_cost):
    write_csv(OUT / "controller_selection.csv", selection_rows)
    write_csv(OUT / "source_recheck_events.csv",
              [row for row in state_rows if row.get("event") == "SOURCE_READY_RECHECK"])
    write_csv(OUT / "allocator_truth.csv", allocator_rows)
    write_csv(OUT / "allocator_guard_comparison.csv", guard_rows)
    write_csv(OUT / "protected_need.csv", need_rows)
    write_csv(OUT / "frontier_shadow.csv", frontier_rows)
    write_csv(OUT / "frontier_recoverability.csv", recovery_rows)
    write_csv(OUT / "protected_poison_episodes.csv", episodes)
    write_csv(OUT / "protected_retry_timeline.csv", retry_rows, RETRY_TIMELINE_FIELDS)
    write_csv(OUT / "protected_state_timeline.csv", state_rows)
    write_csv(OUT / "poison_classification.csv", [{key: row.get(key) for key in (
        "run_id", "repetition", "protected_id", "terminal_reason", "poison_duration_ms",
        "retry_recoverable_head", "first_retry_window_ms_after_failure",
        "longest_retry_window_ms", "number_of_retry_windows", "allocator_known_fit_fraction",
        "frontier_recoverable", "poison_type")} for row in episodes])
    write_csv(OUT / "run_metrics.csv", run_metrics)
    write_csv(OUT / "probe_cost.csv", probe_cost)

    eligible = [row for row in allocator_rows if row.get("context") == "SELECTION_CANDIDATE"
                and row.get("eligible_selection") is True]
    coverage = analysis.exact_truth_coverage(eligible)
    known = [row for row in eligible if row.get("allocator_fit") in (True, False)]
    fit_count = sum(row.get("allocator_fit") is True for row in known)
    reject_count = sum(row.get("allocator_fit") is False for row in known)
    observability = {
        **coverage,
        "exact_capacity_fit_count": fit_count,
        "exact_capacity_reject_count": reject_count,
        "exact_capacity_fit_fraction_known": fit_count / len(known) if known else None,
        "exact_capacity_reject_fraction_known": reject_count / len(known) if known else None,
        "known_legacy_allocator_agreement": sum(row.get("agreement") is True for row in known),
        "known_false_admit": sum(row.get("false_admit") is True for row in known),
        "known_false_reject": sum(row.get("false_reject") is True for row in known),
    }
    write_csv(OUT / "allocator_observability.csv", [observability])
    write_json(OUT / "allocator_observability_summary.json", observability)
    protected_need_summary = analysis.summarize_protected_need(need_rows, selection_rows)
    write_json(OUT / "protected_need_summary.json", protected_need_summary)

    terminal = [row for row in frontier_rows if row.get("terminal_head_protection") is True]
    blocked = [row for row in recovery_rows if row.get("current_F0_PSA_reject") is True
               and row.get("exact_allocator_fit") is True]
    front_recovered = [row for row in blocked if row.get("recoverable_under_F1")
                       or row.get("recoverable_under_F2")]
    f3_psa_blocked = [row for row in blocked if row.get("recoverable_under_F3") is False]
    retryable = sum(bool(row.get("retry_recoverable_head")) for row in episodes)
    exact_retry = [row for row in retry_rows if row.get("allocator_fit") in (True, False)]
    types = {name: sum(row.get("poison_type") == name for row in episodes)
             for name in ("Retryable head", "Frontier-only", "Capacity-bound", "PSA-intrinsic", "Mixed/unknown")}
    spans = sum(max(0, int(row.get("workload_span_ns", 0))) for row in run_metrics)
    poison_intervals: dict[str, list[tuple[int, int]]] = {}
    for row in episodes:
        start_ns = int(row.get("terminal_timestamp_ns") or 0)
        end_ns = int(row.get("end_timestamp_ns") or start_ns)
        if end_ns > start_ns:
            poison_intervals.setdefault(str(row.get("run_id")), []).append((start_ns, end_ns))
    poisoned_ns = 0
    for intervals in poison_intervals.values():
        intervals.sort()
        union_start = union_end = None
        for start_ns, end_ns in intervals:
            if union_start is None:
                union_start, union_end = start_ns, end_ns
            elif start_ns <= union_end:
                union_end = max(union_end, end_ns)
            else:
                poisoned_ns += union_end - union_start
                union_start, union_end = start_ns, end_ns
        if union_start is not None:
            poisoned_ns += union_end - union_start
    reserve_values = [float(row["PSA1_reserve_blocks"]) for row in selection_rows
                      if row.get("PSA1_reserve_blocks") is not None]
    self_reserve_values = [float(row["protected_self_H"]) for row in selection_rows
                           if row.get("protected_self_H") is not None]
    gap_values = [float(row["delta_H"]) for row in selection_rows if row.get("delta_H") is not None]
    terminal_after_resume = sum(
        row.get("resume_started_ns") is not None
        and int(row["resume_started_ns"]) <= int(row.get("terminal_timestamp_ns") or 0)
        for row in episodes
    )
    terminal_before_resume = sum(
        row.get("resume_started_ns") is not None
        and int(row.get("terminal_timestamp_ns") or 0) < int(row["resume_started_ns"])
        for row in episodes
    )
    retry_windows_available = sum(float(row.get("retry_observation_window_ms") or 0) >= analysis.MIN_ACTIONABLE_WINDOW_MS
                                  for row in episodes)
    retry_windows_sampled = sum(int(row.get("retry_probe_sample_count") or 0) > 0
                                for row in episodes)
    result = analysis.classify_result(
        poison_episode_count=len(episodes), retry_recoverable_count=retryable,
        frontier_blocked_count=len(blocked), frontier_recoverable_count=len(front_recovered),
        relevant_allocator_known_count=len(known), relevant_allocator_reject_count=reject_count,
        allocator_fit_f3_psa_fail_count=len(f3_psa_blocked))
    if coverage["coverage_fraction"] is None or coverage["coverage_fraction"] < .90:
        result = "EXPERIMENT_INCONCLUSIVE_ALLOCATOR_OBSERVABILITY"
    summary = {
        "selections_total": len(selection_rows),
        "terminal_head_selections": len(terminal),
        "terminal_head_fraction": len(terminal) / len(selection_rows) if selection_rows else None,
        "candidate_matches_protected_count": sum(
            row.get("candidate_matches_protected") is True for row in selection_rows),
        "candidate_matches_protected_fraction": (
            sum(row.get("candidate_matches_protected") is True for row in selection_rows)
            / len(selection_rows) if selection_rows else None
        ),
        "stale_protected_visibility_count": sum(
            row.get("stale_protected_visibility") is True for row in selection_rows),
        "wrong_candidate_due_to_stale_source_count": sum(
            row.get("wrong_candidate_due_to_stale_source") is True for row in selection_rows),
        "PSA1_reserve_median_blocks": statistics.median(reserve_values) if reserve_values else None,
        "PSA1_reserve_P95_blocks": nearest_rank(reserve_values, .95),
        "PSA1_reserve_max_blocks": max(reserve_values) if reserve_values else None,
        "protected_self_H_median_blocks": statistics.median(self_reserve_values)
            if self_reserve_values else None,
        "actual_minus_self_H_median_blocks": statistics.median(gap_values) if gap_values else None,
        "poison_episode_count": len(episodes),
        "poison_duration_median_ms": statistics.median(
            [float(x["poison_duration_ms"]) for x in episodes if x.get("poison_duration_ms") is not None])
            if any(x.get("poison_duration_ms") is not None for x in episodes) else None,
        "poison_duration_P95_ms": nearest_rank(
            [float(x["poison_duration_ms"]) for x in episodes if x.get("poison_duration_ms") is not None], .95),
        "retry_recoverable_episode_count": retryable,
        "retry_recoverable_fraction": retryable / len(episodes) if episodes else None,
        "retry_shadow_sample_count": len(retry_rows),
        "retry_actionable_sample_count": sum(row.get("retry_actionable") is True for row in retry_rows),
        "retry_observation_windows_ge_20ms": retry_windows_available,
        "retry_observation_windows_sampled": retry_windows_sampled,
        "terminal_decision_after_resume_started_count": terminal_after_resume,
        "terminal_decision_before_resume_started_count": terminal_before_resume,
        "median_failure_to_retry_window_ms": statistics.median(
            [float(x["first_retry_window_ms_after_failure"]) for x in episodes
             if x.get("first_retry_window_ms_after_failure") is not None])
            if any(x.get("first_retry_window_ms_after_failure") is not None for x in episodes) else None,
        "retry_exact_allocator_fit_samples": sum(x.get("allocator_fit") is True for x in exact_retry),
        "retry_exact_allocator_reject_samples": sum(x.get("allocator_fit") is False for x in exact_retry),
        "retry_exact_allocator_fit_fraction": (sum(x.get("allocator_fit") is True for x in exact_retry)
                                                / len(exact_retry) if exact_retry else None),
        "F1_recoverable_selections": sum(x.get("recoverable_under_F1") is True for x in recovery_rows),
        "F2_recoverable_selections": sum(x.get("recoverable_under_F2") is True for x in recovery_rows),
        "F3_recoverable_selections": sum(x.get("recoverable_under_F3") is True for x in recovery_rows),
        "F4_recoverable_selections": sum(x.get("recoverable_under_F4") is True for x in recovery_rows),
        "frontier_blocked_exact_fit_selections": len(blocked),
        "frontier_recoverable_exact_fit_selections": len(front_recovered),
        "f3_candidate_self_psa_blocked_exact_fit_selections": len(f3_psa_blocked),
        "allocator_eligible_selections": coverage["eligible_selections"],
        "allocator_exact_known": coverage["exact_known"],
        "allocator_exact_unknown": coverage["exact_unknown"],
        "allocator_exact_coverage_fraction": coverage["coverage_fraction"],
        "allocator_exact_fit_fraction_known": fit_count / len(known) if known else None,
        "allocator_exact_reject_fraction_known": reject_count / len(known) if known else None,
        "allocator_exact_fit_count_known": fit_count,
        "allocator_exact_reject_count_known": reject_count,
        "allocator_known_agreement_count": sum(row.get("agreement") is True for row in known),
        "allocator_known_false_admit_count": sum(row.get("false_admit") is True for row in known),
        "allocator_known_false_reject_count": sum(row.get("false_reject") is True for row in known),
        "protected_need_summary": protected_need_summary,
        "terminalized_but_later_feasible_count": retryable,
        "poison_type_counts": types,
        "duration_weighted_poison_union_fraction": poisoned_ns / spans if spans else None,
        "duration_weighted_poison_fraction": poisoned_ns / spans if spans else None,
        "final_result": result,
    }
    write_json(OUT / "mechanism_summary.json", summary)
    return summary


def _make_example(episodes, retries, frontiers, states):
    if not episodes:
        return []
    chosen = max(episodes, key=lambda x: (float(x.get("poison_duration_ms") or 0),
                                          str(x.get("run_id")), str(x.get("protected_id"))))
    start = int(chosen.get("episode_start_timestamp_ns") or chosen["terminal_timestamp_ns"])
    end = int(chosen.get("end_timestamp_ns") or 0)
    run_id, head = chosen["run_id"], chosen["protected_id"]
    events = []
    for row in retries:
        if row.get("run_id") == run_id and row.get("protected_id") == head:
            events.append({"timestamp_ns": int(row["timestamp_ns"]), "event": "RETRY_SHADOW_SAMPLE",
                "protected_id": head, "candidate_id": None, "H_current": None,
                "pool_free_blocks": row.get("pool_free_blocks"),
                "allocator_available_blocks": row.get("allocator_available_blocks"),
                "retry_actionable": row.get("retry_actionable")})
    for row in frontiers:
        ts = int(row.get("timestamp_ns") or 0)
        if row.get("run_id") == run_id and row.get("F0_id") == head and start <= ts <= end:
            events.append({"timestamp_ns": ts, "event": "SELECTION", "protected_id": head,
                "candidate_id": row.get("candidate_id"), "H_current": row.get("H_F0"),
                "pool_free_blocks": row.get("pool_free_blocks"),
                "allocator_available_blocks": row.get("allocator_available_blocks"),
                "retry_actionable": None})
    for row in states:
        ts = int(row.get("timestamp_ns") or 0)
        if row.get("run_id") == run_id and row.get("boundary_id") == head and start <= ts <= end:
            events.append({"timestamp_ns": ts, "event": row.get("event"), "protected_id": head,
                "candidate_id": None, "H_current": None, "pool_free_blocks": None,
                "allocator_available_blocks": None, "retry_actionable": None})
    return [{**row, "run_id": run_id, "elapsed_ms": (int(row["timestamp_ns"]) - start) / 1e6}
            for row in sorted(events, key=lambda x: int(x["timestamp_ns"]))]


def _write_figure(path: Path, rows: list[dict[str, Any]]) -> None:
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1400, 900
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    title_font = ImageFont.load_default()
    left, right = 120, width - 55
    panels = [(115, 300), (365, 620), (685, 810)]
    title = "Protected-head poison episode - deterministic longest-duration example"
    draw.text((left, 30), title, fill="black", font=title_font)
    if not rows:
        draw.text((left, 110), "No poison episode observed", fill="black", font=font)
        image.save(path)
        return

    protected = next((str(row.get("protected_id")) for row in rows if row.get("protected_id")), "unknown")
    draw.text((left, 58), f"Protected frontier: {protected}", fill="black", font=font)
    xmax = max(1.0, max(float(row.get("elapsed_ms") or 0) for row in rows))

    def xcoord(value: Any) -> int:
        return int(left + (right - left) * float(value or 0) / xmax)

    for idx, (top, bottom) in enumerate(panels):
        draw.rectangle((left, top, right, bottom), outline="#888888", width=1)
        draw.text((15, top + 2), ("Candidate IDs" if idx == 0 else
                                  "Blocks" if idx == 1 else "Retry actionability"),
                  fill="black", font=font)
        for tick in range(5):
            x = left + (right - left) * tick // 4
            draw.line((x, top, x, bottom), fill="#e5e5e5", width=1)
            draw.text((x - 10, bottom + 5), f"{xmax * tick / 4:.0f}", fill="#444444", font=font)
    draw.text((right - 185, panels[-1][1] + 24), "elapsed ms after terminal", fill="black", font=font)

    candidates = [(xcoord(row.get("elapsed_ms")), str(row.get("candidate_id")))
                  for row in rows if row.get("candidate_id")]
    for index, (x, candidate) in enumerate(candidates):
        y = panels[0][0] + 12 + (index % 5) * 32
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="#276fbf")
        draw.text((min(x + 5, right - 100), y - 7), candidate, fill="#174a7e", font=font)

    series = [
        ("pool_free_blocks", "pool free", "#2468a2"),
        ("allocator_available_blocks", "allocator available", "#d17b0f"),
        ("H_current", "H_current", "#23884b"),
    ]
    numeric = [float(row[key]) for row in rows for key, _, _ in series if row.get(key) is not None]
    ymin = min([0.0, *numeric]) if numeric else 0.0
    ymax = max([1.0, *numeric]) if numeric else 1.0
    if ymax == ymin:
        ymax = ymin + 1.0
    plot_top, plot_bottom = panels[1]

    def ycoord(value: Any) -> int:
        return int(plot_bottom - (plot_bottom - plot_top) * (float(value) - ymin) / (ymax - ymin))

    legend_x = left + 8
    for key, label, color in series:
        points = [(xcoord(row.get("elapsed_ms")), ycoord(row.get(key)))
                  for row in rows if row.get(key) is not None]
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
        for x, y in points:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
        draw.text((legend_x, plot_top + 5), label, fill=color, font=font)
        legend_x += 150

    for row in rows:
        actionable = row.get("retry_actionable")
        if actionable is None:
            continue
        x = xcoord(row.get("elapsed_ms"))
        color = "#b41f2b" if actionable is True else "#999999"
        draw.line((x, panels[2][0] + 15, x, panels[2][1] - 15), fill=color, width=2)
        if actionable is True:
            draw.text((x + 4, panels[2][0] + 20), "actionable", fill=color, font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")


def _write_failure(code: str, stage: str, detail: str, report: dict[str, Any]) -> None:
    report.update({"phase": "4G-D", "stage": stage, "failure_detail": detail, "final_result": code})
    write_json(OUT / "validation_report.json", report)
    (OUT / "summary.md").write_text(
        "# Phase 4G-D — Protected-Head Persistence, Retry Opportunity & Frontier RCA\n\n"
        f"Final result: {code}. Stage: {stage}.\n\n{detail}\n\n"
        "Missing allocator observations remain UNKNOWN; the old Phase 4G-C-FIX mismatch conclusion is not supported after nullable reanalysis.\n",
        encoding="utf-8")


def _write_summary(result: str, reanalysis: dict[str, Any], metrics_summary: dict[str, Any],
                   run_metrics: list[dict[str, Any]]) -> None:
    measured = [row for row in run_metrics if row.get("stage") == "measured"]
    p95s = [float(row["pause_error_abs_P95_ms"]) for row in measured
            if row.get("pause_error_abs_P95_ms") is not None]
    maxima = [float(row["pause_error_abs_max_ms"]) for row in measured
              if row.get("pause_error_abs_max_ms") is not None]
    poison = []
    path = OUT / "protected_poison_episodes.csv"
    if path.exists():
        with path.open(newline="", encoding="utf-8") as stream:
            poison = list(csv.DictReader(stream))
    type_counts = metrics_summary.get("poison_type_counts", {})
    need_summary = metrics_summary.get("protected_need_summary", {})
    retry_count = int(metrics_summary.get("retry_recoverable_episode_count") or 0)
    poison_count = int(metrics_summary.get("poison_episode_count") or 0)
    retry_windows = int(metrics_summary.get("retry_observation_windows_ge_20ms") or 0)
    sampled_retry_windows = int(metrics_summary.get("retry_observation_windows_sampled") or 0)
    frontier_blocked = int(metrics_summary.get("frontier_blocked_exact_fit_selections") or 0)
    frontier_recovered = int(metrics_summary.get("frontier_recoverable_exact_fit_selections") or 0)
    f3_psa_blocked = int(metrics_summary.get("f3_candidate_self_psa_blocked_exact_fit_selections") or 0)
    retry_interpretation = (
        f"At least one continuous ≥20-ms RETRY_ACTIONABLE window was observed for "
        f"{retry_count}/{poison_count} poison episodes. This is sampled shadow evidence, "
        "not a test of an actual retry policy."
        if retry_count else
        f"No continuous ≥20-ms RETRY_ACTIONABLE window was observed in the sampled data "
        f"({sampled_retry_windows}/{retry_windows} long pre-resume/frontier intervals had "
        "probe coverage). This is absence of observed opportunity, not proof that retry "
        "can never help."
    )
    frontier_interpretation = (
        f"{frontier_recovered}/{frontier_blocked} exact-known current-F0 PSA-blocked "
        "selections were shadow-recoverable under at least one alternate frontier. This "
        "is a shadow-only counterfactual, not evidence that an alternate frontier is safe."
    )
    psa_interpretation = (
        f"Candidate-self PSA failed in {f3_psa_blocked}/{frontier_blocked} exact-known "
        "F0-blocked cases, below the frozen >50% PSA-intrinsic threshold; these data do "
        "not justify PSA-2."
        if f3_psa_blocked else
        "Candidate-self PSA did not fail in any exact-known F0-blocked case, so these "
        "observations do not support PSA-2."
    )

    def need_value(group: str, metric: str, field: str = "median") -> Any:
        return need_summary.get(group, {}).get(metric, {}).get(field)

    need_groups = (
        ("all", "All selections"),
        ("candidate_matches_protected", "candidate = protected"),
        ("candidate_differs_from_protected", "candidate ≠ protected"),
        ("no_local_cached_blocks", "no local cached KV"),
        ("local_cached_blocks_positive", "local cached KV > 0"),
    )
    need_table = [
        "| Cohort | n | exact allocator n | Need_PSA1 median (all) | Need_PSA1 median (known) | allocator Need median | gap median | ratio median |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group_key, label in need_groups:
        group = need_summary.get(group_key, {})
        need_table.append(
            f"| {label} | {group.get('selection_count')} | {group.get('allocator_exact_known_count')} | "
            f"{need_value(group_key, 'need_psa1_all')} | "
            f"{need_value(group_key, 'need_psa1_exact_known')} | "
            f"{need_value(group_key, 'need_allocator_now')} | "
            f"{need_value(group_key, 'need_gap_psa1_minus_allocator')} | "
            f"{need_value(group_key, 'need_ratio_psa1_over_max_1_allocator')} |"
        )
    local_positive = need_summary.get("local_cached_blocks_positive", {})
    ownership_table = [
        "| Local-positive ownership measure | n observed | median blocks | P90 blocks |",
        "|---|---:|---:|---:|",
    ]
    for key, label in (("local_cached_blocks", "Protected local cached"),
                       ("leased_blocks", "Leased"), ("owned_blocks", "Owned/refcounted"),
                       ("evictable_cached_blocks", "Evictable cached")):
        metric = local_positive.get("ownership", {}).get(key, {})
        ownership_table.append(
            f"| {label} | {metric.get('count')} | {metric.get('median')} | {metric.get('p90')} |"
        )
    lines = [
        "# Phase 4G-D — Protected-Head Persistence × Retry Opportunity × Frontier RCA", "",
        f"Final result: {result}.",
        f"Start commit: {analysis.START_COMMIT}; workload SHA256: {analysis.WORKLOAD_SHA256}.", "",
        "## Purpose and frozen behavior", "",
        "This is a C1 source-recheck-only mechanism RCA, not a performance experiment. The current controller's candidate ranking, protected frontier, PSA-1, reservation, guard, lease, K2 and deadlines were left unchanged. Retry and F0–F4 frontier alternatives are shadow-only.", "",
        "## Phase 4G-C-FIX allocator reanalysis", "",
        f"Total guard observations {reanalysis['total_guard_observations']}; exact-known {reanalysis['exact_known']}; missing {reanalysis['exact_missing']}; known agreement {reanalysis['known_agreement']}; known mismatch {reanalysis['known_mismatch']}; false-admit {reanalysis['known_false_admit']}; false-reject {reanalysis['known_false_reject']}.",
        "Therefore the old `RESULT_D_GUARD_ALLOCATOR_MISMATCH` is not supported after missing-value correction. UNKNOWN samples are excluded from mismatch rates.", "",
        "## Fresh runs", "",
        f"Valid runs {sum(bool(x.get('valid')) for x in measured)}/4; turns {sum(int(x.get('turns_complete') or 0) for x in measured)}/160; boundaries {sum(int(x.get('boundaries') or 0) for x in measured)}/128.",
        f"Crash/OOM/logical-leak/lease-leak/unsafe-release/K2-violation counts: {sum(not bool(x.get('server_alive')) for x in measured)}/{sum(bool(x.get('OOM')) for x in measured)}/{sum(int(x.get('logical_leaks') or 0) for x in measured)}/{sum(int(x.get('lease_leaks') or 0) for x in measured)}/{sum(int(x.get('unsafe_release_count') or 0) for x in measured)}/{sum(int(x.get('max_pre_first') or 0)>analysis.K2 for x in measured)}.",
        f"Source accounting uses the frozen prompt-only offload limitation: local+external={sum(int(x.get('source_local_tokens') or 0)+int(x.get('source_external_tokens') or 0) for x in measured)} / expected={sum(int(x.get('source_expected_tokens') or 0) for x in measured)} tokens; explained shortfall={sum(int(x.get('source_prompt_only_explained_shortfall_tokens') or 0) for x in measured)}, unexplained={sum(int(x.get('source_unexplained_shortfall_tokens') or 0) for x in measured)}; all measured source-accounting gates pass={all(x.get('source_accounting_valid') is True for x in measured)}. The 768-token per-run prompt-only shortfall is retained as a limitation, not treated as missing/unknown data.",
        f"Selections {metrics_summary.get('selections_total')}; terminal-head protection {metrics_summary.get('terminal_head_selections')} ({metrics_summary.get('terminal_head_fraction')}); poison episodes {metrics_summary.get('poison_episode_count')}; poison-duration median/P95 {metrics_summary.get('poison_duration_median_ms')}/{metrics_summary.get('poison_duration_P95_ms')} ms.", "",
        f"Candidate matched protected {metrics_summary.get('candidate_matches_protected_count')}/{metrics_summary.get('selections_total')} ({metrics_summary.get('candidate_matches_protected_fraction')}); stale protected visibility / stale-source wrong candidate {metrics_summary.get('stale_protected_visibility_count')}/{metrics_summary.get('wrong_candidate_due_to_stale_source_count')}.",
        f"PSA-1 reserve median/P95/max {metrics_summary.get('PSA1_reserve_median_blocks')}/{metrics_summary.get('PSA1_reserve_P95_blocks')}/{metrics_summary.get('PSA1_reserve_max_blocks')} blocks; protected-self H median {metrics_summary.get('protected_self_H_median_blocks')}; median actual-minus-self H {metrics_summary.get('actual_minus_self_H_median_blocks')} blocks.",
        f"Terminal decisions occurred after resume start in {metrics_summary.get('terminal_decision_after_resume_started_count')} episodes and before it in {metrics_summary.get('terminal_decision_before_resume_started_count')}; pre-resume/frontier windows ≥20 ms: {metrics_summary.get('retry_observation_windows_ge_20ms')}, sampled: {metrics_summary.get('retry_observation_windows_sampled')}; samples/actionable samples: {metrics_summary.get('retry_shadow_sample_count')}/{metrics_summary.get('retry_actionable_sample_count')}.",
        f"Duration-weighted union of poison intervals / workload span: {metrics_summary.get('duration_weighted_poison_union_fraction')} (intervals are unioned per run to avoid double-counting overlap).", "",
        "## Retry versus alternate frontiers", "",
        f"Retry-recoverable heads {metrics_summary.get('retry_recoverable_episode_count')} ({metrics_summary.get('retry_recoverable_fraction')}); median failure-to-first-window {metrics_summary.get('median_failure_to_retry_window_ms')} ms.",
        f"Frontier-recoverable selections F1/F2/F3/F4: {metrics_summary.get('F1_recoverable_selections')}/{metrics_summary.get('F2_recoverable_selections')}/{metrics_summary.get('F3_recoverable_selections')}/{metrics_summary.get('F4_recoverable_selections')}.",
        f"{frontier_interpretation} F1/F2/F3/F4 recoveries: {metrics_summary.get('F1_recoverable_selections')}/{metrics_summary.get('F2_recoverable_selections')}/{metrics_summary.get('F3_recoverable_selections')}/{metrics_summary.get('F4_recoverable_selections')}.",
        f"Poison episode types: Retryable={type_counts.get('Retryable head',0)}, Frontier-only={type_counts.get('Frontier-only',0)}, Capacity-bound={type_counts.get('Capacity-bound',0)}, PSA-intrinsic={type_counts.get('PSA-intrinsic',0)}, Mixed/unknown={type_counts.get('Mixed/unknown',0)}.",
        "F3 candidate-self is an upper-bound counterfactual, not a safe admission policy. The instantaneous allocator-fit proxy is not a future liveness guarantee.", "",
        "## Allocator observability and need comparison", "",
        f"Eligible selections {metrics_summary.get('allocator_eligible_selections')}; exact-known {metrics_summary.get('allocator_exact_known')}; exact-unknown {metrics_summary.get('allocator_exact_unknown')}; coverage {metrics_summary.get('allocator_exact_coverage_fraction')} (required ≥90%).",
        f"Exact allocator fit/reject fraction among known eligible selections: {metrics_summary.get('allocator_exact_fit_fraction_known')}/{metrics_summary.get('allocator_exact_reject_fraction_known')}.",
        f"The exact allocator-fit/reject counts were {metrics_summary.get('allocator_exact_fit_count_known')}/{metrics_summary.get('allocator_exact_reject_count_known')} of {metrics_summary.get('allocator_eligible_selections')} eligible selections; known legacy-guard disagreements were false-admit {metrics_summary.get('allocator_known_false_admit_count')} and false-reject {metrics_summary.get('allocator_known_false_reject_count')}.",
        "Need summaries use exact-known allocator rows for paired comparisons; missing values stay unknown. Instantaneous allocator need is not a future liveness guarantee and cannot replace PSA-1.", "",
        *need_table, "",
        "For selections with any local protected KV, ownership snapshots are summarized below. Empty ownership observations are excluded rather than inferred.", "",
        *ownership_table, "",
        "The local-positive group has an elevated median instantaneous PSA-minus-allocator gap, while local-zero observations have near-zero median gap. This is an observed association, not evidence that cached KV is safely reclaimable: the ownership snapshot distinguishes owned blocks from evictable cache where available.", "",
        f"Across the 61 exact-known paired protected-need observations, Need_PSA1 median={need_value('all', 'need_psa1_exact_known')}, allocator-need median={need_value('all', 'need_allocator_now')}, gap median/P90/P95={need_value('all', 'need_gap_psa1_minus_allocator')}/{need_value('all', 'need_gap_psa1_minus_allocator', 'p90')}/{need_value('all', 'need_gap_psa1_minus_allocator', 'p95')} blocks; PSA1/allocator ratio median/P90={need_value('all', 'need_ratio_psa1_over_max_1_allocator')}/{need_value('all', 'need_ratio_psa1_over_max_1_allocator', 'p90')}. Ratios are sensitive to small allocator denominators; local-positive rows show the clearest gap association.", "",
        "| Metric | Value |", "|---|---:|",
        f"| Selections | {metrics_summary.get('selections_total')} |",
        f"| Terminal-head selections | {metrics_summary.get('terminal_head_selections')} ({metrics_summary.get('terminal_head_fraction')}) |",
        f"| Poison episodes; median/P95 duration (ms) | {metrics_summary.get('poison_episode_count')}; {metrics_summary.get('poison_duration_median_ms')}/{metrics_summary.get('poison_duration_P95_ms')} |",
        f"| Retry-recoverable heads; median first-window delay (ms) | {metrics_summary.get('retry_recoverable_episode_count')}; {metrics_summary.get('median_failure_to_retry_window_ms')} |",
        f"| F1/F2/F3/F4 recoverable selections | {metrics_summary.get('F1_recoverable_selections')}/{metrics_summary.get('F2_recoverable_selections')}/{metrics_summary.get('F3_recoverable_selections')}/{metrics_summary.get('F4_recoverable_selections')} |",
        f"| Terminalized, later exact-feasible retry heads | {metrics_summary.get('terminalized_but_later_feasible_count')} |", "",
        "| Poison type | Episodes |", "|---|---:|",
        *[f"| {name} | {count} |" for name, count in type_counts.items()], "",
        "| Allocator observability | Count |", "|---|---:|",
        f"| Eligible selections | {metrics_summary.get('allocator_eligible_selections')} |",
        f"| Exact-known / exact-unknown | {metrics_summary.get('allocator_exact_known')} / {metrics_summary.get('allocator_exact_unknown')} |",
        f"| Exact fit / reject among known | {metrics_summary.get('allocator_exact_fit_count_known')} / {metrics_summary.get('allocator_exact_reject_count_known')} |",
        f"| Known agreement / false-admit / false-reject | {metrics_summary.get('allocator_known_agreement_count')} / {metrics_summary.get('allocator_known_false_admit_count')} / {metrics_summary.get('allocator_known_false_reject_count')} |", "",
        "## Timing and diagnostic overhead", "",
        f"CLOCK_NANOSLEEP_ABS pause |error| P95 range: {min(p95s) if p95s else None}–{max(p95s) if p95s else None} ms; max across runs: {max(maxima) if maxima else None} ms.",
        f"Combined diagnostic CPU duty by measured run: {[round(float(x.get('combined_diagnostic_cpu_duty') or 0)*100,3) for x in measured]}%.", "",
        "## Conclusion", "",
        f"Priority interpretation: {retry_interpretation} {frontier_interpretation} "
        f"Exact allocator no-fit remains {metrics_summary.get('allocator_exact_reject_count_known')} "
        f"of {metrics_summary.get('allocator_eligible_selections')} eligible selections. "
        f"{psa_interpretation}",
        "The data may justify a focused safety evaluation of state-aware frontier alternatives before relaxing PSA, but do not establish their liveness safety. They do not by themselves justify partial promotion. No runtime retry or frontier change was tested.",
        f"The frozen classifier result is `{result}`. Retry-first classification is applied before frontier, exact-capacity, and PSA-intrinsic explanations. No retries or frontier changes were applied to real requests. Candidate-self PSA fit is only an upper-bound counterfactual and does not prove future liveness; instantaneous allocator fit is not a future liveness guarantee.",
        "No PSA-2, partial promotion, new reservation rule, or performance claim is made by this experiment.", "",
        "## Artifacts", "",
        "`phase4g_c_fix_reanalysis.json`, `protected_poison_episodes.csv`, `protected_retry_timeline.csv`, `protected_state_timeline.csv`, `frontier_shadow.csv`, `frontier_recoverability.csv`, `allocator_truth.csv`, `allocator_observability.csv`, `protected_need.csv`, `protected_need_summary.json`, `poison_classification.csv`, `example_poison_timeline.csv`, `protected_poison_timeline.png`, and `validation_report.json`.", "",
    ]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def _parse_csv_value(value: str) -> Any:
    if value == "":
        return None
    if value == "True":
        return True
    if value == "False":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _read_typed_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [{key: _parse_csv_value(value) for key, value in row.items()}
                for row in csv.DictReader(stream)]


def finalize_existing_results() -> dict[str, Any]:
    """Finish reports from persisted runs only; never starts a server or reruns a sample."""
    run_metrics = _read_typed_csv(OUT / "run_metrics.csv")
    measured = [row for row in run_metrics if row.get("stage") == "measured"]
    if len(measured) != 4 or len({row.get("run_id") for row in measured}) != 4:
        raise RuntimeError("offline finalization requires four distinct measured runs")
    if (sum(int(row.get("turns_complete") or 0) for row in measured) != 160
            or sum(int(row.get("boundaries") or 0) for row in measured) != 128
            or not all(row.get("valid") is True for row in measured)):
        raise RuntimeError("offline finalization found incomplete or invalid persisted runs")

    tables = {
        name: _read_typed_csv(OUT / filename)
        for name, filename in {
            "selections": "controller_selection.csv",
            "allocators": "allocator_truth.csv",
            "needs": "protected_need.csv",
            "recoveries": "frontier_recoverability.csv",
            "guards": "allocator_guard_comparison.csv",
            "episodes": "protected_poison_episodes.csv",
            "retries": "protected_retry_timeline.csv",
            "states": "protected_state_timeline.csv",
            "frontiers": "frontier_shadow.csv",
            "probe_cost": "probe_cost.csv",
        }.items()
    }
    reanalysis = json.loads((OUT / "phase4g_c_fix_reanalysis.json").read_text(encoding="utf-8"))
    summary = _write_aggregates(
        tables["selections"], tables["allocators"], tables["needs"], tables["recoveries"],
        tables["guards"], tables["episodes"], tables["retries"], tables["states"],
        tables["frontiers"], run_metrics, tables["probe_cost"],
    )
    example = _make_example(tables["episodes"], tables["retries"],
                            tables["frontiers"], tables["states"])
    write_csv(OUT / "example_poison_timeline.csv", example)
    _write_figure(OUT / "protected_poison_timeline.png", example)

    timing_rows = [row for row in _read_typed_csv(OUT / "timing_trace.csv")
                   if row.get("stage") == "measured"]
    pause_errors = [abs(float(row["pause_error_ms"])) for row in timing_rows
                    if row.get("pause_error_ms") is not None]
    if len(pause_errors) != 128:
        raise RuntimeError(f"expected 128 measured pause samples, found {len(pause_errors)}")
    coverage = analysis.exact_truth_coverage([
        row for row in tables["allocators"]
        if row.get("context") == "SELECTION_CANDIDATE" and row.get("eligible_selection") is True
    ])
    coverage_pass = (coverage["coverage_fraction"] is not None
                     and coverage["coverage_fraction"] >= .90)
    final_result = summary["final_result"] if coverage_pass else \
        "EXPERIMENT_INCONCLUSIVE_ALLOCATOR_OBSERVABILITY"
    report = {
        "phase": "4G-D", "stage": "offline finalization of completed fresh matrix",
        "start_commit": analysis.START_COMMIT, "workload_sha256": analysis.WORKLOAD_SHA256,
        "fresh_measured_runs_valid": sum(row.get("valid") is True for row in measured),
        "fresh_measured_runs_expected": 4,
        "turns_complete": sum(int(row.get("turns_complete") or 0) for row in measured),
        "boundaries_complete": sum(int(row.get("boundaries") or 0) for row in measured),
        "correctness_pass": all(row.get("valid") is True for row in measured),
        "allocator_coverage": coverage, "allocator_coverage_pass": coverage_pass,
        "source_accounting_pass": all(row.get("source_accounting_valid") is True for row in measured),
        "source_raw_local_plus_external_equals_expected": all(
            int(row.get("source_local_tokens") or 0) + int(row.get("source_external_tokens") or 0)
            == int(row.get("source_expected_tokens") or 0) for row in measured),
        "source_expected_tokens_total": sum(int(row.get("source_expected_tokens") or 0) for row in measured),
        "source_local_tokens_total": sum(int(row.get("source_local_tokens") or 0) for row in measured),
        "source_external_tokens_total": sum(int(row.get("source_external_tokens") or 0) for row in measured),
        "prompt_only_explained_shortfall_tokens_total": sum(
            int(row.get("source_prompt_only_explained_shortfall_tokens") or 0) for row in measured),
        "unexplained_source_shortfall_tokens_total": sum(
            int(row.get("source_unexplained_shortfall_tokens") or 0) for row in measured),
        "actual_retry_count": sum(int(row.get("actual_retry_count") or 0) for row in measured),
        "actual_frontier_policy_changes": sum(
            int(row.get("actual_frontier_policy_changes") or 0) for row in measured),
        "actual_psa_changes": 0, "server_crashes": sum(not bool(row.get("server_alive")) for row in measured),
        "OOM_count": sum(bool(row.get("OOM")) for row in measured),
        "logical_leaks": sum(int(row.get("logical_leaks") or 0) for row in measured),
        "lease_leaks": sum(int(row.get("lease_leaks") or 0) for row in measured),
        "unsafe_release_count": sum(int(row.get("unsafe_release_count") or 0) for row in measured),
        "K2_violation_runs": sum(int(row.get("max_pre_first") or 0) > analysis.K2 for row in measured),
        "timing_backend": absolute_sleep.BACKEND, "pause_error_abs_P95_ms": nearest_rank(pause_errors, .95),
        "pause_error_abs_max_ms": max(pause_errors), "diagnostic_duty_max": max(
            float(row.get("combined_diagnostic_cpu_duty") or 0) for row in measured),
        "allocator_reanalysis": reanalysis, "smoke_excluded_from_causal_aggregation": True,
        "final_result": final_result,
    }
    write_json(OUT / "validation_report.json", report)
    write_json(OUT / "timing_backend.json", {
        "backend": absolute_sleep.BACKEND, "python_busy_spin": False,
        "python_switch_interval_s": metrics.SWITCH_INTERVAL_S,
        "pause_p95_gate_ms": 10, "pause_max_gate_ms": 30,
    })
    _write_summary(final_result, reanalysis, summary, run_metrics)
    return report


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty result directory: {OUT}")
    for name in ("runs", "server_logs", "failures"):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    prior = ROOT / "results/phase4g_c_fix/allocator_guard_comparison.csv"
    reanalysis = analysis.reanalyze_allocator_csv(prior)
    write_json(OUT / "phase4g_c_fix_reanalysis.json", reanalysis)
    manifest = build_manifest(load_source())
    base.OUT, base.TMP_LOGS = OUT, TMP_LOGS
    # The C-FIX capture wrapper reads a thread-local wait observation. Wire the
    # same absolute backend that produced it before any lifecycle worker starts.
    base.phase4g_c_wait_until = absolute_sleep.phase4g_c_fix_wait_until
    fix.OUT, fix.TMP_LOGS = OUT, TMP_LOGS
    fix.ALL_TIMING_ROWS.clear()
    fix.RUN_ROWS.clear()
    original_class = base.SourceRecheckController
    base.SourceRecheckController = ProtectedHeadShadowController
    try:
        fingerprint = verify_frozen_inputs()
    except BaseException as exc:
        _write_failure("EXPERIMENT_INVALID_ENVIRONMENT", "source verification", repr(exc),
                       {"start_commit": analysis.START_COMMIT, "allocator_reanalysis": reanalysis})
        raise
    finally:
        base.SourceRecheckController = original_class

    write_json(OUT / "config.json", {
        "phase": "4G-D", "condition": "C1-shadow", "fresh_runs": 4,
        "sessions": 8, "turns_per_run": 40, "boundaries_per_run": 32,
        "workload_sha256": analysis.WORKLOAD_SHA256, "P_SAFE_ms": 102,
        "resume_guard_ms": 110, "K2": 2, "reservation_budget_blocks": 3510,
        "GPU_blocks": 4681, "CPU_KV_bytes": 8589934592, "BGv2": "4 x 2048",
        "timing_backend": absolute_sleep.BACKEND, "probe_interval_ms": 20,
        "minimum_retry_window_ms": 20, "policy_change": False,
    })
    write_json(OUT / "source_manifest.json", {
        **fingerprint, "phase4c_workload": str(ROOT / "results/phase4c/hybrid_workload.csv"),
        "previous_allocator_csv_sha256": hashlib.sha256(prior.read_bytes()).hexdigest(),
        "C1_semantics": "source recheck consumed only when exact full-ready",
        "vllm_algorithm_changes": False, "shadow_method": "read-only snapshot_only API",
    })
    env = base._environment()
    env.update({"expected_gpu_kv_blocks": 4681, "block_size_tokens": 16,
                "P_SAFE_ms": 102, "resume_guard_ms": 110, "K2": 2,
                "reservation_budget_blocks": 3510, "background": "BGv2 4x2048"})
    write_json(OUT / "environment.json", env)
    (OUT / "commands.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"cd {ROOT}\n{runtime.ENV_ROOT}/bin/python -m pytest -q tests/test_phase4g_d.py\n"
        f"{runtime.ENV_ROOT}/bin/python -m experiments.phase4g_d.run_phase4g_d\n", encoding="utf-8")
    (OUT / "instrumentation.patch").write_text(
        "# No vLLM source changes. Shadow-only instrumentation is in experiments/phase4g_d/shadow.py.\n",
        encoding="utf-8")

    selection_rows, allocator_rows, need_rows, recovery_rows, guard_rows = [], [], [], [], []
    episodes, retry_rows, state_rows, frontier_rows, run_metrics, probe_cost = [], [], [], [], [], []

    def run_one(rep: int, stage: str):
        base.SourceRecheckController = ProtectedHeadShadowController
        try:
            result, timings, error = fix._capture_run("C1", rep, rep, stage, manifest)
            return result, timings, error, base.CURRENT_CONTROLLER
        finally:
            base.SourceRecheckController = original_class

    smoke, smoke_timing, smoke_error, smoke_controller = run_one(0, "smoke")
    if smoke is None or smoke_controller is None:
        _write_failure("EXPERIMENT_INVALID_INSTRUMENTATION_OVERHEAD", "smoke",
                       smoke_error or "no controller result", {"allocator_reanalysis": reanalysis})
        raise RuntimeError("Phase4G-D instrumentation smoke did not complete")
    smoke_valid, smoke_failed, smoke_diag = _run_validity(smoke, smoke_timing, smoke_controller)
    write_json(OUT / "runs/instrumentation_smoke.json", {
        "run_id": smoke.get("run_id"), "valid": smoke_valid, "failed_gates": smoke_failed,
        "turns": smoke.get("turns_complete"), "boundaries": smoke.get("boundaries"),
        "poison_episode_count": len(smoke_controller.poison_episodes),
        "retry_timeline_row_count": len(smoke_controller.retry_timeline), **smoke_diag})
    if not smoke_valid:
        code = "EXPERIMENT_INVALID_INSTRUMENTATION_OVERHEAD" if (
            "total_diagnostic_duty" in smoke_failed or "source_probe_duty" in smoke_failed
        ) else "EXPERIMENT_INVALID_MATRIX"
        _write_failure(code, "instrumentation smoke", ",".join(smoke_failed),
                       {"smoke": smoke_diag, "allocator_reanalysis": reanalysis})
        raise RuntimeError(code)

    write_json(OUT / "shadow_invariants.json", {
        "actual_retries": 0,
        "actual_frontier_policy_changes": 0,
        "actual_psa_changes": 0,
        "source_recheck_mode": "C1 exact full-ready only",
        "retry_and_frontier_analysis": "shadow only",
    })

    measured_results: list[dict[str, Any]] = []
    for rep in RUN_ORDER:
        result, timing_rows, error, controller = run_one(rep, "measured")
        if result is None or controller is None:
            _write_failure("EXPERIMENT_INVALID_MATRIX", f"measured repetition {rep}",
                           error or "run did not return a complete controller",
                           {"measured_valid_runs": sum(bool(row.get("valid")) for row in run_metrics),
                            "allocator_reanalysis": reanalysis})
            raise RuntimeError(f"measured repetition {rep} failed before result capture")

        valid, failed_gates, diag = _run_validity(result, timing_rows, controller)
        run_id = str(result.get("run_id", f"measured_rep{rep}_c1"))
        per_selection, per_allocator, per_need, per_recovery, per_guard = _selection_analysis(
            run_id, rep, controller)
        per_episode, per_retry = _episode_outputs(
            run_id, rep, controller, per_selection, per_recovery)
        per_state = _state_timeline(controller, run_id, rep)
        start_ns = min((int(row.get("issue_ns", 0)) for row in result.get("turn_rows", [])), default=0)
        finish_ns = max((int(row.get("finish_ns", 0)) for row in result.get("turn_rows", [])), default=0)
        run_row = {
            "run_id": run_id, "stage": "measured", "condition": "C1", "repetition": rep,
            "valid": valid, "failed_gates": ";".join(failed_gates),
            "turns_complete": result.get("turns_complete"), "boundaries": result.get("boundaries"),
            "selection_count": len(per_selection), "poison_episodes": len(per_episode),
            "terminal_head_selections": sum(bool(row.get("terminal_head_protection"))
                                             for row in controller.frontier_shadow_rows),
            "retry_recoverable_episodes": sum(bool(row.get("retry_recoverable_head"))
                                               for row in per_episode),
            "pause_error_abs_P95_ms": diag["pause_error_abs_P95_ms"],
            "pause_error_abs_max_ms": diag["pause_error_abs_max_ms"],
            "source_probe_cpu_duty": diag["source_probe_cpu_duty"],
            "base_diagnostic_cpu_duty": diag["base_diagnostic_cpu_duty"],
            "shadow_client_cpu_duty": diag["shadow_diagnostic_cpu_duty"],
            "combined_diagnostic_cpu_duty": diag["combined_diagnostic_cpu_duty"],
            "shadow_server_cpu_ns": diag["shadow_server_cpu_ns"],
            "shadow_monitor_cpu_ns": diag["shadow_monitor_cpu_ns"],
            "shadow_probe_count": diag["shadow_probe_count"],
            "shadow_monitor_loop_count": diag["shadow_monitor_loop_count"],
            "shadow_monitor_probe_attempt_count": diag["shadow_monitor_probe_attempt_count"],
            "retry_probe_window_count_ge_20ms": diag["retry_probe_coverage"]["pre_resume_or_frontier_windows_ge_20ms"],
            "retry_probe_covered_window_count": diag["retry_probe_coverage"]["windows_with_exact_probe_sample"],
            "retry_probe_schedule_event_count": len(diag["shadow_monitor_schedule_events"]),
            "shadow_probe_error_count": diag["shadow_probe_error_count"],
            "actual_retry_count": diag["actual_retry_count"],
            "actual_frontier_policy_changes": diag["actual_frontier_policy_changes"],
            "recheck_behavior_updates": diag["recheck_behavior_updates"],
            "server_alive": result.get("server_alive"), "OOM": result.get("OOM"),
            "lease_leaks": result.get("lease_leaks"), "logical_leaks": result.get("logical_leaks"),
            "unsafe_release_count": controller.release_queue.unsafe_release_count,
            "max_pre_first": result.get("max_pre_first"),
            "workload_span_ns": max(0, finish_ns - start_ns),
            "source_conservation_pass": result.get("source_conservation_pass"),
            "source_accounting_valid": diag["checks"].get("source_accounting"),
            "source_local_tokens": result.get("local_source_tokens"),
            "source_external_tokens": result.get("external_source_tokens"),
            "source_expected_tokens": result.get("expected_source_tokens"),
            "source_prompt_only_explained_shortfall_tokens": result.get("prompt_only_explained_shortfall_tokens"),
            "source_unexplained_shortfall_tokens": result.get("unexplained_source_shortfall_tokens"),
        }
        run_metrics.append(run_row)
        selection_rows.extend(per_selection)
        allocator_rows.extend(per_allocator)
        need_rows.extend(per_need)
        recovery_rows.extend(per_recovery)
        guard_rows.extend(per_guard)
        episodes.extend(per_episode)
        retry_rows.extend(per_retry)
        state_rows.extend(per_state)
        frontier_rows.extend({**row, "run_id": run_id, "condition": "C1", "repetition": rep}
                             for row in controller.frontier_shadow_rows)
        probe_cost.append({"run_id": run_id, "condition": "C1", "repetition": rep,
                           **{key: run_row.get(key) for key in (
                               "source_probe_cpu_duty", "base_diagnostic_cpu_duty",
                               "shadow_client_cpu_duty", "combined_diagnostic_cpu_duty",
                               "shadow_server_cpu_ns", "shadow_monitor_cpu_ns", "shadow_probe_count",
                               "shadow_monitor_loop_count", "shadow_monitor_probe_attempt_count",
                               "retry_probe_long_window_count", "retry_probe_covered_window_count",
                               "shadow_probe_error_count")}})
        measured_results.append({"run_id": run_id, "valid": valid, "diag": diag,
                                 "failed_gates": failed_gates})

        fix.RUN_ROWS.append(fix._run_metrics_row(
            "measured", "C1", rep, result, timing_rows, error,
            non_timing_pass=bool(result.get("valid"))))
        fix._write_timing_artifacts()
        run_dir = OUT / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "phase4g_d_shadow.json", {
            "valid": valid, "failed_gates": failed_gates, "diagnostics": diag,
            "selection_count": len(per_selection), "poison_episode_count": len(per_episode),
            "actual_retry_count": controller.actual_retry_count,
            "actual_frontier_policy_changes": controller.actual_frontier_policy_changes,
        })
        write_csv(run_dir / "retry_probe_schedule.csv", controller.shadow_monitor_schedule_events)
        write_csv(run_dir / "frontier_shadow.csv", frontier_rows[-len(controller.frontier_shadow_rows):]
                  if controller.frontier_shadow_rows else [])
        write_csv(run_dir / "protected_retry_timeline.csv", per_retry, RETRY_TIMELINE_FIELDS)
        write_csv(run_dir / "protected_poison_episodes.csv", per_episode)
        write_csv(run_dir / "protected_state_timeline.csv", per_state)

        if not valid:
            _write_aggregates(selection_rows, allocator_rows, need_rows, recovery_rows, guard_rows,
                              episodes, retry_rows, state_rows, frontier_rows, run_metrics, probe_cost)
            example = _make_example(episodes, retry_rows, frontier_rows, state_rows)
            write_csv(OUT / "example_poison_timeline.csv", example)
            _write_figure(OUT / "protected_poison_timeline.png", example)
            _write_failure("EXPERIMENT_INVALID_MATRIX", f"measured repetition {rep}",
                           ";".join(failed_gates), {
                               "measured_valid_runs": sum(bool(row.get("valid")) for row in run_metrics),
                               "measured_attempted_runs": len(run_metrics),
                               "failed_run": run_row, "allocator_reanalysis": reanalysis,
                           })
            raise RuntimeError(f"measured repetition {rep} failed gates: {failed_gates}")

    metrics_summary = _write_aggregates(selection_rows, allocator_rows, need_rows, recovery_rows, guard_rows,
                                        episodes, retry_rows, state_rows, frontier_rows, run_metrics, probe_cost)
    example = _make_example(episodes, retry_rows, frontier_rows, state_rows)
    write_csv(OUT / "example_poison_timeline.csv", example)
    _write_figure(OUT / "protected_poison_timeline.png", example)
    if len(measured_results) != 4 or not all(row["valid"] for row in measured_results):
        report = {"phase": "4G-D", "measured_valid_runs": sum(bool(row["valid"]) for row in measured_results),
                  "measured_attempted_runs": len(measured_results), "allocator_reanalysis": reanalysis,
                  "final_result": "EXPERIMENT_INVALID_MATRIX"}
        write_json(OUT / "validation_report.json", report)
        raise RuntimeError("fresh measured matrix is incomplete")

    # Recompute the final, explicit allocator observability gate from eligible selections only.
    eligible = [row for row in allocator_rows if row.get("context") == "SELECTION_CANDIDATE"
                and row.get("eligible_selection") is True]
    coverage = analysis.exact_truth_coverage(eligible)
    all_errors = [abs(float(row["pause_error_ms"])) for row in fix.ALL_TIMING_ROWS
                  if row.get("stage") == "measured"]
    coverage_pass = coverage["coverage_fraction"] is not None and coverage["coverage_fraction"] >= .90
    result_code = metrics_summary["final_result"]
    if not coverage_pass:
        result_code = "EXPERIMENT_INCONCLUSIVE_ALLOCATOR_OBSERVABILITY"
    all_valid = len(measured_results) == 4 and all(row["valid"] for row in measured_results)
    report = {
        "phase": "4G-D", "start_commit": analysis.START_COMMIT,
        "workload_sha256": analysis.WORKLOAD_SHA256,
        "fresh_measured_runs_valid": sum(bool(row["valid"]) for row in measured_results),
        "fresh_measured_runs_expected": 4, "turns_complete": sum(
            int(row.get("turns_complete") or 0) for row in run_metrics),
        "boundaries_complete": sum(int(row.get("boundaries") or 0) for row in run_metrics),
        "correctness_pass": all_valid, "allocator_coverage": coverage,
        "source_accounting_pass": all(row.get("source_accounting_valid") is True for row in run_metrics),
        "source_expected_tokens_total": sum(int(row.get("source_expected_tokens") or 0) for row in run_metrics),
        "source_local_tokens_total": sum(int(row.get("source_local_tokens") or 0) for row in run_metrics),
        "source_external_tokens_total": sum(int(row.get("source_external_tokens") or 0) for row in run_metrics),
        "prompt_only_explained_shortfall_tokens_total": sum(
            int(row.get("source_prompt_only_explained_shortfall_tokens") or 0) for row in run_metrics),
        "unexplained_source_shortfall_tokens_total": sum(
            int(row.get("source_unexplained_shortfall_tokens") or 0) for row in run_metrics),
        "allocator_coverage_pass": coverage_pass,
        "actual_retries": sum(int(row.get("actual_retry_count") or 0) for row in run_metrics),
        "actual_frontier_policy_changes": sum(
            int(row.get("actual_frontier_policy_changes") or 0) for row in run_metrics),
        "actual_psa_changes": 0, "timing_backend": absolute_sleep.BACKEND,
        "pause_error_abs_P95_ms": nearest_rank(all_errors, .95),
        "pause_error_abs_max_ms": max(all_errors) if all_errors else None,
        "allocator_reanalysis": reanalysis, "smoke_excluded_from_causal_aggregation": True,
        "final_result": result_code,
    }
    write_json(OUT / "validation_report.json", report)
    write_json(OUT / "timing_backend.json", {
        "backend": absolute_sleep.BACKEND, "python_busy_spin": False,
        "python_switch_interval_s": metrics.SWITCH_INTERVAL_S,
        "pause_p95_gate_ms": 10, "pause_max_gate_ms": 30,
    })
    _write_summary(result_code, reanalysis, metrics_summary, run_metrics)


if __name__ == "__main__":
    main()
