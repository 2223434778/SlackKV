"""Paired Phase 4G-C source visibility and allocator-truth RCA."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from experiments.phase4d_a import runtime
from experiments.phase4d_b.experiment import validate_runtime
from experiments.phase4g_a import run_phase4g_a as lifecycle
from experiments.phase4g_a.workload import ROOT, SOURCE, build_manifest, load_source
from experiments.phase4g_a_fix import background as bgv2
from experiments.phase4g_c.analysis import (
    GPU_BLOCKS, K2, P_SAFE_MS, RECHECK_INTERVAL_MS, RESERVATION_BUDGET_BLOCKS,
    RESUME_GUARD_MS, SOURCE_SHA, START_COMMIT, classify_attempt,
    classify_capacity_reject, classify_matrix_results, normalize_incomplete_result,
    source_coverage_diagnostic,
    run_scoped_diagnostic_counts, summarize, unique_source_trace_rows,
    validate_fresh_matrix,
)
from experiments.phase4g_c.controller import SourceRecheckController
from experiments.phase4g_b import run_phase4g_b as instrumentation

OUT = ROOT / "results/phase4g_c"
VLLM_SOURCE = Path("/media/a6000/D/ZhaoBinHe/slackkv-phase2c-fix-sourceprobe")
VLLM_COMMIT = "0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665"
VLLM_PRE_PHASE4G_C_DIFF_SHA = "65a78bb3d925e7539f8133574c465e75847c1b54fbac299f5e4bcc4f5b9446e3"
GPU_KV_BYTES, CPU_KV_BYTES, BLOCK_SIZE = 2147483648, 8589934592, 16
RUN_ORDER = [
    (1, "C0"), (1, "C1"), (2, "C1"), (2, "C0"),
    (3, "C0"), (3, "C1"), (4, "C1"), (4, "C0"),
]
TMP_LOGS = Path("/tmp/slackkv-phase4g-c-server-logs")
ORIGINAL_START_SERVER = runtime.start_server
CURRENT_CONTROLLER: SourceRecheckController | None = None
PAUSE_WAIT_SPIN_WINDOW_S = 0.050
PAUSE_THREAD_SWITCH_INTERVAL_S = 0.0005


@contextmanager
def precise_client_thread_switching():
    """Reduce client-thread handoff jitter while deadlines are being issued."""
    previous = sys.getswitchinterval()
    sys.setswitchinterval(PAUSE_THREAD_SWITCH_INTERVAL_S)
    try:
        yield
    finally:
        sys.setswitchinterval(previous)


def phase4g_c_wait_until(
    target_s: float,
    monotonic=time.monotonic,
    sleeper=time.sleep,
) -> None:
    """Wait against an absolute monotonic deadline with one short spin tail."""
    while True:
        remaining = target_s - monotonic()
        if remaining <= 0:
            return
        if remaining > PAUSE_WAIT_SPIN_WINDOW_S:
            sleeper(remaining - PAUSE_WAIT_SPIN_WINDOW_S)
            continue
        while monotonic() < target_s:
            pass
        return


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _record_run_failure(label: str, payload: dict[str, Any], condition: str, repetition: int) -> Path:
    failure_dir = OUT / "failures"
    failure_dir.mkdir(parents=True, exist_ok=True)
    failure_path = failure_dir / f"{label}.json"
    suffix = 1
    while failure_path.exists():
        failure_path = failure_dir / f"{label}_attempt{suffix}.json"
        suffix += 1
    log_index = repetition * 10 + (condition == "C1")
    log_path = TMP_LOGS / "server_logs" / f"measured_rep{log_index}_d.log"
    if log_path.exists():
        diagnostic = [
            line[:1600] for line in log_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
            if any(marker in line for marker in (
                "ERROR", "OOM", "Traceback", "AttributeError", "RuntimeError",
                "EngineDeadError",
            ))
        ]
        payload = {
            **payload,
            "diagnostic_log": str(log_path),
            "server_diagnostic_lines": diagnostic[-80:],
        }
    write_json(failure_path, payload)
    return failure_path


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def vllm_diff_sha() -> str:
    tracked = subprocess.check_output(["git", "diff", "--binary"], cwd=VLLM_SOURCE)
    route = VLLM_SOURCE / "vllm/entrypoints/serve/dev/slackkv_prefetch/api_router.py"
    return sha256(tracked + b"\x00" + route.read_bytes())


def frozen_checks() -> dict[str, Any]:
    if sha256(SOURCE.read_bytes()) != SOURCE_SHA:
        raise RuntimeError("EXPERIMENT_INVALID_WORKLOAD: frozen Phase4C SHA mismatch")
    turns = load_source()
    manifest = build_manifest(turns)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    upstream = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=VLLM_SOURCE, text=True).strip()
    version = subprocess.check_output(
        [str(runtime.ENV_ROOT / "bin" / "python"), "-c", "import vllm; print(vllm.__version__)"],
        text=True, stderr=subprocess.STDOUT,
    ).strip().splitlines()[-1]
    route = VLLM_SOURCE / "vllm/entrypoints/serve/dev/slackkv_prefetch/api_router.py"
    if not route.is_file():
        raise RuntimeError("EXPERIMENT_INVALID_ENVIRONMENT: research prefetch endpoint missing")
    offload_config_source = VLLM_SOURCE / "vllm/v1/kv_offload/base.py"
    offload_config_text = offload_config_source.read_text(encoding="utf-8")
    if 'extra_config.get("offload_prompt_only", True)' not in offload_config_text:
        raise RuntimeError("EXPERIMENT_INVALID_ENVIRONMENT: prompt-only offload default changed")
    if head != START_COMMIT or upstream != VLLM_COMMIT or version != "0.23.0":
        raise RuntimeError(f"EXPERIMENT_INVALID_ENVIRONMENT: SlackKV={head} vLLM={upstream}/{version}")
    if len(turns) != 40 or len(manifest) != 32:
        raise RuntimeError("EXPERIMENT_INVALID_WORKLOAD: turn/boundary shape")
    model_config = json.loads((runtime.MODEL / "config.json").read_text(encoding="utf-8"))
    if int(model_config.get("max_position_embeddings", 0)) < 32768:
        raise RuntimeError("EXPERIMENT_INVALID_ENVIRONMENT: model context")
    return {
        "start_commit": head, "workload_sha256": SOURCE_SHA, "vllm_upstream_commit": upstream,
        "vllm_version": version, "vllm_diff_sha256": vllm_diff_sha(),
        "vllm_pre_phase4g_c_diff_sha256": VLLM_PRE_PHASE4G_C_DIFF_SHA,
        "vllm_research_endpoint_sha256": sha256(route.read_bytes()),
        "offload_prompt_only": True,
        "offload_prompt_only_configured": False,
        "offload_prompt_only_default_source": str(offload_config_source),
        "model_snapshot_path": str(runtime.MODEL), "turns": len(turns),
        "boundaries": len(manifest), "episodes": 8,
    }


def marker_events(path: Path, marker: str) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if marker not in line:
            continue
        try:
            rows.append(json.loads(line.split(marker, 1)[1]))
        except (ValueError, IndexError):
            rows.append({"event": "PARSE_ERROR", "line_sha256": sha256(line.encode())})
    return rows


def _start_server(label: str, connector: bool = True, gpu_bytes: int = GPU_KV_BYTES, out: Path | None = None):
    TMP_LOGS.mkdir(parents=True, exist_ok=True)
    return ORIGINAL_START_SERVER(label, connector=connector, gpu_bytes=gpu_bytes, out=TMP_LOGS)


def run_one(condition: str, repetition: int, manifest: list[dict[str, Any]], stage: str) -> dict[str, Any]:
    global CURRENT_CONTROLLER
    CURRENT_CONTROLLER = None
    label = f"{stage}_rep{repetition}_{condition.lower()}"
    original_start, original_send = runtime.start_server, instrumentation._install_trace_params(manifest)
    trace_send = runtime.send
    original_bg, original_controller = lifecycle.start_background, lifecycle.DynamicController
    original_wait_until = lifecycle.wait_until
    original_out, original_x_tokens = lifecycle.OUT, lifecycle.X_OUTPUT_TOKENS

    class _Controller(SourceRecheckController):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, condition=condition, repetition=repetition, **kwargs)
            global CURRENT_CONTROLLER
            CURRENT_CONTROLLER = self

    def _allocator_traced_send(prompt, max_tokens, *, stream=False, params=None, on_first=None):
        params = dict(params or {})
        if params.get("slackkv_trace_resume") is True:
            params["slackkv_allocator_trace"] = True
            params["slackkv_allocator_trace_once"] = True
        return trace_send(prompt, max_tokens, stream=stream, params=params, on_first=on_first)

    runtime.send = _allocator_traced_send
    runtime.start_server = _start_server
    lifecycle.OUT = OUT / "runs" / label
    lifecycle.DynamicController = _Controller
    lifecycle.start_background = instrumentation._bgv2_background
    lifecycle.X_OUTPUT_TOKENS = bgv2.BG_TOTAL_OUTPUT_TOKENS
    lifecycle.wait_until = phase4g_c_wait_until
    try:
        with precise_client_thread_switching():
            result = lifecycle.run_lifecycle(
                "D", stage="measured", repetition=repetition * 10 + (condition == "C1"),
            )
        controller = CURRENT_CONTROLLER
        if controller is None:
            raise RuntimeError("controller was not constructed")
        log = TMP_LOGS / "server_logs" / f"measured_rep{repetition * 10 + (condition == 'C1')}_d.log"
        probe = marker_events(log, "SLACKKV_READY_PROBE ")
        capacity = marker_events(log, "SLACKKV_CAPACITY_TRACE ")
        snapshots = marker_events(log, "SLACKKV_SELECTION_SNAPSHOT ")
        allocator = marker_events(log, "SLACKKV_ALLOCATOR_TRUTH ")
        source_rows = [row for row in probe if row.get("event") == "SAMPLE"]
        costs = [row for row in probe if row.get("event") == "PROBE_COST"]
        span_ns = (
            max(int(row["timestamp_monotonic_ns"]) for row in costs)
            - min(int(row["timestamp_monotonic_ns"]) for row in costs) + 20_000_000
            if costs else 0
        )
        cpu_ns = sum(int(row.get("probe_duration_ns", 0)) for row in costs)
        allocator_cpu_ns = sum(
            int(row.get("capacity_trace_cpu_ns", 0))
            for row in [*capacity, *snapshots]
        ) + sum(int(row.get("instrumentation_cpu_ns", 0)) for row in allocator)
        run_dir = OUT / "runs" / label
        run_dir.mkdir(parents=True, exist_ok=True)
        result.update({
            "run_id": label, "condition": condition, "repetition": repetition,
            "client_thread_switch_interval_s": PAUSE_THREAD_SWITCH_INTERVAL_S,
            "controller_selection": controller.selection_rows,
            "controller_visible_ready_ids": sorted(controller.offload_events),
            "source_recheck_events": controller.source_rechecks,
            "recheck_behavior_updates": controller.recheck_behavior_updates,
            "ready_probe_events": probe, "capacity_trace": capacity,
            "selection_snapshots": snapshots, "allocator_truth": allocator,
            "source_probe_cpu_duty": cpu_ns / span_ns if span_ns else None,
            "allocator_truth_cpu_duty": allocator_cpu_ns / span_ns if span_ns else None,
            "diagnostic_cpu_duty": (cpu_ns + allocator_cpu_ns) / span_ns if span_ns else None,
            "selection_count": len(controller.selection_rows),
            "stale_protected_visibility_count": sum(
                not row["protected_controller_visible_ready"] and row["protected_exact_source_ready"]
                for row in controller.selection_rows
            ),
            "candidate_matches_protected_count": sum(
                bool(row["candidate_matches_protected"]) for row in controller.selection_rows
            ),
            "median_psa_reserve": statistics.median(
                [int(row["PSA1_reserve_blocks"]) for row in controller.selection_rows]
            ) if controller.selection_rows else None,
            "max_psa_reserve": max(
                (int(row["PSA1_reserve_blocks"]) for row in controller.selection_rows), default=0
            ),
            "successful_promotions": sum(
                (item.get("response") or {}).get("status") == "ok"
                for item in controller.promotions.values()
            ),
            "turns_complete": result.get("turns_complete"),
            "boundaries": result.get("boundaries"),
            "recheck_behavior_updates_expected": 0 if condition == "C0" else None,
            "background_tail_finish_ms": (
                (int(result["X_finish_ns"]) - max(int(row["finish_ns"]) for row in result["turn_rows"])) / 1e6
                if result.get("X_finish_ns") != "NA" and result.get("turn_rows") else None
            ),
        })
        raw_source_rows = result.get("source_trace", [])
        source_rows, duplicate_source_trace_events = unique_source_trace_rows(raw_source_rows)
        local_source_tokens = sum(int(row.get("local_cached_tokens", 0)) for row in source_rows)
        external_source_tokens = sum(int(row.get("external_cached_tokens", 0)) for row in source_rows)
        expected_source_tokens = sum(int(row["required_prefix_tokens"]) for row in manifest)
        turn_by_episode = {
            (str(row["episode_id"]), int(row["turn_index"])): row
            for row in result.get("turn_rows", [])
        }
        source_by_session = {
            str(row["session"]): row for row in source_rows
        }
        source_coverage = []
        for boundary in manifest:
            episode = str(boundary["episode_id"])
            boundary_index = int(boundary["boundary_index"])
            previous_turn = turn_by_episode.get((episode, boundary_index), {})
            source = source_by_session.get(f"{episode}:turn:{boundary_index + 1}", {})
            source_coverage.append({
                "boundary_id": str(boundary["boundary_id"]),
                **source_coverage_diagnostic(
                    full_prefix_tokens=int(boundary["required_prefix_tokens"]),
                    previous_prompt_tokens=int(previous_turn.get("prompt_tokens", 0)),
                    local_source_tokens=int(source.get("local_cached_tokens", 0)),
                    external_source_tokens=int(source.get("external_cached_tokens", 0)),
                    block_size=BLOCK_SIZE,
                ),
            })
        prompt_only_unavailable = sum(
            int(row["prompt_only_decode_tail_tokens"]) for row in source_coverage
        )
        explained_shortfall = sum(
            int(row["prompt_only_explained_shortfall_tokens"]) for row in source_coverage
        )
        unexplained_shortfall = sum(
            int(row["unexplained_source_shortfall_tokens"]) for row in source_coverage
        )
        result.update({
            "local_source_tokens": local_source_tokens,
            "external_source_tokens": external_source_tokens,
            "source_trace_raw_event_count": len(raw_source_rows),
            "source_trace_duplicate_event_count": duplicate_source_trace_events,
            "source_trace_unique_resume_count": len(source_rows),
            "expected_source_tokens": expected_source_tokens,
            "source_conservation_pass": local_source_tokens + external_source_tokens == expected_source_tokens,
            "source_coverage": source_coverage,
            "prompt_only_decode_tail_tokens": prompt_only_unavailable,
            "prompt_only_explained_shortfall_tokens": explained_shortfall,
            "unexplained_source_shortfall_tokens": unexplained_shortfall,
        })
        if len(result.get("turn_rows", [])) != 40:
            result["valid"] = False
            result["invalid_reason"] = "turn_count_mismatch"
        if result["source_probe_cpu_duty"] is None or result["diagnostic_cpu_duty"] is None:
            result["valid"] = False
            result["invalid_reason"] = "missing_probe_cost_window"
        elif result["source_probe_cpu_duty"] >= .05 or result["diagnostic_cpu_duty"] >= .07:
            result["valid"] = False
            result["invalid_reason"] = "EXPERIMENT_INVALID_INSTRUMENTATION_OVERHEAD"
        if condition == "C0" and controller.recheck_behavior_updates != 0:
            result["valid"] = False
            result["invalid_reason"] = "C0_consumed_source_recheck"
        if result.get("background_tail_finish_ms", -1) <= 0:
            result["valid"] = False
            result["invalid_reason"] = "BGv2_did_not_cover_final_request_finish"
        write_json(run_dir / "run.json", _compact_run_payload(result))
        compact = [
            line[:1600] for line in log.read_text(encoding="utf-8", errors="replace").splitlines()
            if any(marker in line for marker in (
                "SLACKKV_", "ERROR", "OOM", "GPU KV cache size",
            ))
        ] if log.exists() else []
        output_logs = OUT / "server_logs"
        output_logs.mkdir(parents=True, exist_ok=True)
        (output_logs / f"{label}_compact.log").write_text(
            "\n".join(compact[-120:]) + ("\n" if compact else ""), encoding="utf-8"
        )
        write_json(run_dir / "diagnostics.json", {
            "run_id": label, "valid": result.get("valid"),
            "error": result.get("invalid_reason"),
            "last_server_lines": compact[-120:],
        })
        if not result.get("valid"):
            raise RuntimeError(f"RUN_INVALID:{label}:{result.get('invalid_reason')}")
        return result
    except BaseException as exc:
        _record_run_failure(label, {
            "run_id": label, "condition": condition, "repetition": repetition,
            "error": repr(exc),
        }, condition, repetition)
        raise
    finally:
        runtime.start_server = original_start
        runtime.send = original_send
        lifecycle.start_background = original_bg
        lifecycle.DynamicController = original_controller
        lifecycle.wait_until = original_wait_until
        lifecycle.OUT = original_out
        lifecycle.X_OUTPUT_TOKENS = original_x_tokens


def run_order_check() -> None:
    expected = [
        (1, "C0"), (1, "C1"), (2, "C1"), (2, "C0"),
        (3, "C0"), (3, "C1"), (4, "C1"), (4, "C0"),
    ]
    if RUN_ORDER != expected:
        raise RuntimeError("Phase4G-C paired run order changed")


def _event_boundary(event: dict[str, Any]) -> str:
    value = str(event.get("boundary_id") or event.get("candidate_id") or "")
    if value.startswith("E") and ":b" in value:
        return value
    session = str(event.get("session") or event.get("lease_key") or "")
    episode, sep, turn = session.partition(":turn:")
    return f"{episode}:b{int(turn)-1}" if sep and turn.isdigit() and int(turn) > 0 else value


def _selection_export(row: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    return {
        "condition": run["condition"], "repetition": run["repetition"],
        **{key: value for key, value in row.items() if key != "selection_snapshot"},
    }


def _allocator_export(event: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    return {"condition": run["condition"], "repetition": run["repetition"],
            "run_id": run["run_id"], **event}


def _compact_run_payload(result: dict[str, Any]) -> dict[str, Any]:
    compact = dict(result)
    compact["controller_selection"] = [
        {key: value for key, value in row.items() if key != "selection_snapshot"}
        for row in result.get("controller_selection", [])
    ]
    omitted = {
        "candidate_source_exact", "protected_source_exact",
        "candidate_allocator_preflight", "protected_self_allocator_preflight",
        "protected_local_ownership",
    }
    compact["capacity_trace"] = [
        {key: value for key, value in row.items() if key not in omitted}
        for row in result.get("capacity_trace", [])
    ]
    compact["selection_snapshots"] = [
        {key: value for key, value in row.items() if key not in omitted}
        for row in result.get("selection_snapshots", [])
    ]
    return compact


def aggregate(results: list[dict[str, Any]], manifest: list[dict[str, Any]]) -> dict[str, Any]:
    selection_rows: list[dict[str, Any]] = []
    recheck_rows: list[dict[str, Any]] = []
    allocator_rows: list[dict[str, Any]] = []
    guard_rows: list[dict[str, Any]] = []
    need_rows: list[dict[str, Any]] = []
    counterfactual_rows: list[dict[str, Any]] = []
    race_rows: list[dict[str, Any]] = []
    funnel_rows: list[dict[str, Any]] = []
    promotion_rows: list[dict[str, Any]] = []
    run_metrics: list[dict[str, Any]] = []
    source_coverage_rows: list[dict[str, Any]] = []
    probe_cost_rows: list[dict[str, Any]] = []
    for run in results:
        condition = str(run["condition"])
        meta = {"condition": condition, "repetition": run["repetition"], "run_id": run["run_id"]}
        selections = list(run.get("controller_selection", []))
        probe = list(run.get("ready_probe_events", []))
        source_rows = [row for row in probe if row.get("event") == "SAMPLE"]
        costs = [row for row in probe if row.get("event") == "PROBE_COST"]
        recheck_rows.extend({**meta, **row} for row in run.get("source_recheck_events", []))
        selection_rows.extend(_selection_export(row, run) for row in selections)
        exact_events = list(run.get("allocator_truth", []))
        allocator_rows.extend(_allocator_export(row, run) for row in exact_events)
        capacity_events = [row for row in run.get("capacity_trace", [])
                           if row.get("event") == "CAPACITY_GUARD"]
        allocator_by_request: dict[str, list[dict[str, Any]]] = {}
        for row in exact_events:
            allocator_by_request.setdefault(str(row.get("request_id", "")), []).append(row)
        by_candidate = {str(row.get("candidate_id")): row for row in selections}
        promotion_map = run.get("promotions", {})
        progress_rejects = 0
        capacity_rejects = 0
        promotion_ok = 0
        useful = 0
        remote_wait = 0
        ready_miss = 0
        attempts = []
        for candidate_id, promotion in promotion_map.items():
            response = promotion.get("response") or {}
            status = response.get("status")
            if status == "progress_rejected":
                progress_rejects += 1
            elif status == "capacity_rejected":
                capacity_rejects += 1
            elif status == "ok":
                promotion_ok += 1
            attempts.append((str(candidate_id), promotion, response))
        resume_arrival_by_boundary: dict[str, float] = {}
        remote_wait_boundaries: set[str] = set()
        for event in run.get("resume_trace", []):
            boundary = _event_boundary(event)
            ts = event.get("timestamp_monotonic", event.get("timestamp"))
            if ts is None:
                continue
            if event.get("event") == "RESUME_ARRIVED":
                resume_arrival_by_boundary[boundary] = float(ts)
            elif event.get("event") == "WAITING_FOR_REMOTE_KVS":
                remote_wait_boundaries.add(boundary)
        for candidate_id, promotion, response in attempts:
            stamps = response.get("phase3m_timestamps") or {}
            t6 = stamps.get("T6")
            arrival = resume_arrival_by_boundary.get(candidate_id)
            classification = "NOT_READY"
            if response.get("status") == "ok" and t6 is not None and arrival is not None:
                if float(t6) < arrival and candidate_id not in remote_wait_boundaries:
                    classification = "PROMOTION_USEFUL"
                    useful += 1
                elif float(t6) < arrival:
                    classification = "PROMOTION_REMOTE_WAIT"
                    remote_wait += 1
                else:
                    classification = "PROMOTION_READY_MISS"
                    ready_miss += 1
            attempts_row = {
                **meta, "boundary_id": candidate_id, "status": response.get("status"),
                "selection_id": promotion.get("selection_id"),
                "client_issue_ns": response.get("client_issue_ns"),
                "client_return_ns": response.get("client_return_ns"),
                "T0": stamps.get("T0"), "T1": stamps.get("T1"),
                "T6": t6, "T7": stamps.get("T7"),
                "issue_to_T6_ms": (
                    (float(t6) - float(stamps.get("T0", t6))) * 1000 if t6 is not None else None
                ),
                "transfer_bytes": response.get("transfer_bytes", 0),
                "external_tokens_loaded": response.get("external_tokens_loaded", 0),
                "promotion_usefulness": classification,
            }
            promotion_rows.append(attempts_row)
            selection = by_candidate.get(candidate_id, {})
            if response.get("status") == "progress_rejected" and condition == "C0":
                snap = selection.get("selection_snapshot", {})
                shadow = snap.get("protected_self_allocator_preflight", {})
                protected_exact = snap.get("protected_source_exact", {})
                free_at_selection = int(snap.get("free_blocks_before", -1))
                self_h = int(selection.get("protected_self_H", 0))
                protected_blocks = int(selection.get("protected_required_blocks", 0))
                self_psa = free_at_selection - protected_blocks >= self_h
                self_logical = bool(selection.get("self_logical_fit", False))
                self_legacy = free_at_selection >= protected_blocks
                self_alloc = bool(shadow.get("allocator_fit", False))
                timing = bool(selection.get("protected_self_timing_permits", False))
                would_recover = bool(
                    selection.get("protected_exact_source_ready") and self_psa and self_logical
                    and self_legacy and self_alloc and timing
                )
                counterfactual_rows.append({
                    **meta, "actual_candidate": candidate_id, "protected": selection.get("protected_id"),
                    "actual_H": selection.get("PSA1_reserve_blocks"),
                    "self_H": selection.get("protected_self_H"),
                    "protected_exact_ready": selection.get("protected_exact_source_ready"),
                    "self_logical_fit": self_logical, "self_PSA_fit": self_psa,
                    "self_legacy_physical_fit": self_legacy,
                    "self_exact_allocator_fit": shadow.get("allocator_fit"),
                    "timing_permits": timing, "would_recover": would_recover,
                })
        for event in capacity_events:
            candidate_id = str(event.get("candidate_id") or _event_boundary(event))
            selection = by_candidate.get(candidate_id, {})
            exact = event.get("candidate_allocator_preflight") or {}
            observed_step = next(
                (row for row in allocator_by_request.get(str(event.get("request_id", "")), [])
                 if row.get("gate") == "STEP"),
                None,
            )
            comparison = classify_attempt(
                legacy_required_blocks=int(event.get("candidate_required_blocks", event.get("required_blocks", 0))),
                pool_free_blocks=int(event.get("free_blocks_before", 0)),
                allocator_required_blocks=int(exact.get("allocator_required_blocks", 0)),
                allocator_available_blocks=int(exact.get("allocator_available_blocks", 0)),
            )
            guard_rows.append({
                **meta, "boundary_id": candidate_id,
                "event": event.get("event"), "decision": event.get("decision"),
                "legacy_required_blocks": event.get("candidate_required_blocks", event.get("required_blocks")),
                "pool_free_blocks": event.get("free_blocks_before"),
                "legacy_guard_decision": comparison["legacy_guard_decision"],
                "allocator_num_blocks_to_allocate": exact.get("allocator_required_blocks"),
                "allocator_required_blocks": exact.get("allocator_required_blocks"),
                "allocator_available_blocks": exact.get("allocator_available_blocks"),
                "allocator_reserved_blocks": exact.get("inflight_reserved_blocks"),
                "allocator_fit": exact.get("allocator_fit"),
                "allocator_watermark_blocks": exact.get("watermark_blocks", 0),
                "full_sequence_required_blocks": exact.get("full_sequence_required_blocks"),
                "full_sequence_fit": exact.get("full_sequence_fit"),
                "actual_allocate_slots_required_blocks": (
                    observed_step.get("allocator_required_blocks") if observed_step else None
                ),
                "actual_allocate_slots_available_blocks": (
                    observed_step.get("allocator_available_blocks") if observed_step else None
                ),
                "actual_allocate_slots_fit": observed_step.get("allocator_fit") if observed_step else None,
                "shadow_vs_actual_fit_agreement": (
                    bool(exact.get("allocator_fit")) == bool(observed_step.get("allocator_fit"))
                    if observed_step else None
                ),
                "agreement": comparison["agreement"],
                "false_admit": comparison["false_admit"],
                "false_reject": comparison["false_reject"],
                "selection_exact_allocator_fit": selection.get("selection_exact_allocator_fit"),
                "selection_to_scheduler_ms": event.get("selection_to_scheduler_ms"),
                "selection_snapshot_to_attempt_ms": event.get(
                    "selection_snapshot_to_attempt_ms"
                ),
                "free_delta_blocks": (
                    int(event.get("free_blocks_before", 0))
                    - int(selection.get("selection_free_blocks", 0))
                ) if selection else None,
                "reserved_delta_blocks": (
                    int(exact.get("inflight_reserved_blocks", 0))
                    - int((selection.get("selection_snapshot") or {}).get(
                        "candidate_allocator_preflight", {}
                    ).get("inflight_reserved_blocks", 0))
                ) if selection else None,
            })
            race = bool(selection and selection.get("selection_exact_allocator_fit") is True
                        and exact.get("allocator_fit") is False)
            if event.get("decision") == "CAPACITY_REJECT":
                race_classification = classify_capacity_reject(
                    selection.get("selection_exact_allocator_fit"), bool(exact.get("allocator_fit"))
                )
                race_rows.append({
                    **meta, "boundary_id": candidate_id,
                    "classification": race_classification,
                    "selection_exact_fit": selection.get("selection_exact_allocator_fit"),
                    "attempt_exact_fit": exact.get("allocator_fit"),
                    "selection_to_scheduler_ms": event.get("selection_snapshot_to_attempt_ms"),
                    "free_delta_blocks": guard_rows[-1].get("free_delta_blocks"),
                    "reserved_delta_blocks": guard_rows[-1].get("reserved_delta_blocks"),
                })
        for row in selections:
            snapshot = row.get("selection_snapshot", {})
            protected = snapshot.get("protected_source_exact", {})
            ownership = snapshot.get("protected_local_ownership", {})
            shadow = snapshot.get("protected_self_allocator_preflight", {})
            protected_key = next(
                (str(item["lease_key"]) for item in manifest
                 if str(item["boundary_id"]) == str(row["protected_id"])),
                "",
            )
            actual_resume_preflight = next(
                (event for event in exact_events
                 if event.get("session") == protected_key and event.get("gate") == "STEP"),
                {},
            )
            need_rows.append({
                **meta, "selection_id": row["selection_id"],
                "protected_id": row["protected_id"],
                "progress_blocks": row["protected_progress_blocks"],
                "required_blocks": row.get("protected_required_blocks"),
                "local_cached_blocks": protected.get("local_cached_blocks"),
                "external_ready_blocks": protected.get("external_ready_needed_blocks"),
                "leased_blocks": ownership.get("current_protected_lease_blocks"),
                "owned_blocks": ownership.get("owned_by_running_or_waiting"),
                "owned_by_running_blocks": ownership.get("owned_by_running_request"),
                "owned_by_waiting_blocks": ownership.get("owned_by_waiting_request"),
                "evictable_cached_blocks": ownership.get("cached_but_free_evictable"),
                "other_refcounted_blocks": ownership.get("other_refcounted"),
                "PSA1_need": row.get("PSA1_reserve_blocks"),
                "allocator_need_now": shadow.get("allocator_required_blocks"),
                "allocator_available_now": shadow.get("allocator_available_blocks"),
                "actual_resume_allocator_need": actual_resume_preflight.get("allocator_required_blocks"),
                "actual_resume_allocator_available": actual_resume_preflight.get("allocator_available_blocks"),
                "actual_resume_allocator_fit": actual_resume_preflight.get("allocator_fit"),
                "need_gap_blocks": (
                    int(row.get("PSA1_reserve_blocks", 0)) -
                    int(shadow.get("allocator_required_blocks", 0))
                ) if shadow.get("allocator_required_blocks") is not None else None,
                "need_ratio": (
                    float(row.get("PSA1_reserve_blocks", 0)) /
                    max(1, int(shadow.get("allocator_required_blocks", 0)))
                ) if shadow.get("allocator_required_blocks") is not None else None,
            })
        for job in manifest:
            bid = str(job["boundary_id"])
            source_events = [row for row in run.get("source_events", []) if row.get("boundary_id") == bid]
            probes = [row for row in source_rows if row.get("boundary_id") == bid]
            initial = next((row for row in source_events if row.get("event") in {
                "SOURCE_READY", "SOURCE_NOT_READY"
            }), {})
            attempts_for_job = [item for item in attempts if item[0] == bid]
            selection_for_job = [row for row in selections if row.get("candidate_id") == bid]
            cap_for_job = [row for row in capacity_events
                           if str(row.get("candidate_id") or _event_boundary(row)) == bid]
            response_status = attempts_for_job[0][2].get("status") if attempts_for_job else None
            cap_decision = cap_for_job[-1].get("decision") if cap_for_job else None
            funnel_rows.append({
                **meta, "boundary_id": bid,
                "source_initial": initial.get("event"),
                "source_eventually_full_ready": any(row.get("source_fully_ready") for row in probes),
                "source_controller_visible": any(row.get("event") == "SOURCE_READY" for row in source_events)
                    or bid in set(run.get("controller_visible_ready_ids", [])),
                "selected_candidate": bool(selection_for_job),
                "candidate_is_protected": any(row.get("candidate_matches_protected") for row in selection_for_job),
                "logical_reservation_pass": any(row.get("event") == "PROMOTION_ATTEMPT"
                                                 for row in run.get("decisions", []) if row.get("boundary_id") == bid),
                "PSA_pass": response_status != "progress_rejected" if response_status else None,
                "legacy_physical_guard_pass": cap_decision in {"ADMIT", "PROGRESS_REJECT"},
                "exact_allocator_pass": (
                    (cap_for_job[-1].get("candidate_allocator_preflight") or {}).get("allocator_fit")
                    if cap_for_job else None
                ),
                "promotion_result": response_status,
                "promotion_usefulness": next((row["promotion_usefulness"] for row in promotion_rows
                                              if row.get("condition") == condition
                                              and row.get("boundary_id") == bid), None),
                "resume_arrived": bid in resume_arrival_by_boundary,
            })
        duty_probe = run.get("source_probe_cpu_duty")
        duty_total = run.get("diagnostic_cpu_duty")
        scoped_counts = run_scoped_diagnostic_counts(
            run["run_id"], condition, guard_rows, counterfactual_rows,
        )
        run_metrics.append({
            "condition": condition, "repetition": run["repetition"], "run_id": run["run_id"],
            "valid": run["valid"], "turns_complete": run["turns_complete"],
            "boundaries": run["boundaries"], "selection_count": len(selections),
            "stale_protected_visibility": sum(
                not row["protected_controller_visible_ready"] and row["protected_exact_source_ready"]
                for row in selections
            ),
            "candidate_matches_protected": sum(bool(row["candidate_matches_protected"]) for row in selections),
            "candidate_matches_protected_rate": (
                sum(bool(row["candidate_matches_protected"]) for row in selections) / len(selections)
                if selections else None
            ),
            "median_psa_reserve": statistics.median(
                [int(row["PSA1_reserve_blocks"]) for row in selections]
            ) if selections else None,
            "p95_psa_reserve": summarize([int(row["PSA1_reserve_blocks"]) for row in selections])["p95"],
            "max_psa_reserve": max((int(row["PSA1_reserve_blocks"]) for row in selections), default=0),
            "progress_rejects": progress_rejects, "capacity_rejects": capacity_rejects,
            "exact_allocator_rejects": scoped_counts["exact_allocator_rejects"],
            "successful_promotions": promotion_ok, "useful_promotions": useful,
            "promotion_remote_waits": remote_wait, "promotion_ready_misses": ready_miss,
            "missed_protected_self_promotion_count": scoped_counts[
                "missed_protected_self_promotion_count"
            ],
            "recheck_behavior_updates": run["recheck_behavior_updates"],
            "source_probe_cpu_duty": duty_probe, "diagnostic_cpu_duty": duty_total,
            "max_pre_first": run["max_pre_first"], "lease_leaks": run["lease_leaks"],
            "logical_leaks": run["logical_leaks"], "OOM": run["OOM"],
            "server_alive": run["server_alive"],
            "bg_tail_finish_ms": run.get("background_tail_finish_ms"),
            "target_source_tokens": int(run.get("expected_source_tokens", 0)),
            "source_trace_raw_event_count": int(run.get("source_trace_raw_event_count", 0)),
            "source_trace_unique_resume_count": int(run.get("source_trace_unique_resume_count", 0)),
            "source_trace_duplicate_event_count": int(run.get("source_trace_duplicate_event_count", 0)),
            "actual_source_hit_tokens": int(run.get("local_source_tokens", 0))
                + int(run.get("external_source_tokens", 0)),
            "full_prefix_source_shortfall_tokens": int(run.get("expected_source_tokens", 0))
                - int(run.get("local_source_tokens", 0)) - int(run.get("external_source_tokens", 0)),
            "prompt_only_decode_tail_tokens": int(run.get("prompt_only_decode_tail_tokens", 0)),
            "prompt_only_explained_shortfall_tokens": int(
                run.get("prompt_only_explained_shortfall_tokens", 0)
            ),
            "unexplained_source_shortfall_tokens": int(
                run.get("unexplained_source_shortfall_tokens", 0)
            ),
            "full_prefix_source_hit_exact": bool(run.get("source_conservation_pass", False)),
        })
        source_coverage_rows.extend({**meta, **row} for row in run.get("source_coverage", []))
        probe_cost_rows.extend({
            **meta, **row,
        } for row in costs)
    return {
        "controller_selection": selection_rows,
        "source_recheck_events": recheck_rows,
        "allocator_truth": allocator_rows,
        "allocator_guard_comparison": guard_rows,
        "promotion_metrics": promotion_rows,
        "protected_need": need_rows,
        "progress_reject_counterfactual": counterfactual_rows,
        "promotion_funnel": funnel_rows,
        "run_metrics": run_metrics,
        "source_coverage": source_coverage_rows,
        "selection_race": race_rows,
        "probe_cost": probe_cost_rows,
    }


def _condition_summary(metrics: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    rows = [row for row in metrics if row["condition"] == condition]
    total_selections = sum(int(row["selection_count"]) for row in rows)
    psa_values = [
        int(item["PSA1_reserve_blocks"]) for run in RUN_RESULTS
        if run.get("condition") == condition for item in run.get("controller_selection", [])
    ]
    pooled_median = statistics.median(psa_values) if psa_values else None
    pooled_p95 = summarize([float(value) for value in psa_values])["p95"] if psa_values else None
    capacity_attempts = sum(
        row.get("condition") == condition and row.get("event") == "CAPACITY_GUARD"
        for row in GUARD_COMPARISON_ROWS
    )
    allocator_rejects = sum(
        row.get("condition") == condition and row.get("event") == "CAPACITY_GUARD"
        and row.get("allocator_fit") is False
        for row in GUARD_COMPARISON_ROWS
    )
    return {
        "condition": condition, "run_count": len(rows),
        "selection_count": total_selections,
        "stale_protected_visibility_count": sum(int(row["stale_protected_visibility"]) for row in rows),
        "stale_protected_visibility_fraction": (
            sum(int(row["stale_protected_visibility"]) for row in rows) / total_selections
            if total_selections else 0.0
        ),
        "candidate_matches_protected_count": sum(int(row["candidate_matches_protected"]) for row in rows),
        "candidate_matches_protected_rate": (
            sum(int(row["candidate_matches_protected"]) for row in rows) / total_selections
            if total_selections else 0.0
        ),
        "median_psa_reserve": pooled_median,
        "p95_psa_reserve": pooled_p95,
        "max_psa_reserve": max(psa_values, default=0),
        "progress_rejects": sum(int(row["progress_rejects"]) for row in rows),
        "capacity_rejects": sum(int(row["capacity_rejects"]) for row in rows),
        "exact_allocator_rejects": sum(int(row["exact_allocator_rejects"]) for row in rows),
        "successful_promotions": sum(int(row["successful_promotions"]) for row in rows),
        "useful_promotions": sum(int(row["useful_promotions"]) for row in rows),
        "source_probe_cpu_duty_max": max(
            (float(row["source_probe_cpu_duty"] or 0) for row in rows), default=0.0
        ),
        "diagnostic_cpu_duty_max": max(
            (float(row["diagnostic_cpu_duty"] or 0) for row in rows), default=0.0
        ),
        "missed_protected_self_promotion_count": sum(
            int(row["missed_protected_self_promotion_count"]) for row in rows
        ),
        "source_ready_selection_opportunities": sum(
            sum(bool(item["protected_exact_source_ready"]) for item in run.get("controller_selection", []))
            for run in RUN_RESULTS if run.get("condition") == condition
        ),
        "psa_blocked_source_ready_fraction": (
            sum(int(row["progress_rejects"]) for row in rows)
            / max(1, sum(
                sum(bool(item["protected_exact_source_ready"]) for item in run.get("controller_selection", []))
                for run in RUN_RESULTS if run.get("condition") == condition
            ))
        ),
        "median_allocator_need_now": statistics.median(
            [float(row["allocator_need_now"]) for row in PROTECTED_NEED_ROWS
             if row.get("condition") == condition and row.get("allocator_need_now") is not None]
        ) if any(row.get("condition") == condition and row.get("allocator_need_now") is not None
                 for row in PROTECTED_NEED_ROWS) else 0,
        "exact_allocator_reject_fraction": (
            allocator_rejects / max(1, capacity_attempts)
        ),
        "false_admit_count": sum(
            bool(row.get("false_admit")) for row in GUARD_COMPARISON_ROWS
            if row.get("condition") == condition and row.get("event") == "CAPACITY_GUARD"
        ),
        "false_reject_count": sum(
            bool(row.get("false_reject")) for row in GUARD_COMPARISON_ROWS
            if row.get("condition") == condition and row.get("event") == "CAPACITY_GUARD"
        ),
        "attempt_count": capacity_attempts,
    }


RUN_RESULTS: list[dict[str, Any]] = []
PROTECTED_NEED_ROWS: list[dict[str, Any]] = []
GUARD_COMPARISON_ROWS: list[dict[str, Any]] = []


def _environment() -> dict[str, Any]:
    import os
    import sys
    env: dict[str, Any] = {
        "python": sys.version, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "model": str(runtime.MODEL), "GPU_KV_bytes": GPU_KV_BYTES, "CPU_KV_bytes": CPU_KV_BYTES,
        "block_size": BLOCK_SIZE, "GPU_blocks": GPU_BLOCKS, "P_SAFE_ms": P_SAFE_MS,
        "resume_guard_ms": RESUME_GUARD_MS, "K2": K2,
        "reservation_budget_blocks": RESERVATION_BUDGET_BLOCKS,
    }
    try:
        env["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used",
             "--format=csv,noheader"], text=True, stderr=subprocess.STDOUT,
        ).strip()
        env["nvidia_smi_topology"] = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True)
        mem = Path("/proc/meminfo")
        env["meminfo"] = mem.read_text(encoding="utf-8").splitlines()[:8] if mem.exists() else "NA"
    except (OSError, subprocess.CalledProcessError) as exc:
        env["resource_snapshot_error"] = repr(exc)
    return env


def _write_static_artifacts(fingerprint: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(load_source())
    write_json(OUT / "config.json", {
        "phase": "4G-C", "conditions": ["C0", "C1"],
        "workload_sha256": SOURCE_SHA, "P_SAFE_ms": P_SAFE_MS,
        "resume_guard_ms": RESUME_GUARD_MS, "K2": K2,
        "reservation_budget_blocks": RESERVATION_BUDGET_BLOCKS,
        "source_recheck_interval_ms": RECHECK_INTERVAL_MS,
        "pause_wait_spin_window_s": PAUSE_WAIT_SPIN_WINDOW_S,
        "client_thread_switch_interval_s": PAUSE_THREAD_SWITCH_INTERVAL_S,
        "offload_prompt_only": "pinned default true; not overridden",
        "background": "BGv2: 4 x 2048 output tokens, 9500-token prompt",
        "workload_turns": 40, "boundaries_per_run": 32,
        "matrix_run_order": RUN_ORDER,
    })
    write_json(OUT / "source_manifest.json", {
        **fingerprint, "phase4c_workload": str(SOURCE),
        "runtime_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "vllm_source": str(VLLM_SOURCE), "vllm_upstream_commit": VLLM_COMMIT,
        "vllm_pre_phase4g_c_diff_sha256": VLLM_PRE_PHASE4G_C_DIFF_SHA,
        "vllm_research_endpoint_sha256": fingerprint["vllm_research_endpoint_sha256"],
        "manifest_summary": {
            "sessions": 8, "turns": 40, "boundaries": len(manifest),
            "reusable_tokens": sum(int(row["required_prefix_tokens"]) for row in manifest),
            "reusable_blocks": sum(int(row["required_blocks"]) for row in manifest),
        },
    })
    write_json(OUT / "environment.json", _environment())
    (OUT / "commands.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"cd {ROOT}\n"
        f"{runtime.ENV_ROOT}/bin/python -m pytest -q tests/test_phase4g_c.py\n"
        f"{runtime.ENV_ROOT}/bin/python -m experiments.phase4g_c.run_phase4g_c\n",
        encoding="utf-8",
    )
    patch = subprocess.check_output(["git", "diff", "--binary"], cwd=VLLM_SOURCE)
    route = VLLM_SOURCE / "vllm/entrypoints/serve/dev/slackkv_prefetch/api_router.py"
    route_patch = subprocess.run(
        ["git", "diff", "--binary", "--no-index", "/dev/null", str(route)],
        cwd=VLLM_SOURCE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    ).stdout
    patch += route_patch
    (OUT / "instrumentation.patch").write_bytes(patch)


def finalize(results: list[dict[str, Any]], manifest: list[dict[str, Any]]) -> dict[str, Any]:
    global RUN_RESULTS, PROTECTED_NEED_ROWS, GUARD_COMPARISON_ROWS
    if not validate_fresh_matrix(RUN_ORDER, results):
        raise RuntimeError("EXPERIMENT_INVALID_MATRIX: refusing partial or invalid aggregation")
    RUN_RESULTS = results
    data = aggregate(results, manifest)
    PROTECTED_NEED_ROWS = data["protected_need"]
    GUARD_COMPARISON_ROWS = data["allocator_guard_comparison"]
    summaries = {name: _condition_summary(data["run_metrics"], name) for name in ("C0", "C1")}
    results_class = classify_matrix_results(summaries["C0"], summaries["C1"])
    if len(results) != 8 or not all(row.get("valid") for row in results):
        results_class = ["INCOMPLETE_MATRIX"]
    for filename, key in (
        ("controller_selection.csv", "controller_selection"),
        ("source_recheck_events.csv", "source_recheck_events"),
        ("allocator_truth.csv", "allocator_truth"),
        ("allocator_guard_comparison.csv", "allocator_guard_comparison"),
        ("protected_need.csv", "protected_need"),
        ("progress_reject_counterfactual.csv", "progress_reject_counterfactual"),
        ("promotion_funnel.csv", "promotion_funnel"),
        ("run_metrics.csv", "run_metrics"),
        ("source_coverage.csv", "source_coverage"),
        ("selection_race.csv", "selection_race"),
        ("probe_cost.csv", "probe_cost"),
    ):
        write_csv(OUT / filename, data[key])
    need_values = [
        float(row["allocator_need_now"]) for row in data["protected_need"]
        if row.get("allocator_need_now") not in (None, "")
    ]
    need_groups = {}
    for matches in (True, False):
        ids = {str(row["selection_id"]) for row in data["controller_selection"]
               if bool(row["candidate_matches_protected"]) is matches}
        group = [row for row in data["protected_need"] if str(row.get("selection_id")) in ids]
        gaps = [float(row["need_gap_blocks"]) for row in group if row.get("need_gap_blocks") is not None]
        ratios = [float(row["need_ratio"]) for row in group if row.get("need_ratio") is not None]
        need_groups["candidate_matches_protected" if matches else "candidate_differs_from_protected"] = {
            "count": len(group), "gap": summarize(gaps), "ratio": summarize(ratios),
        }
    local_cache_dependence = []
    for label, predicate in (
        ("none", lambda row: int(row.get("local_cached_blocks") or 0) == 0),
        ("partial", lambda row: 0 < int(row.get("local_cached_blocks") or 0)
         < int(row.get("required_blocks") or 0)),
        ("full", lambda row: int(row.get("local_cached_blocks") or 0)
         >= int(row.get("required_blocks") or 0)),
    ):
        group = [row for row in data["protected_need"] if predicate(row)]
        local_cache_dependence.append({
            "local_residency": label, "count": len(group),
            "median_local_blocks": statistics.median(
                [int(row["local_cached_blocks"]) for row in group]
            ) if group else None,
            "median_PSA1_need": statistics.median(
                [float(row["PSA1_need"]) for row in group if row.get("PSA1_need") is not None]
            ) if group else None,
            "median_allocator_need_now": statistics.median(
                [float(row["allocator_need_now"]) for row in group if row.get("allocator_need_now") is not None]
            ) if any(row.get("allocator_need_now") is not None for row in group) else None,
            "median_need_gap": statistics.median(
                [float(row["need_gap_blocks"]) for row in group if row.get("need_gap_blocks") is not None]
            ) if any(row.get("need_gap_blocks") is not None for row in group) else None,
        })
    ownership_summary = {}
    for field in ("local_cached_blocks", "leased_blocks", "owned_by_running_blocks",
                  "owned_by_waiting_blocks", "evictable_cached_blocks", "other_refcounted_blocks"):
        values = [int(row[field]) for row in data["protected_need"]
                  if row.get(field) not in (None, "", "NA")]
        ownership_summary[field] = {
            "available_count": len(values), "median_blocks": statistics.median(values) if values else None,
        }
    mismatches = [
        row for row in data["allocator_guard_comparison"]
        if row.get("event") == "CAPACITY_GUARD"
    ]
    shadow_actual = [row for row in mismatches if row.get("shadow_vs_actual_fit_agreement") is not None]
    report = {
        "phase": "4G-C", "start_commit": START_COMMIT,
        "workload_sha256": SOURCE_SHA, "valid_runs": len(results),
        "expected_runs": 8, "requests_completed": sum(int(r.get("turns_complete", 0)) for r in results),
        "boundaries_completed": sum(int(r.get("boundaries", 0)) for r in results),
        "server_crashes": sum(not bool(r.get("server_alive")) for r in results),
        "OOM_count": sum(bool(r.get("OOM")) for r in results),
        "lease_leaks": sum(int(r.get("lease_leaks", 0)) for r in results),
        "logical_leaks": sum(int(r.get("logical_leaks", 0)) for r in results),
        "K2_violations": sum(int(r.get("max_pre_first", 0)) > K2 for r in results),
        "source_conservation_failures": sum(not bool(r.get("source_conservation_pass")) for r in results),
        "source_hit_shortfall_tokens": sum(
            int(r.get("expected_source_tokens", 0))
            - int(r.get("local_source_tokens", 0))
            - int(r.get("external_source_tokens", 0)) for r in results
        ),
        "prompt_only_explained_shortfall_tokens": sum(
            int(r.get("prompt_only_explained_shortfall_tokens", 0)) for r in results
        ),
        "unexplained_source_shortfall_tokens": sum(
            int(r.get("unexplained_source_shortfall_tokens", 0)) for r in results
        ),
        "conditions": summaries, "selection_count": len(data["controller_selection"]),
        "protected_need_count": len(data["protected_need"]),
        "allocator_truth_count": len(data["allocator_truth"]),
        "legacy_guard_attempt_count": len(mismatches),
        "legacy_guard_agreement_fraction": (
            sum(bool(row.get("agreement")) for row in mismatches) / len(mismatches) if mismatches else None
        ),
        "legacy_false_admit_count": sum(bool(row.get("false_admit")) for row in mismatches),
        "legacy_false_reject_count": sum(bool(row.get("false_reject")) for row in mismatches),
        "shadow_actual_allocator_comparison_count": len(shadow_actual),
        "shadow_actual_allocator_fit_agreement_fraction": (
            sum(bool(row.get("shadow_vs_actual_fit_agreement")) for row in shadow_actual)
            / len(shadow_actual) if shadow_actual else None
        ),
        "structural_capacity_rejects": sum(
            row.get("classification") == "STRUCTURAL_CAPACITY_REJECT" for row in data["selection_race"]
        ),
        "admission_race_rejects": sum(
            row.get("classification") == "ADMISSION_RACE" for row in data["selection_race"]
        ),
        "counterfactual_progress_rejects": len(data["progress_reject_counterfactual"]),
        "counterfactual_recoverable_by_recheck": sum(
            bool(row.get("would_recover")) for row in data["progress_reject_counterfactual"]
        ),
        "need_conservatism_by_candidate_identity": need_groups,
        "local_cache_dependence": local_cache_dependence,
        "protected_ownership_summary": ownership_summary,
        "final_results": results_class,
        "instrumentation_overhead_pass": all(
            float(r.get("source_probe_cpu_duty") or 1) < .05
            and float(r.get("diagnostic_cpu_duty") or 1) < .07
            for r in results
        ),
    }
    write_json(OUT / "validation_report.json", report)
    _write_summary(summaries, data, report, results_class, need_groups, need_values)
    return report


def _write_summary(
    summaries: dict[str, dict[str, Any]], data: dict[str, Any], report: dict[str, Any],
    results_class: list[str], need_groups: dict[str, Any], need_values: list[float],
) -> None:
    c0, c1 = summaries["C0"], summaries["C1"]
    cf = data["progress_reject_counterfactual"]
    exact_ready = sum(bool(row.get("protected_exact_ready")) for row in cf)
    self_psa_fit = sum(bool(row.get("self_PSA_fit")) for row in cf)
    self_alloc_fit = sum(bool(row.get("self_exact_allocator_fit")) for row in cf)
    recover = sum(bool(row.get("would_recover")) for row in cf)
    comparisons = data["allocator_guard_comparison"]
    mismatches = [row for row in comparisons if row.get("event") == "CAPACITY_GUARD"]
    agreement = sum(bool(row.get("agreement")) for row in mismatches)
    agreement_text = f"{agreement}/{len(mismatches)} ({agreement / len(mismatches):.1%})" if mismatches else "NA"
    legacy_need = [float(row["legacy_required_blocks"]) for row in comparisons
                   if row.get("legacy_required_blocks") is not None]
    allocator_required = [float(row["allocator_required_blocks"]) for row in comparisons
                          if row.get("allocator_required_blocks") is not None]
    allocator_available = [float(row["allocator_available_blocks"]) for row in comparisons
                           if row.get("allocator_available_blocks") is not None]
    pool_free = [float(row["pool_free_blocks"]) for row in comparisons if row.get("pool_free_blocks") is not None]
    psa_needs = [float(row["PSA1_need"]) for row in data["protected_need"] if row.get("PSA1_need") is not None]
    gap_values = [float(row["need_gap_blocks"]) for row in data["protected_need"]
                  if row.get("need_gap_blocks") is not None]
    f = lambda values, q: summarize(values)[q] if values else "NA"
    text = [
        "# Phase 4G-C: Source Recheck × PSA × Allocator-Truth RCA",
        "",
        "Final result(s): " + ", ".join(results_class) + ".",
        f"Start commit: {START_COMMIT}. Workload SHA256: {SOURCE_SHA}.",
        "",
        "This is a mechanism RCA, not a latency-performance comparison. C0 retains one-shot source visibility; C1 consumes only a periodic exact full-ready transition. Candidate ranking, deadlines, latest-feasible-start, reservation, PSA-1, physical guard, leases, K2, BGv2, and workload are frozen.",
        "",
        "## Paired mechanism summary",
        "",
        "| Metric | C0 Current | C1 Recheck-only |",
        "|---|---:|---:|",
        f"| Promotion selections | {c0['selection_count']} | {c1['selection_count']} |",
        f"| Stale protected visibility | {c0['stale_protected_visibility_count']} ({c0['stale_protected_visibility_fraction']:.1%}) | {c1['stale_protected_visibility_count']} ({c1['stale_protected_visibility_fraction']:.1%}) |",
        f"| Candidate = protected | {c0['candidate_matches_protected_count']} ({c0['candidate_matches_protected_rate']:.1%}) | {c1['candidate_matches_protected_count']} ({c1['candidate_matches_protected_rate']:.1%}) |",
        f"| PSA reserve median / P95 / max (blocks) | {c0['median_psa_reserve']} / {c0['p95_psa_reserve']} / {c0['max_psa_reserve']} | {c1['median_psa_reserve']} / {c1['p95_psa_reserve']} / {c1['max_psa_reserve']} |",
        f"| Progress rejects | {c0['progress_rejects']} | {c1['progress_rejects']} |",
        f"| Capacity rejects | {c0['capacity_rejects']} | {c1['capacity_rejects']} |",
        f"| Exact allocator no-fit attempts | {c0['exact_allocator_rejects']} | {c1['exact_allocator_rejects']} |",
        f"| Successful / useful promotions | {c0['successful_promotions']} / {c0['useful_promotions']} | {c1['successful_promotions']} / {c1['useful_promotions']} |",
        "",
        "## Allocator truth",
        "",
        "The pinned vLLM 0.23.0 allocate_slots path calls the coordinator's get_num_blocks_to_allocate() and compares that actual required count to pool free minus in-flight reservations. Its separate full-sequence admission gate uses the pool-free count. This phase records those exact runtime inputs and outcomes; it does not substitute the cached-prefix full footprint for incremental allocator demand. No allocator decision branch was changed.",
        "",
        "| Quantity | Median | P95 |",
        "|---|---:|---:|",
        f"| Legacy full-prefix required blocks | {f(legacy_need, 'median')} | {f(legacy_need, 'p95')} |",
        f"| Allocator required blocks | {f(allocator_required, 'median')} | {f(allocator_required, 'p95')} |",
        f"| Pool free blocks | {f(pool_free, 'median')} | {f(pool_free, 'p95')} |",
        f"| Allocator available blocks | {f(allocator_available, 'median')} | {f(allocator_available, 'p95')} |",
        f"| PSA-1 protected reserve | {f(psa_needs, 'median')} | {f(psa_needs, 'p95')} |",
        f"| PSA-1 minus allocator need | {f(gap_values, 'median')} | {f(gap_values, 'p95')} |",
        "",
        f"Conservatism by identity: candidate=protected gap {need_groups['candidate_matches_protected']['gap']} and ratio {need_groups['candidate_matches_protected']['ratio']}; candidate differs gap {need_groups['candidate_differs_from_protected']['gap']} and ratio {need_groups['candidate_differs_from_protected']['ratio']}.",
        "",
        f"Protected local-residency strata and median PSA/allocator/gap: {report['local_cache_dependence']}. Exact ownership medians: {report['protected_ownership_summary']}.",
        "",
        "Instantaneous allocator requirement is not itself a future liveness guarantee: unleased cache blocks can be evicted by a candidate allocation. These are conservatism comparisons, not permission to replace PSA-1.",
        "",
        "## C0 progress-reject counterfactual",
        "",
        f"Among {len(cf)} C0 progress rejects, protected source was exact full-ready in {exact_ready}; protected-as-candidate passed the existing PSA test in {self_psa_fit}, exact allocator preflight in {self_alloc_fit}, and all recorded timing/logical/physical tests in {recover}. See progress_reject_counterfactual.csv for per-case facts.",
        "",
        f"Legacy guard and exact allocator agreed on {agreement_text}; false-admit={report['legacy_false_admit_count']}, false-reject={report['legacy_false_reject_count']}. Capacity rejects split into structural={report['structural_capacity_rejects']} and selection-to-attempt race={report['admission_race_rejects']}.",
        "",
        f"Where a guarded prefetch proceeded into allocate_slots, shadow-vs-observed preflight fit agreed on {report['shadow_actual_allocator_comparison_count']} observations at fraction {report['shadow_actual_allocator_fit_agreement_fraction']}.",
        "",
        "## Run validity and interpretation",
        "",
        f"Fresh measured runs valid: {report['valid_runs']}/8; turns complete: {report['requests_completed']}/320; boundaries: {report['boundaries_completed']}/256. Crashes={report['server_crashes']}, OOM={report['OOM_count']}, logical leaks={report['logical_leaks']}, lease leaks={report['lease_leaks']}, K2 violations={report['K2_violations']}.",
        f"Source-hit coverage is diagnostic, not a Phase 4G-C run gate: {report['source_conservation_failures']}/8 runs had fewer local+external cache-hit tokens than the full target ({report['source_hit_shortfall_tokens']} tokens short); {report['prompt_only_explained_shortfall_tokens']} are explained by the pinned OffloadingConnector default offload_prompt_only=true, and {report['unexplained_source_shortfall_tokens']} remain unexplained. Per-boundary evidence is in source_coverage.csv.",
        f"Instrumentation overhead: {'PASS' if report['instrumentation_overhead_pass'] else 'FAIL'}; maximum source-probe CPU duty C0/C1={c0['source_probe_cpu_duty_max']:.2%}/{c1['source_probe_cpu_duty_max']:.2%}; maximum total diagnostic CPU duty C0/C1={c0['diagnostic_cpu_duty_max']:.2%}/{c1['diagnostic_cpu_duty_max']:.2%}.",
        "",
        "The causal funnel is recorded per boundary in promotion_funnel.csv. Readiness requires every required block; no partial or percentage threshold is used. Protected KV ownership is split into running, waiting, leased, evictable, and other-refcounted blocks only where the pinned runtime exposes exact block-ID evidence; otherwise the artifact keeps NA.",
        "",
        "## Frozen questions",
        "",
        f"1. C0 stale protected visibility: {c0['stale_protected_visibility_count']}/{c0['selection_count']} selections.",
        f"2. Protected-self recoveries: {recover}/{len(cf)} C0 progress-reject cases; exact readiness count {exact_ready}.",
        f"3. PSA-1 median reserve: C0={c0['median_psa_reserve']}, C1={c1['median_psa_reserve']} blocks.",
        f"4. Exact allocator fit and legacy mismatch directions are tabulated in allocator_guard_comparison.csv.",
        f"5. Exact full-target cache-hit equality is diagnostic, not a request-correctness gate; the source-hit shortfall is separately attributed in source_coverage.csv.",
        f"6. Runtime instrumentation overhead gate: {'PASS' if report['instrumentation_overhead_pass'] else 'FAIL'}.",
        f"7. Result taxonomy: {', '.join(results_class)}.",
        f"8. Source recheck sufficiency: {'supported at the frozen thresholds' if 'RESULT_A_SOURCE_VISIBILITY_PSA_INTERACTION' in results_class else 'not established by the frozen thresholds'}.",
        f"9. PSA-2: {'motivated for a later separately authorized phase' if 'RESULT_B_PSA1_INTRINSIC_CONSERVATISM' in results_class else 'not supported as the primary explanation by this result alone'}. Allocator-aligned admission: {'indicated for separate follow-up' if any(x in results_class for x in ('RESULT_C_ALLOCATOR_CAPACITY_DOMINANT', 'RESULT_D_GUARD_ALLOCATOR_MISMATCH')) else 'not established as dominant'}. Partial promotion was not tested or implemented.",
        "",
        "This experiment distinguishes mechanism evidence within the frozen hybrid trace-driven workload; it does not establish a unique root cause outside this workload/runtime and does not compare performance.",
        "",
    ]
    (OUT / "summary.md").write_text("\n".join(text), encoding="utf-8")


def _write_incomplete(
    stage: str, reason: str, completed: list[dict[str, Any]],
    failed_run_id: str | None = None,
) -> None:
    result_code, failure_detail = normalize_incomplete_result(reason)
    write_json(OUT / "validation_report.json", {
        "phase": "4G-C", "start_commit": START_COMMIT, "workload_sha256": SOURCE_SHA,
        "stage": stage, "completed_runs": len(completed), "expected_measured_runs": 8,
        "attempted_runs": len(completed) + int(failed_run_id is not None),
        "failed_run_id": failed_run_id,
        "completed_run_ids": [row.get("run_id") for row in completed],
        "failure_detail": failure_detail, "final_results": [result_code],
    })
    (OUT / "summary.md").write_text(
        "# Phase 4G-C: Source Recheck × PSA × Allocator-Truth RCA\n\n"
        f"Final status: {result_code}. The run stopped during {stage}; no causal or performance comparison was performed.\n\n"
        f"Start commit: {START_COMMIT}. Frozen workload SHA256: {SOURCE_SHA}.\n\n"
        f"Failure detail: {failure_detail or 'not recorded'}. Completed fresh runs before stop: {len(completed)}.\n",
        encoding="utf-8",
    )


def main() -> None:
    run_order_check()
    fingerprint = frozen_checks()
    _write_static_artifacts(fingerprint)
    turns = load_source()
    manifest = build_manifest(turns)
    smoke_results: list[dict[str, Any]] = []
    for condition in ("C0", "C1"):
        try:
            smoke_results.append(run_one(condition, 0, manifest, "smoke"))
        except BaseException as exc:
            write_csv(OUT / "semantic_smoke.csv", [
                {"condition": row["condition"], "valid": row["valid"],
                 "X_tail_finish_ms": row.get("background_tail_finish_ms"),
                 "reason": row.get("invalid_reason", "")}
                for row in smoke_results
            ])
            _write_incomplete("semantic smoke", repr(exc), smoke_results)
            raise
    write_csv(OUT / "semantic_smoke.csv", [
        {
            "condition": row["condition"], "valid": row["valid"],
            "turns_complete": row["turns_complete"], "boundaries": row["boundaries"],
            "full_prefix_source_hit_exact": row["source_conservation_pass"],
            "source_hit_shortfall_tokens": int(row["expected_source_tokens"])
                - int(row["local_source_tokens"]) - int(row["external_source_tokens"]),
            "prompt_only_explained_shortfall_tokens": row["prompt_only_explained_shortfall_tokens"],
            "unexplained_source_shortfall_tokens": row["unexplained_source_shortfall_tokens"],
            "K2_max_pre_first": row["max_pre_first"],
            "source_probe_cpu_duty": row["source_probe_cpu_duty"],
            "diagnostic_cpu_duty": row["diagnostic_cpu_duty"],
            "recheck_events": len(row["source_recheck_events"]),
            "recheck_behavior_updates": row["recheck_behavior_updates"],
            "BG_tokens": row.get("background_validation", {}).get("tokens"),
            "BG_tail_finish_ms": row.get("background_tail_finish_ms"),
        } for row in smoke_results
    ])
    smoke_margin = min(float(row["background_tail_finish_ms"]) for row in smoke_results)
    if (
        len(smoke_results) != 2 or not all(row.get("valid") for row in smoke_results)
        or smoke_margin <= 0
        or smoke_results[0]["recheck_behavior_updates"] != 0
        or smoke_results[1]["recheck_behavior_updates"] <= 0
    ):
        _write_incomplete("semantic smoke", "EXPERIMENT_INVALID_MATRIX", smoke_results)
        raise RuntimeError("EXPERIMENT_INVALID_MATRIX: smoke correctness gate")

    measured: list[dict[str, Any]] = []
    for repetition, condition in RUN_ORDER:
        try:
            run = run_one(condition, repetition, manifest, "measured")
        except BaseException as exc:
            _write_incomplete(
                "measured matrix", f"EXPERIMENT_INVALID_MATRIX:{exc}", measured,
                failed_run_id=f"measured_rep{repetition}_{condition.lower()}",
            )
            raise
        measured.append(run)
        write_json(OUT / "progress.json", {
            "completed_measured_runs": len(measured), "expected_measured_runs": 8,
            "completed": [{"repetition": r["repetition"], "condition": r["condition"],
                           "valid": r["valid"]} for r in measured],
        })
    report = finalize(measured, manifest)
    report["smoke_valid"] = len(smoke_results)
    report["smoke_min_BG_tail_finish_ms"] = smoke_margin
    write_json(OUT / "validation_report.json", report)
    write_csv(OUT / "semantic_smoke_summary.csv", [
        {"condition": row["condition"], "valid": row["valid"],
         "BG_tail_finish_ms": row["background_tail_finish_ms"],
         "recheck_events": len(row["source_recheck_events"])}
        for row in smoke_results
    ])
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    main()
