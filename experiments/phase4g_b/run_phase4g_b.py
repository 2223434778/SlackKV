"""Phase 4G-B read-only source-readiness/headroom mechanism RCA."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from experiments.phase3f.scheduler import latest_feasible_start, merge_blackouts
from experiments.phase4d_a import runtime
from experiments.phase4d_a.experiment import x_prompt
from experiments.phase4d_b.experiment import validate_runtime
from experiments.phase4g_a import run_phase4g_a as lifecycle
from experiments.phase4g_a.controller import DynamicController
from experiments.phase4g_a.workload import ROOT, SOURCE, SOURCE_SHA, build_manifest, load_source, nearest_rank
from experiments.phase4g_a_fix import background as bgv2
from experiments.phase4g_b.analysis import (
    P_SAFE_MS, PROBE_INTERVAL_MS, RESERVATION_BUDGET_BLOCKS, RESUME_GUARD_MS,
    classify_result, feasibility_snapshot, longest_contiguous_true_ms,
    partial_opportunity, progress_reserve_blocks, source_readiness_milestones,
    FROZEN_RUN_ORDER, validate_matrix, background_is_valid,
)

OUT = ROOT / "results/phase4g_b"
TMP_LOGS = Path("/tmp/slackkv-phase4g-b-server-logs")
START_COMMIT = "3c887a710cca06142a00c4ca35ca152e7b590bc0"
VLLM_SOURCE = Path("/media/a6000/D/ZhaoBinHe/slackkv-phase2c-fix-sourceprobe")
VLLM_COMMIT = "0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665"
VLLM_BASE_DIFF_SHA = "4ec03c5f7bcb331485becea471d6d46b28b2b9ec3f8b0a2997f62e7e8bf15618"
GPU_KV_BYTES, CPU_KV_BYTES, GPU_BLOCKS, BLOCK_SIZE = 2147483648, 8589934592, 4681, 16
EXPECTED_TURNS, EXPECTED_BOUNDARIES, BG_TOKENS = 40, 32, bgv2.BG_TOTAL_OUTPUT_TOKENS
RUN_ORDER = FROZEN_RUN_ORDER
ORIGINAL_START_SERVER = runtime.start_server
CURRENT_CONTROLLER: Any | None = None
CURRENT_CONTROLLER_SNAPSHOTS: list[dict[str, Any]] = []
CURRENT_BG_FUTURE: Any | None = None


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def vllm_diff_sha() -> str:
    return sha256(subprocess.check_output(["git", "diff", "--binary"], cwd=VLLM_SOURCE))


def frozen_checks() -> dict[str, Any]:
    if sha256(SOURCE.read_bytes()) != SOURCE_SHA:
        raise RuntimeError("EXPERIMENT_INVALID_WORKLOAD:Phase4C workload SHA")
    rows, manifest = load_source(), build_manifest(load_source())
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    upstream = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=VLLM_SOURCE, text=True).strip()
    diff_sha = vllm_diff_sha()
    changed = set(subprocess.check_output(["git", "diff", "--name-only"], cwd=VLLM_SOURCE, text=True).splitlines())
    expected_changed = {
        "vllm/distributed/kv_transfer/kv_connector/v1/offloading/common.py",
        "vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py",
        "vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py",
        "vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py",
        "vllm/entrypoints/serve/__init__.py",
        "vllm/v1/core/sched/output.py",
        "vllm/v1/core/sched/scheduler.py",
        "vllm/v1/engine/core.py",
        "vllm/v1/worker/gpu_model_runner.py",
        "vllm/v1/worker/kv_connector_model_runner_mixin.py",
    }
    if head != START_COMMIT:
        raise RuntimeError(f"EXPERIMENT_INVALID_ENVIRONMENT:SlackKV HEAD {head}")
    if upstream != VLLM_COMMIT or changed != expected_changed:
        raise RuntimeError(f"EXPERIMENT_INVALID_ENVIRONMENT:vLLM source {upstream} changed={sorted(changed)}")
    baseline_names = {"scheduler.py", "offloading_scheduler.py", "offloading_connector.py", "core.py"}
    if not all((Path("/tmp/phase4gb-vllm-baseline") / name).is_file() for name in baseline_names):
        raise RuntimeError("EXPERIMENT_INVALID_INSTRUMENTATION: baseline snapshots missing")
    model_config_path = runtime.MODEL / "config.json"
    if not runtime.MODEL.is_dir() or not model_config_path.is_file():
        raise RuntimeError("EXPERIMENT_INVALID_ENVIRONMENT: pinned local model snapshot missing")
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    max_positions = int(model_config.get("max_position_embeddings", 0))
    if max_positions < 32768:
        raise RuntimeError(f"EXPERIMENT_INVALID_ENVIRONMENT:model context {max_positions}")
    vllm_version = subprocess.check_output(
        [str(runtime.ENV_ROOT / "bin" / "python"), "-c", "import vllm; print(vllm.__version__)"],
        text=True, stderr=subprocess.STDOUT,
    ).strip().splitlines()[-1]
    if vllm_version != "0.23.0":
        raise RuntimeError(f"EXPERIMENT_INVALID_ENVIRONMENT:vLLM version {vllm_version}")
    if len(rows) != EXPECTED_TURNS or len(manifest) != EXPECTED_BOUNDARIES:
        raise RuntimeError("EXPERIMENT_INVALID_WORKLOAD:turn/boundary count")
    return {"start_commit": head, "workload_sha256": SOURCE_SHA,
            "vllm_upstream_commit": upstream, "vllm_baseline_diff_sha256": VLLM_BASE_DIFF_SHA,
            "vllm_current_diff_sha256": diff_sha, "vllm_diff_files": sorted(changed),
            "vllm_version": vllm_version, "model_snapshot_path": str(runtime.MODEL),
            "model_max_position_embeddings": max_positions,
            "turns": len(rows), "boundaries": len(manifest), "episodes": 8}


def marker_events(path: Path, marker: str) -> list[dict[str, Any]]:
    events = []
    if not path.exists():
        return events
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if marker in line:
            try:
                events.append(json.loads(line.split(marker, 1)[1]))
            except (ValueError, IndexError):
                events.append({"event": "PARSE_ERROR", "line_sha256": sha256(line.encode())})
    return events


def _start_server(label: str, connector: bool = True, gpu_bytes: int = GPU_KV_BYTES, out: Path | None = None):
    TMP_LOGS.mkdir(parents=True, exist_ok=True)
    return ORIGINAL_START_SERVER(label, connector=connector, gpu_bytes=gpu_bytes, out=TMP_LOGS)


class _EmptyFuture:
    def __init__(self, first_ns: int):
        self.first_ns = first_ns

    def result(self, timeout: float | None = None):
        return {"tokens": [], "first_ns": self.first_ns, "done_ns": time.monotonic_ns(), "chunks": []}


class _EmptyPool:
    def shutdown(self, wait: bool = True, cancel_futures: bool = False):
        return None


def _no_background():
    first = time.monotonic_ns()
    return _EmptyPool(), _EmptyFuture(first), first


def _bgv2_background():
    global CURRENT_BG_FUTURE
    ready, first = threading.Event(), {}
    prompt = x_prompt()
    if len(prompt) != bgv2.BG_PROMPT_TOKENS:
        raise RuntimeError("RUN_INVALID_BACKGROUND:BGv2 prompt token count")
    def on_first(ns: int):
        first["ns"] = int(ns)
        ready.set()
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(bgv2.run_chunks, runtime, prompt, on_first)
    CURRENT_BG_FUTURE = future
    if not ready.wait(120):
        pool.shutdown(wait=False, cancel_futures=True)
        raise TimeoutError("BGv2 first-token timeout")
    return pool, future, first["ns"]


class _ObservedController(DynamicController):
    def __init__(self, *args, **kwargs):
        global CURRENT_CONTROLLER
        super().__init__(*args, **kwargs)
        CURRENT_CONTROLLER = self

    def run(self, done: threading.Event):
        stop = threading.Event()
        def observer():
            while not stop.is_set():
                now = time.monotonic_ns()
                with self.lock:
                    active = dict(self.ledger.active)
                    pending = [(key, state) for key, state in self.registered.items()
                               if state.get("first_token_ns") is None]
                    protected = min(pending, key=lambda pair: (pair[1]["deadline_ns"], pair[0]), default=(None, None))
                    CURRENT_CONTROLLER_SNAPSHOTS.append({
                        "timestamp_monotonic_ns": now, "reserved_blocks": sum(active.values()),
                        "active_reservations": active, "protected_boundary_id": protected[0],
                        "protected_deadline_ns": protected[1]["deadline_ns"] if protected[1] else None,
                    })
                stop.wait(PROBE_INTERVAL_MS / 1000)
        thread = threading.Thread(target=observer, daemon=True)
        thread.start()
        try:
            super().run(done)
        finally:
            stop.set()
            thread.join(timeout=1)
            CURRENT_CONTROLLER_SNAPSHOTS.append({
                "timestamp_monotonic_ns": time.monotonic_ns(),
                "reserved_blocks": self.ledger.reserved_blocks,
                "active_reservations": dict(self.ledger.active),
                "protected_boundary_id": None, "protected_deadline_ns": None,
            })


def _install_trace_params(manifest: list[dict[str, Any]]):
    by_id = {str(row["boundary_id"]): row for row in manifest}
    original_send = runtime.send
    def wrapped(prompt, max_tokens, *, stream=False, params=None, on_first=None):
        params = dict(params or {})
        boundary = params.get("slackkv_trace_next_boundary")
        if isinstance(boundary, str) and boundary in by_id:
            job = by_id[boundary]
            params.update({
                "slackkv_trace_next_pause_ms": float(job["proxy_pause_ms"]),
                "slackkv_trace_next_isl": int(job["resume_isl"]),
                "slackkv_trace_next_lease_key": str(job["lease_key"]),
            })
        if params.get("slackkv_trace_resume") is True:
            session = str(params.get("slackkv_session", ""))
            episode, sep, number = session.rpartition(":turn:")
            boundary_id = f"{episode}:b{int(number) - 1}" if sep and number.isdigit() else ""
            if boundary_id not in by_id:
                raise RuntimeError(f"invalid probe resume identity: {session}")
            params["slackkv_trace_boundary_id"] = boundary_id
            params["slackkv_trace_resume_issue_ns"] = time.monotonic_ns()
        return original_send(prompt, max_tokens, stream=stream, params=params, on_first=on_first)
    runtime.send = wrapped
    return original_send


def _sample_gap_ms(samples: list[dict[str, Any]]) -> float:
    grouped: dict[str, set[int]] = {}
    for row in samples:
        grouped.setdefault(str(row.get("boundary_id")), set()).add(int(row["timestamp_monotonic_ns"]))
    gaps = [(right - left) / 1e6 for values in grouped.values()
            for left, right in zip(sorted(values), sorted(values)[1:])]
    return max(gaps, default=0.0)


def _controller_at(rows: list[dict[str, Any]], ns: int) -> dict[str, Any]:
    prior = [row for row in rows if int(row["timestamp_monotonic_ns"]) <= ns]
    return max(prior, key=lambda row: int(row["timestamp_monotonic_ns"])) if prior else {
        "reserved_blocks": 0, "active_reservations": {},
    }


def run_one(condition: str, policy: str, background: bool, stage: str, repetition: int,
            manifest: list[dict[str, Any]]) -> dict[str, Any]:
    global CURRENT_CONTROLLER, CURRENT_CONTROLLER_SNAPSHOTS, CURRENT_BG_FUTURE
    CURRENT_CONTROLLER, CURRENT_BG_FUTURE = None, None
    CURRENT_CONTROLLER_SNAPSHOTS = []
    original_start, original_send = runtime.start_server, _install_trace_params(manifest)
    original_bg, original_controller = lifecycle.start_background, lifecycle.DynamicController
    original_out, original_x_count = lifecycle.OUT, lifecycle.X_OUTPUT_TOKENS
    runtime.start_server = _start_server
    lifecycle.OUT = OUT
    lifecycle.DynamicController = _ObservedController
    lifecycle.start_background = _bgv2_background if background else _no_background
    lifecycle.X_OUTPUT_TOKENS = BG_TOKENS if background else 0
    label = f"{stage}_rep{repetition}_{policy.lower()}"
    try:
        result = lifecycle.run_lifecycle(policy, stage=stage, repetition=repetition)
        server_log = TMP_LOGS / "server_logs" / f"{label}.log"
        probe = marker_events(server_log, "SLACKKV_READY_PROBE ")
        costs = [row for row in probe if row.get("event") == "PROBE_COST"]
        samples = [row for row in probe if row.get("event") == "SAMPLE"]
        if costs:
            span_ns = max(int(row["timestamp_monotonic_ns"]) for row in costs) - min(int(row["timestamp_monotonic_ns"]) for row in costs) + 20_000_000
            duration_ns = sum(int(row.get("probe_duration_ns", 0)) for row in costs)
        else:
            span_ns = duration_ns = 0
        result.update({
            "run_id": f"{stage}_rep{repetition}_{condition}", "condition": condition,
            "policy": policy, "background_enabled": background,
            "ready_probe_events": probe,
            "probe_error_count": sum(row.get("event") == "PROBE_ERROR" for row in probe),
            "controller_snapshots": CURRENT_CONTROLLER_SNAPSHOTS,
            "probe_cost_summary": {
                "sample_count": len(costs), "boundary_sample_count": len(samples),
                "total_probe_duration_ms": duration_ns / 1e6,
                "observation_span_ms": span_ns / 1e6,
                "probe_cpu_duty_fraction": duration_ns / span_ns if span_ns else None,
                "probe_duration_P50_ms": statistics.median([int(row.get("probe_duration_ns", 0)) for row in costs]) / 1e6 if costs else None,
                "probe_duration_P95_ms": nearest_rank([int(row.get("probe_duration_ns", 0)) for row in costs], .95) / 1e6 if costs else None,
                "probe_duration_max_ms": max([int(row.get("probe_duration_ns", 0)) for row in costs], default=0) / 1e6,
                "probe_wall_duration_P50_ms": statistics.median([int(row.get("probe_wall_duration_ns", 0)) for row in costs]) / 1e6 if costs else None,
                "probe_wall_duration_P95_ms": nearest_rank([int(row.get("probe_wall_duration_ns", 0)) for row in costs], .95) / 1e6 if costs else None,
                "probe_wall_duration_max_ms": max([int(row.get("probe_wall_duration_ns", 0)) for row in costs], default=0) / 1e6,
                "max_probe_sample_gap_ms": _sample_gap_ms(samples),
            },
        })
        if background:
            bg = CURRENT_BG_FUTURE.result(timeout=5) if CURRENT_BG_FUTURE is not None else {}
            chunks = bg.get("chunks", [])
            bg_summary = bgv2.summarize_chunks(chunks)
            tail_first_margin = float(result.get("X_tail_margin_ms", 0) or 0)
            last_turn_finish = max((int(row["finish_ns"]) for row in result.get("turn_rows", [])), default=0)
            tail_finish_margin = ((int(bg.get("done_ns", 0)) - last_turn_finish) / 1e6
                                  if last_turn_finish else 0.0)
            required_tail = 3000.0 if stage == "smoke" else 0.0
            result["background_chunks"] = chunks
            result["background_validation"] = {
                "chunks": len(chunks), "tokens": bg_summary["total_generated_tokens"],
                "max_inter_chunk_idle_gap_ms": bg_summary["max_inter_chunk_idle_gap_ms"],
                "tail_last_first_token_ms": tail_first_margin,
                "tail_last_request_finish_ms": tail_finish_margin,
                "valid": background_is_valid(
                    generated_tokens=int(bg_summary["total_generated_tokens"]),
                    expected_tokens=BG_TOKENS,
                    max_idle_gap_ms=float(bg_summary["max_inter_chunk_idle_gap_ms"]),
                    allowed_gap_ms=bgv2.MAX_IDLE_GAP_MS,
                    tail_margin_ms=tail_first_margin,
                    minimum_tail_ms=required_tail,
                ),
            }
        else:
            result["background_chunks"] = []
            result["background_validation"] = {"enabled": False, "valid": True}
            for field in ("X_first_ns", "X_finish_ns", "X_decode_ms", "X_TPOT_ms", "X_tail_margin_ms"):
                result[field] = "NA"
        result["source_events"] = result.get("source_events", [])
        result["valid"] = bool(
            result.get("valid") and result["probe_error_count"] == 0 and bool(samples)
            and (not background or result["background_validation"]["valid"])
            and (policy != "A" or (int(result.get("promotion_attempts", -1)) == 0
                 and int(result.get("lease_event_count", -1)) == 0
                 and int(result.get("logical_leaks", -1)) == 0))
        )
        run_path = OUT / "runs" / label / "run.json"
        write_json(run_path, result)
        return result
    except BaseException as exc:
        log = TMP_LOGS / "server_logs" / f"{label}.log"
        write_json(OUT / "failures" / f"{label}.json", {
            "condition": condition, "policy": policy, "stage": stage,
            "repetition": repetition, "background": background, "error": repr(exc),
            "last_server_lines": log.read_text(encoding="utf-8", errors="replace").splitlines()[-80:] if log.exists() else [],
            "controller_snapshots": CURRENT_CONTROLLER_SNAPSHOTS,
        })
        raise
    finally:
        runtime.start_server = original_start
        runtime.send = original_send
        lifecycle.start_background = original_bg
        lifecycle.DynamicController = original_controller
        lifecycle.OUT = original_out
        lifecycle.X_OUTPUT_TOKENS = original_x_count


def derive_timeline(results: list[dict[str, Any]], manifest: list[dict[str, Any]]):
    by_job = {str(row["boundary_id"]): row for row in manifest}
    timeline, milestones, gaps = [], [], []
    for run in results:
        meta = {key: run[key] for key in ("run_id", "condition", "policy", "repetition", "background_enabled")}
        samples = [row for row in run.get("ready_probe_events", []) if row.get("event") == "SAMPLE"]
        source_by_boundary: dict[str, list[dict[str, Any]]] = {}
        for event in run.get("source_events", []):
            if isinstance(event.get("boundary_id"), str):
                source_by_boundary.setdefault(event["boundary_id"], []).append(event)
        boundary_client = {str(row["boundary_id"]): row for row in run.get("boundary_rows", [])}
        grouped: dict[str, list[dict[str, Any]]] = {}
        for sample in samples:
            grouped.setdefault(str(sample["boundary_id"]), []).append(sample)
        origin_ns = int(run["X_first_ns"]) if run.get("X_first_ns") != "NA" else (
            min((int(row["timestamp_monotonic_ns"]) for row in samples), default=0))
        for boundary_id, rows in grouped.items():
            if boundary_id not in by_job:
                continue
            job = by_job[boundary_id]
            client = boundary_client.get(boundary_id, {})
            deadline_ns = int(client.get("target_resume_issue_ns", 0))
            if not deadline_ns:
                deadline_ns = int(rows[0]["estimated_deadline_ns"])
            safe_start_ns = deadline_ns - round(P_SAFE_MS * 1e6)
            rows.sort(key=lambda row: int(row["timestamp_monotonic_ns"]))
            initial = next((event for event in source_by_boundary.get(boundary_id, [])
                            if event.get("event") in {"SOURCE_READY", "SOURCE_NOT_READY"}), None)
            initial_status = initial.get("event", "NA") if initial else "NA"
            ready_rows = [row for row in rows if bool(row.get("source_fully_ready"))]
            source_in_time = any(bool(row.get("source_fully_ready")) and int(row["timestamp_monotonic_ns"]) <= safe_start_ns for row in rows)
            job_timeline: list[dict[str, Any]] = []
            for sample in rows:
                ts = int(sample["timestamp_monotonic_ns"])
                siblings = [row for row in samples if str(row.get("sample_id")) == str(sample.get("sample_id"))]
                deadlines = {str(row["boundary_id"]): (int(row["estimated_deadline_ns"]) - origin_ns) / 1e6 for row in siblings}
                ready_jobs = [(str(row["boundary_id"]), deadlines[str(row["boundary_id"])])
                              for row in siblings if row.get("source_fully_ready")]
                now_ms = (ts - origin_ns) / 1e6
                start = latest_feasible_start(now_ms, ready_jobs,
                    merge_blackouts(deadlines, RESUME_GUARD_MS), P_SAFE_MS) if ready_jobs else now_ms
                timing_fit = bool(ready_jobs) and now_ms + .5 >= start
                controller_state = _controller_at(run.get("controller_snapshots", []), ts)
                reserved = int(controller_state.get("reserved_blocks", 0)) if run["policy"] == "D" else 0
                protected = sample.get("protected_frontier")
                required = int(sample["required_blocks"])
                h = progress_reserve_blocks(
                    int(sample.get("protected_progress_blocks", 0)),
                    int(sample.get("protected_lease_blocks", 0)),
                    candidate_matches_protected=boundary_id == protected,
                    candidate_blocks=required,
                )
                values = feasibility_snapshot(
                    required_blocks=required,
                    local_blocks=int(sample.get("local_cached_blocks", 0)),
                    external_ready_blocks=int(sample.get("external_ready_needed_blocks", 0)),
                    free_blocks=int(sample.get("free_blocks", 0)), progress_reserve=h,
                    reserved_blocks=reserved, logical_enabled=run["policy"] == "D",
                    source_ready=bool(sample.get("source_fully_ready")), timing_open=timing_fit,
                )
                free = int(sample.get("free_blocks", 0))
                timeline_row = {
                    **meta, **sample,
                    "source_ready": bool(sample.get("source_fully_ready")),
                    "timestamp_ms": (ts - origin_ns) / 1e6,
                    "deadline_ms": (deadline_ns - origin_ns) / 1e6,
                    "safe_start_ms": (safe_start_ns - origin_ns) / 1e6,
                    "initial_one_shot_status": initial_status,
                    "initial_not_ready": initial_status == "SOURCE_NOT_READY",
                    "source_in_time": source_in_time,
                    "progress_reserve_blocks": h,
                    "logical_reserved_blocks": reserved,
                    "logical_fit": values["logical_fit"],
                    "timing_fit": timing_fit,
                    "timing_target_ms": start,
                    "physical_fit_external_needed": values["physical_fit"],
                    "psa_safe_fit_external_needed": values["psa_safe_fit"],
                    "physical_fit_full_prefix": free >= required,
                    "psa_safe_fit_full_prefix": free - required >= h,
                    "full_feasible": bool(sample.get("source_fully_ready") and free - required >= h
                        and values["logical_fit"] and timing_fit),
                    "partial_opportunity_fraction": partial_opportunity(
                        required, int(sample.get("external_ready_needed_blocks", 0)), free, h),
                }
                timeline.append(timeline_row)
                job_timeline.append(timeline_row)
            full_by_safe = next((row for row in ready_rows if int(row["timestamp_monotonic_ns"]) <= safe_start_ns), None)
            full_eventual = ready_rows[0] if ready_rows else None
            safe_samples = [row for row in rows if int(row["timestamp_monotonic_ns"]) <= safe_start_ns]
            def safe_predicate(row):
                required = int(row.get("required_blocks", 0))
                h = progress_reserve_blocks(
                    int(row.get("protected_progress_blocks", 0)), int(row.get("protected_lease_blocks", 0)),
                    candidate_matches_protected=boundary_id == row.get("protected_frontier"),
                    candidate_blocks=required,
                )
                return bool(row.get("source_fully_ready")) and int(row.get("free_blocks", 0)) - required >= h
            actionable_rows = [row for row in job_timeline if int(row["timestamp_monotonic_ns"]) <= safe_start_ns]
            actionable_ms = longest_contiguous_true_ms(actionable_rows, lambda row: bool(row["full_feasible"]))
            source_milestone = source_readiness_milestones(rows, int(job["required_blocks"]))
            deadline_samples = [row for row in rows if int(row["timestamp_monotonic_ns"]) <= deadline_ns]
            full_safe_window = longest_contiguous_true_ms(deadline_samples, safe_predicate)
            milestone = {
                **meta, "boundary_id": boundary_id,
                "initial_one_shot_status": initial_status,
                "initial_not_ready": initial_status == "SOURCE_NOT_READY",
                "initial_ready_blocks": initial.get("ready_prefix_blocks", "NA") if initial else "NA",
                "required_blocks": int(job["required_blocks"]), **source_milestone,
                "eventual_full_ready": bool(full_eventual),
                "full_ready_by_deadline": bool(full_eventual and int(full_eventual["timestamp_monotonic_ns"]) <= deadline_ns),
                "full_ready_by_safe_start": bool(full_by_safe),
                "source_in_time": source_in_time,
                "full_psa_safe_window_ms": full_safe_window,
                "actionable_full_window_ms": actionable_ms,
                "partial_ge_25pct_20ms": longest_contiguous_true_ms([row for row in job_timeline if int(row["timestamp_monotonic_ns"]) <= deadline_ns], lambda row: float(row.get("partial_opportunity_fraction", 0)) >= .25) >= 20,
                "partial_ge_50pct_50ms": longest_contiguous_true_ms([row for row in job_timeline if int(row["timestamp_monotonic_ns"]) <= deadline_ns], lambda row: float(row.get("partial_opportunity_fraction", 0)) >= .50) >= 50,
                "partial_ge_75pct_50ms": longest_contiguous_true_ms([row for row in job_timeline if int(row["timestamp_monotonic_ns"]) <= deadline_ns], lambda row: float(row.get("partial_opportunity_fraction", 0)) >= .75) >= 50,
                "partial_ge_90pct_20ms": longest_contiguous_true_ms([row for row in job_timeline if int(row["timestamp_monotonic_ns"]) <= deadline_ns], lambda row: float(row.get("partial_opportunity_fraction", 0)) >= .90) >= 20,
                "source_sample_count": len(rows),
                "source_probe_max_gap_ms": _sample_gap_ms(rows),
            }
            milestones.append(milestone)
            category = ("SOURCE_READY_AT_INITIAL_CHECK" if initial_status == "SOURCE_READY" else
                "MISSED_SOURCE_RECHECK" if milestone["full_ready_by_safe_start"] else
                "SOURCE_READY_AFTER_SAFE_START" if milestone["full_ready_by_deadline"] else
                "SOURCE_NEVER_FULL_BEFORE_DEADLINE")
            gaps.append({
                **meta, "boundary_id": boundary_id,
                "initial_one_shot_status": initial_status,
                "initial_ready_blocks": milestone["initial_ready_blocks"],
                "required_blocks": int(job["required_blocks"]),
                "max_covered_blocks": max((int(row.get("covered_blocks", 0)) for row in rows), default=0),
                "eventual_full_ready": milestone["eventual_full_ready"],
                "full_ready_by_safe_start": milestone["full_ready_by_safe_start"],
                "observation_gap_category": category,
                "initial_deficit_blocks": max(0, int(job["required_blocks"]) - int(milestone["initial_ready_blocks"] or 0)) if milestone["initial_ready_blocks"] != "NA" else "NA",
            })
    return timeline, milestones, gaps


def _sample_gap_ms(rows: list[dict[str, Any]]) -> float:
    times = sorted({int(row["timestamp_monotonic_ns"]) for row in rows})
    return max(((b - a) / 1e6 for a, b in zip(times, times[1:])), default=0.0)


def _histograms(gaps: list[dict[str, Any]], timeline: list[dict[str, Any]]):
    deficits = []
    bins = ((0, 0, "0"), (1, 1, "1"), (2, 4, "2-4"), (5, 16, "5-16"), (17, 64, "17-64"), (65, 10**9, ">64"))
    target = [row for row in gaps if row["condition"] == "M2"]
    for lo, hi, label in bins:
        selected = [row for row in target if isinstance(row.get("initial_deficit_blocks"), (int, float))
                    and lo <= int(row["initial_deficit_blocks"]) <= hi]
        deficits.append({
            "initial_deficit_bin": label, "boundary_count": len(selected),
            "eventual_full_ready": sum(bool(row["eventual_full_ready"]) for row in selected),
            "full_ready_by_safe_start": sum(bool(row["full_ready_by_safe_start"]) for row in selected),
        })
    readiness = [{"category": cat, "count": sum(row["observation_gap_category"] == cat for row in target)}
                 for cat in ("SOURCE_READY_AT_INITIAL_CHECK", "MISSED_SOURCE_RECHECK",
                             "SOURCE_READY_AFTER_SAFE_START", "SOURCE_NEVER_FULL_BEFORE_DEADLINE")]
    headroom = []
    for condition in ("M0", "M1", "M2", "M3"):
        rows = [row for row in timeline if row["condition"] == condition and row.get("source_fully_ready")]
        bins2 = ((0, .25, "<25%"), (.25, .5, "25-49%"), (.5, .75, "50-74%"),
                 (.75, 1, "75-99%"), (1, float("inf"), ">=100%"))
        for lo, hi, label in bins2:
            selected = [row for row in rows if lo <= int(row["free_blocks"]) / max(1, int(row["required_blocks"])) < hi]
            headroom.append({"condition": condition, "headroom_fraction": label, "sample_count": len(selected)})
    return deficits, readiness, headroom


def _blockers(milestones: list[dict[str, Any]], timeline: list[dict[str, Any]], results: list[dict[str, Any]]):
    output = []
    actual_decisions = {
        (str(run["run_id"]), str(row.get("boundary_id")))
        for run in results for row in run.get("decisions", [])
        if row.get("event") == "PROMOTION_ATTEMPT"
    }
    for item in milestones:
        condition = item["condition"]
        rows = [row for row in timeline if row["run_id"] == item["run_id"] and row["boundary_id"] == item["boundary_id"]]
        source_in_time = bool(item["source_in_time"])
        physical_exists = any(row["source_fully_ready"] and row["physical_fit_full_prefix"] for row in rows)
        safe_window = float(item.get("full_psa_safe_window_ms", 0) or 0) >= 20
        source_safe_overlap = any(row["source_fully_ready"] and row["psa_safe_fit_full_prefix"] for row in rows)
        source_seen = any(row["source_fully_ready"] for row in rows)
        safe_seen = any(row["psa_safe_fit_full_prefix"] for row in rows)
        output.append({
            "condition": condition, "run_id": item["run_id"], "boundary_id": item["boundary_id"],
            "source_late": not source_in_time,
            "physical_capacity_blocker": source_in_time and not physical_exists,
            "psa_safe_headroom_blocker": source_in_time and physical_exists and not safe_window,
            "temporal_source_capacity_no_overlap": source_in_time and source_seen and safe_seen and not source_safe_overlap,
            "jit_timing_blocker": safe_window and not any(row["timing_fit"] and row["full_feasible"] for row in rows),
            "partial_ge_50pct_50ms": bool(item["partial_ge_50pct_50ms"]),
            "controller_missed_shadow_actionable_window": condition == "M3"
                and float(item.get("actionable_full_window_ms", 0) or 0) >= 20
                and (str(item["run_id"]), str(item["boundary_id"])) not in actual_decisions,
        })
    return output


def _actual_and_consistency(results: list[dict[str, Any]], timeline: list[dict[str, Any]]):
    attempts, consistency = [], []
    for run in results:
        meta = {"run_id": run["run_id"], "condition": run["condition"], "policy": run["policy"], "repetition": run["repetition"]}
        for row in run.get("decisions", []):
            attempts.append({**meta, **row})
        for row in run.get("capacity_trace", []):
            attempts.append({**meta, **row})
            if row.get("event") != "CAPACITY_GUARD":
                continue
            lease_key = str(row.get("lease_key", ""))
            episode, sep, suffix = lease_key.partition(":turn:")
            bid = f"{episode}:b{int(suffix)-1}" if sep and suffix.isdigit() else "NA"
            ts = int(float(row.get("timestamp_monotonic", 0)) * 1e9)
            candidates = [sample for sample in timeline if sample["run_id"] == run["run_id"] and sample["boundary_id"] == bid]
            closest = min(candidates, key=lambda sample: abs(int(sample["timestamp_monotonic_ns"]) - ts), default=None)
            free = int(row.get("free_blocks_before", -1))
            required = int(row.get("candidate_required_blocks", row.get("required_blocks", -1)))
            reserve = int(row.get("progress_reserve_blocks", 0))
            decision = str(row.get("decision", "NA"))
            invariant = ((decision == "ADMIT" and free >= required and free-required >= reserve)
                         or (decision == "CAPACITY_REJECT" and free < required)
                         or (decision == "PROGRESS_REJECT" and free >= required and free-required < reserve))
            consistency.append({
                **meta, "boundary_id": bid, "decision": decision, "free_blocks": free,
                "required_blocks": required, "progress_reserve_blocks": reserve,
                "probe_sample_matched_within_40ms": closest is not None and abs(int(closest["timestamp_monotonic_ns"])-ts) <= 40_000_000,
                "probe_sample_id": closest.get("sample_id") if closest else "NA",
                "capacity_invariant": invariant,
            })
        for key, promotion in run.get("promotions", {}).items():
            response = promotion.get("response") or {}
            stamps = response.get("phase3m_timestamps") or {}
            attempts.append({**meta, "event": "PROMOTION_RESPONSE", "boundary_id": key,
                "status": response.get("status", "NA"), "submitted_ns": promotion.get("submitted_ns", "NA"),
                "T0": stamps.get("T0", "NA"), "T1": stamps.get("T1", "NA"),
                "T6": stamps.get("T6", "NA"), "T7": stamps.get("T7", "NA"),
                "transfer_bytes": response.get("transfer_bytes", 0),
                "external_tokens_loaded": response.get("external_tokens_loaded", 0)})
    return attempts, consistency


def aggregate(results: list[dict[str, Any]], manifest: list[dict[str, Any]]) -> dict[str, Any]:
    timeline, milestones, gaps = derive_timeline(results, manifest)
    source_events, controller_rows, run_metrics, probe_cost = [], [], [], []
    for run in results:
        meta = {"run_id": run["run_id"], "condition": run["condition"], "policy": run["policy"], "repetition": run["repetition"]}
        source_events.extend({**meta, **row} for row in run.get("source_events", []))
        controller_rows.extend({**meta, **row} for row in run.get("controller_snapshots", []))
        run_metrics.append({**meta, "valid": run.get("valid"), "turns_complete": run.get("turns_complete"),
            "boundaries": run.get("boundaries"), "promotion_attempts": run.get("promotion_attempts"),
            "source_ready_count": run.get("source_ready_count"), "source_not_ready_count": run.get("source_not_ready_count"),
            "probe_error_count": run.get("probe_error_count"), **run.get("probe_cost_summary", {}),
            "background_enabled": run.get("background_enabled"),
            "BG_tokens": run.get("background_validation", {}).get("tokens", 0),
            "BG_max_gap_ms": run.get("background_validation", {}).get("max_inter_chunk_idle_gap_ms", "NA"),
            "BG_tail_first_token_ms": run.get("BG_tail_first_token_ms", "NA"),
            "BG_tail_finish_ms": run.get("BG_tail_request_finish_ms", "NA")})
        probe_cost.extend({**meta, **row} for row in run.get("ready_probe_events", []) if row.get("event") == "PROBE_COST")
    deficits, readiness, headroom = _histograms(gaps, timeline)
    blockers = _blockers(milestones, timeline, results)
    attempts, consistency = _actual_and_consistency(results, timeline)
    condition_summary = []
    for cond in ("M0", "M1", "M2", "M3"):
        runs = [row for row in results if row["condition"] == cond]
        sm = [row for row in milestones if row["condition"] == cond]
        tl = [row for row in timeline if row["condition"] == cond]
        decisions = [event for run in runs for event in run.get("decisions", [])]
        condition_summary.append({
            "condition": cond, "policy": "A" if cond in {"M0", "M2"} else "D",
            "background": cond in {"M2", "M3"}, "runs": len(runs),
            "valid_runs": sum(bool(run.get("valid")) for run in runs),
            "promotion_attempts": sum(int(run.get("promotion_attempts", 0)) for run in runs),
            "initial_not_ready": sum(bool(row["initial_not_ready"]) for row in sm),
            "full_ready_by_safe_start": sum(bool(row["full_ready_by_safe_start"]) for row in sm),
            "actionable_ge_20ms": sum(float(row.get("actionable_full_window_ms", 0) or 0) >= 20 for row in sm),
            "source_in_time": sum(bool(row["source_in_time"]) for row in sm),
            "missed_source_rechecks": sum(row["observation_gap_category"] == "MISSED_SOURCE_RECHECK"
                                           for row in gaps if row["condition"] == cond),
            "capacity_rejects": sum(row.get("event") == "PROMOTION_CAPACITY_REJECTED" for row in decisions),
            "progress_rejects": sum(row.get("event") == "PROMOTION_PROGRESS_REJECTED" for row in decisions),
            "reservation_skips": sum(row.get("event") == "RESERVATION_SKIP" for row in decisions),
            "mean_free_blocks": statistics.mean([int(row["free_blocks"]) for row in tl]) if tl else "NA",
            "mean_external_ready_needed_blocks": statistics.mean([int(row["external_ready_needed_blocks"]) for row in tl]) if tl else "NA",
            "mean_progress_reserve_blocks": statistics.mean([int(row["progress_reserve_blocks"]) for row in tl]) if tl else "NA",
        })
    return {"timeline": timeline, "milestones": milestones, "gaps": gaps,
            "source_events": source_events, "controller_rows": controller_rows,
            "run_metrics": run_metrics, "probe_cost": probe_cost, "deficits": deficits,
            "readiness": readiness, "headroom": headroom, "blockers": blockers,
            "attempts": attempts, "consistency": consistency,
            "partial": milestones, "condition_summary": condition_summary}


def classify_aggregate(data: dict[str, Any]) -> str:
    rows = [row for row in data["milestones"] if row["condition"] == "M2"]
    source_time = [row for row in rows if row["source_in_time"]]
    physically_fit = []
    for row in source_time:
        samples = [sample for sample in data["timeline"]
                   if sample["run_id"] == row["run_id"]
                   and sample["boundary_id"] == row["boundary_id"]
                   and sample["source_fully_ready"]]
        if any(sample["physical_fit_full_prefix"] for sample in samples):
            physically_fit.append(row)
    physical_without_safe = sum(float(row["full_psa_safe_window_ms"]) < 20 for row in physically_fit)
    return classify_result(
        m2_not_ready_count=sum(bool(row["initial_not_ready"]) for row in rows),
        m2_later_full_by_safe_start_count=sum(bool(row["full_ready_by_safe_start"]) for row in rows),
        m2_actionable_recoveries=sum(float(row.get("actionable_full_window_ms", 0) or 0) >= 20 for row in rows),
        m2_source_in_time_count=len(source_time),
        m2_source_in_time_without_safe_window=sum(float(row["full_psa_safe_window_ms"]) < 20 for row in source_time),
        m2_source_in_time_with_partial=sum(bool(row["partial_ge_50pct_50ms"]) for row in source_time),
        m2_source_in_time_physical_fit_count=len(physically_fit),
        m2_physical_fit_without_safe_window=physical_without_safe,
    )


def _plot_artifacts(data: dict[str, Any]) -> None:
    from PIL import Image, ImageDraw, ImageFont

    OUT.mkdir(parents=True, exist_ok=True)
    colors = {"M0": (76, 120, 168), "M1": (245, 133, 24), "M2": (84, 162, 75), "M3": (228, 87, 86)}
    font = ImageFont.load_default()

    def canvas(title: str, xlabel: str, ylabel: str, width: int = 1100, height: int = 650):
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        left, top, right, bottom = 90, 60, width - 35, height - 85
        draw.text((left, 20), title, fill="black", font=font)
        draw.line((left, top, left, bottom, right, bottom), fill=(40, 40, 40), width=2)
        draw.text((left, height - 55), xlabel, fill="black", font=font)
        draw.text((10, top), ylabel, fill="black", font=font)
        for fraction in (0, .25, .5, .75, 1):
            y = int(bottom - fraction * (bottom - top))
            draw.line((left, y, right, y), fill=(232, 232, 232), width=1)
            draw.text((left - 34, y - 5), f"{fraction:.2g}", fill=(80, 80, 80), font=font)
        return image, draw, (left, top, right, bottom)

    def ecdf_plot(filename: str, title: str, xlabel: str, field: str, threshold: float | None = None):
        series: dict[str, list[float]] = {}
        for cond in ("M0", "M1", "M2", "M3"):
            values = [float(row[field]) for row in data["milestones"]
                      if row["condition"] == cond and row.get(field) != "NA"]
            if values:
                series[cond] = sorted(values)
        xmax = max([max(values) for values in series.values()] + ([threshold] if threshold else [1.0]) + [1.0])
        image, draw, (left, top, right, bottom) = canvas(title, xlabel, "ECDF")
        def xy(x: float, y: float):
            return (left + x / xmax * (right - left), bottom - y * (bottom - top))
        for cond, values in series.items():
            points = [xy(value, (index + 1) / len(values)) for index, value in enumerate(values)]
            if len(points) == 1:
                x, y = points[0]; draw.ellipse((x-4, y-4, x+4, y+4), fill=colors[cond])
            else:
                draw.line(points, fill=colors[cond], width=3)
            legend_y = top + (list(series).index(cond) * 22)
            draw.line((right - 135, legend_y + 6, right - 112, legend_y + 6), fill=colors[cond], width=3)
            draw.text((right - 106, legend_y), cond, fill="black", font=font)
        if threshold is not None:
            x, _ = xy(threshold, 0)
            draw.line((x, top, x, bottom), fill=(30, 30, 30), width=2)
            draw.text((min(x + 4, right - 140), top + 92), f"gate={threshold:g}ms", fill="black", font=font)
        image.save(OUT / filename)

    ecdf_plot("source_readiness_ecdf.png", "Observed source readiness (20 ms probes)",
              "First fully-ready sample, ms after first boundary sample", "ready_minus_0_ms")
    ecdf_plot("safe_headroom_ecdf.png", "Full-prefix source-ready + PSA-safe window before deadline",
              "Contiguous safe-window duration (ms)", "full_psa_safe_window_ms", threshold=20)

    chosen = {}
    for cond in ("M2", "M3"):
        candidates = [row for row in data["timeline"] if row["condition"] == cond]
        if candidates:
            first_run = min(row["run_id"] for row in candidates)
            first_boundary = min(row["boundary_id"] for row in candidates if row["run_id"] == first_run)
            chosen[cond] = sorted((row for row in candidates if row["run_id"] == first_run
                                   and row["boundary_id"] == first_boundary),
                                  key=lambda row: int(row["timestamp_monotonic_ns"]))
    height = max(500, 360 * len(chosen))
    image = Image.new("RGB", (1100, height), "white")
    draw = ImageDraw.Draw(image)
    for panel, (cond, rows) in enumerate(chosen.items()):
        top, bottom, left, right = 50 + panel * 350, 330 + panel * 350, 90, 1060
        draw.text((left, top - 25), f"{cond}: {rows[0]['run_id']} {rows[0]['boundary_id']}", fill="black", font=font)
        if not rows:
            continue
        xmin = min(float(row["timestamp_ms"]) for row in rows)
        xmax = max(float(row["timestamp_ms"]) for row in rows) or xmin + 1
        ymax = max(4681, max(int(row["required_blocks"]) for row in rows))
        xmap = lambda x: left + (x - xmin) / (xmax - xmin) * (right - left)
        ymap = lambda y: bottom - y / ymax * (bottom - top)
        draw.line((left, top, left, bottom, right, bottom), fill=(40, 40, 40), width=2)
        draw.line([(xmap(float(row["timestamp_ms"])), ymap(int(row["free_blocks"]))) for row in rows], fill=colors[cond], width=3)
        draw.line([(xmap(float(row["timestamp_ms"])), ymap(int(row["required_blocks"]))) for row in rows], fill=(90, 90, 90), width=2)
        for row in rows:
            if row["source_fully_ready"]:
                x, y = xmap(float(row["timestamp_ms"])), ymap(int(row["required_blocks"]))
                draw.ellipse((x-3, y-3, x+3, y+3), fill=(45, 160, 100))
        safe_x = xmap(float(rows[0]["safe_start_ms"]))
        draw.line((safe_x, top, safe_x, bottom), fill=(0, 0, 0), width=1)
        draw.text((left, bottom + 8), "time from run origin (ms); green points = source fully ready", fill="black", font=font)
    image.save(OUT / "source_vs_headroom_example_timelines.png")

    blocker_names = ("source_late", "physical_capacity_blocker", "psa_safe_headroom_blocker",
                     "temporal_source_capacity_no_overlap", "jit_timing_blocker",
                     "controller_missed_shadow_actionable_window")
    conds = ("M0", "M1", "M2", "M3")
    image, draw, (left, top, right, bottom) = canvas("Shadow blocker attribution", "Condition", "Count", height=650)
    max_value = max((sum(bool(row.get(name)) for row in data["blockers"] if row["condition"] == cond)
                     for name in blocker_names for cond in conds), default=1) or 1
    group_w = (right - left) / len(conds)
    bar_w = group_w / (len(blocker_names) + 1)
    palette = [(76,120,168), (245,133,24), (84,162,75), (228,87,86), (114,183,178), (176,122,161)]
    for ci, cond in enumerate(conds):
        draw.text((left + ci * group_w + group_w/3, bottom + 14), cond, fill="black", font=font)
        for bi, name in enumerate(blocker_names):
            value = sum(bool(row.get(name)) for row in data["blockers"] if row["condition"] == cond)
            x0 = left + ci * group_w + bi * bar_w
            y0 = bottom - value / max_value * (bottom - top)
            draw.rectangle((x0, y0, x0 + bar_w - 2, bottom), fill=palette[bi])
    for bi, name in enumerate(blocker_names):
        x0, y0 = left + (bi % 3) * 300, height - 42 + (bi // 3) * 18
        draw.rectangle((x0, y0 + 2, x0 + 10, y0 + 12), fill=palette[bi])
        draw.text((x0 + 14, y0), name, fill="black", font=font)
    image.save(OUT / "blocker_breakdown.png")


def _write_instrumentation_patch(path: Path) -> str:
    import difflib
    files = {
        "vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py": "offloading_scheduler.py",
        "vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py": "offloading_connector.py",
        "vllm/v1/core/sched/scheduler.py": "scheduler.py",
        "vllm/v1/engine/core.py": "core.py",
    }
    chunks = []
    for relative, baseline_name in files.items():
        baseline = Path("/tmp/phase4gb-vllm-baseline") / baseline_name
        current = VLLM_SOURCE / relative
        if not baseline.exists() or not current.exists():
            raise RuntimeError(f"missing instrumentation patch input: {relative}")
        left = baseline.read_text(encoding="utf-8").splitlines(keepends=True)
        right = current.read_text(encoding="utf-8").splitlines(keepends=True)
        chunks.extend(difflib.unified_diff(left, right, fromfile=f"a/{relative}", tofile=f"b/{relative}"))
    text = "".join(chunks)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return sha256(text.encode())


def _write_summary(data: dict[str, Any], result: str, smoke: dict[str, Any], frozen: dict[str, Any]) -> None:
    rows = data["condition_summary"]
    lines = [
        "# Phase 4G-B — Source Readiness × Safe Headroom RCA", "",
        f"**Classification:** `{result}`", "",
        "This is a mechanism RCA, not a performance comparison. It distinguishes whether reusable KV has reached the CPU tier from whether full-prefix promotion can fit while preserving the PSA-1 progress reserve.", "",
        "## Frozen inputs and gates", "",
        f"- SlackKV start commit: `{frozen['start_commit']}`.",
        f"- Frozen Phase 4C workload SHA-256: `{frozen['workload_sha256']}`.",
        f"- vLLM upstream: `{frozen['vllm_upstream_commit']}`; pre-instrumentation source-diff SHA-256: `{frozen['vllm_baseline_diff_sha256']}`.",
        f"- Workload: {frozen['episodes']} episodes, {frozen['turns']} turns, {frozen['boundaries']} boundaries; block size {BLOCK_SIZE}; GPU KV {GPU_KV_BYTES} bytes / {GPU_BLOCKS} blocks.",
        f"- Probe cadence: {PROBE_INTERVAL_MS:g} ms. Instrumentation smoke duty: {smoke.get('probe_cpu_duty_fraction', 'NA')} (required < 0.05); probe errors: {smoke.get('probe_error_count', 'NA')}.",
        "- No request prompts, token arrays, controller decisions, or vLLM scheduler decisions were modified by the readiness probes.", "",
        "## Condition matrix", "",
        "| Condition | Policy | Background | Runs valid | Source initially not ready | Full source ready by deadline−P_SAFE | Actionable full window ≥20 ms | Capacity rejects | PSA rejects | Reservation skips |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append("| {condition} | {policy} | {background} | {valid_runs}/{runs} | {initial_not_ready} | {full_ready_by_safe_start} | {actionable_ge_20ms} | {capacity_rejects} | {progress_rejects} | {reservation_skips} |".format(**row))
    lines.extend(["", "## Interpretation", "",
        f"The frozen classification is `{result}`. Source readiness is based on an exact per-key read-only view of GPU-local cached blocks plus CPU offload-policy ready state; headroom uses the scheduler block pool's exact free-block count. These are reported as separate gates.", "",
        "A ‘missed source recheck’ means the initial one-shot observation said source-not-ready, while the 20 ms periodic probe subsequently observed all required blocks ready before the conservative safe-start (deadline minus 102 ms). A full-prefix PSA-safe window additionally requires free blocks to cover the entire prefix plus the observed protected-request progress reserve; an actionable window also requires the frozen queue-aware timing and logical-reservation constraints.", "",
        "Partial-prefix Q is an opportunity diagnostic only. It does not establish that issuing a partial promotion would be correct or beneficial. The M0/M2 no-JIT rows are shadow analyses; their counterfactual reservation state is not a replay of hypothetical promotions. Actual D decisions are reported separately and are never replaced by shadow calculations.", "",
        "No P95, throughput, winner, or performance-superiority claim is made. The background condition is the Phase 4G-A-FIX fixed-work BGv2 protocol; it is treated as a load condition, not an agent workload metric.", "",
        "## Probe caveat", "",
        "Readiness probe cost is measured inside the server and summarized separately. The instrumentation smoke is accepted only if total probe CPU duty is below 5% and the probe produces samples without errors. Any remaining source-key coverage or sampling gaps are exposed in the boundary-level milestone and observation-gap artifacts.", ""])
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_artifacts(results: list[dict[str, Any]], manifest: list[dict[str, Any]], frozen: dict[str, Any], smoke: dict[str, Any]) -> dict[str, Any]:
    validate_matrix(results)
    data = aggregate(results, manifest)
    result = classify_aggregate(data)
    tables = {
        "mechanism_timeline.csv": data["timeline"],
        "source_milestones.csv": data["milestones"],
        "readiness_observation_gap.csv": data["gaps"],
        "source_events.csv": data["source_events"],
        "controller_timeline.csv": data["controller_rows"],
        "run_metrics.csv": data["run_metrics"],
        "probe_cost.csv": data["probe_cost"],
        "source_deficit_histogram.csv": data["deficits"],
        "source_readiness_histogram.csv": data["readiness"],
        "safe_headroom_histogram.csv": data["headroom"],
        "blocker_matrix.csv": data["blockers"],
        "actual_attempts.csv": data["attempts"],
        "attempt_consistency.csv": data["consistency"],
        "partial_opportunity.csv": data["partial"],
        "condition_summary.csv": data["condition_summary"],
    }
    for name, rows in tables.items():
        write_csv(OUT / name, rows)
    counterfactual_fields = (
        "run_id", "condition", "policy", "repetition", "boundary_id", "timestamp_monotonic_ns",
        "timestamp_ms", "deadline_ms", "safe_start_ms", "source_fully_ready", "local_cached_blocks",
        "external_ready_blocks", "external_ready_needed_blocks", "missing_blocks", "required_blocks",
        "free_blocks", "used_blocks", "progress_reserve_blocks", "physical_fit_external_needed",
        "psa_safe_fit_external_needed", "physical_fit_full_prefix", "psa_safe_fit_full_prefix",
        "logical_reserved_blocks", "logical_fit", "timing_fit", "timing_target_ms", "full_feasible",
        "partial_opportunity_fraction", "active_lease_count", "active_lease_unique_blocks",
        "num_running", "num_waiting",
    )
    write_csv(OUT / "counterfactual_constraints.csv", [
        {key: row.get(key, "NA") for key in counterfactual_fields} for row in data["timeline"]
    ])
    write_csv(OUT / "safe_headroom_timeline.csv", [
        row for row in data["timeline"]
        if row.get("source_fully_ready") or row.get("physical_fit_full_prefix") or row.get("psa_safe_fit_full_prefix")
    ])
    write_csv(OUT / "boundary_metrics.csv", data["milestones"])
    patch_sha = _write_instrumentation_patch(OUT / "instrumentation.patch")
    cfg = {
        "phase": "4G-B", "probe_interval_ms": PROBE_INTERVAL_MS,
        "p_safe_ms": P_SAFE_MS, "resume_guard_ms": RESUME_GUARD_MS,
        "gpu_kv_bytes": GPU_KV_BYTES, "cpu_external_kv_bytes": CPU_KV_BYTES,
        "gpu_blocks": GPU_BLOCKS, "block_size_tokens": BLOCK_SIZE,
        "reservation_budget_blocks": RESERVATION_BUDGET_BLOCKS,
        "policies": {"M0": "OnDemand-K2 / no background", "M1": "FullSlackKV-PSA1-K2 / no background",
                     "M2": "OnDemand-K2 / BGv2", "M3": "FullSlackKV-PSA1-K2 / BGv2"},
        "frozen_run_order": [{"condition": cond, "policy": pol, "background": bg} for cond, pol, bg in RUN_ORDER],
        "matrix_measured_runs": len(RUN_ORDER), "background_output_tokens": BG_TOKENS,
        "classification": result, "instrumentation_patch_sha256": patch_sha,
    }
    write_json(OUT / "config.json", cfg)
    source = {
        "phase": "Phase 4G-B", "start_commit": frozen["start_commit"],
        "phase4c_workload_path": str(SOURCE.relative_to(ROOT)),
        "phase4c_workload_sha256": frozen["workload_sha256"],
        "vllm_source_path": str(VLLM_SOURCE), "vllm_upstream_commit": frozen["vllm_upstream_commit"],
        "vllm_preinstrumentation_diff_sha256": frozen["vllm_baseline_diff_sha256"],
        "new_trace_only_instrumentation_patch_sha256": patch_sha,
        "no_scheduler_algorithm_changes": True,
    }
    write_json(OUT / "source_manifest.json", source)
    env = {
        "python": subprocess.check_output(["python", "--version"], text=True, stderr=subprocess.STDOUT).strip(),
        "torch": subprocess.check_output(["python", "-c", "import torch; print(torch.__version__); print(torch.version.cuda)"], text=True).strip().splitlines(),
        "vllm": frozen["vllm_version"],
        "vllm_working_diff_sha256": frozen["vllm_current_diff_sha256"],
        "model": str(runtime.MODEL), "model_config_max_position_embeddings": frozen["model_max_position_embeddings"],
        "dtype": "float16", "tensor_parallel_size": 1, "cuda_visible_devices": "0",
        "gpu_kv_bytes": GPU_KV_BYTES, "cpu_external_kv_bytes": CPU_KV_BYTES,
        "prefix_caching": True, "block_size_tokens": BLOCK_SIZE,
        "nvidia_smi": subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,temperature.gpu,power.draw,clocks.sm,clocks.mem", "--format=csv,noheader"], text=True, capture_output=True).stdout.strip(),
        "nvidia_smi_topology": subprocess.run(["nvidia-smi", "topo", "-m"], text=True, capture_output=True).stdout,
    }
    write_json(OUT / "environment.json", env)
    (OUT / "commands.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "cd /media/a6000/D/ZhaoBinHe/SlackKV\n"
        f"{runtime.ENV_ROOT}/bin/python -m pytest -q tests/test_phase4g_b.py\n"
        f"{runtime.ENV_ROOT}/bin/python -m py_compile experiments/phase4g_b/*.py\n"
        f"{runtime.ENV_ROOT}/bin/python -m experiments.phase4g_b.run_phase4g_b --run\n", encoding="utf-8")
    _plot_artifacts(data)
    _write_summary(data, result, smoke, frozen)
    report = {
        "phase": "4G-B", "start_commit": frozen["start_commit"],
        "workload_sha256": frozen["workload_sha256"],
        "instrumentation_smoke": {"valid": smoke.get("valid"), "probe_error_count": smoke.get("probe_error_count"),
                                  "probe_cpu_duty_fraction": smoke.get("probe_cost_summary", {}).get("probe_cpu_duty_fraction"),
                                  "max_probe_sample_gap_ms": smoke.get("probe_cost_summary", {}).get("max_probe_sample_gap_ms")},
        "measured_runs_completed": len(results), "expected_measured_runs": len(RUN_ORDER),
        "measured_runs_valid": sum(bool(row.get("valid")) for row in results),
        "all_boundary_probe_samples_present": all(
            len({row.get("boundary_id") for row in run.get("ready_probe_events", []) if row.get("event") == "SAMPLE"}) == EXPECTED_BOUNDARIES
            for run in results),
        "condition_summaries": data["condition_summary"],
        "result": result,
        "interpretation_scope": "Mechanism RCA only; no latency or throughput comparison.",
    }
    write_json(OUT / "validation_report.json", report)
    return report


def write_initial_context(frozen: dict[str, Any]) -> None:
    write_json(OUT / "config.json", {
        "phase": "4G-B", "probe_interval_ms": PROBE_INTERVAL_MS,
        "p_safe_ms": P_SAFE_MS, "resume_guard_ms": RESUME_GUARD_MS,
        "gpu_kv_bytes": GPU_KV_BYTES, "cpu_external_kv_bytes": CPU_KV_BYTES,
        "gpu_blocks": GPU_BLOCKS, "block_size_tokens": BLOCK_SIZE,
        "reservation_budget_blocks": RESERVATION_BUDGET_BLOCKS,
        "policies": {"M0": "OnDemand-K2 / no background", "M1": "FullSlackKV-PSA1-K2 / no background",
                     "M2": "OnDemand-K2 / BGv2", "M3": "FullSlackKV-PSA1-K2 / BGv2"},
        "frozen_run_order": [{"condition": cond, "policy": pol, "background": bg} for cond, pol, bg in RUN_ORDER],
        "expected_measured_runs": len(RUN_ORDER), "background_output_tokens": BG_TOKENS,
    })
    write_json(OUT / "source_manifest.json", {
        "phase": "Phase 4G-B", "start_commit": frozen["start_commit"],
        "phase4c_workload_path": str(SOURCE.relative_to(ROOT)),
        "phase4c_workload_sha256": frozen["workload_sha256"],
        "vllm_source_path": str(VLLM_SOURCE), "vllm_upstream_commit": frozen["vllm_upstream_commit"],
        "vllm_version": frozen["vllm_version"],
        "vllm_preinstrumentation_diff_sha256": frozen["vllm_baseline_diff_sha256"],
        "vllm_working_diff_sha256_before_run": frozen["vllm_current_diff_sha256"],
        "vllm_diff_files": frozen["vllm_diff_files"],
        "model_snapshot_path": frozen["model_snapshot_path"],
        "model_max_position_embeddings": frozen["model_max_position_embeddings"],
        "no_scheduler_algorithm_changes": True,
    })
    env = {
        "python": subprocess.check_output([str(runtime.ENV_ROOT / "bin" / "python"), "--version"], text=True, stderr=subprocess.STDOUT).strip(),
        "torch": subprocess.check_output([str(runtime.ENV_ROOT / "bin" / "python"), "-c", "import torch; print(torch.__version__); print(torch.version.cuda)"], text=True, stderr=subprocess.STDOUT).strip().splitlines()[-2:],
        "vllm": frozen["vllm_version"], "vllm_source_commit": frozen["vllm_upstream_commit"],
        "model": frozen["model_snapshot_path"], "max_position_embeddings": frozen["model_max_position_embeddings"],
        "dtype": "float16", "tensor_parallel_size": 1, "cuda_visible_devices": "0",
        "gpu_kv_bytes": GPU_KV_BYTES, "cpu_external_kv_bytes": CPU_KV_BYTES,
        "prefix_caching": True, "block_size_tokens_expected": BLOCK_SIZE,
        "nvidia_smi": subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,temperature.gpu,power.draw,clocks.sm,clocks.mem", "--format=csv,noheader"], text=True, capture_output=True).stdout.strip(),
        "nvidia_smi_topology": subprocess.run(["nvidia-smi", "topo", "-m"], text=True, capture_output=True).stdout,
    }
    write_json(OUT / "environment.json", env)
    (OUT / "commands.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "cd /media/a6000/D/ZhaoBinHe/SlackKV\n"
        f"{runtime.ENV_ROOT}/bin/python -m pytest -q tests/test_phase4g_b.py\n"
        f"{runtime.ENV_ROOT}/bin/python -m py_compile experiments/phase4g_b/*.py\n"
        f"{runtime.ENV_ROOT}/bin/python -m experiments.phase4g_b.run_phase4g_b --run\n", encoding="utf-8")


def execute() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    smoke_repetition = 0
    if any(OUT.iterdir()):
        prior_report_path = OUT / "validation_report.json"
        prior_run_paths = sorted((OUT / "runs").glob("smoke_rep*_a/run.json"))
        if not (prior_report_path.is_file() and prior_run_paths):
            raise RuntimeError(f"refusing to overwrite non-empty result directory: {OUT}")
        prior_report = json.loads(prior_report_path.read_text(encoding="utf-8"))
        prior_run_path = prior_run_paths[-1]
        prior_run = json.loads(prior_run_path.read_text(encoding="utf-8"))
        if prior_report.get("result") != "EXPERIMENT_INVALID_INSTRUMENTATION_SMOKE" or prior_run.get("valid") is not False:
            raise RuntimeError(f"refusing to overwrite non-retryable result directory: {OUT}")
        prior_rep = int(prior_run_path.parent.name.removeprefix("smoke_rep").removesuffix("_a"))
        smoke_repetition = prior_rep + 1
        write_json(OUT / "failures" / f"instrumentation_smoke_attempt_{prior_rep}.json", {
            "validation_report": prior_report,
            "run_id": prior_run.get("run_id"),
            "probe_error_count": prior_run.get("probe_error_count"),
            "probe_cost_summary": prior_run.get("probe_cost_summary"),
            "source_probe_events": len(prior_run.get("ready_probe_events", [])),
            "run_json": str(prior_run_path.relative_to(ROOT)),
        })
    frozen = frozen_checks()
    write_initial_context(frozen)
    manifest = build_manifest(load_source())
    smoke = run_one("M2", "A", True, "smoke", smoke_repetition, manifest)
    duty = smoke.get("probe_cost_summary", {}).get("probe_cpu_duty_fraction")
    if not smoke.get("valid") or duty is None or float(duty) >= .05:
        write_json(OUT / "validation_report.json", {
            "phase": "4G-B", "start_commit": frozen["start_commit"],
            "workload_sha256": frozen["workload_sha256"],
            "instrumentation_smoke_valid": bool(smoke.get("valid")),
            "probe_cpu_duty_fraction": duty, "required_probe_cpu_duty_fraction_lt": .05,
            "result": "EXPERIMENT_INVALID_INSTRUMENTATION_SMOKE",
        })
        raise RuntimeError("EXPERIMENT_INVALID_INSTRUMENTATION_SMOKE")
    results = []
    for index, (condition, policy, background) in enumerate(RUN_ORDER, start=1):
        repetition = index
        run = run_one(condition, policy, background, "measured", repetition, manifest)
        if not run.get("valid"):
            write_json(OUT / "validation_report.json", {
                "phase": "4G-B", "start_commit": frozen["start_commit"],
                "workload_sha256": frozen["workload_sha256"],
                "instrumentation_smoke_valid": True, "completed_before_stop": len(results),
                "failed_run_id": run.get("run_id"), "result": "INCOMPLETE_INVALID_RUN",
            })
            raise RuntimeError(f"INCOMPLETE_INVALID_RUN:{run.get('run_id')}")
        results.append(run)
        # Persist recoverable progress after every run; never aggregate a partial matrix as complete.
        write_json(OUT / "progress.json", {"completed_runs": len(results), "expected_runs": len(RUN_ORDER),
                                             "run_ids": [item["run_id"] for item in results]})
    return write_artifacts(results, manifest, frozen, smoke)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="validate only frozen source and workload")
    parser.add_argument("--run", action="store_true", help="run the frozen Phase 4G-B protocol")
    args = parser.parse_args()
    if args.preflight:
        print(json.dumps(frozen_checks(), indent=2, sort_keys=True))
        return
    if not args.run:
        parser.error("choose --preflight or --run")
    report = execute()
    print(json.dumps(report, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
