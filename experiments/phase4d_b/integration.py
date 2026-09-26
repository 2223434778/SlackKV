from __future__ import annotations

import csv
import json
import signal
import statistics
import threading
import time
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from experiments.phase3f.scheduler import latest_feasible_start, merge_blackouts
from experiments.phase4d_a import runtime
from experiments.phase4d_a.helpers import ReservationLedger
from experiments.phase4d_a_fix import integration as legacy
from experiments.phase4d_a_fix.release_fix import (
    LeaseTraceReader,
    RuntimeReleaseQueue,
    EXPECTED_GPU_BLOCKS,
    FROZEN_P_SAFE_MS,
    parse_lease_trace,
    service_envelope,
)
from experiments.phase4d_a.run_policy import parse_trace
from experiments.phase4d_b.capacity import (
    capacity_invariant,
    parse_capacity_trace,
    validate_capacity_rejection,
    rejected_request_lifecycle_valid,
)
from experiments.phase4d_b.capacity_client import capacity_prefetch
from experiments.phase4d_b.controller import harvest_completed_promotions
from experiments.phase4d_b.experiment import (
    OUT,
    RESERVATION_BUDGET,
    frozen_jobs,
    reconstructed_data,
    validate_runtime,
)
from experiments.phase4d_b.g1 import available_memory_bytes, check_native_result

WATCHDOG_SECONDS = 30.0
PREFETCH_WATCHDOG_SECONDS = 5.0
MIN_AVAILABLE_BYTES = 16 * 1024**3

CURRENT_CAPACITY_RESULTS: dict[str, dict[str, Any]] = {}


class FinalDrainReleaseQueue(RuntimeReleaseQueue):
    @property
    def pending_release_count(self) -> int:
        return max(super().pending_release_count, len(self.ledger.active))


def drain_runtime_releases(
    queue: RuntimeReleaseQueue,
    states: dict[str, dict[str, Any]],
    lease_reader: LeaseTraceReader,
    now_ns: int,
) -> list[dict[str, Any]]:
    for episode_id, state in states.items():
        first_ns = state.get("first_token_ns")
        if (
            first_ns is not None
            and state.get("leased")
            and episode_id in queue.ledger.active
        ):
            queue.enqueue_first_token(episode_id, int(first_ns))
    queue.observe_lease_events(lease_reader.read_events())
    return queue.drain(now_ns)


def _append_decision(OUT_DIR: Path, rows: list[dict[str, Any]]) -> None:
    if rows:
        legacy.write_csv(OUT_DIR / "reservation_decisions.csv", rows)


def schedule_promotions(
    jobs: list[dict[str, Any]],
    data: dict[str, dict[str, Any]],
    states: dict[str, dict[str, Any]],
    lock: threading.Lock,
    t0_s: float,
    ledger: ReservationLedger,
    queue: RuntimeReleaseQueue,
    lease_reader: LeaseTraceReader,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    out_dir: Path = legacy.OUT
    pending = {job["episode_id"] for job in jobs}
    deadlines = {job["episode_id"]: float(job["deadline_ms"]) for job in jobs}
    blocks = {job["episode_id"]: int(job["required_blocks"]) for job in jobs}
    blackouts = merge_blackouts(deadlines, 110.0)
    decisions: list[dict[str, Any]] = []
    release_rows: list[dict[str, Any]] = []
    capacity_events: list[dict[str, Any]] = []
    promotions: dict[str, dict[str, Any]] = {}
    pool = ThreadPoolExecutor(max_workers=len(jobs))

    def drain_and_record() -> None:
        now_ns = time.monotonic_ns()
        new_rows = drain_runtime_releases(queue, states, lease_reader, now_ns)
        for row in new_rows:
            release_rows.append(row)
            decisions.append(
                {
                    "episode_id": row["episode_id"],
                    "decision": row["event"],
                    "physical_release_ms": row["physical_release_ms"],
                    "logical_release_ms": row["logical_release_ms"],
                    "reserved_blocks": row["reserved_blocks_after"],
                }
            )
        if new_rows:
            _append_decision(out_dir, decisions)

    def harvest() -> None:
        before = len(capacity_events)
        new_rows = harvest_completed_promotions(
            promotions,
            states,
            ledger,
            capacity_events,
            now_monotonic=time.monotonic(),
        )
        for row in new_rows:
            result = promotions[row["episode_id"]].get("result")
            if result is not None:
                CURRENT_CAPACITY_RESULTS[str(promotions[row["episode_id"]]["lease_key"])] = dict(result)
            decisions.append(dict(row))
            if row["event"] == "PROMOTION_PHYSICALLY_ADMITTED":
                item = promotions[row["episode_id"]]
                check_native_result(
                    item["result"],
                    int(data[row["episode_id"]]["job"]["required_prefix_tokens"]),
                )
            elif row["event"] == "CAPACITY_REJECT_RELEASE":
                if not validate_capacity_rejection(item_result(promotions[row["episode_id"]])):
                    raise RuntimeError(
                        f"FAST_REJECT_INVALID:{row['episode_id']}"
                    )
        if len(capacity_events) != before:
            _append_decision(out_dir, decisions)

    try:
        while pending:
            harvest()
            drain_and_record()
            now_ms = (time.monotonic() - t0_s) * 1000.0
            earliest = min(pending, key=lambda episode: (deadlines[episode], episode))
            target = latest_feasible_start(
                now_ms,
                [(episode, deadlines[episode]) for episode in pending],
                blackouts,
                FROZEN_P_SAFE_MS,
            )
            if target > now_ms + 0.5:
                time.sleep(min(0.01, (target - now_ms) / 1000.0))
                harvest()
                drain_and_record()
                continue

            with lock:
                stale = states[earliest].get("resume_started_ns") is not None
                if not stale:
                    admitted = ledger.reserve(earliest, blocks[earliest])
                else:
                    admitted = False
                if stale:
                    states[earliest]["promoted"] = False
                    states[earliest]["leased"] = False
            if stale:
                decisions.append(
                    {
                        "episode_id": earliest,
                        "decision": "STALE_PROMOTION_CANCELLED",
                        "target_start_ms": target,
                        "reason": "ACTUAL_RESUME_PRECEDED_PREFETCH_COMMIT",
                    }
                )
                _append_decision(out_dir, decisions)
                pending.remove(earliest)
                continue
            if not admitted:
                decisions.append(
                    {
                        "episode_id": earliest,
                        "decision": "RESERVATION_SKIP",
                        "target_start_ms": target,
                        "required_blocks": blocks[earliest],
                        "reserved_blocks": ledger.reserved_blocks,
                        "budget_blocks": ledger.budget_blocks,
                    }
                )
                states[earliest]["promoted"] = False
                states[earliest]["leased"] = False
                _append_decision(out_dir, decisions)
                pending.remove(earliest)
                continue

            lease_key = data[earliest]["job"]["lease_key"]
            item: dict[str, Any] = {
                "target_start_ms": target,
                "lease_key": lease_key,
                "required_blocks": blocks[earliest],
                "submitted_monotonic_s": time.monotonic(),
                "harvested": False,
            }

            def issue_one(episode_id: str = earliest, lease: str = lease_key) -> dict[str, Any]:
                response = capacity_prefetch(
                    data[episode_id]["prefix"],
                    required_blocks=blocks[episode_id],
                    lease_key=lease,
                    timeout_s=PREFETCH_WATCHDOG_SECONDS,
                )
                response["sent_monotonic_s"] = int(response["client_issue_ns"]) / 1e9
                response["completed_monotonic_s"] = int(response["client_return_ns"]) / 1e9
                response["client_e2e_ms"] = float(response["client_response_ms"])
                return response

            with lock:
                states[earliest]["promoted"] = True
                states[earliest]["leased"] = False
                item["future"] = pool.submit(issue_one)
            promotions[earliest] = item
            decisions.append(
                {
                    "episode_id": earliest,
                    "decision": "RESERVATION_PROMOTE",
                    "target_start_ms": target,
                    "required_blocks": blocks[earliest],
                    "reserved_blocks": ledger.reserved_blocks,
                    "budget_blocks": ledger.budget_blocks,
                    "lease_key": lease_key,
                }
            )
            _append_decision(out_dir, decisions)
            pending.remove(earliest)

        while any(not item["future"].done() for item in promotions.values()):
            harvest()
            drain_and_record()
            now = time.monotonic()
            for episode_id, item in promotions.items():
                if (
                    not item.get("harvested")
                    and not item["future"].done()
                    and now - float(item["submitted_monotonic_s"])
                    >= PREFETCH_WATCHDOG_SECONDS
                ):
                    raise TimeoutError(f"PREFETCH_NONPROGRESS:{episode_id}")
            time.sleep(0.01)
        harvest()
        drain_and_record()
        for episode_id, item in promotions.items():
            if not item.get("harvested"):
                raise RuntimeError(f"pending guarded promotion was not harvested: {episode_id}")
        return promotions, decisions, release_rows
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def item_result(item: dict[str, Any]) -> dict[str, Any]:
    return item.get("result") or item["future"].result()


def _copy_run_artifacts(run_dir: Path, run_name: str, stage: str, repetition: int) -> dict[str, Any]:
    detail_path = run_dir / "release_fix_integration.json"
    if not detail_path.exists():
        raise RuntimeError("integration harness did not persist its result")
    detail = json.loads(detail_path.read_text(encoding="utf-8"))
    run_metrics = {
        key: value
        for key, value in detail.items()
        if key not in {
            "resume_rows",
            "native_rows",
            "reservation_decisions",
            "runtime_release_events",
            "lease_events",
            "scheduler_trace",
            "service_envelope",
        }
    }
    run_metrics.update({"run_name": run_name, "stage": stage, "repetition": repetition})

    def read_rows(name: str) -> list[dict[str, Any]]:
        path = run_dir / name
        if not path.exists():
            return []
        with path.open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    for source, dest in (
        ("session_metrics.csv", "session_metrics.csv"),
        ("prefetch_metrics.csv", "prefetch_metrics.csv"),
        ("reservation_decisions.csv", "reservation_decisions.csv"),
        ("lease_events.csv", "lease_events.csv"),
        ("scheduler_trace.csv", "scheduler_trace.csv"),
        ("runtime_release_events.csv", "reservation_events.csv"),
    ):
        rows = read_rows(source)
        if source == "prefetch_metrics.csv":
            response_fields = (
                "reason", "required_blocks", "free_blocks", "total_blocks",
                "external_tokens_loaded", "transfer_bytes", "model_forward_count",
                "generated_token_count", "mixed_prefetch_model_step_count",
            )
            for row in rows:
                endpoint = CURRENT_CAPACITY_RESULTS.get(str(row.get("lease_key")), {})
                for field in response_fields:
                    if field in endpoint:
                        row[field] = endpoint[field]
            legacy.write_csv(run_dir / source, rows)
        for row in rows:
            row.update({"run_name": run_name, "stage": stage, "repetition": repetition})
        root_file = OUT / dest
        existing = []
        if root_file.exists():
            with root_file.open(encoding="utf-8", newline="") as stream:
                existing = list(csv.DictReader(stream))
        legacy.write_csv(root_file, existing + rows)

    prefetch_rows = read_rows("prefetch_metrics.csv")
    issue_rows = []
    for row in prefetch_rows:
        timestamps = {}
        try:
            timestamps = json.loads(row.get("timestamps_json", "{}"))
        except (TypeError, json.JSONDecodeError):
            pass
        issue_rows.append(row)
    cap_events = parse_capacity_trace(
        "\n".join(
            line
            for path in (run_dir / "server_logs").glob("*.log")
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
            if "SLACKKV_CAPACITY_TRACE" in line
        )
    )
    capacity_rows = []
    for event in cap_events:
        key = event.get("lease_key")
        corresponding = next((row for row in prefetch_rows if row.get("lease_key") == key), {})
        capacity_rows.append(
            {
                **event,
                "stage": stage,
                "repetition": repetition,
                "run_name": run_name,
                "episode_id": str(key or "").split(":turn:", 1)[0],
                "endpoint_status": corresponding.get("status", "NA"),
                "response_ms": corresponding.get("client_e2e_ms", "NA"),
            }
        )
    capacity_path = OUT / "capacity_trace.csv"
    existing_capacity = []
    if capacity_path.exists():
        with capacity_path.open(encoding="utf-8", newline="") as stream:
            existing_capacity = list(csv.DictReader(stream))
    legacy.write_csv(capacity_path, existing_capacity + capacity_rows)
    timeline = []
    for row in capacity_rows:
        timeline.append({**row, "timeline_event": "CAPACITY_" + str(row["decision"])})
    for row in read_rows("lease_events.csv"):
        timeline.append(
            {
                "stage": stage,
                "repetition": repetition,
                "run_name": run_name,
                "episode_id": str(row.get("lease_key", "")).split(":turn:", 1)[0],
                "timestamp_monotonic": row.get("timestamp_monotonic"),
                "timeline_event": row.get("event"),
                "required_blocks": row.get("num_blocks"),
                "lease_key": row.get("lease_key"),
            }
        )
    for row in read_rows("scheduler_trace.csv"):
        if row.get("event") in {
            "RESUME_ADMITTED",
            "FIRST_TOKEN",
            "REQUEST_FINISHED",
            "WAITING_FOR_REMOTE_KVS",
        }:
            timeline.append(
                {
                    "stage": stage,
                    "repetition": repetition,
                    "run_name": run_name,
                    "episode_id": str(row.get("session", "")).split(":turn:", 1)[0],
                    "timestamp_monotonic": row.get("timestamp_monotonic_raw"),
                    "timeline_event": row.get("event"),
                    "request_id": row.get("request_id"),
                }
            )
    timeline_path = OUT / "capacity_timeline.csv"
    existing_timeline = []
    if timeline_path.exists():
        with timeline_path.open(encoding="utf-8", newline="") as stream:
            existing_timeline = list(csv.DictReader(stream))
    legacy.write_csv(timeline_path, existing_timeline + timeline)

    return {
        "detail": detail,
        "run_metrics": run_metrics,
        "session_rows": read_rows("session_metrics.csv"),
        "prefetch_rows": prefetch_rows,
        "trace_rows": read_rows("scheduler_trace.csv"),
        "lease_rows": read_rows("lease_events.csv"),
        "capacity_events": cap_events,
    }


def run_one(stage: str, repetition: int, jobs: list[dict[str, Any]], data: dict[str, dict[str, Any]]) -> dict[str, Any]:
    run_name = f"{stage.lower()}_rep{repetition}"
    run_dir = OUT / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    CURRENT_CAPACITY_RESULTS.clear()
    run_context: dict[str, Any] = {"t0_ns": None, "process": None, "log": None, "healthy_before_shutdown": False, "exit_code": None, "stop_seen": False}

    original_out = legacy.OUT
    original_schedule = legacy.schedule_promotions
    original_drain = legacy.drain_releases
    original_queue = legacy.RuntimeReleaseQueue
    original_start_background = legacy.start_background
    original_start_server = runtime.start_server
    original_stop_server = runtime.stop_server

    def start_server(label: str, **kwargs: Any):
        actual_label = f"{label}_{run_name}"
        proc, log = original_start_server(actual_label, **kwargs)
        run_context["process"] = proc
        run_context["log"] = log
        run_context["actual_label"] = actual_label
        return proc, log

    def stop_server(proc, log_path: Path, out: Path, label: str) -> int:
        if proc is run_context["process"] and not run_context["stop_seen"]:
            try:
                with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=0.5):
                    run_context["healthy_before_shutdown"] = True
            except Exception:
                run_context["healthy_before_shutdown"] = False
            run_context["stop_seen"] = True
            actual_label = run_context["actual_label"]
        else:
            actual_label = label
        code = original_stop_server(proc, log_path, out, actual_label)
        run_context["exit_code"] = code
        return code

    def start_background(*args: Any, **kwargs: Any):
        result = original_start_background(*args, **kwargs)
        run_context["t0_ns"] = int(result[2]["first_ns"])
        return result

    legacy.OUT = run_dir
    legacy.schedule_promotions = schedule_promotions
    legacy.drain_releases = drain_runtime_releases
    legacy.RuntimeReleaseQueue = FinalDrainReleaseQueue
    legacy.start_background = start_background
    runtime.start_server = start_server
    runtime.stop_server = stop_server

    result_box: dict[str, Any] = {}
    error_box: dict[str, BaseException] = {}

    def run_worker() -> None:
        try:
            result_box["detail"] = legacy.run_release_fix(jobs, data)
        except BaseException as exc:
            error_box["error"] = exc

    worker = threading.Thread(target=run_worker, name=f"phase4d-b-{run_name}", daemon=True)
    worker.start()
    overall_deadline = time.monotonic() + 900
    watchdog_expired = False
    while worker.is_alive():
        t0_ns = run_context["t0_ns"]
        if t0_ns is not None and (time.monotonic_ns() - t0_ns) / 1e9 >= WATCHDOG_SECONDS:
            watchdog_expired = True
            proc = run_context["process"]
            if proc is not None and proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
            break
        if time.monotonic() >= overall_deadline:
            watchdog_expired = True
            proc = run_context["process"]
            if proc is not None and proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
            break
        time.sleep(0.1)
    worker.join(timeout=45)
    runtime.start_server = original_start_server
    runtime.stop_server = original_stop_server
    legacy.OUT = original_out
    legacy.schedule_promotions = original_schedule
    legacy.drain_releases = original_drain
    legacy.RuntimeReleaseQueue = original_queue
    legacy.start_background = original_start_background

    log_path = run_context.get("log")
    raw_log = ""
    if log_path and Path(log_path).exists():
        raw_log = Path(log_path).read_text(encoding="utf-8", errors="replace")
    oom = any(marker in raw_log.lower() for marker in ("cuda out of memory", "outofmemoryerror"))
    crash = "EngineCore encountered a fatal error" in raw_log or "EngineCore.*exited" in raw_log
    healthy = bool(run_context["healthy_before_shutdown"])
    if watchdog_expired or "detail" not in result_box:
        row = {
            "stage": stage,
            "repetition": repetition,
            "run_name": run_name,
            "valid": False,
            "watchdog_expired": watchdog_expired,
            "server_alive_before_shutdown": healthy,
            "OOM": oom,
            "server_crash": crash,
            "error": str(error_box.get("error", "watchdog expired or integration raised")),
            "resume_completed_count": 0,
            "capacity_reject_count": 0,
            "pending_guarded_prefetch": "NA",
        }
        legacy.write_csv(
            OUT / "integration_runs.csv",
            _append_existing(OUT / "integration_runs.csv", row),
        )
        _keep_compact_log(run_dir, raw_log, run_name)
        return row

    detail = result_box["detail"]
    artifacts = _copy_run_artifacts(run_dir, run_name, stage, repetition)
    capacity_events = artifacts["capacity_events"]
    prefetch_rows = artifacts["prefetch_rows"]
    event_by_key = {str(event.get("lease_key")): event for event in capacity_events}
    capacity_valid = all(capacity_invariant(event) for event in capacity_events)
    status_by_key = {str(row.get("lease_key")): row for row in prefetch_rows}
    all_promotions_valid = True
    for key, event in event_by_key.items():
        response = status_by_key.get(key, {})
        if event.get("decision") == "ADMIT":
            all_promotions_valid &= response.get("status") == "ok"
        elif event.get("decision") == "REJECT":
            all_promotions_valid &= (
                response.get("status") == "capacity_rejected"
                and int(float(response.get("transfer_bytes", 0))) == 0
                and int(float(response.get("external_tokens_loaded", 0))) == 0
                and int(float(response.get("model_forward_count", 0))) == 0
                and int(float(response.get("generated_token_count", 0))) == 0
            )
    rejected = [
        row for row in prefetch_rows if row.get("status") == "capacity_rejected"
    ]
    rejected_events = [event for event in capacity_events if event.get("decision") == "REJECT"]
    trace_rows = artifacts["trace_rows"]
    lease_rows = artifacts["lease_rows"]
    connector_only_lines = [line for line in raw_log.splitlines() if "SLACKKV_CONNECTOR_ONLY_STEP" in line]
    rejected_lifecycle_valid = (
        len(rejected) == len(rejected_events)
        and all(
            rejected_request_lifecycle_valid(
                event,
                status_by_key.get(str(event.get("lease_key")), {}),
                trace_rows,
                lease_rows,
                connector_only_lines,
            )
            for event in rejected_events
        )
    )
    tokens_ok = len(artifacts["session_rows"]) == 8 and all(
        int(float(row.get("generated_token_count", -1)))
        == int(float(row.get("resume_osl", -2)))
        for row in artifacts["session_rows"]
    )
    source_ok = bool(detail.get("source_conservation"))
    k2_ok = int(detail.get("K2_max_pre_first", 999)) <= 2
    lease_leaks = int(detail.get("physical_lease_leak_count", 999))
    logical_leaks = int(detail.get("logical_reservation_leak_count", 999))
    overcommit = bool(detail.get("reservation_overcommit")) or bool(detail.get("lease_overcommit"))
    release_ok = (
        int(detail.get("unsafe_release_count", 999)) == 0
        and bool(detail.get("physical_before_logical_valid"))
        and lease_leaks == 0
        and logical_leaks == 0
    )
    finished = int(detail.get("resume_completed_count", 0)) == 8
    completed_within_watchdog = False
    t0_ns = run_context["t0_ns"]
    if t0_ns is not None and artifacts["session_rows"]:
        finish_values = []
        for row in artifacts["session_rows"]:
            try:
                finish_values.append(float(row["request_finished_ms"]))
            except (KeyError, TypeError, ValueError):
                pass
        completed_within_watchdog = len(finish_values) == 8 and max(finish_values) <= WATCHDOG_SECONDS * 1000
    x_active = bool(detail.get("X_active_through_last_first_token"))
    valid = (
        finished
        and completed_within_watchdog
        and tokens_ok
        and source_ok
        and rejected_lifecycle_valid
        and k2_ok
        and not overcommit
        and release_ok
        and all_promotions_valid
        and capacity_valid
        and x_active
        and healthy
        and not oom
        and not crash
        and not watchdog_expired
        and all(float(row.get("client_e2e_ms", 0)) < PREFETCH_WATCHDOG_SECONDS * 1000 for row in prefetch_rows)
    )
    row = {
        "stage": stage,
        "repetition": repetition,
        "run_name": run_name,
        "valid": valid,
        "watchdog_expired": watchdog_expired,
        "server_alive_before_shutdown": healthy,
        "OOM": oom,
        "server_crash": crash,
        "resume_completed_count": detail.get("resume_completed_count"),
        "token_correct_count": detail.get("resume_correct_count"),
        "source_conservation": source_ok,
        "K2_max_pre_first": detail.get("K2_max_pre_first"),
        "reservation_peak_blocks": detail.get("reservation_peak_blocks"),
        "physical_lease_peak_blocks": detail.get("physical_lease_peak_blocks"),
        "logical_leaks": logical_leaks,
        "lease_leaks": lease_leaks,
        "rejected_lifecycle_valid": rejected_lifecycle_valid,
        "rejected_request_no_wait": all(not any(r.get("request_id") == e.get("request_id") and r.get("event") == "WAITING_FOR_REMOTE_KVS" for r in trace_rows) for e in rejected_events),
        "rejected_request_no_lease": all(not any(r.get("request_id") == e.get("request_id") and r.get("event") == "LEASE_ACQUIRED" for r in lease_rows) for e in rejected_events),
        "rejected_request_no_connector_step": all(not any(str(e.get("request_id")) in line for line in connector_only_lines) for e in rejected_events),
        "unsafe_release_count": detail.get("unsafe_release_count"),
        "capacity_attempt_count": len(capacity_events),
        "capacity_admit_count": sum(e.get("decision") == "ADMIT" for e in capacity_events),
        "capacity_reject_count": sum(e.get("decision") == "REJECT" for e in capacity_events),
        "capacity_invariant_valid": capacity_valid,
        "native_promotion_valid": all_promotions_valid,
        "post_first_release_valid": release_ok,
        "x_active_through_last_first_token": x_active,
        "completed_within_30s": completed_within_watchdog,
        "pending_guarded_prefetch": 0,
        "error": "",
    }
    legacy.write_csv(
        OUT / "integration_runs.csv",
        _append_existing(OUT / "integration_runs.csv", row),
    )
    issue_to_t6 = [
        float(item["issue_to_T6_ms"])
        for item in artifacts["prefetch_rows"]
        if item.get("status") == "ok"
        and item.get("issue_to_T6_ms") not in (None, "", "NA")
    ]
    audit = service_envelope(issue_to_t6, FROZEN_P_SAFE_MS) if issue_to_t6 else {
        "promotion_count": 0,
        "overrun_count": 0,
        "budget_ms": FROZEN_P_SAFE_MS,
        "status": "NO_SUCCESSFUL_PROMOTIONS",
    }
    _append_json_list(OUT / "service_envelope_audit.json", stage, repetition, audit)
    _keep_compact_log(run_dir, raw_log, run_name)
    return row


def _append_existing(path: Path, row: dict[str, Any]) -> list[dict[str, Any]]:
    existing = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as stream:
            existing = list(csv.DictReader(stream))
    return existing + [row]


def _append_json_list(path: Path, stage: str, repetition: int, record: dict[str, Any]) -> None:
    existing: list[dict[str, Any]] = []
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
    existing.append({"stage": stage, "repetition": repetition, **record})
    legacy.write_json(path, existing)


def _keep_compact_log(run_dir: Path, raw: str, run_name: str) -> None:
    marker_terms = (
        "SLACKKV_CAPACITY_TRACE",
        "SLACKKV_LEASE_TRACE",
        "SLACKKV_RESUME_TRACE",
        "SLACKKV_NATIVE_PREFETCH_COMPLETE",
        "SLACKKV_CONNECTOR_ONLY_STEP",
        "ERROR",
        "OOM",
    )
    log_dir = run_dir / "server_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    compact = [line[:1200] for line in raw.splitlines() if any(term in line for term in marker_terms)]
    (log_dir / f"{run_name}_diagnostic.log").write_text("\n".join(compact[-1000:]) + "\n", encoding="utf-8")
    for path in log_dir.glob("*.log"):
        if not path.name.endswith("_compact.log") and path.name != f"{run_name}_diagnostic.log":
            path.unlink(missing_ok=True)


def run_g2(repeats: int = 3) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    jobs = frozen_jobs()
    data = reconstructed_data(jobs)
    records = []
    smoke = run_one("G2-S", 0, jobs, data)
    records.append(smoke)
    if not smoke.get("valid"):
        return {"smoke": smoke, "repetitions": [], "pass": False}
    for repetition in range(1, repeats + 1):
        row = run_one("G2-R", repetition, jobs, data)
        records.append(row)
        if not row.get("valid"):
            return {"smoke": smoke, "repetitions": records[1:], "pass": False}
    return {"smoke": smoke, "repetitions": records[1:], "pass": all(r["valid"] for r in records)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["g2"], required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    report = run_g2(args.repeats)
    legacy.write_json(OUT / "g2_report.json", report)
    if not report["pass"]:
        raise SystemExit(1)
