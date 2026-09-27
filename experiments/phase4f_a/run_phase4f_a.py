from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import subprocess
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from experiments.phase2d.helpers import source_tokens_delta
from experiments.phase3f.scheduler import latest_feasible_start, merge_blackouts
from experiments.phase4d_a import runtime
from experiments.phase4d_a.experiment import (
    materialize_and_evict, source_snapshot, write_csv, write_json,
)
from experiments.phase4d_a.helpers import ReservationLedger
from experiments.phase4d_a.run_policy import stream_resume
from experiments.phase4d_a_fix.release_fix import LeaseTraceReader, RuntimeReleaseQueue
from experiments.phase4d_b.experiment import reconstructed_data, validate_runtime
from experiments.phase4e_a_fix.run_phase4e_a_fix import warmup, start_background_512
from experiments.phase4e_a_rca.analysis import parse_structured_log
from experiments.phase4f_a.logic import (
    BUDGET_BLOCKS, EXPECTED_PROGRESS, GPU_BLOCKS, K, P_SAFE_MS,
    WORKLOAD_SHA, progress_blocks, protected_episode, psa_trace_valid,
    rejected_response_valid, verified_jobs,
)
from experiments.phase4f_a.progress_client import progress_prefetch
from experiments.phase4d_a.helpers import sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/phase4f_a"
START_COMMIT = "5e94128bb69e13bf3c7772ed1e5de32f56f0701b"
WAITING_MARKER = "SLACKKV_WAITING_TRACE"
RESUME_MARKER = "SLACKKV_RESUME_TRACE"
LEASE_MARKER = "SLACKKV_LEASE_TRACE"
CAPACITY_MARKER = "SLACKKV_CAPACITY_TRACE"


def append_csv(path: Path, new_rows: list[dict[str, Any]]) -> None:
    if not new_rows:
        return
    old: list[dict[str, Any]] = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as stream:
            old = list(csv.DictReader(stream))
    write_csv(path, old + new_rows)


def events(log: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        "waiting": parse_structured_log(log, WAITING_MARKER),
        "resume": parse_structured_log(log, RESUME_MARKER),
        "lease": parse_structured_log(log, LEASE_MARKER),
        "capacity": parse_structured_log(log, CAPACITY_MARKER),
    }


def unique_resume_counts(trace: list[dict[str, Any]]) -> dict[str, int]:
    """Count requests, not scheduler admission events (a preempted request can re-enter)."""
    names = {"RESUME_ARRIVED": "resume_arrived",
             "RESUME_ADMITTED": "resume_admitted",
             "FIRST_TOKEN": "first_tokens",
             "REQUEST_FINISHED": "finished"}
    return {column: len({row["request_id"] for row in trace
                         if row.get("event") == event})
            for event, column in names.items()}


def compact_log(log: Path, label: str) -> None:
    if not log.exists():
        return
    markers = (WAITING_MARKER, RESUME_MARKER, LEASE_MARKER, CAPACITY_MARKER,
               "GPU KV cache size", "ERROR", "OOM")
    selected = [line[:2500] for line in log.read_text(encoding="utf-8", errors="replace").splitlines()
                if any(marker in line for marker in markers)]
    target = OUT / "server_logs"
    target.mkdir(parents=True, exist_ok=True)
    (target / f"{label}_compact.txt").write_text("\n".join(selected[-1500:]) + "\n", encoding="utf-8")
    log.unlink()


def prefetch_record(
    episode: str, result: dict[str, Any], job: dict[str, Any],
    *, progress_guard: bool,
) -> dict[str, Any]:
    status = result["status"]
    if status == "ok":
        if int(result.get("external_tokens_loaded", -1)) != int(job["required_prefix_tokens"]):
            raise RuntimeError(f"native token mismatch:{episode}")
        if int(result.get("transfer_bytes", 0)) <= 0:
            raise RuntimeError(f"native transfer missing:{episode}")
        if "T6" not in (result.get("phase3m_timestamps") or {}):
            raise RuntimeError(f"native T6 missing:{episode}")
        if any(int(result.get(field, -1)) != 0 for field in (
            "model_forward_count", "generated_token_count", "mixed_prefetch_model_step_count"
        )):
            raise RuntimeError(f"native model work:{episode}")
    elif status in ("capacity_rejected", "progress_rejected"):
        if not rejected_response_valid(result, status):
            raise RuntimeError(f"invalid rejection:{episode}:{status}")
    else:
        raise RuntimeError(f"unknown prefetch status:{status}")
    return {
        "episode_id": episode, "status": status,
        "progress_guard": progress_guard,
        "required_blocks": int(job["required_blocks"]),
        "transfer_bytes": int(result.get("transfer_bytes", 0)),
        "external_tokens_loaded": int(result.get("external_tokens_loaded", 0)),
        "client_response_ms": result["client_response_ms"],
        "client_issue_ns": result["client_issue_ns"],
        "client_return_ns": result["client_return_ns"],
    }


def save_event_artifacts(stage: str, repetition: int, policy: str, snapshot: dict[str, list[dict[str, Any]]], t0_s: float) -> None:
    for key, filename in (("waiting", "waiting_trace.csv"), ("resume", "resume_trace.csv"),
                          ("lease", "lease_events.csv"), ("capacity", "capacity_trace.csv")):
        append_csv(OUT / filename, [
            {"stage": stage, "repetition": repetition, "policy": policy,
             "timestamp_ms": (float(event["timestamp_monotonic"]) - t0_s) * 1000,
             **event}
            for event in snapshot[key]
        ])


def progress_admission_rows(
    stage: str, repetition: int, policy: str,
    capacities: list[dict[str, Any]], jobs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_key = {job["lease_key"]: job for job in jobs}
    output = []
    for event in capacities:
        if event.get("progress_guard") is not True:
            continue
        candidate = by_key.get(event.get("lease_key"))
        protected = by_key.get(event.get("protected_lease_key"))
        output.append({
            "stage": stage, "repetition": repetition, "policy": policy,
            "candidate_episode": candidate["episode_id"] if candidate else "NA",
            "candidate_lease_key": event.get("lease_key"),
            "protected_episode": protected["episode_id"] if protected else "NA",
            "protected_lease_key": event.get("protected_lease_key"),
            "protected_deadline_ms": protected["deadline_ms"] if protected else "NA",
            "protected_progress_blocks": event.get("protected_progress_blocks"),
            "existing_protected_lease_blocks": event.get("protected_existing_lease_blocks"),
            "hypothetical_protected_lease_blocks": event.get("protected_hypothetical_lease_blocks"),
            "progress_reserve_blocks": event.get("progress_reserve_blocks"),
            "free_blocks_before": event.get("free_blocks_before"),
            "candidate_required_blocks": event.get("candidate_required_blocks"),
            "free_blocks_after_candidate": event.get("free_blocks_after_candidate"),
            "decision": event.get("decision"),
        })
    return output


def verify_snapshot(snapshot: dict[str, list[dict[str, Any]]], records: list[dict[str, Any]]) -> int:
    capacity = snapshot["capacity"]
    psa = [event for event in capacity if event.get("progress_guard") is True]
    violations = sum(not psa_trace_valid(event) for event in psa)
    if violations:
        raise RuntimeError(f"PSA_TRACE_INVARIANT_FAIL:{violations}")
    if len(capacity) != len(records):
        raise RuntimeError(f"capacity trace/endpoint count mismatch:{len(capacity)}:{len(records)}")
    rejected_keys = {event["lease_key"] for event in capacity
                     if event["decision"] in ("PROGRESS_REJECT", "CAPACITY_REJECT", "REJECT")}
    if any(event.get("event") == "LEASE_ACQUIRED" and event.get("lease_key") in rejected_keys
           for event in snapshot["lease"]):
        raise RuntimeError("rejected promotion acquired lease")
    return violations


def precondition_setup(label: str, jobs, data):
    run_dir = OUT / "runs" / label
    run_dir.mkdir(parents=True, exist_ok=True)
    proc, log = runtime.start_server(label, out=run_dir)
    _, block_size = runtime.wait_ready(proc)
    capacity = validate_runtime(log, block_size)
    if capacity["total_gpu_kv_blocks"] != GPU_BLOCKS:
        raise RuntimeError("EXPERIMENT_INVALID_ENVIRONMENT:GPU blocks")
    warmup(label)
    prepared = materialize_and_evict(proc, log, label, jobs, data, run_dir)
    if not prepared["pressure"]["capacity_exceeded"] or any(
        not row["validation"] for row in prepared["materialization"]
    ):
        raise RuntimeError("RUN_INVALID_PRECONDITION:materialization/pressure")
    return proc, log, run_dir


def sequential_attempts(
    attempts: list[tuple[str, bool]], jobs: list[dict[str, Any]],
    data: dict[str, dict[str, Any]], log: Path,
) -> list[dict[str, Any]]:
    records = []
    states = {job["episode_id"]: {"first_token_ns": None} for job in jobs}
    protected = protected_episode(jobs, states)
    if protected != "E04":
        raise RuntimeError("frozen protected frontier is not E04")
    for episode, enabled in attempts:
        job = data[episode]["job"]
        protected_job = data[protected]["job"]
        result = progress_prefetch(
            data[episode]["prefix"],
            required_blocks=int(job["required_blocks"]),
            lease_key=job["lease_key"],
            protected_lease_key=protected_job["lease_key"],
            protected_progress_blocks=progress_blocks(int(protected_job["turn1_isl"])),
            progress_guard=enabled,
        )
        record = prefetch_record(episode, result, job, progress_guard=enabled)
        if record["status"] == "ok":
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                lease = events(log)["lease"]
                required = ("LEASE_ACQUIRED", "LEASE_HELD_AFTER_PREFETCH_FREE")
                if all(any(event.get("event") == name
                           and event.get("lease_key") == job["lease_key"]
                           and int(event.get("num_blocks", -1)) == int(job["required_blocks"])
                           for event in lease) for name in required):
                    break
                time.sleep(0.02)
            else:
                raise RuntimeError(f"lease precondition missing:{episode}")
        records.append(record)
    return records


def run_micro(
    name: str, attempts: list[tuple[str, bool]],
    expected_statuses: list[str], jobs, data,
) -> dict[str, Any]:
    label = f"P0_{name}"
    proc, log, run_dir = precondition_setup(label, jobs, data)
    t0_s = time.monotonic()
    try:
        records = sequential_attempts(attempts, jobs, data, log)
        snapshot = events(log)
        violations = verify_snapshot(snapshot, records)
        actual = [record["status"] for record in records]
        valid = actual == expected_statuses and proc.poll() is None
        row = {"stage": "P0", "repetition": 1, "policy": name,
               "attempted": "|".join(e for e, _ in attempts),
               "statuses": "|".join(actual), "expected": "|".join(expected_statuses),
               "PSA_invariant_violations": violations, "valid": valid}
        append_csv(OUT / "micro_runs.csv", [row])
        append_csv(OUT / "prefetch_metrics.csv", [{"stage": "P0", "repetition": 1,
                    "policy": name, **record} for record in records])
        save_event_artifacts("P0", 1, name, snapshot, t0_s)
        append_csv(OUT / "progress_admission.csv", progress_admission_rows(
            "P0", 1, name, snapshot["capacity"], jobs))
        if not valid:
            raise RuntimeError(f"P0 micro failed:{name}:{row}")
        return row
    finally:
        if proc.poll() is None:
            runtime.stop_server(proc, log, run_dir, label)
        compact_log(log, label)


def run_jit(
    stage: str, repetition: int, *, logical_reservation: bool,
    jobs: list[dict[str, Any]], data: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    policy = "FULL_PSA1" if logical_reservation else "GUARDED_LEASE_PSA1"
    label = f"{stage}_{policy}_rep{repetition}"
    proc, log, run_dir = precondition_setup(label, jobs, data)
    xpool: ThreadPoolExecutor | None = None
    resume_pool: ThreadPoolExecutor | None = None
    promotion_pool: ThreadPoolExecutor | None = None
    try:
        source_before = source_snapshot()
        states = {job["episode_id"]: {"resume_started_ns": None,
                  "first_token_ns": None, "leased": False, "promoted": False}
                  for job in jobs}
        lock = threading.Lock()
        xpool, xfuture, xmeta = start_background_512()
        t0_s = int(xmeta["first_ns"]) / 1e9
        resume_pool = ThreadPoolExecutor(max_workers=8)
        resumes = {episode: resume_pool.submit(
            stream_resume, entry["job"]["lease_key"], entry, t0_s,
            float(entry["job"]["deadline_ms"]), True, states, lock)
            for episode, entry in data.items()}
        promotion_pool = ThreadPoolExecutor(max_workers=8)
        ledger = ReservationLedger(BUDGET_BLOCKS) if logical_reservation else None
        keys = {job["episode_id"]: job["lease_key"] for job in jobs}
        blocks = {job["episode_id"]: int(job["required_blocks"]) for job in jobs}
        release_queue = RuntimeReleaseQueue(ledger, keys, blocks, t0_s) if ledger else None
        lease_reader = LeaseTraceReader(log) if ledger else None
        deadlines = {job["episode_id"]: float(job["deadline_ms"]) for job in jobs}
        blackouts = merge_blackouts(deadlines, 110.0)
        pending = set(deadlines)
        futures: dict[str, tuple[Future, float]] = {}
        records: list[dict[str, Any]] = []
        decisions: list[dict[str, Any]] = []
        releases: list[dict[str, Any]] = []

        def drain() -> None:
            if release_queue is None or lease_reader is None or ledger is None:
                return
            for episode in list(ledger.active):
                first = states[episode]["first_token_ns"]
                if first is not None:
                    release_queue.enqueue_first_token(episode, int(first))
            release_queue.observe_lease_events(lease_reader.read_events())
            releases.extend(release_queue.drain(time.monotonic_ns()))

        def harvest() -> None:
            for episode, (future, submitted) in list(futures.items()):
                if not future.done():
                    if time.monotonic() - submitted > 5.0:
                        raise RuntimeError(f"PREFETCH_NONPROGRESS:{episode}")
                    continue
                result = future.result()
                job = data[episode]["job"]
                record = prefetch_record(episode, result, job, progress_guard=True)
                records.append(record)
                status = record["status"]
                if status == "ok":
                    with lock:
                        states[episode]["leased"] = True
                        states[episode]["promoted"] = True
                elif ledger is not None:
                    before = ledger.reserved_blocks
                    ledger.release(episode)
                    decisions.append({"episode_id": episode,
                                      "decision": status.upper() + "_RELEASE",
                                      "reserved_before": before,
                                      "reserved_after": ledger.reserved_blocks})
                decisions.append({"episode_id": episode, "decision": status.upper(),
                                  "timestamp_monotonic": time.monotonic()})
                del futures[episode]

        while pending or futures:
            harvest()
            drain()
            if pending:
                now_ms = (time.monotonic() - t0_s) * 1000
                episode = min(pending, key=lambda item: (deadlines[item], item))
                target = latest_feasible_start(
                    now_ms, [(item, deadlines[item]) for item in pending],
                    blackouts, P_SAFE_MS)
                if target <= now_ms + 0.5:
                    pending.remove(episode)
                    with lock:
                        stale = states[episode]["resume_started_ns"] is not None
                        reserved = (not stale and
                                    (ledger.reserve(episode, blocks[episode])
                                     if ledger else True))
                        protected = protected_episode(jobs, states)
                    if stale or protected is None:
                        if ledger and reserved:
                            ledger.release(episode)
                        decisions.append({"episode_id": episode,
                                          "decision": "STALE_PROMOTION_CANCELLED"})
                    elif not reserved:
                        decisions.append({"episode_id": episode,
                                          "decision": "RESERVATION_SKIP"})
                    else:
                        protected_job = data[protected]["job"]
                        future = promotion_pool.submit(
                            progress_prefetch, data[episode]["prefix"],
                            required_blocks=blocks[episode],
                            lease_key=keys[episode],
                            protected_lease_key=keys[protected],
                            protected_progress_blocks=progress_blocks(
                                int(protected_job["turn1_isl"])),
                            timeout_s=5.0)
                        futures[episode] = (future, time.monotonic())
                        decisions.append({"episode_id": episode,
                                          "decision": "PROMOTION_ATTEMPT",
                                          "protected_episode": protected,
                                          "target_start_ms": target})
            time.sleep(0.005)
        responses = {episode: future.result(timeout=30)
                     for episode, future in resumes.items()}
        end = time.monotonic() + 5.0
        while ledger is not None and ledger.active and time.monotonic() < end:
            drain()
            time.sleep(0.01)
        x = xfuture.result(timeout=15)
        source_after = source_snapshot()
        local = int(round(source_tokens_delta(source_before, source_after, "local_cache_hit")))
        external = int(round(source_tokens_delta(source_before, source_after, "external_kv_transfer")))
        snapshot = events(log)
        verify_snapshot(snapshot, records)
        counts = unique_resume_counts(snapshot["resume"])
        max_pre = max((int(event.get("num_slackkv_pre_first_token") or 0)
                       for event in snapshot["resume"]), default=0)
        lease_acquired = [event for event in snapshot["lease"]
                          if event.get("event") == "LEASE_ACQUIRED"]
        lease_released = [event for event in snapshot["lease"]
                          if event.get("event") == "LEASE_RELEASED"]
        tail_ms = ((x["done_ns"] - max(row["first_token_ns"]
                    for row in responses.values())) / 1e6)
        valid = (len(responses) == 8
                 and all(row["generated_token_count"] == int(data[episode]["job"]["resume_osl"])
                         for episode, row in responses.items())
                 and counts == {"resume_arrived": 8, "resume_admitted": 8,
                                "first_tokens": 8, "finished": 8}
                 and local + external == 168016 and max_pre <= K
                 and len(x["tokens"]) == 512 and tail_ms > 0
                 and len(lease_acquired) == len(lease_released)
                 and (ledger is None or not ledger.active)
                 and (release_queue is None or release_queue.unsafe_release_count == 0)
                 and proc.poll() is None)
        row = {"stage": stage, "repetition": repetition, "policy": policy,
               "resume_arrived": counts["resume_arrived"],
               "resume_admitted": counts["resume_admitted"],
               "first_tokens": counts["first_tokens"], "finished": counts["finished"],
               "promotion_attempts": len(records),
               "promotion_admits": sum(r["status"] == "ok" for r in records),
               "capacity_rejects": sum(r["status"] == "capacity_rejected" for r in records),
               "progress_rejects": sum(r["status"] == "progress_rejected" for r in records),
               "reservation_skips": sum(d["decision"] == "RESERVATION_SKIP" for d in decisions),
               "lease_acquired": len(lease_acquired), "lease_released": len(lease_released),
               "logical_leaks": len(ledger.active) if ledger else 0,
               "unsafe_release": release_queue.unsafe_release_count if release_queue else 0,
               "local_source_tokens": local, "external_source_tokens": external,
               "source_conservation": local + external == 168016,
               "K2_max_pre_first": max_pre, "X_tail_margin_ms": tail_ms,
               "valid": valid}
        append_csv(OUT / "liveness_runs.csv", [row])
        append_csv(OUT / "prefetch_metrics.csv", [{"stage": stage, "repetition": repetition,
                   "policy": policy, **record} for record in records])
        append_csv(OUT / "reservation_events.csv", [{"stage": stage, "repetition": repetition,
                   "policy": policy, **decision} for decision in decisions + releases])
        save_event_artifacts(stage, repetition, policy, snapshot, t0_s)
        append_csv(OUT / "progress_admission.csv", progress_admission_rows(
            stage, repetition, policy, snapshot["capacity"], jobs))
        if not valid:
            raise RuntimeError(f"{stage} invalid:{row}")
        return row
    finally:
        if proc.poll() is None:
            runtime.stop_server(proc, log, run_dir, label)
        if promotion_pool is not None:
            promotion_pool.shutdown(wait=False, cancel_futures=True)
        if resume_pool is not None:
            resume_pool.shutdown(wait=False, cancel_futures=True)
        if xpool is not None:
            xpool.shutdown(wait=False, cancel_futures=True)
        compact_log(log, label)


def run_deterministic(
    stage: str, repetition: int, *, psa: bool, x_enabled: bool,
    selected: tuple[str, ...], expect_progress: bool,
    jobs: list[dict[str, Any]], data: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    policy = "PSA1" if psa else "PSA_OFF"
    label = f"{stage}_{policy}_rep{repetition}"
    proc, log, run_dir = precondition_setup(label, jobs, data)
    resume_pool: ThreadPoolExecutor | None = None
    xpool: ThreadPoolExecutor | None = None
    try:
        if x_enabled:
            xpool, xfuture, xmeta = start_background_512()
            if not xmeta.get("first_ns"):
                raise RuntimeError("X has no first token")
        attempts = [(episode, psa) for episode in selected]
        records = sequential_attempts(attempts, jobs, data, log)
        expected = (["ok", "ok", "progress_rejected"]
                    if psa and selected == ("E00", "E01", "E02")
                    else ["ok"] * len(selected))
        if [record["status"] for record in records] != expected:
            raise RuntimeError(f"{stage}:wrong sequential decisions:{records}")
        if x_enabled and xfuture.done():
            raise RuntimeError("X ended before resume_t0")
        source_before = source_snapshot()
        t0_s = time.monotonic()
        states = {job["episode_id"]: {
            "resume_started_ns": None, "first_token_ns": None,
            "leased": any(record["episode_id"] == job["episode_id"]
                          and record["status"] == "ok" for record in records),
        } for job in jobs}
        lock = threading.Lock()
        resume_pool = ThreadPoolExecutor(max_workers=8)
        futures = {episode: resume_pool.submit(
            stream_resume, entry["job"]["lease_key"], entry, t0_s,
            float(entry["job"]["deadline_ms"]), True, states, lock,
        ) for episode, entry in data.items()}
        wait(list(futures.values()), timeout=30 if expect_progress else 12)
        snapshot = events(log)
        counts = unique_resume_counts(snapshot["resume"])
        violations = verify_snapshot(snapshot, records)
        first_e04_failure = next((event for event in snapshot["waiting"]
                                  if event.get("event") == "WAITING_ALLOCATE_NONE"
                                  and event.get("session") == "E04:turn:1"), None)
        acquired = sum(event.get("event") == "LEASE_ACQUIRED" for event in snapshot["lease"])
        released = sum(event.get("event") == "LEASE_RELEASED" for event in snapshot["lease"])
        alive = proc.poll() is None
        text = log.read_text(encoding="utf-8", errors="replace")
        oom = "cuda out of memory" in text.lower() or "outofmemoryerror" in text.lower()
        server_error = " ERROR " in text
        max_pre = max((int(event.get("num_slackkv_pre_first_token") or 0)
                       for event in snapshot["resume"]), default=0)
        if expect_progress:
            complete = all(future.done() and future.exception() is None for future in futures.values())
            response_ok = complete and all(
                future.result()["generated_token_count"] == int(data[episode]["job"]["resume_osl"])
                for episode, future in futures.items()
            )
            source_after = source_snapshot()
            local = int(round(source_tokens_delta(source_before, source_after, "local_cache_hit")))
            external = int(round(source_tokens_delta(source_before, source_after, "external_kv_transfer")))
            source_valid = local + external == 168016
            valid = (complete and response_ok and counts == {
                "resume_arrived": 8, "resume_admitted": 8,
                "first_tokens": 8, "finished": 8,
            } and source_valid and max_pre <= K and acquired == released
                and alive and not oom and not server_error)
        else:
            local = external = "NA"
            source_valid = "NA"
            valid = (counts["resume_arrived"] == 8 and counts["resume_admitted"] == 0
                     and counts["first_tokens"] == 0 and counts["finished"] == 0
                     and first_e04_failure is not None
                     and first_e04_failure.get("will_break_waiting_loop") is True
                     and acquired == len(selected) and released == 0
                     and alive and not oom and not server_error)
        if x_enabled:
            x = xfuture.result(timeout=15)
            valid &= len(x["tokens"]) == 512
        row = {
            "stage": stage, "repetition": repetition, "policy": policy,
            "resume_arrived": counts["resume_arrived"],
            "resume_admitted": counts["resume_admitted"],
            "first_tokens": counts["first_tokens"], "finished": counts["finished"],
            "promotion_attempts": len(records),
            "promotion_admits": sum(record["status"] == "ok" for record in records),
            "capacity_rejects": sum(record["status"] == "capacity_rejected" for record in records),
            "progress_rejects": sum(record["status"] == "progress_rejected" for record in records),
            "lease_acquired": acquired, "lease_released": released,
            "HOL_allocate_none_count": sum(event.get("event") == "WAITING_ALLOCATE_NONE"
                                           and event.get("session") == "E04:turn:1"
                                           for event in snapshot["waiting"]),
            "E04_first_failure_free_blocks": first_e04_failure.get("free_blocks") if first_e04_failure else "NA",
            "local_source_tokens": local, "external_source_tokens": external,
            "source_conservation": source_valid, "K2_max_pre_first": max_pre,
            "server_alive": alive, "OOM": oom, "server_error": server_error,
            "PSA_invariant_violations": violations,
            "classification": "PROGRESS" if valid and expect_progress else
                              "LEASE_HOL_NONPROGRESS" if valid else "INVALID",
        }
        append_csv(OUT / "liveness_runs.csv", [row])
        append_csv(OUT / "prefetch_metrics.csv", [{"stage": stage, "repetition": repetition,
                    "policy": policy, **record} for record in records])
        save_event_artifacts(stage, repetition, policy, snapshot, t0_s)
        append_csv(OUT / "progress_admission.csv", progress_admission_rows(
            stage, repetition, policy, snapshot["capacity"], jobs))
        if not valid:
            raise RuntimeError(f"{stage} invalid:{row}")
        return row
    finally:
        if proc.poll() is None:
            runtime.stop_server(proc, log, run_dir, label)
        if resume_pool is not None:
            resume_pool.shutdown(wait=False, cancel_futures=True)
        if xpool is not None:
            xpool.shutdown(wait=False, cancel_futures=True)
        compact_log(log, label)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    actual_sha = sha256_file(ROOT / "results/phase4c/hybrid_workload.csv")
    if actual_sha != WORKLOAD_SHA:
        raise RuntimeError("EXPERIMENT_INVALID_ENVIRONMENT:workload SHA")
    jobs = verified_jobs()
    data = reconstructed_data(jobs)
    write_json(OUT / "config.json", {
        "phase": "4F-A", "start_commit": START_COMMIT,
        "workload_sha256": actual_sha, "gpu_blocks": GPU_BLOCKS,
        "reservation_budget_blocks": BUDGET_BLOCKS, "P_SAFE_ms": P_SAFE_MS,
        "resume_guard_ms": 110, "K": K, "X_output_tokens": 512,
        "P4_repetitions": 8, "P5_repetitions": 4,
    })
    write_json(OUT / "source_manifest.json", {
        "source_workload": "results/phase4c/hybrid_workload.csv",
        "source_workload_sha256": actual_sha,
        "frozen_jobs": "results/phase4d_a/boundary0_jobs.csv",
        "vllm_upstream_commit": "0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665",
        "start_commit": START_COMMIT,
    })
    completed: list[str] = []
    try:
        print("P0 physical reject", flush=True)
        run_micro("physical_reject", [("E00", False), ("E01", False),
                  ("E02", False), ("E04", True)],
                  ["ok", "ok", "ok", "capacity_rejected"], jobs, data)
        completed.append("P0_physical")
        print("P0 progress reject", flush=True)
        run_micro("progress_reject", [("E00", True), ("E01", True),
                  ("E02", True)], ["ok", "ok", "progress_rejected"], jobs, data)
        completed.append("P0_progress")
        print("P0 protected candidate", flush=True)
        run_micro("candidate_aware", [("E04", True)], ["ok"], jobs, data)
        completed.append("P0_candidate")
        print("P1 baseline", flush=True)
        run_deterministic("P1", 0, psa=False, x_enabled=False,
                          selected=("E00", "E01", "E02"), expect_progress=False,
                          jobs=jobs, data=data)
        completed.append("P1_baseline")
        for repetition in range(1, 3):
            print(f"P1 PSA {repetition}/2", flush=True)
            run_deterministic("P1", repetition, psa=True, x_enabled=False,
                              selected=("E00", "E01", "E02"), expect_progress=True,
                              jobs=jobs, data=data)
            completed.append(f"P1_PSA_{repetition}")
        print("P2 baseline", flush=True)
        try:
            run_deterministic("P2", 0, psa=False, x_enabled=True,
                              selected=("E00", "E01", "E02"), expect_progress=False,
                              jobs=jobs, data=data)
        except RuntimeError as exc:
            if "P2 invalid:" not in str(exc):
                raise
            print(f"P2 baseline did not reproduce: {exc}", flush=True)
        completed.append("P2_baseline_attempt")
        for repetition in range(1, 4):
            print(f"P2 PSA {repetition}/3", flush=True)
            run_deterministic("P2", repetition, psa=True, x_enabled=True,
                              selected=("E00", "E01", "E02"), expect_progress=True,
                              jobs=jobs, data=data)
            completed.append(f"P2_PSA_{repetition}")
        for repetition in range(1, 3):
            print(f"P3 candidate-aware {repetition}/2", flush=True)
            run_deterministic("P3", repetition, psa=True, x_enabled=True,
                              selected=("E04", "E00", "E05"), expect_progress=True,
                              jobs=jobs, data=data)
            completed.append(f"P3_{repetition}")
        for repetition in range(1, 9):
            print(f"P4 real JIT {repetition}/8", flush=True)
            row = run_jit("P4", repetition, logical_reservation=False,
                          jobs=jobs, data=data)
            if row["promotion_admits"] < 1:
                raise RuntimeError("PSA_LIVENESS_PASS_BUT_NO_USEFUL_ADMISSION")
            completed.append(f"P4_{repetition}")
        for repetition in range(1, 5):
            print(f"P5 FullSlackKV {repetition}/4", flush=True)
            run_jit("P5", repetition, logical_reservation=True,
                    jobs=jobs, data=data)
            completed.append(f"P5_{repetition}")
    except Exception as exc:
        write_json(OUT / "failure.json", {
            "completed_gates": completed, "error_type": type(exc).__name__,
            "error": str(exc),
        })
        raise
    finally:
        write_json(OUT / "execution_progress.json", {"completed_gates": completed})


if __name__ == "__main__":
    main()
