from __future__ import annotations

import csv
import json
import math
import re
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from experiments.phase3f.native_client import native_prefetch
from experiments.phase3f.scheduler import latest_feasible_start, merge_blackouts
from experiments.phase4d_a import runtime
from experiments.phase4d_a.experiment import (
    OUT as OLD_OUT, ROOT, build_resume_prompts, first_boundary_jobs, materialize_and_evict,
    source_snapshot, write_csv, write_json,
)
from experiments.phase4d_a.helpers import (
    ReservationLedger, FROZEN_SHA256, source_conserved, sha256_file,
)
from experiments.phase4d_a.run_policy import parse_trace, start_background, stream_resume
from experiments.phase4d_a_fix.release_fix import (
    EXPECTED_GPU_BLOCKS, FROZEN_P_SAFE_MS, RuntimeReleaseQueue, parse_lease_trace,
    LeaseTraceReader, service_envelope,
)

OUT = ROOT / "results/phase4d_a_fix"
POLICY = "SlackKV-Oracle-ReserveLease-K2"


def validate_frozen_inputs(jobs: list[dict[str, Any]]) -> None:
    if sha256_file(ROOT / "results/phase4c/hybrid_workload.csv") != FROZEN_SHA256:
        raise RuntimeError("EXPERIMENT_INVALID_WORKLOAD:Phase4C hash changed")
    if len(jobs) != 8 or sum(j["required_prefix_tokens"] for j in jobs) != 168016:
        raise RuntimeError("EXPERIMENT_INVALID_WORKLOAD:job/token invariant")
    if sum(j["required_blocks"] for j in jobs) != 10501:
        raise RuntimeError("EXPERIMENT_INVALID_WORKLOAD:block invariant")
    minimum_mib = sum(j["required_prefix_tokens"] for j in jobs) * 28672 / 2**20
    if not math.isclose(minimum_mib, 4594.1875, abs_tol=1e-9):
        raise RuntimeError("EXPERIMENT_INVALID_WORKLOAD:minimum H2D changed")


def drain_releases(
    queue: RuntimeReleaseQueue,
    states: dict[str, dict[str, Any]],
    lease_reader: LeaseTraceReader,
    now_ns: int,
) -> list[dict[str, Any]]:
    for episode, state in states.items():
        first_ns = state.get("first_token_ns")
        if first_ns is not None:
            queue.enqueue_first_token(episode, int(first_ns))
    queue.observe_lease_events(lease_reader.read_events())
    return queue.drain(now_ns)


def persist_decisions(decisions: list[dict[str, Any]]) -> None:
    if decisions:
        write_csv(OUT / "reservation_decisions.csv", decisions)


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
    pending = {j["episode_id"] for j in jobs}
    deadlines = {j["episode_id"]: float(j["deadline_ms"]) for j in jobs}
    blocks = {j["episode_id"]: int(j["required_blocks"]) for j in jobs}
    blackouts = merge_blackouts(deadlines, 110.0)
    decisions: list[dict[str, Any]] = []
    releases: list[dict[str, Any]] = []
    promoted: dict[str, dict[str, Any]] = {}
    pool = ThreadPoolExecutor(max_workers=len(jobs))
    while pending:
        now_ns = time.monotonic_ns()
        now_ms = (now_ns / 1e9 - t0_s) * 1000
        new_releases = drain_releases(queue, states, lease_reader, now_ns)
        for row in new_releases:
            releases.append(row)
            decisions.append({"episode_id": row["episode_id"], "decision": row["event"],
                              "physical_release_ms": row["physical_release_ms"],
                              "logical_release_ms": row["logical_release_ms"],
                              "reserved_blocks": row["reserved_blocks_after"]})
            persist_decisions(decisions)
        earliest = min(pending, key=lambda e: (deadlines[e], e))
        target = latest_feasible_start(now_ms, [(e, deadlines[e]) for e in pending], blackouts, FROZEN_P_SAFE_MS)
        if target > now_ms + 0.5:
            time.sleep(min(0.01, (target - now_ms) / 1000))
            continue
        with lock:
            stale = states[earliest].get("resume_started_ns") is not None
        if stale:
            states[earliest]["promoted"] = False
            decisions.append({"episode_id": earliest, "decision": "STALE_PROMOTION_CANCELLED",
                              "target_start_ms": target})
            persist_decisions(decisions)
            pending.remove(earliest)
            continue
        if not ledger.reserve(earliest, blocks[earliest]):
            states[earliest]["promoted"] = False
            decisions.append({"episode_id": earliest, "decision": "RESERVATION_SKIP",
                              "target_start_ms": target, "reserved_blocks": ledger.reserved_blocks,
                              "required_blocks": blocks[earliest], "budget_blocks": ledger.budget_blocks})
            persist_decisions(decisions)
            pending.remove(earliest)
            continue
        lease_key = data[earliest]["job"]["lease_key"]
        states[earliest]["leased"] = True
        future = pool.submit(native_prefetch, data[earliest]["prefix"], lease_key=lease_key, timeout_s=120.0)
        states[earliest]["promoted"] = True
        promoted[earliest] = {"future": future, "target_start_ms": target, "lease_key": lease_key,
                              "required_blocks": blocks[earliest]}
        decisions.append({"episode_id": earliest, "decision": "RESERVATION_PROMOTE",
                          "target_start_ms": target, "required_blocks": blocks[earliest],
                          "reserved_blocks": ledger.reserved_blocks, "budget_blocks": ledger.budget_blocks})
        persist_decisions(decisions)
        pending.remove(earliest)
    # Persist each actual endpoint response before waiting on the next promotion.
    completed_results: dict[str, Any] = {}
    for episode, item in promoted.items():
        try:
            item["result"] = item["future"].result(timeout=180)
        except Exception as exc:
            write_json(OUT / "native_prefetch_partial.json", {
                "completed": completed_results,
                "failed_episode": episode,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "issued_promotions": sorted(promoted),
            })
            raise
        completed_results[episode] = {
            "lease_key": item["lease_key"],
            "required_blocks": item["required_blocks"],
            "target_start_ms": item["target_start_ms"],
            "result": item["result"],
        }
        write_json(OUT / "native_prefetch_partial.json", {
            "completed": completed_results,
            "failed_episode": None,
            "issued_promotions": sorted(promoted),
        })
    pool.shutdown(wait=True)
    # Drain at admission boundaries and once more after all resume completions in run_release_fix.
    return promoted, decisions, releases


def lease_peak(events: list[dict[str, Any]]) -> tuple[int, int]:
    active: dict[str, int] = {}
    peak = 0
    for event in sorted(events, key=lambda item: float(item["timestamp_monotonic"])):
        key = str(event.get("lease_key", ""))
        if event.get("event") == "LEASE_ACQUIRED":
            active[key] = int(event["num_blocks"])
        elif event.get("event") == "LEASE_RELEASED":
            active.pop(key, None)
        peak = max(peak, sum(active.values()))
    return peak, len(active)


def run_release_fix(jobs: list[dict[str, Any]], data: dict[str, dict[str, Any]]) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    label = "release_fix_slackkv_oracle_reservelease_k2_retry2"
    proc, log = runtime.start_server(label, connector=True, out=OUT)
    states = {j["episode_id"]: {"resume_started_ns": None, "first_token_ns": None,
                                "leased": False, "promoted": False} for j in jobs}
    resume_rows: list[dict[str, Any]] = []
    try:
        _, block = runtime.wait_ready(proc)
        if block != 16:
            raise RuntimeError("EXPERIMENT_INVALID_ENVIRONMENT:block_size")
        capacity = __import__("experiments.phase4d_a.run_policy", fromlist=["capacity_from_server"]).capacity_from_server(log, jobs)
        if capacity["total_gpu_kv_blocks"] != EXPECTED_GPU_BLOCKS:
            raise RuntimeError("EXPERIMENT_INVALID_ENVIRONMENT:GPU KV blocks changed")
        materialize_and_evict(proc, log, label, jobs, data, OUT)
        source_before = source_snapshot()
        ledger = ReservationLedger(capacity["reservation_budget_blocks"])
        episode_to_key = {e: data[e]["job"]["lease_key"] for e in data}
        required = {e: int(data[e]["job"]["required_blocks"]) for e in data}
        t0_box: dict[str, float] = {}
        lock = threading.Lock()
        xpool, xfuture, xmeta = start_background()
        t0_ns = int(xmeta["first_ns"])
        t0_s = t0_ns / 1e9
        t0_box["s"] = t0_s
        queue = RuntimeReleaseQueue(ledger, episode_to_key, required, t0_s)
        lease_reader = LeaseTraceReader(log)
        pool = ThreadPoolExecutor(max_workers=len(jobs))
        resume_futures = {}
        for episode, entry in data.items():
            lease_key = entry["job"]["lease_key"]
            resume_futures[episode] = pool.submit(
                stream_resume, lease_key, entry, t0_s, float(entry["job"]["deadline_ms"]),
                True, states, lock,
            )
        promotions, decisions, release_rows = schedule_promotions(
            jobs, data, states, lock, t0_s, ledger, queue, lease_reader,
        )
        for episode, future in resume_futures.items():
            row = future.result(timeout=1800)
            row["episode_id"] = episode
            row["resume_osl"] = int(data[episode]["expected_output_count"])
            row["expected_output_sha256"] = data[episode]["expected_output_sha256"]
            row["output_identity"] = row["output_sha256"] == row["expected_output_sha256"]
            row["effective_resume_latency_ms"] = max(0.0, (row["first_token_ns"] / 1e9 - t0_s) * 1000 - row["deadline_ms"])
            resume_rows.append(row)
        pool.shutdown(wait=True)
        # Final drain is deliberately after all real resume completions.
        final_deadline = time.monotonic() + 5
        while queue.pending_release_count and time.monotonic() < final_deadline:
            now_ns = time.monotonic_ns()
            new = drain_releases(queue, states, lease_reader, now_ns)
            for item in new:
                release_rows.append(item)
                decisions.append({"episode_id": item["episode_id"], "decision": item["event"],
                                  "physical_release_ms": item["physical_release_ms"],
                                  "logical_release_ms": item["logical_release_ms"],
                                  "reserved_blocks": item["reserved_blocks_after"]})
                persist_decisions(decisions)
            if queue.pending_release_count:
                time.sleep(0.05)
        xresult = xfuture.result(timeout=900)
        xpool.shutdown(wait=True)
        source_after = source_snapshot()
        from experiments.phase2d.helpers import source_tokens_delta
        local = int(round(source_tokens_delta(source_before, source_after, "local_cache_hit")))
        external = int(round(source_tokens_delta(source_before, source_after, "external_kv_transfer")))
        runtime.stop_server(proc, log, OUT, label)
        trace, lease_events = parse_trace(log, t0_ns)
        parsed_lease_events = parse_lease_trace(log.read_text(encoding="utf-8", errors="replace"))
        trace_by = {}
        for event in trace:
            session = str(event.get("session") or "")
            episode = session.split(":turn:", 1)[0]
            if episode in data:
                trace_by.setdefault(episode, {}).setdefault(event["event"], event)
        native_rows = []
        service_values = []
        for episode, item in promotions.items():
            result = item["result"]
            timestamps = result.get("phase3m_timestamps") or {}
            issue_s = float(result["sent_monotonic_s"])
            t6_s = float(timestamps["T6"]) if "T6" in timestamps else None
            issue_to_t6 = (t6_s - issue_s) * 1000 if t6_s is not None else None
            if issue_to_t6 is not None:
                service_values.append(issue_to_t6)
            arrival = trace_by.get(episode, {}).get("RESUME_ARRIVED", {}).get("timestamp_monotonic_raw")
            remote = "WAITING_FOR_REMOTE_KVS" in trace_by.get(episode, {})
            classification = ("PROMOTION_INTERNAL_READY_MISS" if t6_s is None or arrival is None or t6_s >= arrival
                              else "PROMOTION_REMOTE_WAIT" if remote else "PROMOTION_USEFUL")
            native_rows.append({
                "episode_id": episode, "lease_key": item["lease_key"], "required_blocks": item["required_blocks"],
                "target_start_ms": item["target_start_ms"], "client_issue_ms": (issue_s - t0_s) * 1000,
                "T0_ms": (float(timestamps["T0"]) - t0_s) * 1000 if "T0" in timestamps else "NA",
                "T6_ms": (t6_s - t0_s) * 1000 if t6_s is not None else "NA",
                "T7_ms": (float(timestamps["T7"]) - t0_s) * 1000 if "T7" in timestamps else "NA",
                "issue_to_T6_ms": issue_to_t6 if issue_to_t6 is not None else "NA",
                "arrival_ms": (float(arrival) - t0_s) * 1000 if arrival else "NA",
                "arrival_minus_T6_ms": (float(arrival) - t6_s) * 1000 if arrival and t6_s else "NA",
                "external_tokens_loaded": result.get("external_tokens_loaded"),
                "transfer_bytes": result.get("transfer_bytes"), "status": result.get("status"),
                "model_forward_count": result.get("model_forward_count"),
                "generated_token_count": result.get("generated_token_count"),
                "mixed_prefetch_model_step_count": result.get("mixed_prefetch_model_step_count"),
                "promotion_classification": classification,
            })
        for episode, row in trace_by.items():
            pass
        for row in resume_rows:
            episode = row["episode_id"]
            events = trace_by.get(episode, {})
            row.update({
                "scheduler_arrival_ms": events.get("RESUME_ARRIVED", {}).get("timestamp_ms", "NA"),
                "scheduler_admission_ms": events.get("RESUME_ADMITTED", {}).get("timestamp_ms", "NA"),
                "first_token_ms": events.get("FIRST_TOKEN", {}).get("timestamp_ms", "NA"),
                "request_finished_ms": events.get("REQUEST_FINISHED", {}).get("timestamp_ms", "NA"),
            })
        counts = {}
        for event in parsed_lease_events:
            counts.setdefault(event["lease_key"], []).append(event["event"])
        promoted_keys = {item["lease_key"] for item in promotions.values()}
        physical_release_count = sum(names.count("LEASE_RELEASED") for key, names in counts.items() if key in promoted_keys)
        lease_peak_blocks, physical_lease_leaks = lease_peak(parsed_lease_events)
        firsts = [int(r["first_token_ns"]) for r in resume_rows]
        x_active = bool(firsts) and max(firsts) < int(xresult["done_ns"])
        max_pre = max([int(e.get("num_slackkv_pre_first_token") or 0) for e in trace] + [0])
        native_correct = all(
            row["status"] == "ok"
            and int(row["external_tokens_loaded"]) == int(data[row["episode_id"]]["job"]["required_prefix_tokens"])
            and int(row["transfer_bytes"]) > 0
            and int(row["model_forward_count"]) == 0
            and int(row["generated_token_count"]) == 0
            and int(row["mixed_prefetch_model_step_count"]) == 0
            for row in native_rows
        )
        token_correct = len(resume_rows) == 8 and all(r["generated_token_count"] == r["resume_osl"] for r in resume_rows)
        conservation = source_conserved(local, external, 168016)
        runtime_events = queue.events
        later_promoted = [
            d["episode_id"] for index, d in enumerate(decisions)
            if d.get("decision") == "RESERVATION_PROMOTE"
            and d["episode_id"] in {"E06", "E07", "E03"}
            and any(prev.get("decision") == "RUNTIME_RELEASE_AFTER_PHYSICAL_LEASE" for prev in decisions[:index])
        ]
        release_ordering = all(
            float(event["logical_release_ms"]) >= float(event["physical_release_ms"])
            and float(event["logical_minus_physical_ms"]) >= 0
            for event in runtime_events
        )
        exact_sequence = all(counts.get(key) == [
            "LEASE_ACQUIRED", "LEASE_HELD_AFTER_PREFETCH_FREE", "LEASE_RELEASE_READY", "LEASE_RELEASED"
        ] for key in promoted_keys)
        integration = {
            "policy": POLICY, "repetition": 0, "valid": True,
            "resume_complete": len(resume_rows) == 8,
            "token_correct": token_correct,
            "resume_completed_count": len(resume_rows),
            "resume_correct_count": sum(r["generated_token_count"] == r["resume_osl"] for r in resume_rows),
            "local_tokens": local, "external_tokens": external, "source_conservation": conservation,
            "promotion_count": len(promotions), "promotion_episode_ids": sorted(promotions),
            "reservation_peak_blocks": ledger.peak_blocks, "physical_lease_peak_blocks": lease_peak_blocks,
            "physical_release_count": physical_release_count, "logical_release_count": len(runtime_events),
            "runtime_release_count": len(runtime_events), "unsafe_release_count": queue.unsafe_release_count,
            "logical_reservation_leak_count": queue.logical_leaks,
            "physical_lease_leak_count": physical_lease_leaks,
            "lease_leak_count": physical_lease_leaks,
            "later_job_promoted_after_release": bool(later_promoted),
            "later_job_episode_ids": later_promoted,
            "K2_max_pre_first": max_pre,
            "K2_valid": max_pre <= 2,
            "reservation_overcommit": ledger.peak_blocks > capacity["reservation_budget_blocks"],
            "lease_overcommit": lease_peak_blocks > capacity["reservation_budget_blocks"],
            "physical_before_logical_valid": release_ordering,
            "lease_sequences_valid": exact_sequence,
            "native_correct": native_correct,
            "X_active_through_last_first_token": x_active,
            "P_SAFE_MS": FROZEN_P_SAFE_MS,
            "issue_to_T6_ms": service_values,
            "resume_rows": resume_rows,
            "native_rows": native_rows,
            "reservation_decisions": decisions,
            "runtime_release_events": runtime_events,
            "lease_events": parsed_lease_events,
            "scheduler_trace": trace,
            "capacity": capacity,
            "promotion_memory_time_MiB_s": sum(
                int(row["external_tokens_loaded"]) * 28672 / 2**20 *
                max(0.0, float(next(r for r in resume_rows if r["episode_id"] == row["episode_id"])["scheduler_admission_ms"]) - float(row["T6_ms"])) / 1000
                for row in native_rows if row["T6_ms"] != "NA"
            ),
            "X_decode_duration_ms": (int(xresult["done_ns"]) - int(xresult["first_ns"])) / 1e6,
        }
        integration["valid"] = all([
            integration["resume_complete"], integration["token_correct"], conservation,
            integration["K2_valid"], not integration["reservation_overcommit"], not integration["lease_overcommit"],
            integration["lease_leak_count"] == 0, integration["logical_reservation_leak_count"] == 0,
            integration["unsafe_release_count"] == 0, release_ordering, exact_sequence, native_correct, x_active,
            bool(later_promoted),
        ])
        integration["service_envelope"] = service_envelope(service_values, FROZEN_P_SAFE_MS) if service_values else None
        OUT.joinpath("integration_logical_reservation_detail.json").write_text(
            json.dumps(integration, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_csv(OUT / "runtime_release_events.csv", runtime_events)
        write_csv(OUT / "reservation_decisions.csv", decisions)
        write_csv(OUT / "lease_events.csv", parsed_lease_events)
        write_csv(OUT / "session_metrics.csv", resume_rows)
        write_csv(OUT / "prefetch_metrics.csv", native_rows)
        write_csv(OUT / "scheduler_trace.csv", trace)
        write_json(OUT / "release_fix_integration.json", {k: v for k, v in integration.items() if k not in {
            "resume_rows", "native_rows", "reservation_decisions", "runtime_release_events", "lease_events", "scheduler_trace"
        }})
        write_json(OUT / "service_envelope_audit.json", integration["service_envelope"] or {
            "promotion_count": 0, "overrun_count": 0, "status": "NO_PROMOTIONS"})
        return integration
    finally:
        runtime.stop_server(proc, log, OUT, label)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "server_logs").mkdir(exist_ok=True)
    jobs = first_boundary_jobs(ROOT / "results/phase4c/hybrid_workload.csv")
    validate_frozen_inputs(jobs)
    data = build_resume_prompts(jobs)
    old_service = json.loads((OLD_OUT / "service_budget.json").read_text(encoding="utf-8"))
    if int(old_service["P_SAFE_MS"]) != FROZEN_P_SAFE_MS:
        raise RuntimeError("frozen P_SAFE mismatch")
    integration = run_release_fix(jobs, data)
    result = "PASS" if integration["valid"] else "FAIL"
    report = {
        "phase": "4D-A-FIX", "start_commit": "86ac84a94b671514c78b69aeb3dd294e137a03a8",
        "workload_sha256": FROZEN_SHA256, "gpu_blocks": EXPECTED_GPU_BLOCKS,
        "p_safe_ms": FROZEN_P_SAFE_MS, "reservation_fix": result,
        "integration_valid": integration["valid"], "liveness_started": False,
        "final_result": "PENDING_LIVENESS" if integration["valid"] else "RESULT_D_RELEASE_FIX_FAILED",
    }
    write_json(OUT / "validation_report.json", report)
    write_json(OUT / "source_manifest.json", {
        "start_commit": report["start_commit"], "phase4c_workload_sha256": FROZEN_SHA256,
        "base_phase4d_a_commit": "86ac84a94b671514c78b69aeb3dd294e137a03a8",
        "vllm_source_commit": "0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665",
        "vllm_changes": "None; Phase 3S lease implementation reused",
    })
    print(json.dumps(report, indent=2))
    if not integration["valid"]:
        raise SystemExit("RESULT_D_RELEASE_FIX_FAILED; do not start Naive liveness")


if __name__ == "__main__":
    main()
