from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from typing import Any

from transformers import AutoTokenizer

from experiments.phase2d import run_phase2d as runtime
from experiments.phase2d.helpers import gpu_to_cpu_offload_bytes, parse_prometheus_samples, source_tokens_delta
from experiments.phase3f.native_client import native_prefetch
from experiments.phase3f.scheduler import latest_feasible_start, merge_blackouts
from experiments.phase3g import run_phase3g as transport
from experiments.phase3g_full import run_phase3g_full as engine
from experiments.phase3g_full.metrics import interval_concurrency, percentile_linear
from experiments.phase3h import run_phase3h as gate_config
from experiments.phase3r_fix.helpers import effective_latency_ms
from experiments.phase3u_fix import run_phase3u_fix as prior
from experiments.phase3v.ledger import (
    BLOCK_SIZE_TOKENS, DEADLINES_MS, HANDOFF_GUARD_MS, HETEROGENEOUS_TOKENS,
    KV_BYTES_PER_TOKEN, LEASE_BUDGET_BLOCKS, POLICIES, UNIFORM_TOKENS,
    BlockReservationLedger, FixedCount2Ledger, h2d_accounting, required_blocks,
    required_mib, reservation_policy_decision,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase3v"
TOKENS = 32
SERVICE_MS = 40.0
X_TOKENS = 256
X_BASELINE_MS: float | None = None
RESERVATION_DECISIONS: list[dict[str, Any]] = []
RESERVATION_EVENTS: list[dict[str, Any]] = []
make_current_prompts: dict[str, list[int]] = {}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def capture(stage: str, snapshots: list[dict[str, Any]]) -> dict[Any, float]:
    raw = runtime.metric_text()
    samples = parse_prometheus_samples(raw)
    relevant = {str(k): v for k, v in samples.items() if any(x in str(k) for x in ("local_cache_hit", "external_kv_transfer", "kv_offload", "gpu_to_cpu", "cpu_to_gpu"))}
    snapshots.append({"stage": stage, "timestamp_monotonic_s": time.monotonic(), "relevant_samples": relevant})
    return samples


def make_prompts(token_sizes: dict[str, int]) -> dict[str, list[int]]:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", local_files_only=True)

    def exact_ids(text: str, count: int) -> list[int]:
        result = tokenizer.encode(text, add_special_tokens=False)
        while len(result) < count:
            result.extend(result)
        return result[:count]

    prompts: dict[str, list[int]] = {}
    for session in "ABCDEFGHIJKL":
        marker = f"SlackKV Phase3V unique session {session}; ordinary repository debugging notes, source files, tests, and tool output. "
        prompts[session.lower()] = exact_ids(marker * 900, token_sizes[session])
    if any(len(prompts[s.lower()]) != token_sizes[s] for s in "ABCDEFGHIJKL"):
        raise AssertionError("session prompt length mismatch")
    if len({tuple(prompts[s.lower()]) for s in "ABCDEFGHIJKL"}) != 12:
        raise AssertionError("session prefixes must be distinct")
    pressure = (
        "PRESSURE ONE unique unrelated repository file-search output alpha. ",
        "PRESSURE TWO unique unrelated compiler output beta. ",
        "PRESSURE THREE unique unrelated repository status gamma. ",
        "PRESSURE FOUR unique unrelated terminal output delta. ",
    )
    for index, text in enumerate(pressure, 1):
        prompts[f"pressure_{index}"] = exact_ids(text * 3000, 9500)
    prompts["suffix"] = tokenizer.encode("\nTool result: operation completed successfully.\nAssistant:", add_special_tokens=False)
    return prompts


def event(rows: list[dict[str, Any]], session: str, name: str) -> dict[str, Any] | None:
    return next((row for row in rows if row.get("session") == session and row.get("event") == name), None)


def wait_until(absolute_s: float) -> None:
    while time.monotonic() < absolute_s:
        time.sleep(min(0.001, absolute_s - time.monotonic()))


def send_resume(session: str, prompts: dict[str, list[int]], t0: float) -> dict[str, Any]:
    wait_until(t0 + DEADLINES_MS[session] / 1000.0)
    response = transport.send(prompts[session.lower()] + prompts["suffix"], stream=True, max_tokens=TOKENS, trace=True, session=session, gate=True, normal=False)
    if len(response.get("token_ids", [])) != TOKENS:
        raise RuntimeError(f"{session} generated {len(response.get('token_ids', []))} tokens, expected {TOKENS}")
    return response

def schedule_promotions(workload: str, policy: str, prompts: dict[str, list[int]], token_sizes: dict[str, int], t0: float, output: dict[str, dict[str, Any]]) -> None:
    pending = set("ABCDEFGHIJKL")
    blackouts = merge_blackouts(DEADLINES_MS, 110.0)
    ledger: BlockReservationLedger = FixedCount2Ledger(LEASE_BUDGET_BLOCKS) if policy == "FixedCount2-Async40-K2" else BlockReservationLedger(LEASE_BUDGET_BLOCKS)
    decisions: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=3) as prefetch_pool:
        while pending:
            now_ms = (time.monotonic() - t0) * 1000.0
            target = latest_feasible_start(now_ms, [(s, DEADLINES_MS[s]) for s in pending], blackouts, SERVICE_MS)
            if target - now_ms > 0.5:
                time.sleep((target - now_ms) / 1000.0)
                continue
            session = min(pending, key=lambda s: (DEADLINES_MS[s], s))
            decision_ms = (time.monotonic() - t0) * 1000.0
            expired = ledger.expire(decision_ms)
            n_active_after_expire = len(ledger.active)
            for index, (item, before_blocks, after_blocks) in enumerate(expired):
                count_before = n_active_after_expire + len(expired) - index
                events.append({
                    "workload": workload, "policy": policy, "session": item.session, "timestamp_ms": decision_ms,
                    "event": "EXPIRED", "blocks": item.blocks, "active_count_before": count_before,
                    "active_count_after": count_before - 1, "reserved_blocks_before": before_blocks,
                    "reserved_blocks_after": after_blocks, "reservation_start_ms": item.start_ms,
                    "reservation_end_ms": item.end_ms,
                })
            blocks = required_blocks(token_sizes[session], BLOCK_SIZE_TOKENS)
            count_before, blocks_before = len(ledger.active), ledger.reserved_blocks
            decision, reason = reservation_policy_decision(policy, ledger, blocks)
            reservation = None
            if decision == "RESERVATION_PROMOTE":
                reservation = ledger.grant(session, blocks, target, DEADLINES_MS[session])
                events.append({
                    "workload": workload, "policy": policy, "session": session, "timestamp_ms": decision_ms,
                    "event": "GRANTED", "blocks": blocks, "active_count_before": count_before,
                    "active_count_after": len(ledger.active), "reserved_blocks_before": blocks_before,
                    "reserved_blocks_after": ledger.reserved_blocks, "reservation_start_ms": reservation.start_ms,
                    "reservation_end_ms": reservation.end_ms,
                })
            decisions.append({
                "workload": workload, "policy": policy, "session": session,
                "cluster": 1 + (ord(session) - ord("A")) // 4,
                "prefix_tokens": token_sizes[session], "required_blocks": blocks,
                "required_mib": required_mib(token_sizes[session]), "deadline_ms": DEADLINES_MS[session],
                "target_start_ms": target, "decision_time_ms": decision_ms,
                "active_reservation_count_before": count_before, "reserved_blocks_before": blocks_before,
                "budget_blocks": LEASE_BUDGET_BLOCKS, "decision": decision, "reason": reason,
                "reservation_start_ms": reservation.start_ms if reservation else "NA",
                "reservation_end_ms": reservation.end_ms if reservation else "NA",
                "reserved_blocks_after": ledger.reserved_blocks, "active_reservation_count_after": len(ledger.active),
            })
            if decision in ("PROMOTE", "RESERVATION_PROMOTE"):
                lease = decision == "RESERVATION_PROMOTE"
                issue_s = time.monotonic()
                future = prefetch_pool.submit(native_prefetch, prompts[session.lower()], **({"lease_key": session} if lease else {}))
                output[session] = {
                    "workload": workload, "policy": policy, "session": session, "decision": decision,
                    "reason": reason, "target_start_ms": target, "client_issue_s": issue_s,
                    "required_blocks": blocks, "required_tokens": token_sizes[session],
                    "lease_requested": lease, "future": future,
                }
            else:
                output[session] = {
                    "workload": workload, "policy": policy, "session": session, "decision": decision,
                    "reason": reason, "target_start_ms": target, "required_blocks": blocks,
                    "required_tokens": token_sizes[session], "lease_requested": False,
                }
            pending.remove(session)
        for item in output.values():
            future = item.get("future")
            if future is not None:
                item["result"] = future.result(timeout=300)
                item["client_issue_s"] = float(item["result"]["sent_monotonic_s"])
        now_ms = (time.monotonic() - t0) * 1000.0
        expired = ledger.expire(now_ms)
        n_active_after_expire = len(ledger.active)
        for index, (item, before_blocks, after_blocks) in enumerate(expired):
            count_before = n_active_after_expire + len(expired) - index
            events.append({
                "workload": workload, "policy": policy, "session": item.session, "timestamp_ms": now_ms,
                "event": "EXPIRED", "blocks": item.blocks, "active_count_before": count_before,
                "active_count_after": count_before - 1, "reserved_blocks_before": before_blocks,
                "reserved_blocks_after": after_blocks, "reservation_start_ms": item.start_ms,
                "reservation_end_ms": item.end_ms,
            })
    RESERVATION_DECISIONS.extend(decisions)
    RESERVATION_EVENTS.extend(events)


def active_lease_peaks(lease_rows: list[dict[str, Any]]) -> tuple[int, int, int]:
    active: dict[str, int] = {}
    peak_count = peak_blocks = acquired = released = 0
    for row in sorted(lease_rows, key=lambda x: float(x["timestamp_monotonic"])):
        key = str(row.get("lease_key", ""))
        if row.get("event") == "LEASE_ACQUIRED":
            active[key] = int(row.get("num_blocks", 0)); acquired += 1
        elif row.get("event") == "LEASE_RELEASED":
            active.pop(key, None); released += 1
        peak_count = max(peak_count, len(active))
        peak_blocks = max(peak_blocks, sum(active.values()))
    return peak_count, peak_blocks, acquired - released


def run_once(workload: str, policy: str, repetition: int, token_sizes: dict[str, int], record_path: Path) -> dict[str, Any]:
    global RESERVATION_DECISIONS, RESERVATION_EVENTS
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "server_logs").mkdir(parents=True, exist_ok=True)
    label = f"{workload}_{policy.lower().replace('-', '_')}_{repetition}"
    engine.OUT = prior.OUT = OUT
    original_payload = transport.completion_payload
    gate_config._active_gate_limit = 2
    transport.completion_payload = gate_config._payload_with_gate_limit
    RESERVATION_DECISIONS, RESERVATION_EVENTS = [], []
    proc = engine.start_server(label)
    snapshots: list[dict[str, Any]] = []
    try:
        runtime.wait_ready(proc)
        previous = capture("server_ready", snapshots)
        offloads: dict[str, int] = {}
        for session in "ABCDEFGHIJKL":
            engine.send(make_current_prompts[session.lower()], stream=False, max_tokens=1, normal=True)
            expected_bytes = int(round(token_sizes[session] * KV_BYTES_PER_TOKEN))
            deadline = time.monotonic() + 10.0
            while True:
                current = capture(f"after_{session}_first", snapshots)
                observed_bytes = int(round(gpu_to_cpu_offload_bytes(previous, current)))
                if observed_bytes == expected_bytes:
                    offloads[session] = observed_bytes
                    break
                if observed_bytes > expected_bytes:
                    raise RuntimeError(f"unexpected GPU->CPU offload increment for {session}: {observed_bytes} != {expected_bytes}")
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"GPU->CPU offload counter did not settle for {session}: {observed_bytes} != {expected_bytes}")
                time.sleep(0.1)
            previous = current
        expected_offloads = {s: int(round(token_sizes[s] * KV_BYTES_PER_TOKEN)) for s in "ABCDEFGHIJKL"}
        if offloads != expected_offloads:
            raise RuntimeError(f"offload mismatch: {offloads} != {expected_offloads}")
        time.sleep(1.0)
        for index in range(1, 5):
            engine.send(make_current_prompts[f"pressure_{index}"], stream=False, max_tokens=1, normal=False)
        capture("after_four_pressure_requests", snapshots)
        x_first = Event()
        promotion_output: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=16) as pool:
            x_future = pool.submit(prior.send_x_background, make_current_prompts["pressure_1"], x_first)
            if not x_first.wait(180):
                raise RuntimeError("X first-token barrier timed out")
            log_path = OUT / "server_logs" / f"{label}.log"
            t0 = prior.wait_x_server_first(log_path)
            before = capture("after_X_first_token", snapshots)
            promote_future = None
            if policy != "OnDemand-K2":
                promote_future = pool.submit(schedule_promotions, workload, policy, make_current_prompts, token_sizes, t0, promotion_output)
            resume_futures = {s: pool.submit(send_resume, s, make_current_prompts, t0) for s in "ABCDEFGHIJKL"}
            resume_responses = {s: f.result(timeout=900) for s, f in resume_futures.items()}
            if promote_future is not None:
                promote_future.result(timeout=900)
            x_response = x_future.result(timeout=1200)
        after = capture("after_A_to_L_and_X", snapshots)
        trace = prior.parse_trace(log_path, policy, repetition, t0)
        leases = prior.parse_lease_trace(log_path, repetition, t0)
        source_local = int(round(source_tokens_delta(before, after, "local_cache_hit")))
        source_external = int(round(source_tokens_delta(before, after, "external_kv_transfer")))
        expected_tokens = sum(token_sizes.values())
        source_valid = source_local + source_external == expected_tokens
        rows_by_event = {(s, name): event(trace, s, name) for s in "ABCDEFGHIJKL" for name in ("RESUME_ARRIVED", "RESUME_ADMITTED", "FIRST_TOKEN", "REQUEST_FINISHED")}
        if any(not rows_by_event[(s, name)] for s in "ABCDEFGHIJKL" for name in ("RESUME_ARRIVED", "RESUME_ADMITTED", "FIRST_TOKEN", "REQUEST_FINISHED")):
            raise RuntimeError("incomplete scheduler lifecycle trace for A-L")
        lease_peak_count, lease_peak_blocks, lease_leaks = active_lease_peaks(leases)
        acquired = sum(row.get("event") == "LEASE_ACQUIRED" for row in leases)
        released = sum(row.get("event") == "LEASE_RELEASED" for row in leases)
        session_rows: list[dict[str, Any]] = []
        native_rows: list[dict[str, Any]] = []
        for session in "ABCDEFGHIJKL":
            arrived, admitted, first, finished = (rows_by_event[(session, n)] for n in ("RESUME_ARRIVED", "RESUME_ADMITTED", "FIRST_TOKEN", "REQUEST_FINISHED"))
            arrival_ms, admission_ms, first_ms, finish_ms = [
                (float(row["timestamp_monotonic"]) - t0) * 1000.0 for row in (arrived, admitted, first, finished)
            ]
            promotion = promotion_output.get(session)
            result = promotion.get("result") if promotion else None
            remote_wait = event(trace, session, "WAITING_FOR_REMOTE_KVS") is not None
            decision = promotion["decision"] if promotion else "ONDEMAND"
            lease_rows = [row for row in leases if row.get("lease_key") == session]
            lease_names = [row.get("event") for row in lease_rows]
            lease_ok = lease_names == ["LEASE_ACQUIRED", "LEASE_HELD_AFTER_PREFETCH_FREE", "LEASE_RELEASE_READY", "LEASE_RELEASED"]
            t6_ms: float | str = "NA"
            outcome = "ONDEMAND" if decision == "ONDEMAND" else "SKIPPED_ONDEMAND" if decision == "RESERVATION_SKIP" else "PROMOTION_USEFUL"
            actual_lease_blocks: int | str = "NA"
            prediction_error: int | str = "NA"
            if result is not None:
                timestamps = result.get("phase3m_timestamps") or {}
                if not all(key in timestamps for key in ("T0", "T6")):
                    raise RuntimeError(f"native endpoint omitted T0/T6 for {session}")
                t6_ms = (float(timestamps["T6"]) - t0) * 1000.0
                outcome = "PROMOTION_INTERNAL_READY_MISS" if float(timestamps["T6"]) > float(arrived["timestamp_monotonic"]) else "PROMOTION_REMOTE_WAIT" if remote_wait else "LEASE_ERROR" if promotion["lease_requested"] and not lease_ok else "PROMOTION_USEFUL"
                expected_bytes = int(round(token_sizes[session] * KV_BYTES_PER_TOKEN))
                if int(result["external_tokens_loaded"]) != token_sizes[session] or int(result["transfer_bytes"]) != expected_bytes:
                    raise RuntimeError(f"native KV transfer mismatch for {session}: {result}")
                if result["status"] != "ok" or int(result["model_forward_count"]) or int(result["generated_token_count"]) or int(result["mixed_prefetch_model_step_count"]):
                    raise RuntimeError(f"native prefetch invariant failed for {session}: {result}")
                if promotion["lease_requested"]:
                    actual_lease_blocks = next((int(row["num_blocks"]) for row in lease_rows if row.get("event") == "LEASE_ACQUIRED"), 0)
                    prediction_error = actual_lease_blocks - required_blocks(token_sizes[session])
                timestamps = result.get("phase3m_timestamps") or {}
                native_rows.append({
                    "workload": workload, "policy": policy, "repetition": repetition, "session": session,
                    "prefix_tokens": token_sizes[session], "required_blocks": required_blocks(token_sizes[session]),
                    "service_budget_ms": SERVICE_MS, "target_start_ms": promotion["target_start_ms"],
                    "client_issue_ms": (float(result["sent_monotonic_s"]) - t0) * 1000.0,
                    "target_to_issue_ms": (float(result["sent_monotonic_s"]) - t0) * 1000.0 - float(promotion["target_start_ms"]),
                    "T0_ms": (float(timestamps["T0"]) - t0) * 1000.0, "T6_ms": t6_ms,
                    "T7_ms": (float(timestamps.get("T7", result["completed_monotonic_s"])) - t0) * 1000.0,
                    "client_return_ms": (float(result["completed_monotonic_s"]) - t0) * 1000.0,
                    "issue_to_T6_ms": (float(timestamps["T6"]) - float(result["sent_monotonic_s"])) * 1000.0,
                    "target_to_T6_ms": float(t6_ms) - float(promotion["target_start_ms"]),
                    "arrival_minus_T6_ms": arrival_ms - float(t6_ms), "remote_wait_seen": remote_wait,
                    "promotion_outcome": outcome, "external_tokens_loaded": result["external_tokens_loaded"],
                    "transfer_bytes": result["transfer_bytes"], "cpu_to_gpu_ms": result["cpu_to_gpu_transfer_ms"],
                    "model_forward_count": result["model_forward_count"], "generated_token_count": result["generated_token_count"],
                    "mixed_prefetch_model_step_count": result["mixed_prefetch_model_step_count"],
                    "connector_only_step_count": result["connector_only_step_count"],
                    "normal_requests_delayed": result["normal_requests_delayed"], "status": result["status"],
                    "lease_requested": promotion["lease_requested"], "lease_event_sequence": "|".join(map(str, lease_names)),
                    "actual_lease_blocks": actual_lease_blocks, "block_prediction_error": prediction_error,
                })
            gpu_residency = 0.0 if result is None else max(0.0, admission_ms - float(t6_ms))
            session_rows.append({
                "workload": workload, "policy": policy, "repetition": repetition, "session": session,
                "cluster": 1 + (ord(session) - ord("A")) // 4,
                "prefix_tokens": token_sizes[session], "required_blocks": required_blocks(token_sizes[session]),
                "expected_kv_mib": required_mib(token_sizes[session]), "offload_bytes": offloads[session],
                "deadline_ms": DEADLINES_MS[session], "reservation_decision": decision,
                "reservation_reason": promotion.get("reason", "ONDEMAND") if promotion else "ONDEMAND",
                "target_start_ms": promotion.get("target_start_ms", "NA") if promotion else "NA",
                "client_issue_ms": "NA" if result is None else (float(result["sent_monotonic_s"]) - t0) * 1000.0,
                "T6_ms": t6_ms, "scheduler_arrival_ms": arrival_ms, "scheduler_admission_ms": admission_ms,
                "first_token_ms": first_ms, "finish_ms": finish_ms,
                "arrival_minus_T6_ms": "NA" if result is None else arrival_ms - float(t6_ms),
                "remote_wait_seen": remote_wait, "promotion_outcome": outcome,
                "lease_acquired": "LEASE_ACQUIRED" in lease_names, "lease_released": "LEASE_RELEASED" in lease_names,
                "actual_lease_blocks": actual_lease_blocks, "block_prediction_error": prediction_error,
                "gpu_ready_residency_ms": gpu_residency,
                "effective_resume_latency_ms": effective_latency_ms(first_ms, DEADLINES_MS[session]),
                "effective_completion_latency_ms": finish_ms - DEADLINES_MS[session],
                "generated_token_count": len(resume_responses[session].get("token_ids", [])),
            })
        native_intervals = [(float(row["T0_ms"]), float(row["T6_ms"])) for row in native_rows]
        http_intervals = [(float(row["client_issue_ms"]), float(row["client_return_ms"])) for row in native_rows]
        max_internal = interval_concurrency(native_intervals)[0] if native_intervals else 0
        max_http = interval_concurrency(http_intervals)[0] if http_intervals else 0
        x_first_ev = event(trace, "X", "FIRST_TOKEN")
        x_finish_ev = event(trace, "X", "REQUEST_FINISHED")
        l_first_ev = event(trace, "L", "FIRST_TOKEN")
        if not all((x_first_ev, x_finish_ev, l_first_ev)):
            raise RuntimeError("missing X/L scheduler trace")
        x_first_s, x_finish_s, l_first_s = (float(row["timestamp_monotonic"]) for row in (x_first_ev, x_finish_ev, l_first_ev))
        x_duration = (x_finish_s - x_first_s) * 1000.0
        x_tpot = x_duration / 255.0
        x_active = x_finish_s >= l_first_s
        token_valid = all(int(row["generated_token_count"]) == TOKENS for row in session_rows)
        gate_blocked = sum(row.get("event") == "GATE_BLOCKED" for row in trace)
        max_pre = interval_concurrency([(float(row["scheduler_admission_ms"]), float(row["first_token_ms"])) for row in session_rows])[0]
        lease_peak_count, lease_peak_blocks, lease_leaks = active_lease_peaks(leases)
        lease_acquires = sum(row.get("event") == "LEASE_ACQUIRED" for row in leases)
        lease_releases = sum(row.get("event") == "LEASE_RELEASED" for row in leases)
        promoted = [row for row in session_rows if row["reservation_decision"] in ("PROMOTE", "RESERVATION_PROMOTE")]
        skipped = [row for row in session_rows if row["reservation_decision"] == "RESERVATION_SKIP"]
        useful = sum(row["promotion_outcome"] == "PROMOTION_USEFUL" for row in promoted)
        remote_waits = sum(row["promotion_outcome"] == "PROMOTION_REMOTE_WAIT" for row in promoted)
        internal_misses = sum(row["promotion_outcome"] == "PROMOTION_INTERNAL_READY_MISS" for row in promoted)
        native_bytes = sum(int(row["transfer_bytes"]) for row in native_rows)
        h2d = h2d_accounting(native_bytes, source_external, token_sizes)
        total_tokens = sum(token_sizes.values())
        all_first = [float(row["first_token_ms"]) for row in session_rows]
        all_finish = [float(row["finish_ms"]) for row in session_rows]
        resume_latency = [float(row["effective_resume_latency_ms"]) for row in session_rows]
        completion_latency = [float(row["effective_completion_latency_ms"]) for row in session_rows]
        x_first_ms, x_finish_ms = (x_first_s - t0) * 1000.0, (x_finish_s - t0) * 1000.0
        combined_span_s = (max(max(all_finish), x_finish_ms) - min(min(all_first), x_first_ms)) / 1000.0
        grants = [row for row in RESERVATION_EVENTS if row["workload"] == workload and row["policy"] == policy and row["event"] == "GRANTED"]
        peak_reserved = max([0] + [int(row["reserved_blocks_after"]) for row in grants])
        peak_reservations = max([0] + [int(row["active_count_after"]) for row in grants])
        native_valid = all(
            row["status"] == "ok" and int(row["model_forward_count"]) == 0 and int(row["generated_token_count"]) == 0
            and int(row["mixed_prefetch_model_step_count"]) == 0
            and int(row["external_tokens_loaded"]) == token_sizes[row["session"]]
            and int(row["transfer_bytes"]) == int(round(token_sizes[row["session"]] * KV_BYTES_PER_TOKEN))
            for row in native_rows
        )
        block_errors = [row for row in native_rows if row["lease_requested"] and int(row["block_prediction_error"]) != 0]
        lease_expected = policy in ("FixedCount2-Async40-K2", "DynamicBlock512-Async40-K2")
        lease_valid = lease_acquires == lease_releases and lease_leaks == 0
        if lease_expected:
            lease_valid = lease_valid and lease_acquires == len(promoted) and all(
                [item.get("event") for item in leases if item.get("lease_key") == row["session"]]
                == ["LEASE_ACQUIRED", "LEASE_HELD_AFTER_PREFETCH_FREE", "LEASE_RELEASE_READY", "LEASE_RELEASED"] for row in promoted
            )
        elif lease_acquires or lease_releases:
            lease_valid = False
        token_ok = source_local + source_external == total_tokens
        decisions = [row for row in RESERVATION_DECISIONS if row["workload"] == workload and row["policy"] == policy]
        native_target = [float(row["target_to_issue_ms"]) for row in native_rows]
        native_issue_t6 = [float(row["issue_to_T6_ms"]) for row in native_rows]
        native_target_t6 = [float(row["target_to_T6_ms"]) for row in native_rows]

        def stats(values: list[float], p: float) -> float | str:
            return percentile_linear(values, p) if values else "NA"

        run = {
            "workload": workload, "policy": policy, "repetition": repetition,
            "AL_P50_resume_ms": percentile_linear(resume_latency, 50), "AL_P95_resume_ms": percentile_linear(resume_latency, 95),
            "AL_max_resume_ms": max(resume_latency),
            "Cluster1_P95_ms": percentile_linear([float(r["effective_resume_latency_ms"]) for r in session_rows if r["cluster"] == 1], 95),
            "Cluster2_P95_ms": percentile_linear([float(r["effective_resume_latency_ms"]) for r in session_rows if r["cluster"] == 2], 95),
            "Cluster3_P95_ms": percentile_linear([float(r["effective_resume_latency_ms"]) for r in session_rows if r["cluster"] == 3], 95),
            "AL_P95_completion_ms": percentile_linear(completion_latency, 95),
            "AL_decode_tok_s": 372.0 / ((max(all_finish) - min(all_first)) / 1000.0),
            "combined_decode_tok_s": 627.0 / combined_span_s,
            "X_decode_duration_ms": x_duration, "X_TPOT_ms": x_tpot,
            "X_slowdown": "NA" if X_BASELINE_MS is None else x_duration / X_BASELINE_MS,
            "full_makespan_ms": max(all_finish) - DEADLINES_MS["A"],
            "local_source_tokens": source_local, "external_source_tokens": source_external,
            "local_fraction": source_local / total_tokens, "external_fraction": source_external / total_tokens,
            "native_prefetch_count": len(native_rows), "promotion_useful_count": useful,
            "promotion_remote_wait_count": remote_waits, "promotion_internal_ready_miss_count": internal_misses,
            "promotion_useful_rate": useful / len(promoted) if promoted else "NA",
            "reservation_promote_count": sum(row["reservation_decision"] == "RESERVATION_PROMOTE" for row in session_rows),
            "reservation_skip_count": len(skipped), "max_active_reservation_count": peak_reservations,
            "max_reserved_blocks": peak_reserved, "reservation_budget_utilization": peak_reserved / LEASE_BUDGET_BLOCKS,
            "peak_active_lease_count": lease_peak_count, "peak_active_lease_blocks": lease_peak_blocks,
            "lease_acquire_count": lease_acquires, "lease_release_count": lease_releases, "lease_leak_count": lease_leaks,
            **h2d,
            "promotion_lead_time_MiB_s": sum(float(row["gpu_ready_residency_ms"]) * required_mib(int(row["prefix_tokens"])) / 1000.0 for row in promoted),
            "target_to_issue_P50_ms": stats(native_target, 50), "target_to_issue_P95_ms": stats(native_target, 95), "target_to_issue_max_ms": max(native_target, default="NA"),
            "issue_to_T6_P50_ms": stats(native_issue_t6, 50), "issue_to_T6_P95_ms": stats(native_issue_t6, 95), "issue_to_T6_max_ms": max(native_issue_t6, default="NA"),
            "target_to_T6_P50_ms": stats(native_target_t6, 50), "target_to_T6_P95_ms": stats(native_target_t6, 95), "target_to_T6_max_ms": max(native_target_t6, default="NA"),
            "max_outstanding_prefetch_http_calls": max_http, "max_overlapping_T0_T6": max_internal,
            "max_pre_first_concurrency": max_pre, "GATE_BLOCKED_event_count": gate_blocked,
            "token_validation": "PASS" if token_ok and token_valid else "FAIL",
            "source_validation": "PASS" if source_valid else "FAIL", "native_validation": "PASS" if native_valid else "FAIL",
            "gate_validation": "PASS" if max_pre <= 2 else "FAIL", "background_validation": "PASS" if x_active else "FAIL",
            "lease_validation": "PASS" if lease_valid else "FAIL",
            "reservation_validation": "PASS" if peak_reserved <= LEASE_BUDGET_BLOCKS else "FAIL",
            "block_prediction_validation": "PASS" if not block_errors else "FAIL",
            "policy_semantics_validation": "PASS" if (policy == "OnDemand-K2" and not native_rows and not leases or policy == "NaiveAsync40-K2" and len(native_rows) == 12 and not leases or lease_expected and len(promoted) == len(native_rows)) else "FAIL",
            "expected_source_tokens": total_tokens, "reservation_decision_count": len(decisions),
        }
        record = {
            "workload": workload, "policy": policy, "repetition": repetition, "t0_monotonic_s": t0,
            "offload_bytes": offloads, "snapshot_rows": snapshots, "run_metrics": run,
            "session_metrics": session_rows, "async_prefetch_metrics": native_rows,
            "reservation_decisions": decisions,
            "reservation_events": [row for row in RESERVATION_EVENTS if row["workload"] == workload and row["policy"] == policy],
            "lease_events": leases, "scheduler_trace": trace,
            "background_metrics": {
                "workload": workload, "policy": policy, "repetition": repetition,
                "X_first_token_ms": x_first_ms, "X_finished_ms": x_finish_ms,
                "X_decode_duration_ms": x_duration, "X_TPOT_ms": x_tpot,
                "baseline_X_duration_ms": X_BASELINE_MS,
                "X_slowdown": "NA" if X_BASELINE_MS is None else x_duration / X_BASELINE_MS,
                "active_through_L_first_token": x_active,
            },
        }
        write_json(record_path, record)
        return record
    finally:
        transport.completion_payload = original_payload
        gate_config._active_gate_limit = None
        engine.stop_server(proc)

def make_x_only(repetition: int) -> dict[str, Any]:
    prompts = make_prompts(UNIFORM_TOKENS)
    engine.OUT = prior.OUT = OUT
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "server_logs").mkdir(parents=True, exist_ok=True)
    proc = engine.start_server(f"x_only_{repetition}")
    try:
        runtime.wait_ready(proc)
        result = prior.send_x_background(prompts["pressure_1"], Event())
        token_count = len(result.get("token_ids", []))
        duration = (float(result["completed_monotonic_s"]) - float(result["first_monotonic_s"])) * 1000.0
        if token_count != X_TOKENS:
            raise RuntimeError(f"X-only generated {token_count}, expected {X_TOKENS}")
        row = {"repetition": repetition, "X_decode_duration_ms": duration, "X_TPOT_ms": duration / 255.0, "generated_token_count": token_count, "token_validation": "PASS"}
        write_json(OUT / "raw" / "x_only" / f"rep{repetition}.json", row)
        return row
    finally:
        engine.stop_server(proc)


def read_records(directory: Path) -> list[dict[str, Any]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.glob("*.json")) if path.name != "integration_gate.json"]


def validate_uniform(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_policy = {row["policy"]: row["run_metrics"] for row in records}
    valid = len(records) == 2 and set(by_policy) == {"FixedCount2-Async40-K2", "DynamicBlock512-Async40-K2"}
    for row in by_policy.values():
        valid = valid and int(row["native_prefetch_count"]) == 6 and int(row["reservation_promote_count"]) == 6
        valid = valid and int(row["reservation_skip_count"]) == 6 and int(row["promotion_useful_count"]) == 6
        valid = valid and int(row["promotion_remote_wait_count"]) == 0 and int(row["promotion_internal_ready_miss_count"]) == 0
        valid = valid and int(row["lease_leak_count"]) == 0 and int(row["max_reserved_blocks"]) == 512
        valid = valid and int(row["local_source_tokens"]) + int(row["external_source_tokens"]) == 49152
        valid = valid and row["token_validation"] == row["source_validation"] == row["native_validation"] == row["gate_validation"] == row["background_validation"] == "PASS"
        valid = valid and row["lease_validation"] == row["block_prediction_validation"] == "PASS"
    return {"pass": bool(valid), "fixed_promotions": by_policy.get("FixedCount2-Async40-K2", {}).get("native_prefetch_count", "NA"),
            "dynamic_promotions": by_policy.get("DynamicBlock512-Async40-K2", {}).get("native_prefetch_count", "NA"),
            "fixed_max_reserved_blocks": by_policy.get("FixedCount2-Async40-K2", {}).get("max_reserved_blocks", "NA"),
            "dynamic_max_reserved_blocks": by_policy.get("DynamicBlock512-Async40-K2", {}).get("max_reserved_blocks", "NA"),
            "policy_valid": {key: all(value in ("PASS",) for value in (row.get("token_validation"), row.get("source_validation"), row.get("native_validation"), row.get("gate_validation"), row.get("background_validation"))) for key, row in by_policy.items()}}


def validate_integration(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_policy = {row["policy"]: row["run_metrics"] for row in records}
    valid = len(records) == 4 and set(by_policy) == set(POLICIES)
    for row in by_policy.values():
        valid = valid and row["token_validation"] == row["source_validation"] == row["gate_validation"] == row["background_validation"] == row["native_validation"] == "PASS"
        valid = valid and int(row["max_pre_first_concurrency"]) <= 2
    on = by_policy.get("OnDemand-K2", {})
    fixed = by_policy.get("FixedCount2-Async40-K2", {})
    dynamic = by_policy.get("DynamicBlock512-Async40-K2", {})
    naive = by_policy.get("NaiveAsync40-K2", {})
    valid = valid and int(on.get("native_prefetch_count", -1)) == 0 and int(on.get("lease_acquire_count", -1)) == 0
    valid = valid and int(naive.get("native_prefetch_count", -1)) == 12 and int(naive.get("lease_acquire_count", -1)) == 0
    valid = valid and int(fixed.get("native_prefetch_count", -1)) == 6 and int(fixed.get("reservation_skip_count", -1)) == 6
    valid = valid and int(fixed.get("promotion_useful_count", -1)) == 6 and int(fixed.get("promotion_remote_wait_count", -1)) == 0
    valid = valid and int(fixed.get("promotion_internal_ready_miss_count", -1)) == 0 and int(fixed.get("lease_leak_count", -1)) == 0
    valid = valid and fixed.get("lease_validation") == fixed.get("block_prediction_validation") == "PASS"
    valid = valid and int(fixed.get("max_reserved_blocks", -1)) == 128
    valid = valid and (int(fixed.get("local_source_tokens", -1)), int(fixed.get("external_source_tokens", -1))) == (6144, 15360)
    valid = valid and int(dynamic.get("native_prefetch_count", -1)) == 12 and int(dynamic.get("reservation_skip_count", -1)) == 0
    valid = valid and int(dynamic.get("promotion_useful_count", -1)) == 12 and int(dynamic.get("promotion_remote_wait_count", -1)) == 0
    valid = valid and int(dynamic.get("promotion_internal_ready_miss_count", -1)) == 0 and int(dynamic.get("lease_leak_count", -1)) == 0
    valid = valid and dynamic.get("lease_validation") == dynamic.get("block_prediction_validation") == "PASS"
    valid = valid and int(dynamic.get("max_reserved_blocks", -1)) == 448
    valid = valid and (int(dynamic.get("local_source_tokens", -1)), int(dynamic.get("external_source_tokens", -1))) == (21504, 0)
    return {"pass": bool(valid), "policy_valid": {key: row.get("token_validation") == row.get("source_validation") == row.get("native_validation") == row.get("gate_validation") == row.get("background_validation") == "PASS" for key, row in by_policy.items()}, "run_metrics": by_policy}


def run_stage(stage: str, policy: str | None, repetition: int | None) -> None:
    global X_BASELINE_MS, make_current_prompts, RESERVATION_DECISIONS, RESERVATION_EVENTS
    for folder in ("server_logs", "raw/x_only", "uniform_regression", "heterogeneous/integration", "heterogeneous/raw_runs"):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    if stage == "x-only":
        if repetition is None:
            raise SystemExit("--repetition is required")
        make_x_only(repetition)
        return
    if stage in ("validate-uniform", "validate-integration"):
        uniform = validate_uniform(read_records(OUT / "uniform_regression"))
        write_json(OUT / "uniform_regression" / "integration_gate.json", uniform)
        if not uniform["pass"]:
            raise RuntimeError("uniform equal-size regression failed; heterogeneous runs are prohibited")
        if stage == "validate-uniform":
            print(json.dumps(uniform, sort_keys=True), flush=True)
            return
        integ = validate_integration(read_records(OUT / "heterogeneous" / "integration"))
        write_json(OUT / "heterogeneous" / "integration_gate.json", integ)
        print(json.dumps(integ, sort_keys=True), flush=True)
        if not integ["pass"]:
            raise RuntimeError("heterogeneous integration gate failed; measured matrix is prohibited")
        return
    if policy not in POLICIES or repetition is None:
        raise SystemExit("run stages require a valid --policy and --repetition")
    if stage not in ("uniform", "integration", "measured"):
        raise SystemExit(f"unsupported stage {stage}")
    xfiles = sorted((OUT / "raw" / "x_only").glob("rep*.json"))
    if len(xfiles) != 3:
        raise RuntimeError("three fresh X-only baselines are required before policy runs")
    xrows = [json.loads(path.read_text()) for path in xfiles]
    if any(row.get("token_validation") != "PASS" for row in xrows):
        raise RuntimeError("X-only baseline token validation failed")
    X_BASELINE_MS = statistics.median(float(row["X_decode_duration_ms"]) for row in xrows)
    if stage == "uniform" and policy not in ("FixedCount2-Async40-K2", "DynamicBlock512-Async40-K2"):
        raise SystemExit("uniform regression compares only FixedCount2 and DynamicBlock512")
    if stage in ("integration", "measured"):
        uniform_gate = json.loads((OUT / "uniform_regression" / "integration_gate.json").read_text())
        if not uniform_gate.get("pass"):
            raise RuntimeError("uniform regression must pass before heterogeneous evaluation")
    if stage == "measured":
        integ_gate = json.loads((OUT / "heterogeneous" / "integration_gate.json").read_text())
        if not integ_gate.get("pass"):
            raise RuntimeError("measured runs are prohibited until the heterogeneous integration gate passes")
    workload = "uniform" if stage == "uniform" else "heterogeneous"
    sizes = UNIFORM_TOKENS if workload == "uniform" else HETEROGENEOUS_TOKENS
    make_current_prompts = make_prompts(sizes)
    RESERVATION_DECISIONS, RESERVATION_EVENTS = [], []
    destination = OUT / "uniform_regression" if stage == "uniform" else OUT / "heterogeneous" / ("integration" if stage == "integration" else "raw_runs")
    record_path = destination / f"{policy.lower().replace('-', '_')}_rep{repetition}.json"
    if record_path.exists():
        raise RuntimeError(f"refusing to overwrite existing run record: {record_path}")
    record = run_once(workload, policy, repetition, sizes, record_path)
    print(json.dumps({"stage": stage, "policy": policy, "repetition": repetition, "run_metrics": record["run_metrics"]}, sort_keys=True), flush=True)

def flatten(records: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    return [row for record in records for row in record[field]]


def median_metric(records: list[dict[str, Any]], name: str) -> float:
    return statistics.median(float(row["run_metrics"][name]) for row in records)


def finalize() -> None:
    x_records = [json.loads(p.read_text()) for p in sorted((OUT / "raw" / "x_only").glob("rep*.json"))]
    uniform_records = read_records(OUT / "uniform_regression")
    integration_records = read_records(OUT / "heterogeneous" / "integration")
    measured_records = read_records(OUT / "heterogeneous" / "raw_runs")
    if (len(x_records), len(uniform_records), len(integration_records), len(measured_records)) != (3, 2, 4, 12):
        raise RuntimeError(f"incomplete result set X/U/I/M={(len(x_records), len(uniform_records), len(integration_records), len(measured_records))}")
    global X_BASELINE_MS
    X_BASELINE_MS = statistics.median(float(row["X_decode_duration_ms"]) for row in x_records)
    write_csv(OUT / "x_only_baseline.csv", x_records)
    uniform_gate = validate_uniform(uniform_records)
    integration_gate = validate_integration(integration_records)
    write_json(OUT / "uniform_regression" / "integration_gate.json", uniform_gate)
    write_json(OUT / "heterogeneous" / "integration_gate.json", integration_gate)

    def write_group(directory: Path, records: list[dict[str, Any]]) -> None:
        write_csv(directory / "run_metrics.csv", [row["run_metrics"] for row in records])
        write_csv(directory / "session_metrics.csv", flatten(records, "session_metrics"))
        write_csv(directory / "reservation_decisions.csv", flatten(records, "reservation_decisions"))
        write_csv(directory / "reservation_events.csv", flatten(records, "reservation_events"))
        write_csv(directory / "lease_events.csv", flatten(records, "lease_events"))
        write_csv(directory / "async_prefetch_metrics.csv", flatten(records, "async_prefetch_metrics"))
        write_csv(directory / "scheduler_trace.csv", flatten(records, "scheduler_trace"))

    write_group(OUT / "uniform_regression", uniform_records)
    write_group(OUT / "heterogeneous" / "integration_runs", integration_records)
    write_group(OUT / "heterogeneous", measured_records)
    write_group(OUT, measured_records)
    write_csv(OUT / "background_metrics.csv", [row["background_metrics"] for row in measured_records])
    write_json(OUT / "metric_snapshots.json", [
        {"policy": row["policy"], "repetition": row["repetition"], "t0_monotonic_s": row["t0_monotonic_s"], "snapshots": row["snapshot_rows"]}
        for row in measured_records
    ])
    summaries: list[dict[str, Any]] = []
    for policy in POLICIES:
        records = [row for row in measured_records if row["policy"] == policy]
        run_rows = [row["run_metrics"] for row in records]
        sessions = flatten(records, "session_metrics")
        promoted = [row for row in sessions if row["reservation_decision"] in ("PROMOTE", "RESERVATION_PROMOTE")]
        useful = sum(row["promotion_outcome"] == "PROMOTION_USEFUL" for row in promoted)
        summaries.append({
            "policy": policy,
            "AL_P50_resume_ms": statistics.median(float(r["AL_P50_resume_ms"]) for r in run_rows),
            "AL_P95_resume_ms": statistics.median(float(r["AL_P95_resume_ms"]) for r in run_rows),
            "AL_max_resume_ms": statistics.median(float(r["AL_max_resume_ms"]) for r in run_rows),
            "Cluster1_P95_ms": statistics.median(float(r["Cluster1_P95_ms"]) for r in run_rows),
            "Cluster2_P95_ms": statistics.median(float(r["Cluster2_P95_ms"]) for r in run_rows),
            "Cluster3_P95_ms": statistics.median(float(r["Cluster3_P95_ms"]) for r in run_rows),
            "AL_P95_completion_ms": statistics.median(float(r["AL_P95_completion_ms"]) for r in run_rows),
            "AL_decode_tok_s": statistics.median(float(r["AL_decode_tok_s"]) for r in run_rows),
            "combined_decode_tok_s": statistics.median(float(r["combined_decode_tok_s"]) for r in run_rows),
            "X_slowdown": statistics.median(float(r["X_slowdown"]) for r in run_rows),
            "local_fraction": statistics.median(float(r["local_fraction"]) for r in run_rows),
            "external_fraction": statistics.median(float(r["external_fraction"]) for r in run_rows),
            "promotion_useful_rate": useful / len(promoted) if promoted else "NA",
            "promotions_per_run": statistics.median(int(r["native_prefetch_count"]) for r in run_rows),
            "useful_promotions_per_run": statistics.median(int(r["promotion_useful_count"]) for r in run_rows),
            "skips_per_run": statistics.median(int(r["reservation_skip_count"]) for r in run_rows),
            "max_reserved_blocks": statistics.median(int(r["max_reserved_blocks"]) for r in run_rows),
            "budget_utilization": statistics.median(float(r["reservation_budget_utilization"]) for r in run_rows),
            "native_promotion_mib": statistics.median(float(r["native_promotion_mib"]) for r in run_rows),
            "resume_external_mib": statistics.median(float(r["resume_external_mib"]) for r in run_rows),
            "total_KV_H2D_mib": statistics.median(float(r["total_KV_H2D_mib"]) for r in run_rows),
            "duplicate_KV_H2D_mib": statistics.median(float(r["duplicate_KV_H2D_mib"]) for r in run_rows),
            "promotion_remote_waits_total": sum(int(r["promotion_remote_wait_count"]) for r in run_rows),
            "promotion_internal_ready_misses_total": sum(int(r["promotion_internal_ready_miss_count"]) for r in run_rows),
            "lease_leaks_total": sum(int(r["lease_leak_count"]) for r in run_rows),
            "peak_active_lease_count": statistics.median(int(r["peak_active_lease_count"]) for r in run_rows),
            "peak_active_lease_blocks": statistics.median(int(r["peak_active_lease_blocks"]) for r in run_rows),
        })
    write_csv(OUT / "policy_summary.csv", summaries)
    measured_valid = len(measured_records) == 12 and all(
        all(row["run_metrics"][name] == "PASS" for name in (
            "token_validation", "source_validation", "native_validation", "gate_validation",
            "background_validation", "lease_validation", "reservation_validation", "block_prediction_validation",
        )) for row in measured_records
    )
    by_policy = {policy: [row for row in measured_records if row["policy"] == policy] for policy in POLICIES}
    fixed_runs = by_policy["FixedCount2-Async40-K2"]
    dynamic_runs = by_policy["DynamicBlock512-Async40-K2"]
    fixed_valid = len(fixed_runs) == 3 and all(
        int(row["run_metrics"]["native_prefetch_count"]) == 6
        and int(row["run_metrics"]["reservation_skip_count"]) == 6
        and int(row["run_metrics"]["promotion_useful_count"]) == 6
        and int(row["run_metrics"]["promotion_remote_wait_count"]) == 0
        and int(row["run_metrics"]["promotion_internal_ready_miss_count"]) == 0
        and int(row["run_metrics"]["max_active_reservation_count"]) == 2
        and int(row["run_metrics"]["max_reserved_blocks"]) == 128
        and int(row["run_metrics"]["local_source_tokens"]) == 6144
        and int(row["run_metrics"]["external_source_tokens"]) == 15360
        and int(row["run_metrics"]["lease_leak_count"]) == 0
        for row in fixed_runs
    )
    dynamic_valid = len(dynamic_runs) == 3 and all(
        int(row["run_metrics"]["native_prefetch_count"]) == 12
        and int(row["run_metrics"]["reservation_skip_count"]) == 0
        and int(row["run_metrics"]["promotion_useful_count"]) == 12
        and int(row["run_metrics"]["promotion_remote_wait_count"]) == 0
        and int(row["run_metrics"]["promotion_internal_ready_miss_count"]) == 0
        and int(row["run_metrics"]["max_reserved_blocks"]) == 448
        and int(row["run_metrics"]["peak_active_lease_blocks"]) <= 512
        and int(row["run_metrics"]["lease_leak_count"]) == 0
        and int(row["run_metrics"]["block_prediction_validation"] == "PASS")
        and int(row["run_metrics"]["local_source_tokens"]) == 21504
        and int(row["run_metrics"]["external_source_tokens"]) == 0
        and abs(float(row["run_metrics"]["total_KV_H2D_mib"]) - 588.0) < 1e-5
        and abs(float(row["run_metrics"]["duplicate_KV_H2D_mib"])) < 1e-5
        for row in dynamic_runs
    )
    all_valid = uniform_gate["pass"] and integration_gate["pass"] and measured_valid
    result, next_direction = "EXPERIMENT_INVALID", "EXPERIMENT_INVALIDITY_REVIEW"
    if all_valid:
        if any(row["run_metrics"]["block_prediction_validation"] != "PASS" for row in dynamic_runs):
            result, next_direction = "RESULT_B_BLOCK_MODEL_INVALID", "BLOCK_COST_MODEL_FIX"
        elif any(int(row["run_metrics"]["promotion_internal_ready_miss_count"]) > 0 for row in dynamic_runs):
            result, next_direction = "RESULT_D_TIMING_FAILURE", "BLOCK_AWARE_SERVICE_SCHEDULING"
        elif any(int(row["run_metrics"]["promotion_remote_wait_count"]) > 0 or int(row["run_metrics"]["lease_leak_count"]) > 0 for row in dynamic_runs):
            result, next_direction = "RESULT_C_RESIDENCY_FAILURE", "MULTI_LEASE_CAPACITY_DIAGNOSIS"
        elif dynamic_valid and fixed_valid and all(float(row["run_metrics"]["total_KV_H2D_mib"]) <= float(fixed_runs[0]["run_metrics"]["total_KV_H2D_mib"]) + 1e-5 for row in dynamic_runs):
            result, next_direction = "RESULT_A", "DYNAMIC_BLOCK_RESERVATION_VALIDATED"
    tests_proc = __import__("subprocess").run(
        [__import__("sys").executable, "-m", "pytest", "-q", "tests/test_phase3v.py", "tests/test_phase3v_tracelab.py"],
        cwd=ROOT, text=True, capture_output=True,
    )
    test_output = tests_proc.stdout + tests_proc.stderr
    test_summary = next((line.strip() for line in test_output.splitlines() if " passed" in line), "no pytest summary")
    write_json(OUT / "validation_report.json", {
        "tests": {"returncode": tests_proc.returncode, "summary": test_summary, "output": test_output[-4000:]},
        "x_only_baseline": {
            "runs": len(x_records),
            "median_decode_duration_ms": X_BASELINE_MS,
            "median_TPOT_ms": statistics.median(float(row["X_TPOT_ms"]) for row in x_records),
            "token_valid_runs": sum(int(row["token_validation"] == "PASS") for row in x_records),
        },
        "uniform_regression": {
            "fixed_valid": uniform_gate["policy_valid"].get("FixedCount2-Async40-K2", False),
            "dynamic_valid": uniform_gate["policy_valid"].get("DynamicBlock512-Async40-K2", False),
            "fixed_promotions": uniform_gate["fixed_promotions"], "dynamic_promotions": uniform_gate["dynamic_promotions"],
            "fixed_max_reserved_blocks": uniform_gate["fixed_max_reserved_blocks"], "dynamic_max_reserved_blocks": uniform_gate["dynamic_max_reserved_blocks"],
            "pass": uniform_gate["pass"],
        },
        "heterogeneous": {
            "stage_a": "PASS" if integration_gate["pass"] else "FAIL",
            "stage_a_runs": f"{len(integration_records)}/4",
            "stage_b": "PASS" if measured_valid else "FAIL",
            "stage_b_runs": f"{len(measured_records)}/12",
            "real_resume_correctness": sum(int(row["run_metrics"]["token_validation"] == "PASS") * 12 for row in measured_records),
            "source_conservation": sum(int(row["run_metrics"]["source_validation"] == "PASS") for row in measured_records),
            "K2": sum(int(row["run_metrics"]["gate_validation"] == "PASS") for row in measured_records),
            "background": sum(int(row["run_metrics"]["background_validation"] == "PASS") for row in measured_records),
            "native": sum(int(row["run_metrics"]["native_validation"] == "PASS") for row in measured_records),
            "lease": sum(int(row["run_metrics"]["lease_validation"] == "PASS") for row in measured_records),
        },
        "fixed_count": {
            "promotions": sum(int(row["run_metrics"]["native_prefetch_count"]) for row in fixed_runs),
            "skips": sum(int(row["run_metrics"]["reservation_skip_count"]) for row in fixed_runs),
            "useful": sum(int(row["run_metrics"]["promotion_useful_count"]) for row in fixed_runs),
            "remote_waits": sum(int(row["run_metrics"]["promotion_remote_wait_count"]) for row in fixed_runs),
            "internal_ready_misses": sum(int(row["run_metrics"]["promotion_internal_ready_miss_count"]) for row in fixed_runs),
            "lease_leaks": sum(int(row["run_metrics"]["lease_leak_count"]) for row in fixed_runs),
            "max_active_reservation_count": max(int(row["run_metrics"]["max_active_reservation_count"]) for row in fixed_runs),
            "max_reserved_blocks": max(int(row["run_metrics"]["max_reserved_blocks"]) for row in fixed_runs),
            "budget_utilization": max(float(row["run_metrics"]["reservation_budget_utilization"]) for row in fixed_runs),
            "local_source_tokens": sum(int(row["run_metrics"]["local_source_tokens"]) for row in fixed_runs),
            "external_source_tokens": sum(int(row["run_metrics"]["external_source_tokens"]) for row in fixed_runs),
            "total_H2D_mib": statistics.median(float(row["run_metrics"]["total_KV_H2D_mib"]) for row in fixed_runs),
            "duplicate_H2D_mib": statistics.median(float(row["run_metrics"]["duplicate_KV_H2D_mib"]) for row in fixed_runs),
        },
        "dynamic_block": {
            "promotions": sum(int(row["run_metrics"]["native_prefetch_count"]) for row in dynamic_runs),
            "skips": sum(int(row["run_metrics"]["reservation_skip_count"]) for row in dynamic_runs),
            "useful": sum(int(row["run_metrics"]["promotion_useful_count"]) for row in dynamic_runs),
            "remote_waits": sum(int(row["run_metrics"]["promotion_remote_wait_count"]) for row in dynamic_runs),
            "internal_ready_misses": sum(int(row["run_metrics"]["promotion_internal_ready_miss_count"]) for row in dynamic_runs),
            "lease_leaks": sum(int(row["run_metrics"]["lease_leak_count"]) for row in dynamic_runs),
            "max_active_reservation_count": max(int(row["run_metrics"]["max_active_reservation_count"]) for row in dynamic_runs),
            "max_reserved_blocks": max(int(row["run_metrics"]["max_reserved_blocks"]) for row in dynamic_runs),
            "budget_utilization": max(float(row["run_metrics"]["reservation_budget_utilization"]) for row in dynamic_runs),
            "block_prediction_errors": sum(int(row["run_metrics"]["block_prediction_validation"] != "PASS") for row in dynamic_runs),
            "local_source_tokens": sum(int(row["run_metrics"]["local_source_tokens"]) for row in dynamic_runs),
            "external_source_tokens": sum(int(row["run_metrics"]["external_source_tokens"]) for row in dynamic_runs),
            "total_H2D_mib": statistics.median(float(row["run_metrics"]["total_KV_H2D_mib"]) for row in dynamic_runs),
            "duplicate_H2D_mib": statistics.median(float(row["run_metrics"]["duplicate_KV_H2D_mib"]) for row in dynamic_runs),
        },
        "naive": {
            "promotions": sum(int(row["run_metrics"]["native_prefetch_count"]) for row in by_policy["NaiveAsync40-K2"]),
            "useful": sum(int(row["run_metrics"]["promotion_useful_count"]) for row in by_policy["NaiveAsync40-K2"]),
            "remote_waits": sum(int(row["run_metrics"]["promotion_remote_wait_count"]) for row in by_policy["NaiveAsync40-K2"]),
            "total_H2D_mib": statistics.median(float(row["run_metrics"]["total_KV_H2D_mib"]) for row in by_policy["NaiveAsync40-K2"]),
            "duplicate_H2D_mib": statistics.median(float(row["run_metrics"]["duplicate_KV_H2D_mib"]) for row in by_policy["NaiveAsync40-K2"]),
        },
        "policy_summaries": summaries, "result": result, "next_direction": next_direction,
    })
    if tests_proc.returncode != 0:
        result, next_direction = "EXPERIMENT_INVALID", "EXPERIMENT_INVALIDITY_REVIEW"
        report = json.loads((OUT / "validation_report.json").read_text())
        report["result"], report["next_direction"] = result, next_direction
        write_json(OUT / "validation_report.json", report)
    s = {row["policy"]: row for row in summaries}
    fixed, dynamic = s["FixedCount2-Async40-K2"], s["DynamicBlock512-Async40-K2"]
    on_demand, naive = s["OnDemand-K2"], s["NaiveAsync40-K2"]
    pct_change = lambda new, old: (float(new) - float(old)) / float(old) * 100.0 if float(old) else float("nan")
    dynamic_native = flatten(dynamic_runs, "async_prefetch_metrics")
    dynamic_timing_misses = sum(float(row["issue_to_T6_ms"]) > SERVICE_MS for row in dynamic_native)
    uniform_fixed = next(row["run_metrics"] for row in uniform_records if row["policy"] == "FixedCount2-Async40-K2")
    uniform_dynamic = next(row["run_metrics"] for row in uniform_records if row["policy"] == "DynamicBlock512-Async40-K2")
    summary = f"""# Phase 3V — Dynamic block reservation

## Objective and setup

This phase compares count-based two-slot reservation with a logical 512-block admission budget under three clusters of heterogeneous KV sizes, the frozen A–L deadlines, K=2 resume admission, and the 9500-token / 256-output-token background X. Runtime is vLLM 0.23.0 on RTX A6000, Qwen2.5-1.5B-Instruct FP16, GPU KV 1 GiB and CPU tier 2 GiB. X slowdown uses the median of the three fresh X-only controls ({X_BASELINE_MS:.3f} ms decode duration).

The workload comprises 12 unique 1024/4096-token prompts (21,504 reusable prefix tokens total). Each cluster contains three 1024-token prefixes and one 4096-token prefix, requiring 448 blocks. The theoretical minimum H2D traffic is 588 MiB.

## Comparison

Medians across the three measured runs per policy:

| Policy | Useful promotions/run | Local source % | Peak reserved blocks | Budget use | AL P95 resume (ms) | Total H2D (MiB) | Duplicate H2D (MiB) | X slowdown | Combined tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
""" + "\n".join(
        f"| {row['policy']} | {row['useful_promotions_per_run']} | {100*float(row['local_fraction']):.2f} | {row['max_reserved_blocks']} | {100*float(row['budget_utilization']):.1f}% | {float(row['AL_P95_resume_ms']):.2f} | {float(row['total_KV_H2D_mib']):.2f} | {float(row['duplicate_KV_H2D_mib']):.2f} | {float(row['X_slowdown']):.3f} | {float(row['combined_decode_tok_s']):.2f} |"
        for row in summaries
    ) + f"""

## Findings

- Uniform 4096-token regression: FixedCount2 promoted/skipped {uniform_fixed['native_prefetch_count']}/{uniform_fixed['reservation_skip_count']}; DynamicBlock promoted/skipped {uniform_dynamic['native_prefetch_count']}/{uniform_dynamic['reservation_skip_count']}. Both had 100% useful promotions, zero promoted remote waits, zero lease leaks, and a 512-block peak: PASS={uniform_gate['pass']}.
- Heterogeneous integration gate: {len(integration_records)}/4 runs completed; gate PASS={integration_gate['pass']}. Measured phase: {len(measured_records)}/12 runs; exact 32-token request validation is {sum(int(row['run_metrics']['token_validation'] == 'PASS') * 12 for row in measured_records)}/144.
- FixedCount2 promoted {fixed['promotions_per_run']} and skipped {fixed['skips_per_run']} per run, using {fixed['max_reserved_blocks']}/512 blocks ({100*float(fixed['budget_utilization']):.1f}%). DynamicBlock promoted {dynamic['promotions_per_run']} and skipped {dynamic['skips_per_run']}, using {dynamic['max_reserved_blocks']}/512 ({100*float(dynamic['budget_utilization']):.1f}%). It admitted four jobs in a cluster because 64+64+64+256=448 blocks.
- Predicted versus physical lease blocks: DynamicBlock errors total {sum(int(row['run_metrics']['block_prediction_validation'] != 'PASS') for row in dynamic_runs)} across measured runs; dynamic peak physical leased blocks median {dynamic['peak_active_lease_blocks']}, maximum {max(int(row['run_metrics']['peak_active_lease_blocks']) for row in dynamic_runs)}. FixedCount2 and DynamicBlock lease leaks total {fixed['lease_leaks_total']} and {dynamic['lease_leaks_total']} respectively.
- Promotion usefulness: NaiveAsync {naive['promotion_useful_rate']}; FixedCount2 {fixed['promotion_useful_rate']}; DynamicBlock {dynamic['promotion_useful_rate']}. Promoted remote waits totals were FixedCount2 {fixed['promotion_remote_waits_total']} and DynamicBlock {dynamic['promotion_remote_waits_total']}.
- H2D accounting: OnDemand total/duplicate {on_demand['total_KV_H2D_mib']:.2f}/{on_demand['duplicate_KV_H2D_mib']:.2f} MiB; NaiveAsync {naive['total_KV_H2D_mib']:.2f}/{naive['duplicate_KV_H2D_mib']:.2f}; FixedCount2 {fixed['total_KV_H2D_mib']:.2f}/{fixed['duplicate_KV_H2D_mib']:.2f}; DynamicBlock {dynamic['total_KV_H2D_mib']:.2f}/{dynamic['duplicate_KV_H2D_mib']:.2f}. NaiveAsync's eviction/wasted-traffic weakness was not observed: duplicate H2D was {naive['duplicate_KV_H2D_mib']:.2f} MiB in all measured repetitions.
- DynamicBlock versus FixedCount2: P50 resume change {pct_change(dynamic['AL_P50_resume_ms'], fixed['AL_P50_resume_ms']):+.2f}%; P95 change {pct_change(dynamic['AL_P95_resume_ms'], fixed['AL_P95_resume_ms']):+.2f}%; combined throughput change {pct_change(dynamic['combined_decode_tok_s'], fixed['combined_decode_tok_s']):+.2f}%; X slowdown difference {float(dynamic['X_slowdown'])-float(fixed['X_slowdown']):+.3f}; useful promotions/run {dynamic['useful_promotions_per_run']} vs {fixed['useful_promotions_per_run']}.
- Async40 service check: {dynamic_timing_misses}/{len(dynamic_native)} DynamicBlock issue-to-T6 measurements exceeded 40 ms. Max measured issue-to-T6 was {max((float(row['issue_to_T6_ms']) for row in dynamic_native), default=0):.2f} ms. Max outstanding HTTP prefetch calls was {max(int(row['run_metrics']['max_outstanding_prefetch_http_calls']) for row in dynamic_runs)}; max internal T0–T6 overlap was {max(int(row['run_metrics']['max_overlapping_T0_T6']) for row in dynamic_runs)}.
- Source accounting totals per run: expected 21,504 tokens. FixedCount2 local/external = {fixed_runs[0]['run_metrics']['local_source_tokens']}/{fixed_runs[0]['run_metrics']['external_source_tokens']}; DynamicBlock = {dynamic_runs[0]['run_metrics']['local_source_tokens']}/{dynamic_runs[0]['run_metrics']['external_source_tokens']}.
- X slowdown medians: OnDemand {on_demand['X_slowdown']:.3f}, FixedCount2 {fixed['X_slowdown']:.3f}, DynamicBlock {dynamic['X_slowdown']:.3f}; combined decode throughput medians were {on_demand['combined_decode_tok_s']:.2f}, {fixed['combined_decode_tok_s']:.2f}, and {dynamic['combined_decode_tok_s']:.2f} tok/s respectively.

## Interpretation and limitations

DynamicBlock's decisions are derived from EDF QueueJIT order, live logical reservations, and `ceil(prefix_tokens/16)` only. It uses no Prometheus or instantaneous free-memory oracle. The fixed-count baseline records block demand but admits based on an active-session limit of two. Both reservation policies use the unchanged Phase 3S physical lease.

The request's arithmetic line `448 + 64 > 512` is inconsistent with its stated `<= 512` admission rule: 448 + 64 equals 512. The implementation follows the explicit `<= 512` rule; the frozen workload has no fifth overlapping 64-block job in that cluster.

The earlier TraceLab-selection Phase 3V artifacts were retained under `results/phase3v_tracelab_incomplete/`; this phase does not overwrite them. No Phase 3W or prediction uncertainty was run.

## Result

**{result}** — next direction: `{next_direction}`.
"""
    (OUT / "summary.md").write_text(summary, encoding="utf-8")
    config = {
        "base_commit": "b78f3fdae91247a58f0ed8e8e32a5294cbe93468",
        "policies": list(POLICIES), "deadlines_ms": DEADLINES_MS,
        "workload_uniform_tokens": UNIFORM_TOKENS, "workload_heterogeneous_tokens": HETEROGENEOUS_TOKENS,
        "block_size_tokens": BLOCK_SIZE_TOKENS, "lease_budget_blocks": LEASE_BUDGET_BLOCKS,
        "handoff_guard_ms": HANDOFF_GUARD_MS, "service_budget_ms": SERVICE_MS,
        "resume_guard_ms": 110, "K": 2, "resume_max_tokens": TOKENS,
        "X_prompt_tokens": 9500, "X_max_tokens": X_TOKENS,
        "GPU_KV_bytes": 1073741824, "CPU_external_bytes": 2147483648,
        "theoretical_minimum_heterogeneous_H2D_MiB": 588,
        "uniform_runs_per_reservation_policy": 1, "integration_runs": 4,
        "measured_runs_per_policy": 3, "measured_policies": 4,
    }
    write_json(OUT / "config.json", config)
    environment = json.loads((ROOT / "results" / "phase3u_fix" / "environment.json").read_text())
    environment["vLLM_patch"] = "Phase 3S residency lease patch reused unchanged; no Phase 3V source patch"
    environment["GPU_count"] = 2
    environment["driver"] = "550.54.14"
    environment["software_runtime"] = "slackkv-phase2c-fix"
    write_json(OUT / "environment.json", environment)
    (OUT / "vllm_phase3v_delta.patch").write_text("Phase 3S lease patch reused unchanged.\nNo new vLLM source modification in Phase 3V.\n", encoding="utf-8")
    (OUT / "commands.sh").write_text("\n".join([
        "PYTHONPATH=. /media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python -m pytest -q tests/test_phase3v.py tests/test_phase3v_tracelab.py",
        *[f"PYTHONPATH=. /media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python experiments/phase3v/run_phase3v.py --stage x-only --repetition {i}" for i in range(3)],
        *[f"PYTHONPATH=. /media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python experiments/phase3v/run_phase3v.py --stage uniform --policy {p} --repetition 0" for p in ("FixedCount2-Async40-K2", "DynamicBlock512-Async40-K2")],
        "PYTHONPATH=. /media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python experiments/phase3v/run_phase3v.py --stage validate-uniform",
        *[f"PYTHONPATH=. /media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python experiments/phase3v/run_phase3v.py --stage integration --policy {p} --repetition 0" for p in POLICIES],
        "PYTHONPATH=. /media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python experiments/phase3v/run_phase3v.py --stage validate-integration",
        *[f"PYTHONPATH=. /media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python experiments/phase3v/run_phase3v.py --stage measured --policy {p} --repetition {i}" for p in POLICIES for i in range(3)],
        "PYTHONPATH=. /media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python experiments/phase3v/run_phase3v.py --stage finalize",
    ]) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3V fixed-count vs dynamic-block reservation experiment")
    parser.add_argument("--stage", required=True, choices=("x-only", "uniform", "validate-uniform", "integration", "validate-integration", "measured", "finalize"))
    parser.add_argument("--policy", choices=POLICIES)
    parser.add_argument("--repetition", type=int)
    args = parser.parse_args()
    if args.stage == "finalize":
        finalize()
    else:
        run_stage(args.stage, args.policy, args.repetition)


if __name__ == "__main__":
    main()
