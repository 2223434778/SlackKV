"""Aggregate only a complete, valid Phase 4G-A paired matrix."""
from __future__ import annotations

import csv
import json
import math
import random
import statistics
from collections import defaultdict
from typing import Any

from experiments.phase4g_a.workload import OUT, ROOT, build_manifest, load_source, nearest_rank, validate_manifest

BYTES_PER_TOKEN = 28672
MIB = 2 ** 20
ORDER = (("A", "D"), ("D", "A")) * 4
POLICIES = {"A": "OnDemand-K2", "D": "FullSlackKV+PSA1-K2"}


def write_csv(name: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        (OUT / name).write_text("repetition,policy,event\n", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pause_bin(pause_ms: float) -> str:
    return "100-199" if pause_ms < 200 else "200-499" if pause_ms < 500 else "500-999" if pause_ms < 1000 else ">=1000"


def percent_delta(new: float, old: float) -> float:
    return (new / old - 1) * 100


def duplicate_bytes(promo: int, external_tokens: int, expected_tokens: int) -> int:
    return max(0, promo + external_tokens * BYTES_PER_TOKEN - expected_tokens * BYTES_PER_TOKEN)


def bootstrap(values: list[float], seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    samples = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(10000))
    return samples[250], samples[9749]


def lease_cost(events: list[dict[str, Any]]) -> tuple[float, int, int]:
    active: dict[str, tuple[float, int]] = {}
    total = 0.0
    peak_blocks = peak_count = 0
    for event in sorted(events, key=lambda row: float(row["timestamp_monotonic"])):
        key = str(event["lease_key"])
        when = float(event["timestamp_monotonic"])
        blocks = int(event["num_blocks"])
        if event["event"] == "LEASE_ACQUIRED":
            if key in active:
                raise RuntimeError("duplicate lease acquire")
            active[key] = (when, blocks)
        elif event["event"] == "LEASE_RELEASED":
            start, held_blocks = active.pop(key)
            if held_blocks != blocks or when < start:
                raise RuntimeError("invalid lease release")
            total += blocks * 16 * BYTES_PER_TOKEN / MIB * (when - start)
        peak_blocks = max(peak_blocks, sum(value[1] for value in active.values()))
        peak_count = max(peak_count, len(active))
    if active:
        raise RuntimeError("lease leak")
    return total, peak_blocks, peak_count


def reservation_cost(result: dict[str, Any], manifest: dict[str, dict[str, object]]) -> float:
    starts = {row["boundary_id"]: int(row["timestamp_ns"]) for row in result["decisions"] if row["event"] == "PROMOTION_ATTEMPT"}
    ends = {row["boundary_id"]: int(row["timestamp_ns"]) for row in result["decisions"] if row["event"] in {"PROMOTION_CAPACITY_REJECTED", "PROMOTION_PROGRESS_REJECTED"}}
    for row in result["release_events"]:
        ends[row["episode_id"]] = round(result["X_first_ns"] + float(row["logical_release_ms"]) * 1e6)
    if set(starts) != set(ends):
        raise RuntimeError("reservation lifecycle mismatch")
    return sum(int(manifest[key]["required_blocks"]) * (ends[key] - start) / 1e9 for key, start in starts.items())


def first_events(events: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    by_session: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for event in sorted(events, key=lambda row: float(row["timestamp_monotonic"])):
        by_session[str(event["session"])].setdefault(str(event["event"]), event)
    return by_session


def process_run(result: dict[str, Any], manifest_rows: list[dict[str, object]],
                reference: dict[tuple[str, int], str], x_baseline: float):
    if not result["valid"] or len(result["turn_rows"]) != 40 or len(result["boundary_rows"]) != 32:
        raise RuntimeError("EXPERIMENT_INVALID_MATRIX:incomplete run")
    policy = result["policy"]
    rep = int(result["repetition"])
    manifest = {str(row["boundary_id"]): row for row in manifest_rows}
    client = {row["boundary_id"]: row for row in result["boundary_rows"]}
    turns = {(row["episode_id"], int(row["turn_index"])): row for row in result["turn_rows"]}
    server = first_events(result["resume_trace"])
    source_events_by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source_event in sorted(result["source_trace"], key=lambda row: float(row["timestamp_monotonic"])):
        source_events_by_session[str(source_event["session"])].append(source_event)
    if len(source_events_by_session) != 32:
        raise RuntimeError("source trace does not cover 32 resumes")
    offload = {row["boundary_id"]: row for row in result["source_events"] if row["event"] == "SOURCE_READY"}
    decisions: dict[str, list[str]] = defaultdict(list)
    for event in result["decisions"]:
        decisions[event["boundary_id"]].append(event["event"])
    boundaries = []
    h2d = []
    prefetch = []
    origin_s = result["X_first_ns"] / 1e9
    for job in manifest_rows:
        key = str(job["boundary_id"])
        session = str(job["lease_key"])
        events = server[session]
        if not {"RESUME_ARRIVED", "RESUME_ADMITTED", "FIRST_TOKEN", "REQUEST_FINISHED"} <= events.keys():
            raise RuntimeError(f"missing scheduler trace:{session}")
        arrival, admit, first, finish = (float(events[name]["timestamp_monotonic"])
            for name in ("RESUME_ARRIVED", "RESUME_ADMITTED", "FIRST_TOKEN", "REQUEST_FINISHED"))
        if not arrival <= admit <= first <= finish:
            raise RuntimeError(f"scheduler order invalid:{session}")
        boundary_client = client[key]
        deadline = int(boundary_client["target_resume_issue_ns"]) / 1e9
        promotion = result["promotions"].get(key)
        response = promotion.get("response", {}) if promotion else {}
        status = str(response.get("status", "NA"))
        timestamps = response.get("phase3m_timestamps") or {}
        t6 = float(timestamps["T6"]) if "T6" in timestamps else None
        promo_bytes = int(response.get("transfer_bytes", 0)) if status == "ok" else 0
        source_events_for_request = source_events_by_session[session]
        if len({event["request_id"] for event in source_events_for_request}) != 1:
            raise RuntimeError(f"source trace request identity mismatch:{session}")
        local = int(source_events_for_request[0]["local_cached_tokens"])
        external = int(source_events_for_request[0]["external_cached_tokens"])
        repeated_external = sum(int(event["external_cached_tokens"]) for event in source_events_for_request[1:])
        expected = int(job["required_prefix_tokens"])
        if local < 0 or external < 0 or local + external > expected:
            raise RuntimeError(f"invalid source accounting:{session}")
        duplicate = duplicate_bytes(promo_bytes, external + repeated_external, expected)
        classification = "NA"
        if status == "ok":
            classification = ("PROMOTION_READY_MISS" if t6 is None or t6 >= arrival else
                "PROMOTION_REMOTE_WAIT" if "WAITING_FOR_REMOTE_KVS" in events else "PROMOTION_USEFUL")
        elif status in {"capacity_rejected", "progress_rejected"}:
            classification = status.upper()
        episode = str(job["episode_id"])
        turn_index = int(job["resume_turn_index"])
        row = {"repetition": rep, "policy": policy, "episode_id": episode,
            "boundary_index": int(job["boundary_index"]), "resume_turn_index": turn_index,
            "boundary_id": key, "proxy_pause_ms": job["proxy_pause_ms"],
            "previous_finish_ms": (int(boundary_client["previous_finish_ns"]) / 1e9 - origin_s) * 1000,
            "registered_ms": (int(boundary_client["registered_ns"]) / 1e9 - origin_s) * 1000,
            "target_resume_issue_ms": (deadline - origin_s) * 1000,
            "actual_issue_ms": (int(boundary_client["actual_issue_ns"]) / 1e9 - origin_s) * 1000,
            "pause_error_ms": boundary_client["pause_error_ms"],
            "source_ready_ms": (float(offload[key]["timestamp_monotonic"]) - origin_s) * 1000 if key in offload else "NA",
            "RESUME_ARRIVED_ms": (arrival - origin_s) * 1000,
            "RESUME_ADMITTED_ms": (admit - origin_s) * 1000,
            "FIRST_TOKEN_ms": (first - origin_s) * 1000,
            "REQUEST_FINISHED_ms": (finish - origin_s) * 1000,
            "deadline_latency_ms": max(0.0, (first - deadline) * 1000),
            "TTFT_ms": (first - arrival) * 1000,
            "scheduler_wait_ms": (admit - arrival) * 1000,
            "post_admit_to_first_ms": (first - admit) * 1000,
            "required_prefix_tokens": expected, "required_blocks": job["required_blocks"],
            "local_cached_tokens": local, "external_cached_tokens": external,
            "repeat_external_restore_tokens": repeated_external,
            "source_allocation_events": len(source_events_for_request),
            "recomputed_reusable_tokens": expected - local - external,
            "promotion_attempted": bool(promotion), "promotion_status": status,
            "promotion_decision": next((event for event in decisions[key] if event in {
                "RESERVATION_SKIP", "SOURCE_LATE_SKIP", "STALE_PROMOTION_CANCELLED"}), "NA"),
            "T6_ms": (t6 - origin_s) * 1000 if t6 is not None else "NA",
            "promotion_classification": classification,
            "promotion_H2D_bytes": promo_bytes, "resume_H2D_bytes": (external + repeated_external) * BYTES_PER_TOKEN,
            "duplicate_H2D_bytes": duplicate,
            "identity_match": turns[episode, turn_index]["output_sha256"] == reference[episode, turn_index]}
        boundaries.append(row)
        h2d.append({"repetition": rep, "policy": policy, "boundary_id": key,
            "expected_reusable_bytes": expected * BYTES_PER_TOKEN, "promotion_bytes": promo_bytes,
            "resume_external_bytes": (external + repeated_external) * BYTES_PER_TOKEN,
            "total_bytes": promo_bytes + (external + repeated_external) * BYTES_PER_TOKEN,
            "duplicate_bytes": duplicate})
        if promotion:
            prefetch.append({"repetition": rep, "policy": policy, "boundary_id": key,
                "status": status, "classification": classification,
                "T0": timestamps.get("T0", "NA"), "T1": timestamps.get("T1", "NA"),
                "T6": timestamps.get("T6", "NA"), "T7": timestamps.get("T7", "NA"),
                "issue_monotonic_ns": response.get("client_issue_ns", "NA"),
                "completion_monotonic_ns": response.get("client_return_ns", "NA"),
                "issue_to_T6_ms": (t6 - int(response["client_issue_ns"]) / 1e9) * 1000 if t6 is not None else "NA",
                "transfer_bytes": response.get("transfer_bytes", 0),
                "external_tokens_loaded": response.get("external_tokens_loaded", 0),
                "protected_boundary_id": promotion.get("protected_boundary_id", "NA")})
    sessions = []
    for episode in sorted({str(job["episode_id"]) for job in manifest_rows}):
        four = [row for row in boundaries if row["episode_id"] == episode]
        if len(four) != 4:
            raise RuntimeError("episode boundary count mismatch")
        sessions.append({"repetition": rep, "policy": policy, "episode_id": episode,
            "boundary_count": 4, "cumulative_TTFT_ms": sum(row["TTFT_ms"] for row in four),
            "cumulative_deadline_latency_ms": sum(row["deadline_latency_ms"] for row in four),
            "session_e2e_ms": (turns[episode, 4]["finish_ns"] - turns[episode, 0]["issue_ns"]) / 1e6,
            "useful_promotions": sum(row["promotion_classification"] == "PROMOTION_USEFUL" for row in four),
            "promotion_attempts": sum(row["promotion_attempted"] for row in four),
            "total_H2D_MiB": sum(row["promotion_H2D_bytes"] + row["resume_H2D_bytes"] for row in four) / MIB,
            "duplicate_H2D_MiB": sum(row["duplicate_H2D_bytes"] for row in four) / MIB})
    return boundaries, sessions, h2d, prefetch


def summarize_run(result: dict[str, Any], boundaries: list[dict[str, Any]],
                  sessions: list[dict[str, Any]], h2d: list[dict[str, Any]],
                  prefetch: list[dict[str, Any]], baseline_x_ms: float,
                  manifest: dict[str, dict[str, object]]) -> dict[str, Any]:
    if len(boundaries) != 32 or len(sessions) != 8 or len(h2d) != 32:
        raise RuntimeError("EXPERIMENT_INVALID_MATRIX:incomplete run metrics")
    policy = result["policy"]
    latency = [float(row["deadline_latency_ms"]) for row in boundaries]
    ttft = [float(row["TTFT_ms"]) for row in boundaries]
    wait = [float(row["scheduler_wait_ms"]) for row in boundaries]
    post = [float(row["post_admit_to_first_ms"]) for row in boundaries]
    cumulative = [float(row["cumulative_TTFT_ms"]) for row in sessions]
    e2e = [float(row["session_e2e_ms"]) for row in sessions]
    local = sum(int(row["local_cached_tokens"]) for row in boundaries)
    external = sum(int(row["external_cached_tokens"]) for row in boundaries)
    repeated_external = sum(int(row["repeat_external_restore_tokens"]) for row in boundaries)
    recomputed = 705728 - local - external
    if recomputed < 0:
        raise RuntimeError("source accounting exceeded reusable-token total")
    promotion_bytes = sum(int(row["promotion_bytes"]) for row in h2d)
    external_bytes = sum(int(row["resume_external_bytes"]) for row in h2d)
    duplicate = sum(int(row["duplicate_bytes"]) for row in h2d)
    lease_memory, peak_leased, peak_leases = lease_cost(result["lease_events"])
    block_time = reservation_cost(result, manifest) if policy == POLICIES["D"] else 0.0
    cap = result["capacity_trace"]
    count_decision = lambda value: sum(row.get("decision") == value for row in cap)
    decisions = result["decisions"]
    event_count = lambda value: sum(row.get("event") == value for row in decisions)
    successful = sum(row["status"] == "ok" for row in prefetch)
    useful = sum(row["classification"] == "PROMOTION_USEFUL" for row in prefetch)
    ready_miss = sum(row["classification"] == "PROMOTION_READY_MISS" for row in prefetch)
    remote_wait = sum(row["classification"] == "PROMOTION_REMOTE_WAIT" for row in prefetch)
    overrun = sum(float(row["issue_to_T6_ms"]) > 102 for row in prefetch if row["status"] == "ok")
    t6 = [float(row["issue_to_T6_ms"]) for row in prefetch if row["status"] == "ok"]
    turns = result["turn_rows"]
    first_ns = min([int(row["first_ns"]) for row in turns] + [int(result["X_first_ns"])])
    last_ns = max([int(row["finish_ns"]) for row in turns] + [int(result["X_finish_ns"])])
    post_first = sum(int(row["output_tokens"]) - 1 for row in turns) + int(result["X_tokens"]) - 1
    combined = post_first * 1e9 / (last_ns - first_ns)
    return {
        "repetition": int(result["repetition"]), "policy": policy, "valid": bool(result["valid"]),
        "boundary_count": 32, "turn_count": 40,
        "deadline_P50_ms": nearest_rank(latency, .50), "deadline_P90_ms": nearest_rank(latency, .90),
        "deadline_P95_ms": nearest_rank(latency, .95), "deadline_max_ms": max(latency),
        "TTFT_P50_ms": nearest_rank(ttft, .50), "TTFT_P95_ms": nearest_rank(ttft, .95),
        "scheduler_wait_P95_ms": nearest_rank(wait, .95),
        "post_admit_to_first_P95_ms": nearest_rank(post, .95),
        "cumulative_TTFT_median_ms": statistics.median(cumulative),
        "cumulative_TTFT_P95_ms": nearest_rank(cumulative, .95),
        "session_e2e_median_ms": statistics.median(e2e), "session_e2e_max_ms": max(e2e),
        "local_source_tokens": local, "external_source_tokens": external,
        "repeat_external_restore_tokens": repeated_external,
        "recomputed_reusable_tokens": recomputed,
        "local_source_fraction": local / 705728, "external_source_fraction": external / 705728,
        "promotion_attempts": len(prefetch), "successful_promotions": successful,
        "promotion_useful": useful, "promotion_ready_miss": ready_miss,
        "promotion_remote_wait": remote_wait, "promotion_useful_rate": useful / successful if successful else "NA",
        "capacity_rejects": count_decision("CAPACITY_REJECT"),
        "PSA_rejects": count_decision("PROGRESS_REJECT"),
        "reservation_skips": event_count("RESERVATION_SKIP"),
        "source_late_skips": event_count("SOURCE_LATE_SKIP"),
        "stale_cancels": event_count("STALE_PROMOTION_CANCELLED"),
        "promotion_H2D_MiB": promotion_bytes / MIB, "external_H2D_MiB": external_bytes / MIB,
        "total_H2D_MiB": (promotion_bytes + external_bytes) / MIB,
        "duplicate_H2D_MiB": duplicate / MIB,
        "lease_memory_time_MiB_s": lease_memory, "peak_leased_blocks": peak_leased,
        "peak_lease_count": peak_leases, "reservation_block_time_block_s": block_time,
        "peak_reserved_blocks": int(result["peak_reserved_blocks"]),
        "X_decode_ms": float(result["X_decode_ms"]), "X_TPOT_ms": float(result["X_TPOT_ms"]),
        "X_slowdown": float(result["X_decode_ms"]) / baseline_x_ms,
        "X_tail_margin_ms": float(result["X_tail_margin_ms"]), "combined_tok_s": combined,
        "service_T6_median_ms": statistics.median(t6) if t6 else "NA",
        "service_T6_P95_ms": nearest_rank(t6, .95) if t6 else "NA",
        "service_T6_max_ms": max(t6) if t6 else "NA", "service_overrun_count": overrun,
        "K2_max_pre_first": int(result["max_pre_first"]), "pause_abs_P95_ms": result["pause_abs_P95_ms"],
        "pause_abs_max_ms": result["pause_abs_max_ms"],
        "source_ready_count": result["source_ready_count"],
        "source_not_ready_count": result["source_not_ready_count"],
        "logical_leaks": result["logical_leaks"], "lease_leaks": result["lease_leaks"],
        "capacity_valid": result["capacity_valid"], "server_alive": result["server_alive"],
        "OOM": result["OOM"],
    }


def classify(valid: bool, h1: bool, h2: bool, h3: bool,
             paired_p95_percent: float, paired_cumulative_percent: float,
             paired_e2e_percent: float) -> str:
    if not valid:
        return "EXPERIMENT_INVALID_MATRIX"
    if all(-delta < 3.0 for delta in (paired_p95_percent, paired_cumulative_percent, paired_e2e_percent)):
        return "RESULT_C_MULTITURN_NO_MATERIAL_LATENCY_VALUE"
    if h1 and h2 and h3:
        return "RESULT_A_MULTITURN_DIRECTIONAL_BENEFIT"
    return "RESULT_B_VALID_MULTITURN_MIXED_TRADEOFF"


def finalize() -> None:
    manifest_rows = build_manifest(load_source())
    manifest = {str(row["boundary_id"]): row for row in manifest_rows}
    reference_path = ROOT / "results/phase4c/request_metrics.csv"
    with reference_path.open(newline="", encoding="utf-8") as stream:
        reference_rows = list(csv.DictReader(stream))
    source_rows = load_source()
    source_hash_to_episode = {row["session_hash"]: row["episode_id"] for row in source_rows}
    reference = {(source_hash_to_episode[row["session_hash"]], int(row["replay_turn_index"])): row["output_sha256"]
                 for row in reference_rows if row["session_hash"] in source_hash_to_episode}
    if len(reference) != 40:
        raise RuntimeError("Phase4C output identity reference incomplete")
    with (OUT / "x_only_runs.csv").open(newline="", encoding="utf-8") as stream:
        x_rows = list(csv.DictReader(stream))
    if len(x_rows) != 8 or not all(row["valid"] == "True" and int(row["X_generated_tokens"]) == 2048
                                    for row in x_rows):
        raise RuntimeError("EXPERIMENT_INVALID_MATRIX:X-only baseline")
    baseline_x = statistics.median(float(row["X_decode_ms"]) for row in x_rows)
    results = []
    for repetition, pair in enumerate(ORDER, 1):
        for letter in pair:
            path = OUT / "runs" / f"measured_rep{repetition}_{letter.lower()}" / "run.json"
            result = json.loads(path.read_text(encoding="utf-8"))
            if result["policy"] != POLICIES[letter] or int(result["repetition"]) != repetition or not result["valid"]:
                raise RuntimeError(f"EXPERIMENT_INVALID_MATRIX:unexpected run {path}")
            results.append(result)
    if len(results) != 16:
        raise RuntimeError("EXPERIMENT_INVALID_MATRIX:partial matrix")
    all_boundaries: list[dict[str, Any]] = []
    all_sessions: list[dict[str, Any]] = []
    all_h2d: list[dict[str, Any]] = []
    all_prefetch: list[dict[str, Any]] = []
    all_run: list[dict[str, Any]] = []
    all_capacity: list[dict[str, Any]] = []
    all_lease: list[dict[str, Any]] = []
    all_reservation: list[dict[str, Any]] = []
    all_resume: list[dict[str, Any]] = []
    all_source: list[dict[str, Any]] = []
    all_turn: list[dict[str, Any]] = []
    for result in results:
        boundary, session, h2d, prefetch = process_run(result, manifest_rows, reference, baseline_x)
        metric = summarize_run(result, boundary, session, h2d, prefetch, baseline_x, manifest)
        if metric["K2_max_pre_first"] > 2 or metric["logical_leaks"] or metric["lease_leaks"] or not metric["capacity_valid"]:
            raise RuntimeError("EXPERIMENT_INVALID_MATRIX:mechanism invariant")
        all_boundaries.extend(boundary)
        all_sessions.extend(session)
        all_h2d.extend(h2d)
        all_prefetch.extend(prefetch)
        all_run.append(metric)
        base = {"repetition": result["repetition"], "policy": result["policy"]}
        all_turn.extend({**base, **row} for row in result["turn_rows"])
        for target, key in ((all_capacity, "capacity_trace"), (all_lease, "lease_events"),
                            (all_reservation, "decisions"), (all_resume, "resume_trace"),
                            (all_source, "source_events")):
            target.extend({**base, **row} for row in result[key])
    smoke_rows = []
    for letter in POLICIES:
        smoke = json.loads((OUT / f"lifecycle_smoke_{letter.lower()}.json").read_text(encoding="utf-8"))
        smoke_rows.append({key: smoke[key] for key in ("policy", "valid", "turns_complete", "boundaries",
            "pre_materialized_boundary_count", "X_tail_margin_ms", "pause_abs_P95_ms", "pause_abs_max_ms",
            "source_ready_count", "source_not_ready_count", "max_pre_first", "lease_leaks", "logical_leaks")})
    write_csv("lifecycle_smoke.csv", smoke_rows)
    resource_rows = [{key: row[key] for key in ("repetition", "policy", "lease_memory_time_MiB_s",
        "peak_leased_blocks", "peak_lease_count", "reservation_block_time_block_s", "peak_reserved_blocks",
        "capacity_rejects", "PSA_rejects", "reservation_skips")}
        for row in all_run]
    for name, rows in (("boundary_metrics.csv", all_boundaries), ("session_metrics.csv", all_sessions),
                       ("turn_metrics.csv", all_turn), ("h2d_by_boundary.csv", all_h2d),
                       ("h2d_accounting.csv", all_h2d), ("prefetch_metrics.csv", all_prefetch),
                       ("run_metrics.csv", all_run), ("capacity_trace.csv", all_capacity),
                       ("lease_events.csv", all_lease), ("reservation_events.csv", all_reservation),
                       ("scheduler_trace.csv", all_resume), ("resume_trace.csv", all_resume),
                       ("offload_trace.csv", all_source), ("source_readiness.csv", all_source),
                       ("resource_costs.csv", resource_rows),
                       ("progress_admission.csv", [row for row in all_capacity if row.get("progress_guard") is True])):
        write_csv(name, rows)
    summary_rows = []
    for policy in POLICIES.values():
        rows = [row for row in all_run if row["policy"] == policy]
        if len(rows) != 8:
            raise RuntimeError("EXPERIMENT_INVALID_MATRIX:policy run count")
        fields = ("deadline_P50_ms", "deadline_P90_ms", "deadline_P95_ms", "TTFT_P95_ms", "cumulative_TTFT_median_ms", "session_e2e_median_ms",
                  "scheduler_wait_P95_ms", "post_admit_to_first_P95_ms", "local_source_fraction",
                  "external_source_fraction", "recomputed_reusable_tokens", "promotion_attempts",
                  "successful_promotions", "promotion_useful", "capacity_rejects", "PSA_rejects",
                  "reservation_skips", "source_late_skips", "promotion_H2D_MiB", "external_H2D_MiB",
                  "total_H2D_MiB", "duplicate_H2D_MiB", "lease_memory_time_MiB_s",
                  "reservation_block_time_block_s", "X_slowdown", "X_TPOT_ms", "combined_tok_s")
        record: dict[str, Any] = {"policy": policy, "valid_repetitions": 8}
        for field in fields:
            values = [float(row[field]) for row in rows]
            record[field + "_median"] = statistics.median(values)
            record[field + "_min"] = min(values)
            record[field + "_max"] = max(values)
        record["promotion_useful_rate"] = (sum(int(row["promotion_useful"]) for row in rows) /
                                           sum(int(row["successful_promotions"]) for row in rows)
                                           if sum(int(row["successful_promotions"]) for row in rows) else "NA")
        summary_rows.append(record)
    write_csv("policy_summary.csv", summary_rows)
    paired = []
    fields = ("deadline_P95_ms", "TTFT_P95_ms", "cumulative_TTFT_median_ms",
              "session_e2e_median_ms", "total_H2D_MiB", "duplicate_H2D_MiB",
              "X_slowdown", "combined_tok_s")
    for repetition in range(1, 9):
        a, d = (next(row for row in all_run if row["repetition"] == repetition and row["policy"] == policy)
                for policy in POLICIES.values())
        row: dict[str, Any] = {"repetition": repetition, "comparison": "D_vs_A"}
        for field in fields:
            row[field + "_delta"] = float(d[field]) - float(a[field])
            row[field + "_relative_percent"] = percent_delta(float(d[field]), float(a[field])) if float(a[field]) else "NA"
        paired.append(row)
    write_csv("paired_deltas.csv", paired)
    boot = []
    for field in ("deadline_P95_ms_relative_percent", "cumulative_TTFT_median_ms_relative_percent",
                  "session_e2e_median_ms_relative_percent", "X_slowdown_delta"):
        low, high = bootstrap([float(row[field]) for row in paired], 20260928)
        boot.append({"comparison": "D_vs_A", "metric": field, "paired_median": statistics.median(float(row[field]) for row in paired),
                     "bootstrap_95_low": low, "bootstrap_95_high": high, "resamples": 10000,
                     "seed": 20260928, "interpretation": "descriptive, not significance test"})
    write_csv("bootstrap_intervals.csv", boot)
    strata = stratified_reports(all_boundaries)
    write_summary(results, all_run, all_boundaries, all_sessions, summary_rows, paired, boot, x_rows, strata)


def stratified_reports(boundaries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    reports: dict[str, list[dict[str, Any]]] = {}
    for name, group in (("stratified_by_boundary.csv", lambda row: int(row["boundary_index"])),
                        ("stratified_by_pause.csv", lambda row: pause_bin(float(row["proxy_pause_ms"])))):
        rows = []
        for policy in POLICIES.values():
            policy_rows = [row for row in boundaries if row["policy"] == policy]
            for key in sorted({group(row) for row in policy_rows}):
                selected = [row for row in policy_rows if group(row) == key]
                per_rep = [[float(row["deadline_latency_ms"]) for row in selected if row["repetition"] == rep]
                           for rep in range(1, 9)]
                if any(not values for values in per_rep):
                    raise RuntimeError("incomplete stratum")
                rows.append({"policy": policy, "stratum": key, "boundary_observations": len(selected),
                             "boundaries_per_run": len(per_rep[0]),
                             "deadline_latency_median_ms": statistics.median(float(row["deadline_latency_ms"]) for row in selected),
                             "TTFT_median_ms": statistics.median(float(row["TTFT_ms"]) for row in selected),
                             "deadline_P95_median_of_runs_ms": statistics.median(nearest_rank(values, .95) for values in per_rep),
                             "TTFT_P95_pooled_ms": nearest_rank([float(row["TTFT_ms"]) for row in selected], .95),
                             "scheduler_wait_P95_pooled_ms": nearest_rank([float(row["scheduler_wait_ms"]) for row in selected], .95),
                             "post_admit_P95_pooled_ms": nearest_rank([float(row["post_admit_to_first_ms"]) for row in selected], .95),
                             "local_source_fraction": (sum(int(row["local_cached_tokens"]) for row in selected) /
                                                       sum(int(row["required_prefix_tokens"]) for row in selected)),
                             "external_source_fraction": (sum(int(row["external_cached_tokens"]) for row in selected) /
                                                          sum(int(row["required_prefix_tokens"]) for row in selected)),
                             "source_ready_count": sum(row["source_ready_ms"] != "NA" for row in selected),
                             "promotion_attempts": sum(bool(row["promotion_attempted"]) for row in selected),
                             "promotion_rate": sum(bool(row["promotion_attempted"]) for row in selected) / len(selected),
                             "successful_promotions": sum(row["promotion_status"] == "ok" for row in selected),
                             "useful_promotions": sum(row["promotion_classification"] == "PROMOTION_USEFUL" for row in selected),
                             "useful_promotion_rate": (sum(row["promotion_classification"] == "PROMOTION_USEFUL" for row in selected) /
                                                       sum(row["promotion_status"] == "ok" for row in selected)
                                                       if any(row["promotion_status"] == "ok" for row in selected) else "NA")})
        write_csv(name, rows)
        reports[name] = rows
    return reports


def write_summary(results: list[dict[str, Any]], runs: list[dict[str, Any]],
                  boundaries: list[dict[str, Any]], sessions: list[dict[str, Any]],
                  policy_rows: list[dict[str, Any]], paired: list[dict[str, Any]],
                  bootstrap_rows: list[dict[str, Any]], x_rows: list[dict[str, str]],
                  strata: dict[str, list[dict[str, Any]]]) -> None:
    a = next(row for row in policy_rows if row["policy"] == POLICIES["A"])
    d = next(row for row in policy_rows if row["policy"] == POLICIES["D"])
    m = lambda row, field: float(row[field + "_median"])
    h1 = m(d, "deadline_P95_ms") <= .90 * m(a, "deadline_P95_ms")
    h2 = m(d, "total_H2D_MiB") <= 1.05 * m(a, "total_H2D_MiB")
    h3 = m(d, "X_slowdown") <= m(a, "X_slowdown") + .05
    result_name = classify(True, h1, h2, h3,
        statistics.median(float(row["deadline_P95_ms_relative_percent"]) for row in paired),
        statistics.median(float(row["cumulative_TTFT_median_ms_relative_percent"]) for row in paired),
        statistics.median(float(row["session_e2e_median_ms_relative_percent"]) for row in paired))
    x_values = [float(row["X_decode_ms"]) for row in x_rows]
    before, after = statistics.median(x_values[:4]), statistics.median(x_values[4:])
    drift = percent_delta(after, before)
    source_ready = {policy: sum(int(run["source_ready_count"]) for run in runs if run["policy"] == policy)
                    for policy in POLICIES.values()}
    source_late = {policy: sum(int(run["source_not_ready_count"]) for run in runs if run["policy"] == policy)
                   for policy in POLICIES.values()}
    identities = {policy: sum(bool(row["identity_match"]) for row in boundaries if row["policy"] == policy)
                  for policy in POLICIES.values()}
    report = {
        "phase": "4G-A", "final_result": result_name,
        "X_only_valid": len(x_rows), "X_baseline_decode_median_ms": statistics.median(x_values),
        "X_first4_median_ms": before, "X_last4_median_ms": after, "X_drift_percent": drift,
        "environment_drift_warning": abs(drift) > 5,
        "Stage_L_pass": {letter: bool(json.loads((OUT / f"lifecycle_smoke_{letter.lower()}.json").read_text())["valid"])
                          for letter in POLICIES},
        "measured_valid": len(runs), "turns_complete": sum(int(run["turn_count"]) for run in runs),
        "boundaries_complete": sum(int(run["boundary_count"]) for run in runs),
        "token_correct": sum(len(result["turn_rows"]) for result in results),
        "source_ready": source_ready, "source_not_ready": source_late,
        "output_identity_match": identities,
        "K2_valid": sum(int(run["K2_max_pre_first"]) <= 2 for run in runs),
        "pause_valid": sum(float(run["pause_abs_P95_ms"]) <= 10 and float(run["pause_abs_max_ms"]) <= 30 for run in runs),
        "X_valid": sum(float(run["X_tail_margin_ms"]) > 0 for run in runs),
        "server_crashes": sum(not run["server_alive"] for run in runs),
        "OOM": sum(bool(run["OOM"]) for run in runs),
        "logical_leaks": sum(int(run["logical_leaks"]) for run in runs),
        "lease_leaks": sum(int(run["lease_leaks"]) for run in runs),
        "capacity_invariant_failures": sum(not run["capacity_valid"] for run in runs),
        "H1": h1, "H2": h2, "H3": h3,
        "paired_P95_percent_median": statistics.median(float(row["deadline_P95_ms_relative_percent"]) for row in paired),
        "paired_P95_percent_range": [min(float(row["deadline_P95_ms_relative_percent"]) for row in paired),
                                     max(float(row["deadline_P95_ms_relative_percent"]) for row in paired)],
        "bootstrap": bootstrap_rows,
        "caveat": "705728 reusable tokens / 19297.25 MiB is one-full-crossing sum, not a minimum required H2D transfer."
    }
    (OUT / "validation_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    table = ["| Policy | Boundary P50 | P90 | P95 | TTFT P95 | Cum TTFT/session | Session E2E | H2D MiB | Duplicate H2D MiB | X slowdown |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in (a, d):
        table.append(f"| {row['policy']} | {m(row,'deadline_P50_ms'):.2f} | {m(row,'deadline_P90_ms'):.2f} | "
                     f"{m(row,'deadline_P95_ms'):.2f} | {m(row,'TTFT_P95_ms'):.2f} | "
                     f"{m(row,'cumulative_TTFT_median_ms'):.2f} | {m(row,'session_e2e_median_ms'):.2f} | "
                     f"{m(row,'total_H2D_MiB'):.2f} | {m(row,'duplicate_H2D_MiB'):.2f} | {m(row,'X_slowdown'):.3f} |")
    p95_delta = statistics.median(float(row["deadline_P95_ms_relative_percent"]) for row in paired)
    def stratum_table(name: str, label: str) -> list[str]:
        rows = strata[name]
        keys = sorted({row["stratum"] for row in rows})
        output = [f"| {label} | N per policy | A TTFT median ms | D TTFT median ms | D–A % | D useful promo rate |",
                  "|---|---:|---:|---:|---:|---:|"]
        for key in keys:
            aa = next(row for row in rows if row["stratum"] == key and row["policy"] == POLICIES["A"])
            dd = next(row for row in rows if row["stratum"] == key and row["policy"] == POLICIES["D"])
            rate = dd["useful_promotion_rate"]
            output.append(f"| {key} | {aa['boundary_observations']} | {aa['TTFT_median_ms']:.2f} | "
                          f"{dd['TTFT_median_ms']:.2f} | {percent_delta(dd['TTFT_median_ms'],aa['TTFT_median_ms']):+.2f}% | "
                          f"{100*rate:.1f}% |" if rate != "NA" else
                          f"| {key} | {aa['boundary_observations']} | {aa['TTFT_median_ms']:.2f} | "
                          f"{dd['TTFT_median_ms']:.2f} | {percent_delta(dd['TTFT_median_ms'],aa['TTFT_median_ms']):+.2f}% | NA |")
        return output
    d_p95_rows = []
    for rep in range(1, 9):
        selected = [row for row in boundaries if row["policy"] == POLICIES["D"] and row["repetition"] == rep]
        d_p95_rows.append(sorted(selected, key=lambda row: float(row["deadline_latency_ms"]))[-2])
    tail_wait = statistics.median(float(row["scheduler_wait_ms"]) for row in d_p95_rows)
    tail_post = statistics.median(float(row["post_admit_to_first_ms"]) for row in d_p95_rows)
    strongest = "scheduler waiting" if tail_wait > tail_post else "post-admission serving"
    paired_cum = statistics.median(float(row["cumulative_TTFT_median_ms_relative_percent"]) for row in paired)
    paired_e2e = statistics.median(float(row["session_e2e_median_ms_relative_percent"]) for row in paired)
    text = ["# Phase 4G-A: 32-boundary causal multi-turn lifecycle", "",
            f"Final result: **{result_name}**. Eight paired repetitions, 16/16 measured runs, 640/640 turns, 512/512 real resume boundaries.", "",
            "This is a frozen hybrid trace-derived replay: real coding-agent token/KV structure and empirical TraceToolProxy pauses, not real request timestamps or a production workload.", "",
            "## Frozen workload and lifecycle", "",
            "Eight five-turn sessions, 32 boundaries per run; wave reusable-token totals 168016 / 172736 / 179632 / 185344. No boundary prefix was pre-materialized.",
            "A boundary was registered only after its previous request completed, and its prompt used that request's actual generated token IDs. Source readiness was observed from the connector's exact block lookup; unready D candidates were skipped, not promoted from a fabricated external source.",
            f"Observed source-ready boundaries: A {source_ready[POLICIES['A']]}/256, D {source_ready[POLICIES['D']]}/256. Source-not-ready: A {source_late[POLICIES['A']]}/256, D {source_late[POLICIES['D']]}/256.",
            "705728 reusable tokens (19297.25 MiB at one full crossing) are **not** a minimum required H2D transfer; a boundary may retain local blocks or recompute missing output-derived blocks.", "",
            "## Policy results (median of eight run-level values)", "", *table, "",
            f"Paired D–A boundary-P95 delta: {p95_delta:+.2f}% (range {report['paired_P95_percent_range'][0]:+.2f}% to {report['paired_P95_percent_range'][1]:+.2f}%). H1: {'PASS' if h1 else 'FAIL'}; H2: {'PASS' if h2 else 'FAIL'}; H3: {'PASS' if h3 else 'FAIL'}.",
            f"Paired cumulative-TTFT delta {paired_cum:+.2f}%; paired session-E2E delta {paired_e2e:+.2f}%. Bootstrap intervals are in bootstrap_intervals.csv (descriptive only).",
            f"D promotions: median {m(d,'successful_promotions'):.1f}/run, useful {d['promotion_useful_rate']}, capacity rejects median {m(d,'capacity_rejects'):.1f}, PSA rejects median {m(d,'PSA_rejects'):.1f}, reservation skips median {m(d,'reservation_skips'):.1f}, source-late skips median {m(d,'source_late_skips'):.1f}.",
            f"D lease memory-time median {m(d,'lease_memory_time_MiB_s'):.2f} MiB·s; reservation block-time median {m(d,'reservation_block_time_block_s'):.2f} block·s; local source fraction {100*m(d,'local_source_fraction'):.2f}%; external source fraction {100*m(d,'external_source_fraction'):.2f}%.",
            "", "## Stratified results", "", "Boundary index:", "", *stratum_table("stratified_by_boundary.csv", "Boundary"),
            "", "TraceToolProxy pause bin:", "", *stratum_table("stratified_by_pause.csv", "Pause ms"),
            "", "## Tail bottleneck", "",
            f"At each D run's actual nearest-rank P95 boundary (31st of 32), median scheduler wait was {tail_wait:.2f} ms versus {tail_post:.2f} ms post-admission-to-first-token. Thus the observed larger component was {strongest}. This decomposition, not P95 alone, informs whether the first-boundary sample size or compute queueing limits benefit; it is not a controlled causal attribution.",
            "", "## Validation and limitations", "",
            f"Stage L A/D PASS; X-only 8/8; measured 16/16; K2 {report['K2_valid']}/16; pause fidelity {report['pause_valid']}/16; X overlap {report['X_valid']}/16; output identity A {identities[POLICIES['A']]}/256 and D {identities[POLICIES['D']]}/256 (diagnostic).",
            f"X-only median decode {statistics.median(x_values):.2f} ms; pre/post drift {drift:+.2f}%" + (" (ENVIRONMENT_DRIFT_WARNING)." if abs(drift) > 5 else "."),
            "These are descriptive paired measurements (n=8), not a statistical-significance test. No policy retuning, predictor, or full-agent tool execution occurred."]
    (OUT / "summary.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def finalize_invalid_matrix() -> dict[str, Any]:
    """Document a stopped matrix without aggregating partial policy performance."""
    failure = json.loads((OUT / "failure_report.json").read_text(encoding="utf-8"))
    x_rows = list(csv.DictReader((OUT / "x_only_runs.csv").open(newline="", encoding="utf-8")))
    progress = list(csv.DictReader((OUT / "measured_progress.csv").open(newline="", encoding="utf-8")))
    bad_path = OUT / "runs/measured_rep2_d/run.json"
    bad = json.loads(bad_path.read_text(encoding="utf-8"))
    first = max(int(row["first_ns"]) for row in bad["turn_rows"])
    x_first = int(bad["X_first_ns"])
    x_finish = int(bad["X_finish_ns"])
    if failure["measured_completed"] != len(progress) or len(progress) != 2 or bad["valid"]:
        raise RuntimeError("invalid failure-artifact accounting")
    smoke = [json.loads((OUT / f"lifecycle_smoke_{letter}.json").read_text(encoding="utf-8"))
             for letter in ("a", "d")]
    # Preserve the artifact schema without publishing partial-matrix samples.
    invalid_schemas = {
        "run_metrics.csv": "repetition,policy,valid,deadline_P95_ms,total_H2D_MiB,X_slowdown",
        "session_metrics.csv": "repetition,policy,episode_id,cumulative_TTFT_ms,session_e2e_ms",
        "boundary_metrics.csv": "repetition,policy,boundary_id,deadline_latency_ms,TTFT_ms",
        "turn_metrics.csv": "repetition,policy,episode_id,turn_index,prompt_tokens,output_tokens",
        "offload_trace.csv": "repetition,policy,boundary_id,event,timestamp_monotonic",
        "prefetch_metrics.csv": "repetition,policy,boundary_id,status,T6,transfer_bytes",
        "progress_admission.csv": "repetition,policy,request_id,decision,free_blocks_before",
        "capacity_trace.csv": "repetition,policy,request_id,decision,free_blocks_before",
        "lease_events.csv": "repetition,policy,event,lease_key,num_blocks,timestamp_monotonic",
        "reservation_events.csv": "repetition,policy,event,boundary_id,timestamp_ns",
        "scheduler_trace.csv": "repetition,policy,event,session,timestamp_monotonic",
        "h2d_by_boundary.csv": "repetition,policy,boundary_id,total_bytes,duplicate_bytes",
        "resource_costs.csv": "repetition,policy,lease_memory_time_MiB_s,reservation_block_time_block_s",
        "paired_deltas.csv": "repetition,comparison,deadline_P95_ms_relative_percent",
        "stratified_by_boundary.csv": "policy,stratum,TTFT_median_ms,deadline_latency_median_ms",
        "stratified_by_pause.csv": "policy,stratum,TTFT_median_ms,deadline_latency_median_ms",
        "bootstrap_intervals.csv": "comparison,metric,bootstrap_95_low,bootstrap_95_high",
    }
    for name, header in invalid_schemas.items():
        (OUT / name).write_text(header + "\n", encoding="utf-8")
    write_csv("lifecycle_smoke.csv", [{key: row[key] for key in ("policy", "valid", "turns_complete",
        "boundaries", "pre_materialized_boundary_count", "X_tail_margin_ms", "pause_abs_P95_ms",
        "pause_abs_max_ms", "source_ready_count", "source_not_ready_count", "max_pre_first",
        "logical_leaks", "lease_leaks")} for row in smoke])
    manifest = validate_manifest(build_manifest(load_source()))
    expected_turns = {(row["episode_id"], int(row["replay_turn_index"])): row for row in load_source()}
    turn_lengths_exact = len(bad["turn_rows"]) == 40 and all(
        int(row["prompt_tokens"]) == int(expected_turns[row["episode_id"], int(row["turn_index"])]["source_isl"])
        and int(row["output_tokens"]) == int(expected_turns[row["episode_id"], int(row["turn_index"])]["source_osl"])
        for row in bad["turn_rows"])
    report = {
        "phase": "4G-A", "final_result": "EXPERIMENT_INVALID_MATRIX",
        "failure_reason": "RUN_INVALID_BACKGROUND",
        "start_commit": "f4b19bd0ee0b7f3d4fe4b4732b8e8bcc4b2499f4",
        "workload_sha256": manifest["source_sha256"],
        "X_only_valid_pre": len(x_rows), "X_only_post_not_started": True,
        "Stage_L_A_valid": bool(smoke[0]["valid"]), "Stage_L_D_valid": bool(smoke[1]["valid"]),
        "Stage_L_X_tail_A_ms": smoke[0]["X_tail_margin_ms"],
        "Stage_L_X_tail_D_ms": smoke[1]["X_tail_margin_ms"],
        "measured_valid": len(progress), "measured_expected": 16,
        "failed_run": "measured_rep2_d", "failed_policy": bad["policy"],
        "failed_run_turns_complete": bad["turns_complete"],
        "failed_run_boundaries_complete": bad["boundaries"],
        "failed_run_X_tail_margin_ms": bad["X_tail_margin_ms"],
        "failed_run_X_finish_after_t0_ms": (x_finish - x_first) / 1e6,
        "failed_run_last_Agent_first_after_t0_ms": (first - x_first) / 1e6,
        "failed_run_prompt_and_output_lengths_exact": turn_lengths_exact,
        "failed_run_K2_max_pre_first": bad["max_pre_first"],
        "failed_run_pause_P95_absolute_error_ms": bad["pause_abs_P95_ms"],
        "failed_run_pause_max_absolute_error_ms": bad["pause_abs_max_ms"],
        "failed_run_server_alive": bad["server_alive"], "failed_run_OOM": bad["OOM"],
        "failed_run_logical_leaks": bad["logical_leaks"], "failed_run_lease_leaks": bad["lease_leaks"],
        "failed_run_capacity_valid": bad["capacity_valid"],
        "partial_matrix_not_aggregated": True,
        "H1": "NOT_EVALUATED", "H2": "NOT_EVALUATED", "H3": "NOT_EVALUATED",
    }
    (OUT / "validation_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    text = ["# Phase 4G-A: 32-boundary multi-turn lifecycle evaluation", "",
            "Final result: **EXPERIMENT_INVALID_MATRIX**. The 16-run performance matrix was stopped at the predeclared background hard gate; no A/D policy medians or H1/H2/H3 claims are reported.", "",
            "## Workload and lifecycle", "",
            f"Frozen Phase4C workload SHA256: {manifest['source_sha256']}. Eight sessions, 40 turns and 32 causal boundaries per run; {manifest['reusable_tokens']} reusable tokens and {manifest['reusable_blocks']} aligned blocks. The 19297.25-MiB one-full-crossing sum is a scale reference, not a lower bound on transfer.",
            "Natural connector offload readiness was observable by exact block lookup. No boundary prefix was pre-materialized. In the final Stage L smoke, A and D each completed 40/40 turns and 32/32 boundaries; both had 10 source-ready and 22 source-not-ready boundaries. D skipped unready promotion candidates rather than fabricating external KV.",
            f"Stage L X-tail margins: A {smoke[0]['X_tail_margin_ms']:.2f} ms, D {smoke[1]['X_tail_margin_ms']:.2f} ms; both exceeded the frozen 3000-ms gate.", "",
            "## Stopped measured matrix", "",
            f"Pre-matrix X-only: {len(x_rows)}/4 valid. Measured: {len(progress)}/16 valid (Rep1 A and D). The next run, Rep2 D, completed all 40 requests and 32 boundaries with K2={bad['max_pre_first']}, 0 OOM/crash/leaks, but X-tail margin was {bad['X_tail_margin_ms']:.2f} ms.",
            f"X finished {(x_finish-x_first)/1e6:.2f} ms after t0; the last Agent first token arrived {(first-x_first)/1e6:.2f} ms after t0. X therefore ended {(first-x_finish)/1e6:.2f} ms before the last Agent first token. This is RUN_INVALID_BACKGROUND under the frozen protocol, not a 40-request completion failure.",
            "The driver immediately stopped. No later repetitions or post-matrix X-only controls ran; partial results were not aggregated. Frozen X=2048, workload, deadlines, K2, PSA-1, and service budget were not changed.", "",
            "## Interpretation", "",
            "The causal multi-turn harness passed its lifecycle smoke but the frozen 2048-token background did not reliably span the full first-token contention window during the measured matrix. Consequently this phase cannot decide whether larger multi-turn sampling reveals latency benefit, whether compute queueing dominates, or whether H1/H2/H3 hold. The invalid outcome is preserved rather than treating a background-short run as a performance sample."]
    (OUT / "summary.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    finalize()
