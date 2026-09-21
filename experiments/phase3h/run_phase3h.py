from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from experiments.phase3g import run_phase3g as transport
from experiments.phase3g_full import run_phase3g_full as base
from experiments.phase3g_full.metrics import (
    decode_throughput_tok_s,
    gate_wait_ms,
    gpu_ready_residency_ms,
    interval_concurrency,
    overlap,
    percentile_linear,
    sequence_hash,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase3h"
SESSIONS = ("A", "B", "C")
POLICIES = (
    ("ResumeAware-K3", None),
    ("FirstTokenGate-K1", 1),
    ("FirstTokenGate-K2", 2),
)
TOKENS = 32


_original_payload = transport.completion_payload
_active_gate_limit: int | None = None


def _payload_with_gate_limit(*args: Any, **kwargs: Any) -> dict[str, Any]:
    payload = _original_payload(*args, **kwargs)
    if kwargs.get("gate") and _active_gate_limit is not None:
        payload["kv_transfer_params"]["slackkv_first_token_gate_limit"] = _active_gate_limit
    return payload


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty output {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_enriched_trace(path: Path, policy: str, repetition: int, t0: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    marker = "SLACKKV_RESUME_TRACE "
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if marker not in line:
            continue
        raw = json.loads(line.split(marker, 1)[1])
        rows.append(
            {
                "policy": policy,
                "repetition": repetition,
                "session": raw.get("session"),
                "request_id": raw.get("request_id"),
                "event": raw.get("event"),
                "step_id": raw.get("step_id"),
                "timestamp_monotonic_raw": raw.get("timestamp_monotonic"),
                "timestamp_ms_from_t0": (float(raw["timestamp_monotonic"]) - t0) * 1000.0,
                "request_status": raw.get("request_status"),
                "num_running": raw.get("num_running"),
                "num_waiting": raw.get("num_waiting"),
                "num_skipped_waiting": raw.get("num_skipped_waiting"),
                "gate_limit": raw.get("gate_limit"),
                "gate_owner_request_ids": json.dumps(raw.get("gate_owner_request_ids")),
                "num_slackkv_pre_first_token": raw.get("num_slackkv_pre_first_token"),
                "num_computed_tokens": raw.get("num_computed_tokens"),
                "num_output_tokens": raw.get("num_output_tokens"),
            }
        )
    return rows


def first_event(rows: list[dict[str, Any]], session: str, event: str) -> dict[str, Any] | None:
    return next((row for row in rows if row["session"] == session and row["event"] == event), None)


def make_session_rows(
    policy: str,
    repetition: int,
    gate_limit: int | None,
    raw_rows: list[dict[str, Any]],
    trace: list[dict[str, Any]],
    promotions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    promo = {row["session"]: row for row in promotions}
    output: list[dict[str, Any]] = []
    for raw in raw_rows:
        session = raw["session"]
        arrived = first_event(trace, session, "RESUME_ARRIVED")
        admitted = first_event(trace, session, "RESUME_ADMITTED")
        first = first_event(trace, session, "FIRST_TOKEN")
        finished = first_event(trace, session, "REQUEST_FINISHED")
        blocked = first_event(trace, session, "GATE_BLOCKED")
        if not all((arrived, admitted, first, finished)):
            raise RuntimeError(f"incomplete trace for {policy}/{repetition}/{session}")
        admission_ms = float(admitted["timestamp_ms_from_t0"])
        first_ms = float(first["timestamp_ms_from_t0"])
        finish_ms = float(finished["timestamp_ms_from_t0"])
        completion = promo.get(session, {}).get("completion_ms")
        output.append(
            {
                "policy": policy,
                "repetition": repetition,
                "session": session,
                "original_deadline_ms": raw["original_deadline_ms"],
                "scheduler_arrival_ms": float(arrived["timestamp_ms_from_t0"]),
                "scheduler_arrival_step": int(arrived["step_id"]),
                "gate_limit": gate_limit if gate_limit is not None else "NA",
                "gate_first_blocked_ms": None if blocked is None else float(blocked["timestamp_ms_from_t0"]),
                "gate_wait_ms": gate_wait_ms(None if blocked is None else float(blocked["timestamp_ms_from_t0"]), admission_ms),
                "gate_blocked_steps": sum(row["session"] == session and row["event"] == "GATE_BLOCKED" for row in trace),
                "scheduler_admission_ms": admission_ms,
                "scheduler_admission_step": int(admitted["step_id"]),
                "first_token_time_ms": first_ms,
                "first_token_step": int(first["step_id"]),
                "request_finished_ms": finish_ms,
                "request_finished_step": int(finished["step_id"]),
                "effective_TTFT_ms": first_ms - float(raw["original_deadline_ms"]),
                "completion_latency_ms": finish_ms - float(raw["original_deadline_ms"]),
                "decode_duration_ms": finish_ms - first_ms,
                "TPOT_ms": (finish_ms - first_ms) / 31.0,
                "promotion_completion_ms": completion if completion is not None else "NA",
                "gpu_ready_residency_ms": gpu_ready_residency_ms(completion, admission_ms),
                "generated_token_count": raw["generated_token_count"],
                "sequence_hash": raw["sequence_hash"],
                "token_count_validation": raw["token_count_validation"],
                "cache_transport_validation": raw["cache_transport_validation"],
            }
        )
    return output


def aggregate(
    policy: str,
    repetition: int,
    gate_limit: int | None,
    rows: list[dict[str, Any]],
    native: list[dict[str, Any]],
    trace: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = sorted(rows, key=lambda row: row["session"])
    pre = [(float(row["scheduler_admission_ms"]), float(row["first_token_time_ms"])) for row in rows]
    decode = [(float(row["first_token_time_ms"]), float(row["request_finished_ms"])) for row in rows]
    pair_pre = [overlap(pre[i], pre[j]) for i in range(3) for j in range(i + 1, 3)]
    pair_decode = [overlap(decode[i], decode[j]) for i in range(3) for j in range(i + 1, 3)]
    max_pre, area_pre = interval_concurrency(pre)
    max_decode, _ = interval_concurrency(decode)
    ttft = [float(row["effective_TTFT_ms"]) for row in rows]
    completion = [float(row["completion_latency_ms"]) for row in rows]
    cache_ok = all(row["cache_transport_validation"] == "PASS" for row in rows)
    token_ok = all(row["token_count_validation"] == "PASS" for row in rows)
    native_ok = all(
        row["external_tokens_loaded"] == 4096
        and row["model_forward_count"] == 0
        and row["generated_token_count"] == 0
        and row["mixed_prefetch_model_step_count"] == 0
        and row["status"] == "ok"
        for row in native
    )
    blocked_c = any(row["session"] == "C" and row["event"] == "GATE_BLOCKED" for row in trace)
    gate_ok = (
        gate_limit is None
        or (
            max_pre <= gate_limit
            and all(int(row["num_slackkv_pre_first_token"]) <= gate_limit for row in trace if row["num_slackkv_pre_first_token"] is not None)
            and (gate_limit != 2 or (max_pre == 2 and blocked_c))
            and (gate_limit != 1 or max_pre == 1)
        )
    )
    return {
        "policy": policy,
        "repetition": repetition,
        "promotion_deadline_miss_count": sum(float(row["deadline_stall_ms"]) > 0 for row in native),
        "blackout_conflict_count": sum(str(row["blackout_conflict"]) == "True" for row in native),
        "long_prefetch_count": sum(str(row["long_prefetch"]) == "True" for row in native),
        "gate_limit": gate_limit if gate_limit is not None else "NA",
        "ABC_pre_first_overlap_pair_count": sum(value > 0 for value in pair_pre),
        "ABC_pre_first_pairwise_overlap_ms": sum(pair_pre),
        "max_concurrent_pre_first_token": max_pre,
        "area_under_pre_first_token_concurrency_ms": area_pre,
        "ABC_total_gate_wait_ms": sum(float(row["gate_wait_ms"]) for row in rows),
        "ABC_max_gate_wait_ms": max(float(row["gate_wait_ms"]) for row in rows),
        "ABC_P50_TTFT_ms": percentile_linear(ttft, 50),
        "ABC_P95_TTFT_ms": percentile_linear(ttft, 95),
        "ABC_max_TTFT_ms": max(ttft),
        "ABC_P95_completion_latency_ms": percentile_linear(completion, 95),
        "ABC_completion_makespan_ms": max(float(row["request_finished_ms"]) for row in rows) - 300.0,
        "ABC_decode_throughput_tok_s": decode_throughput_tok_s(93, [x[0] for x in decode], [x[1] for x in decode]),
        "ABC_total_decode_overlap_ms": sum(pair_decode),
        "max_concurrent_post_first_token_decode": max_decode,
        "median_TPOT_ms": percentile_linear([float(row["TPOT_ms"]) for row in rows], 50),
        "idle_gpu_memory_time_mib_s": sum(float(row["gpu_ready_residency_ms"]) for row in rows) * 112.0 / 1000.0,
        "token_count_validation": "PASS" if token_ok else "FAIL",
        "cache_transport_validation": "PASS" if cache_ok and native_ok else "FAIL",
        "trace_validation": "PASS",
        "gate_invariant_validation": "PASS" if gate_ok else "FAIL",
        "decode_overlap_validation": "PASS" if max_decode >= 2 else "FAIL",
    }


def run_one(policy: str, gate_limit: int | None, repetition: int, prompts: dict[str, list[int]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], float]:
    global _active_gate_limit
    _active_gate_limit = gate_limit
    canonical = "InEngineFirstTokenGate-ResumeAware" if gate_limit is not None else policy
    original_out = base.OUT
    original_payload = transport.completion_payload
    base.OUT = OUT
    transport.completion_payload = _payload_with_gate_limit
    try:
        native, raw_rows, _, _, _, t0 = base.run_policy(canonical, repetition, prompts, SESSIONS)
    finally:
        transport.completion_payload = original_payload
        base.OUT = original_out
        _active_gate_limit = None
    canonical_log = OUT / "server_logs" / f"{canonical.lower().replace('-', '_')}_{repetition}.log"
    final_log = OUT / "server_logs" / f"{policy.lower().replace('-', '_')}_{repetition}.log"
    canonical_log.replace(final_log)
    trace = parse_enriched_trace(final_log, policy, repetition, t0)
    for row in native:
        row["policy"] = policy
        row["repetition"] = repetition
    sessions = make_session_rows(policy, repetition, gate_limit, raw_rows, trace, native)
    run = aggregate(policy, repetition, gate_limit, sessions, native, trace)
    return native, sessions, trace, run, t0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("integration", "full"), required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    prompts = base.make_prompts()
    if args.stage == "integration":
        checks: dict[str, Any] = {}
        for policy, limit in POLICIES[1:]:
            native, sessions, trace, run, t0 = run_one(policy, limit, 99, prompts)
            checks[policy] = {"run": run, "sessions": sessions, "native": native, "t0_monotonic_s": t0}
        valid = all(
            item["run"]["promotion_deadline_miss_count"] == 0
            and item["run"]["blackout_conflict_count"] == 0
            and item["run"]["long_prefetch_count"] == 0
            and item["run"]["token_count_validation"] == "PASS"
            and item["run"]["cache_transport_validation"] == "PASS"
            and item["run"]["gate_invariant_validation"] == "PASS"
            and item["run"]["decode_overlap_validation"] == "PASS"
            for item in checks.values()
        )
        (OUT / "integration_gate.json").write_text(json.dumps({"pass": valid, "checks": checks}, indent=2))
        print(f"INTEGRATION_GATE={'PASS' if valid else 'FAIL'}", flush=True)
        if not valid:
            raise SystemExit(2)
        return
    all_native: list[dict[str, Any]] = []
    all_sessions: list[dict[str, Any]] = []
    all_trace: list[dict[str, Any]] = []
    all_runs: list[dict[str, Any]] = []
    context: list[dict[str, Any]] = []
    for policy, limit in POLICIES:
        for repetition in range(3):
            print(f"RUN {policy} {repetition}", flush=True)
            native, sessions, trace, run, t0 = run_one(policy, limit, repetition, prompts)
            all_native.extend(native)
            all_sessions.extend(sessions)
            all_trace.extend(trace)
            all_runs.append(run)
            context.append({"policy": policy, "repetition": repetition, "t0_monotonic_s": t0})
    write_csv(OUT / "native_prefetch_metrics.csv", all_native)
    write_csv(OUT / "session_metrics.csv", all_sessions)
    write_csv(OUT / "scheduler_trace.csv", all_trace)
    write_csv(OUT / "step_metrics.csv", all_trace)
    write_csv(OUT / "run_metrics.csv", all_runs)
    write_csv(OUT / "run_context.csv", context)
    sequences = {
        f"{row['policy']}:{row['repetition']}:{row['session']}": {
            "hash": row["sequence_hash"],
            "token_count": row["generated_token_count"],
        }
        for row in all_sessions
    }
    (OUT / "generated_sequences.json").write_text(json.dumps(sequences, indent=2))
    print("FULL_DONE", flush=True)


if __name__ == "__main__":
    main()
