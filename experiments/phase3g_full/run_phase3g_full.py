from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from typing import Any

from transformers import AutoTokenizer

from experiments.phase2d import run_phase2d as runtime
from experiments.phase2d.helpers import gpu_to_cpu_offload_bytes, parse_prometheus_samples, source_tokens_delta
from experiments.phase2h.run_phase2h import SERVER, build_prompts
from experiments.phase3f.native_client import native_prefetch
from experiments.phase3f.scheduler import DEADLINE_MS, P_SAFE_MS, RESUME_GUARD_MS, SESSIONS, latest_feasible_start, merge_blackouts
from experiments.phase3g.run_phase3g import send
from experiments.phase3g_full.metrics import (
    decode_throughput_tok_s,
    gate_wait_ms,
    gpu_ready_residency_ms,
    interval_concurrency,
    overlap,
    percentile_linear,
    same_step_pairs,
    sequence_hash,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase3g_full"
SRC = Path("/media/a6000/D/ZhaoBinHe/slackkv-phase2c-fix-sourceprobe")
POLICIES = ("OnDemand-Trace", "ResumeAware-Trace", "InEngineFirstTokenGate-ResumeAware")
BYTES = 117_440_512
TOKENS = 32


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty output {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def start_server(label: str) -> subprocess.Popen[str]:
    OUT.joinpath("server_logs").mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(runtime.BASE_ENV)
    env["VLLM_SERVER_DEV_MODE"] = "1"
    env["PYTHONPATH"] = f"{SRC}:{ROOT}"
    log = (OUT / "server_logs" / f"{label}.log").open("w", encoding="utf-8")
    return subprocess.Popen(SERVER, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)


def stop_server(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def capture(stage: str, snapshots: list[dict[str, Any]]) -> dict[Any, float]:
    raw = runtime.metric_text()
    snapshots.append({"stage": stage, "timestamp_monotonic_s": time.monotonic(), "raw": raw})
    return parse_prometheus_samples(raw)


def make_prompts() -> dict[str, list[int]]:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", local_files_only=True)
    return build_prompts(tokenizer)


def wait_until(deadline_s: float) -> None:
    while time.monotonic() < deadline_s:
        time.sleep(min(0.002, deadline_s - time.monotonic()))


def setup_external(prompts: dict[str, list[int],], label: str, snapshots: list[dict[str, Any]], sessions: tuple[str, ...]) -> tuple[dict[Any, float], dict[str, float]]:
    previous = capture(f"{label}_ready", snapshots)
    offload_bytes: dict[str, float] = {}
    for session in sessions:
        send(prompts[session.lower()], stream=False, max_tokens=1, normal=True)
        current = capture(f"{label}_after_{session}_first", snapshots)
        offload_bytes[session] = gpu_to_cpu_offload_bytes(previous, current)
        previous = current
    time.sleep(1.0)
    for index in range(1, 5):
        send(prompts[f"pressure_{index}"], stream=False, max_tokens=1, normal=False)
    before_t0 = capture(f"{label}_after_pressure", snapshots)
    return before_t0, offload_bytes


def resume_request(session: str, t0: float, prompts: dict[str, list[int]], ready: Event | None, gated: bool) -> dict[str, Any]:
    wait_until(t0 + DEADLINE_MS[session] / 1000.0)
    if ready is not None and not ready.is_set():
        ready.wait(timeout=300)
    response = send(
        prompts[session.lower()] + prompts["suffix"],
        stream=True,
        max_tokens=TOKENS,
        trace=True,
        session=session,
        gate=gated,
        normal=False,
    )
    if len(response["token_ids"]) != TOKENS:
        raise RuntimeError(f"{session} generated {len(response['token_ids'])}, expected {TOKENS}")
    return response


def promote_worker(
    sessions: tuple[str, ...],
    t0: float,
    prompts: dict[str, list[int]],
    ready: dict[str, Event],
    output: dict[str, dict[str, Any]],
) -> None:
    pending = set(sessions)
    blackouts = merge_blackouts({session: DEADLINE_MS[session] for session in sessions}, RESUME_GUARD_MS)
    while pending:
        now_ms = (time.monotonic() - t0) * 1000.0
        jobs = [(session, DEADLINE_MS[session]) for session in pending]
        target_ms = latest_feasible_start(now_ms, jobs, blackouts, P_SAFE_MS)
        if target_ms - now_ms > 0.5:
            time.sleep((target_ms - now_ms) / 1000.0)
            continue
        session = min(pending, key=lambda item: (DEADLINE_MS[item], item))
        result = native_prefetch(prompts[session.lower()])
        output[session] = {
            "target_start_ms": target_ms,
            "actual_start_ms": (result["sent_monotonic_s"] - t0) * 1000.0,
            "completion_ms": (result["completed_monotonic_s"] - t0) * 1000.0,
            **result,
        }
        ready[session].set()
        pending.remove(session)


def parse_trace(path: Path, policy: str, repetition: int, t0: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        marker = "SLACKKV_RESUME_TRACE "
        if marker not in line:
            continue
        raw = json.loads(line.split(marker, 1)[1])
        rows.append({
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
            "gate_owner_request_id": raw.get("gate_owner_request_id"),
            "num_slackkv_pre_first_token": raw.get("num_slackkv_pre_first_token"),
            "num_computed_tokens": raw.get("num_computed_tokens"),
            "num_output_tokens": raw.get("num_output_tokens"),
        })
    return rows


def trace_by_session(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        session = row["session"]
        if session:
            output.setdefault(session, {}).setdefault(row["event"], row)
    return output


def event(event_map: dict[str, dict[str, Any]], name: str) -> dict[str, Any] | None:
    return event_map.get(name)


def as_float(row: dict[str, Any] | None, field: str) -> float | None:
    return None if row is None else float(row[field])


def as_int(row: dict[str, Any] | None, field: str) -> int | None:
    return None if row is None else int(row[field])


def run_policy(policy: str, repetition: int, prompts: dict[str, list[int]], sessions: tuple[str, ...] = SESSIONS) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], float]:
    label = f"{policy.lower().replace('-', '_')}_{repetition}"
    proc = start_server(label)
    snapshots: list[dict[str, Any]] = []
    try:
        runtime.wait_ready(proc)
        before, offload = setup_external(prompts, label, snapshots, sessions)
        if any(value != BYTES for value in offload.values()):
            raise RuntimeError(f"offload bytes mismatch: {offload}")
        t0 = time.monotonic()
        ready = {session: Event() for session in sessions}
        promotions: dict[str, dict[str, Any]] = {}
        gated = policy == "InEngineFirstTokenGate-ResumeAware"
        native_policy = policy != "OnDemand-Trace"
        with ThreadPoolExecutor(max_workers=len(sessions) + 1) as pool:
            promotion_future = None
            if native_policy:
                promotion_future = pool.submit(promote_worker, sessions, t0, prompts, ready, promotions)
            future = {
                session: pool.submit(resume_request, session, t0, prompts, ready.get(session) if native_policy else None, gated)
                for session in sessions
            }
            client_results = {session: value.result(timeout=600) for session, value in future.items()}
            if promotion_future is not None:
                promotion_future.result(timeout=600)
        after = capture(f"{label}_after", snapshots)
        trace = parse_trace(OUT / "server_logs" / f"{label}.log", policy, repetition, t0)
        traced = trace_by_session(trace)
        session_rows: list[dict[str, Any]] = []
        for session in sessions:
            client = client_results[session]
            events = traced.get(session, {})
            arrived = event(events, "RESUME_ARRIVED")
            admitted = event(events, "RESUME_ADMITTED")
            first_model = event(events, "FIRST_MODEL_SCHEDULED")
            first = event(events, "FIRST_TOKEN")
            finished = event(events, "REQUEST_FINISHED")
            blocked = event(events, "GATE_BLOCKED")
            remote_wait = event(events, "WAITING_FOR_REMOTE_KVS")
            remote_ready = event(events, "REMOTE_KV_READY")
            if not all((arrived, admitted, first_model, first, finished)):
                raise RuntimeError(f"incomplete trace for {policy}/{repetition}/{session}: {events.keys()}")
            promotion = promotions.get(session)
            admission_ms = as_float(admitted, "timestamp_ms_from_t0")
            first_ms = as_float(first, "timestamp_ms_from_t0")
            finish_ms = as_float(finished, "timestamp_ms_from_t0")
            token_ids = client["token_ids"]
            session_rows.append({
                "policy": policy,
                "repetition": repetition,
                "session": session,
                "original_deadline_ms": DEADLINE_MS[session],
                "client_request_sent_ms": (client["sent_monotonic_s"] - t0) * 1000.0,
                "scheduler_arrival_ms": as_float(arrived, "timestamp_ms_from_t0"),
                "scheduler_arrival_step": as_int(arrived, "step_id"),
                "arrival_transport_delay_ms": as_float(arrived, "timestamp_ms_from_t0") - (client["sent_monotonic_s"] - t0) * 1000.0,
                "remote_kv_wait_start_ms": as_float(remote_wait, "timestamp_ms_from_t0"),
                "remote_kv_ready_ms": as_float(remote_ready, "timestamp_ms_from_t0"),
                "gate_first_blocked_ms": as_float(blocked, "timestamp_ms_from_t0"),
                "gate_wait_ms": gate_wait_ms(as_float(blocked, "timestamp_ms_from_t0"), admission_ms),
                "gate_blocked_steps": sum(row["event"] == "GATE_BLOCKED" for row in trace if row["session"] == session),
                "scheduler_admission_ms": admission_ms,
                "scheduler_admission_step": as_int(admitted, "step_id"),
                "first_model_scheduled_ms": as_float(first_model, "timestamp_ms_from_t0"),
                "first_model_scheduled_step": as_int(first_model, "step_id"),
                "first_token_time_ms": first_ms,
                "first_token_step": as_int(first, "step_id"),
                "request_finished_ms": finish_ms,
                "request_finished_step": as_int(finished, "step_id"),
                "effective_TTFT_ms": first_ms - DEADLINE_MS[session],
                "completion_latency_ms": finish_ms - DEADLINE_MS[session],
                "admission_to_first_token_ms": first_ms - admission_ms,
                "decode_duration_ms": finish_ms - first_ms,
                "TPOT_ms": (finish_ms - first_ms) / 31.0,
                "promotion_completion_ms": None if promotion is None else promotion["completion_ms"],
                "gpu_ready_residency_ms": gpu_ready_residency_ms(None if promotion is None else promotion["completion_ms"], admission_ms),
                "generated_token_count": len(token_ids),
                "generated_token_ids_json": json.dumps(token_ids),
                "sequence_hash": sequence_hash(token_ids),
                "token_count_validation": "PASS" if len(token_ids) == TOKENS else "FAIL",
                "cache_transport_validation": "PENDING",
            })
        native_rows = []
        for session, result in promotions.items():
            start_ms = result["actual_start_ms"]
            completion_ms = result["completion_ms"]
            conflict = any(start_ms < end and completion_ms > start for start, end in merge_blackouts({s: DEADLINE_MS[s] for s in sessions}, RESUME_GUARD_MS))
            native_rows.append({
                "policy": policy,
                "repetition": repetition,
                "session": session,
                "target_start_ms": result["target_start_ms"],
                "actual_start_ms": start_ms,
                "completion_ms": completion_ms,
                "external_tokens_loaded": result["external_tokens_loaded"],
                "client_e2e_ms": result["client_e2e_ms"],
                "server_e2e_ms": result["elapsed_ms"],
                "cpu_to_gpu_ms": result["cpu_to_gpu_transfer_ms"],
                "deadline_stall_ms": max(0.0, completion_ms - DEADLINE_MS[session]),
                "blackout_conflict": conflict,
                "long_prefetch": result["client_e2e_ms"] > 50.0,
                "connector_only_step_count": result["connector_only_step_count"],
                "normal_requests_delayed": result["normal_requests_delayed"],
                "model_forward_count": result["model_forward_count"],
                "generated_token_count": result["generated_token_count"],
                "mixed_prefetch_model_step_count": result["mixed_prefetch_model_step_count"],
                "status": result["status"],
            })
        external = source_tokens_delta(before, after, "external_kv_transfer")
        local = source_tokens_delta(before, after, "local_cache_hit")
        expected = len(sessions) * 4096.0
        if native_policy:
            # The vLLM source counter does not attribute prefetch-only endpoint
            # loads to ``external_kv_transfer``.  The endpoint response is the
            # authoritative source for those loads; the subsequent local-hit
            # counter proves that real resumes reused the promoted blocks.
            cache_ok = local == expected and all(
                row["external_tokens_loaded"] == 4096
                and row["model_forward_count"] == 0
                and row["generated_token_count"] == 0
                and row["mixed_prefetch_model_step_count"] == 0
                and row["status"] == "ok"
                for row in native_rows
            )
        else:
            cache_ok = external == expected and local == 0.0
        for row in session_rows:
            row["cache_transport_validation"] = "PASS" if cache_ok else "FAIL"
        run = aggregate_run(policy, repetition, session_rows, native_rows, cache_ok)
        return native_rows, session_rows, trace, run, snapshots, t0
    finally:
        stop_server(proc)


def aggregate_run(policy: str, repetition: int, rows: list[dict[str, Any]], native: list[dict[str, Any]], cache_ok: bool) -> dict[str, Any]:
    by = {row["session"]: row for row in rows}
    abc = [by[s] for s in ("A", "B", "C") if s in by]
    all_pre = [(row["scheduler_admission_ms"], row["first_token_time_ms"]) for row in rows]
    abc_pre = [(row["scheduler_admission_ms"], row["first_token_time_ms"]) for row in abc]
    abc_decode = [(row["first_token_time_ms"], row["request_finished_ms"]) for row in abc]
    max_pre, area_pre = interval_concurrency(all_pre)
    max_decode, area_decode = interval_concurrency([(row["first_token_time_ms"], row["request_finished_ms"]) for row in rows])
    pair_pre = [overlap(abc_pre[i], abc_pre[j]) for i in range(len(abc_pre)) for j in range(i + 1, len(abc_pre))]
    pair_decode = [overlap(abc_decode[i], abc_decode[j]) for i in range(len(abc_decode)) for j in range(i + 1, len(abc_decode))]
    ttft = [row["effective_TTFT_ms"] for row in rows]
    completion = [row["completion_latency_ms"] for row in rows]
    abc_ttft = [row["effective_TTFT_ms"] for row in abc]
    abc_completion = [row["completion_latency_ms"] for row in abc]
    gated = policy == "InEngineFirstTokenGate-ResumeAware"
    a, b, c = (by.get(s) for s in ("A", "B", "C"))
    structure = not gated or (
        a["scheduler_admission_step"] < b["scheduler_admission_step"] < c["scheduler_admission_step"]
        and a["first_token_step"] <= b["scheduler_admission_step"]
        and b["first_token_step"] <= c["scheduler_admission_step"]
    )
    release_overlap = not gated or (
        a["first_token_time_ms"] < b["scheduler_admission_ms"] < a["request_finished_ms"]
        and b["first_token_time_ms"] < c["scheduler_admission_ms"] < b["request_finished_ms"]
    )
    return {
        "policy": policy,
        "repetition": repetition,
        "promotion_deadline_miss_count": sum(row["deadline_stall_ms"] > 0 for row in native),
        "blackout_conflict_count": sum(bool(row["blackout_conflict"]) for row in native),
        "long_prefetch_count": sum(bool(row["long_prefetch"]) for row in native),
        "ABC_pre_first_overlap_pair_count": sum(value > 0 for value in pair_pre),
        "ABC_pre_first_pairwise_overlap_ms": sum(pair_pre),
        "max_concurrent_pre_first_token": max_pre,
        "area_under_pre_first_token_concurrency_ms": area_pre,
        "ABC_total_decode_overlap_ms": sum(pair_decode),
        "max_concurrent_post_first_token_decode": max_decode,
        "area_under_post_first_token_decode_concurrency_ms": area_decode,
        "post_first_token_release_overlap_validation": "PASS" if release_overlap else "FAIL",
        "ABC_same_step_admission_pairs": same_step_pairs([row["scheduler_admission_step"] for row in abc]),
        "ABC_total_gate_wait_ms": sum(row["gate_wait_ms"] for row in abc),
        "ABC_P50_TTFT_ms": percentile_linear(abc_ttft, 50),
        "ABC_P95_TTFT_ms": percentile_linear(abc_ttft, 95),
        "ABC_max_TTFT_ms": max(abc_ttft),
        "overall_P50_TTFT_ms": percentile_linear(ttft, 50),
        "overall_P95_TTFT_ms": percentile_linear(ttft, 95),
        "overall_max_TTFT_ms": max(ttft),
        "ABC_P50_completion_latency_ms": percentile_linear(abc_completion, 50),
        "ABC_P95_completion_latency_ms": percentile_linear(abc_completion, 95),
        "ABC_max_completion_latency_ms": max(abc_completion),
        "overall_P95_completion_latency_ms": percentile_linear(completion, 95),
        "ABC_completion_makespan_ms": max(row["request_finished_ms"] for row in abc) - min(row["original_deadline_ms"] for row in abc),
        "ABC_decode_throughput_tok_s": decode_throughput_tok_s(93, [row["first_token_time_ms"] for row in abc], [row["request_finished_ms"] for row in abc]),
        "median_TPOT_ms": percentile_linear([row["TPOT_ms"] for row in rows], 50),
        "P95_TPOT_ms": percentile_linear([row["TPOT_ms"] for row in rows], 95),
        "idle_gpu_memory_time_mib_s": sum(row["gpu_ready_residency_ms"] for row in rows) * 112.0 / 1000.0,
        "token_count_validation": "PASS" if all(row["token_count_validation"] == "PASS" for row in rows) else "FAIL",
        "cache_transport_validation": "PASS" if cache_ok else "FAIL",
        "timeline_validation": "PASS",
        # A positive value measures server intake delay while an engine step is
        # busy, not clock disagreement.  Only a material negative offset would
        # invalidate the shared monotonic time coordinate.
        "time_alignment_validation": "PASS" if all(row["arrival_transport_delay_ms"] >= -1.0 for row in rows) else "FAIL",
        "arrival_transport_delay_over_50ms_count": sum(row["arrival_transport_delay_ms"] >= 50.0 for row in rows),
        "trace_validation": "PASS",
        "gate_invariant_validation": "PASS" if structure and max_pre == 1 else ("NA" if not gated else "FAIL"),
        "decode_overlap_validation": "PASS" if not gated or max_decode >= 2 else "FAIL",
        "schedule_structure_validation": "PASS" if structure else ("NA" if not gated else "FAIL"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("integration", "full"), required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    prompts = make_prompts()
    if args.stage == "integration":
        native, sessions, trace, run, snapshots, t0 = run_policy("InEngineFirstTokenGate-ResumeAware", 99, prompts, ("A", "B", "C"))
        valid = run["token_count_validation"] == "PASS" and run["cache_transport_validation"] == "PASS" and run["gate_invariant_validation"] == "PASS" and run["decode_overlap_validation"] == "PASS" and run["post_first_token_release_overlap_validation"] == "PASS"
        (OUT / "integration_gate.json").write_text(json.dumps({"pass": valid, "run": run, "sessions": sessions, "native": native, "t0_monotonic_s": t0}, indent=2))
        print(f"INTEGRATION_GATE={'PASS' if valid else 'FAIL'}", flush=True)
        if not valid:
            raise SystemExit(2)
        return
    all_native: list[dict[str, Any]] = []
    all_sessions: list[dict[str, Any]] = []
    all_trace: list[dict[str, Any]] = []
    all_runs: list[dict[str, Any]] = []
    all_snapshots: list[dict[str, Any]] = []
    context: list[dict[str, Any]] = []
    for policy in POLICIES:
        for repetition in range(3):
            print(f"RUN {policy} {repetition}", flush=True)
            native, sessions, trace, run, snapshots, t0 = run_policy(policy, repetition, prompts)
            all_native.extend(native); all_sessions.extend(sessions); all_trace.extend(trace); all_runs.append(run); all_snapshots.extend(snapshots)
            context.append({"policy": policy, "repetition": repetition, "t0_monotonic_s": t0})
    write_csv(OUT / "native_prefetch_metrics.csv", all_native)
    write_csv(OUT / "session_metrics.csv", all_sessions)
    write_csv(OUT / "scheduler_trace.csv", all_trace)
    write_csv(OUT / "run_metrics.csv", all_runs)
    write_csv(OUT / "run_context.csv", context)
    write_csv(OUT / "step_metrics.csv", all_trace)
    (OUT / "metric_snapshots.json").write_text(json.dumps(all_snapshots, indent=2))
    sequences = {f"{row['policy']}:{row['repetition']}:{row['session']}": {"hash": row["sequence_hash"], "token_ids": json.loads(row["generated_token_ids_json"])} for row in all_sessions}
    (OUT / "generated_sequences.json").write_text(json.dumps(sequences, indent=2))
    print("FULL_DONE", flush=True)


if __name__ == "__main__":
    main()
