from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from typing import Any

from experiments.phase2d import run_phase2d as runtime
from experiments.phase2d.helpers import gpu_to_cpu_offload_bytes, parse_prometheus_samples, source_tokens_delta
from experiments.phase3f.native_client import native_prefetch
from experiments.phase3g import run_phase3g as transport
from experiments.phase3g_full import run_phase3g_full as base
from experiments.phase3n.helpers import active_x_valid, classify_handoff, internal_pre_return, local_resume_valid

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase3n"
OFFSETS_MS = (5, 20, 30, 40, 70)
BYTES = 117_440_512
TOKENS = 32


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def capture(stage: str, snapshots: list[dict[str, Any]]) -> dict[Any, float]:
    raw = runtime.metric_text()
    snapshots.append({"stage": stage, "timestamp_monotonic_s": time.monotonic(), "raw": raw})
    return parse_prometheus_samples(raw)


def parse_trace(path: Path, repetition: int, offset_ms: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    marker = "SLACKKV_RESUME_TRACE "
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if marker not in line:
            continue
        try:
            raw = json.loads(line.split(marker, 1)[1])
        except json.JSONDecodeError:
            continue
        rows.append({"offset_ms": offset_ms, "repetition": repetition, **raw})
    return rows


def first_event(rows: list[dict[str, Any]], session: str, name: str) -> dict[str, Any] | None:
    for row in rows:
        if row.get("session") == session and row.get("event") == name:
            return row
    return None


def send_x_background(ids: list[int], first_token: Event) -> dict[str, Any]:
    payload = transport.completion_payload(ids, stream=True, max_tokens=128, trace=True, session="X", gate=False, normal=True)
    sent = time.monotonic()
    import urllib.request
    request = urllib.request.Request("http://127.0.0.1:8000/v1/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    token_ids: list[int] = []
    first = None
    with urllib.request.urlopen(request, timeout=900) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            body = line[6:].strip()
            if body == b"[DONE]":
                continue
            choice = (json.loads(body).get("choices") or [{}])[0]
            ids_now = [int(value) for value in (choice.get("token_ids") or [])]
            if ids_now:
                token_ids.extend(ids_now)
                if first is None:
                    first = time.monotonic(); first_token.set()
    done = time.monotonic()
    if first is None:
        raise RuntimeError("X did not generate a first token")
    return {"sent_monotonic_s": sent, "first_monotonic_s": first, "completed_monotonic_s": done, "token_ids": token_ids}


def start_server(label: str):
    base.OUT = OUT
    return base.start_server(label)


def run_once(offset_ms: int, repetition: int, prompts: dict[str, list[int]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    label = f"offset_{offset_ms}_{repetition}"
    proc = start_server(label)
    snapshots: list[dict[str, Any]] = []
    try:
        runtime.wait_ready(proc)
        previous = capture("ready", snapshots)
        base.send(prompts["a"], stream=False, max_tokens=1, normal=True)
        after_a = capture("after_a_first", snapshots)
        offload = gpu_to_cpu_offload_bytes(previous, after_a)
        if offload != BYTES:
            raise RuntimeError(f"A offload bytes {offload}, expected {BYTES}")
        time.sleep(1.0)
        for index in range(1, 5):
            base.send(prompts[f"pressure_{index}"], stream=False, max_tokens=1, normal=False)
        capture("after_pressure", snapshots)
        x_first = Event()
        with ThreadPoolExecutor(max_workers=3) as pool:
            x_future = pool.submit(send_x_background, prompts["pressure_1"], x_first)
            if not x_first.wait(180):
                raise RuntimeError("X first-token barrier timed out")
            prefetch_send_s = time.monotonic()
            prefetch_future = pool.submit(native_prefetch, prompts["a"])
            target = prefetch_send_s + offset_ms / 1000.0
            base.wait_until(target)
            before_resume = capture("before_a_resume", snapshots)
            a_client = base.send(prompts["a"] + prompts["suffix"], stream=True, max_tokens=TOKENS, trace=True, session="A", gate=False, normal=False)
            after_resume = capture("after_a_resume", snapshots)
            native = prefetch_future.result(timeout=300)
            x = x_future.result(timeout=900)
        if len(a_client["token_ids"]) != TOKENS:
            raise RuntimeError(f"A emitted {len(a_client['token_ids'])}, expected {TOKENS}")
        log_path = OUT / "server_logs" / f"{label}.log"
        trace = parse_trace(log_path, repetition, offset_ms)
        arrival = first_event(trace, "A", "RESUME_ARRIVED")
        admitted = first_event(trace, "A", "RESUME_ADMITTED")
        first = first_event(trace, "A", "FIRST_TOKEN")
        finished = first_event(trace, "A", "REQUEST_FINISHED")
        if not all((arrival, admitted, first, finished)):
            raise RuntimeError(f"incomplete A trace: {[row.get('event') for row in trace if row.get('session') == 'A']}")
        phase = native.get("phase3m_timestamps") or {}
        required = ("T0", "T1", "T2", "T3", "T5", "T6", "T7")
        if not all(key in phase for key in required):
            raise RuntimeError(f"missing Phase3M timestamps: {phase}")
        ready_s, engine_done_s, endpoint_return_s = float(phase["T6"]), float(phase["T7"]), float(native["completed_monotonic_s"])
        arrival_s = float(arrival["timestamp_monotonic"])
        classification = classify_handoff(arrival_s, ready_s, endpoint_return_s)
        in_t6_t7, in_t7_return = internal_pre_return(arrival_s, ready_s, engine_done_s, endpoint_return_s)
        remote_wait = first_event(trace, "A", "WAITING_FOR_REMOTE_KVS") is not None
        local = source_tokens_delta(before_resume, after_resume, "local_cache_hit")
        external = source_tokens_delta(before_resume, after_resume, "external_kv_transfer")
        x_first_trace = first_event(trace, "X", "FIRST_TOKEN")
        x_finish_trace = first_event(trace, "X", "REQUEST_FINISHED")
        x_first_s = float(x_first_trace["timestamp_monotonic"]) if x_first_trace else float(x["first_monotonic_s"])
        x_finish_s = float(x_finish_trace["timestamp_monotonic"]) if x_finish_trace else float(x["completed_monotonic_s"])
        x_valid = active_x_valid(x_first_s, prefetch_send_s, endpoint_return_s, x_finish_s)
        native_ok = native["status"] == "ok" and native["external_tokens_loaded"] == 4096 and native["model_forward_count"] == 0 and native["generated_token_count"] == 0 and native["mixed_prefetch_model_step_count"] == 0
        resume_ok = local_resume_valid(local, external, remote_wait)
        native_row = {
            "offset_ms": offset_ms, "repetition": repetition, "prefetch_client_send_s": prefetch_send_s,
            "prefetch_client_return_s": endpoint_return_s, "T0_request_received_s": phase["T0"], "T1_engine_enqueue_s": phase["T1"],
            "T2_scheduler_seen_s": phase["T2"], "T3_connector_start_s": phase["T3"], "T5_transfer_complete_s": phase["T5"],
            "T6_scheduler_complete_s": ready_s, "T7_http_complete_s": engine_done_s,
            "external_tokens_loaded": native["external_tokens_loaded"], "transfer_bytes": native["transfer_bytes"], "client_e2e_ms": native["client_e2e_ms"],
            "server_e2e_ms": native["elapsed_ms"], "cpu_to_gpu_ms": native["cpu_to_gpu_transfer_ms"], "connector_only_step_count": native["connector_only_step_count"],
            "normal_requests_delayed": native["normal_requests_delayed"], "model_forward_count": native["model_forward_count"],
            "generated_token_count": native["generated_token_count"], "mixed_prefetch_model_step_count": native["mixed_prefetch_model_step_count"],
            "status": native["status"], "native_validation": "PASS" if native_ok else "FAIL",
        }
        resume_row = {
            "offset_ms": offset_ms, "repetition": repetition, "classification": classification, "scheduler_arrival_s": arrival_s,
            "client_request_send_s": a_client["sent_monotonic_s"], "scheduler_admission_s": admitted["timestamp_monotonic"], "first_token_s": first["timestamp_monotonic"],
            "request_finished_s": finished["timestamp_monotonic"], "resume_ttft_ms": (float(first["timestamp_monotonic"]) - a_client["sent_monotonic_s"]) * 1000.0,
            "local_hit_tokens": local, "external_hit_tokens": external, "waiting_for_remote_kvs": remote_wait, "generated_token_count": len(a_client["token_ids"]),
            "cache_transport_validation": "PASS" if resume_ok else "FAIL", "x_active_validation": "PASS" if x_valid else "FAIL",
        }
        timing_row = {
            "offset_ms": offset_ms, "repetition": repetition, "classification": classification, "scheduler_arrival_s": arrival_s,
            "arrival_minus_prefetch_send_ms": (arrival_s-prefetch_send_s)*1000.0, "arrival_minus_T6_ms": (arrival_s-ready_s)*1000.0,
            "arrival_minus_T7_ms": (arrival_s-engine_done_s)*1000.0, "arrival_minus_client_return_ms": (arrival_s-endpoint_return_s)*1000.0,
            "T6_ready_latency_ms": (ready_s-prefetch_send_s)*1000.0, "T6_to_T7_ms": (engine_done_s-ready_s)*1000.0,
            "T6_to_client_return_ms": (endpoint_return_s-ready_s)*1000.0, "in_T6_to_T7": in_t6_t7, "in_T7_to_return": in_t7_return,
            "x_first_token_s": x_first_s, "x_finished_s": x_finish_s, "x_active_validation": "PASS" if x_valid else "FAIL",
        }
        return native_row, resume_row, timing_row, trace, snapshots
    finally:
        base.stop_server(proc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offset-ms", type=int, choices=OFFSETS_MS, required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); OUT.joinpath("server_logs").mkdir(exist_ok=True)
    prompts = base.make_prompts()
    native_rows: list[dict[str, Any]] = []; resume_rows: list[dict[str, Any]] = []; timing_rows: list[dict[str, Any]] = []; trace_rows: list[dict[str, Any]] = []; snapshot_rows: list[dict[str, Any]] = []
    for repetition in range(3):
        print(f"RUN offset={args.offset_ms} repetition={repetition}", flush=True)
        native, resume, timing, trace, snapshots = run_once(args.offset_ms, repetition, prompts)
        native_rows.append(native); resume_rows.append(resume); timing_rows.append(timing); trace_rows.extend(trace)
        snapshot_rows.extend({"offset_ms": args.offset_ms, "repetition": repetition, **item} for item in snapshots)
    batch = OUT / "batches" / f"offset_{args.offset_ms}"; batch.mkdir(parents=True, exist_ok=True)
    write_csv(batch / "native_prefetch_metrics.csv", native_rows); write_csv(batch / "resume_metrics.csv", resume_rows); write_csv(batch / "timing_boundary.csv", timing_rows); write_csv(batch / "scheduler_trace.csv", trace_rows)
    (batch / "metric_snapshots.json").write_text(json.dumps(snapshot_rows, indent=2), encoding="utf-8")
    print("OFFSET_DONE", args.offset_ms, flush=True)


if __name__ == "__main__":
    main()
