from __future__ import annotations

import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any

from experiments.phase2d import run_phase2d as runtime
from experiments.phase2d.helpers import (
    gpu_to_cpu_offload_bytes,
    parse_prometheus_samples,
    source_tokens_delta,
)
from experiments.phase3f.native_client import native_prefetch
from experiments.phase3g.run_phase3g import send
from experiments.phase3g_full import run_phase3g_full as base
from experiments.phase3s.helpers import (
    lease_sequence_valid,
    parse_structured_events,
    source_conservation,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase3s"
MODES = ("UnleasedPrefetch", "LeasedPrefetch")
RUNS = 3
OFFLOAD_BYTES = 117_440_512
TOKENS = 32


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def capture(stage: str, snapshots: list[dict[str, Any]]) -> dict[Any, float]:
    raw = runtime.metric_text()
    snapshots.append({"stage": stage, "timestamp_monotonic_s": time.monotonic(), "raw": raw})
    return parse_prometheus_samples(raw)


def parse_resume_trace(text: str, mode: str, repetition: int) -> list[dict[str, Any]]:
    return [{"mode": mode, "repetition": repetition, **row} for row in parse_structured_events(text, "SLACKKV_RESUME_TRACE ")]


def parse_lease_trace(text: str, mode: str, repetition: int) -> list[dict[str, Any]]:
    return [{"mode": mode, "repetition": repetition, **row} for row in parse_structured_events(text, "SLACKKV_LEASE_TRACE ")]


def run_one(mode: str, repetition: int, prompts: dict[str, list[int]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    label = f"{mode.lower()}_{repetition}"
    base.OUT = OUT
    proc = base.start_server(label)
    snapshots: list[dict[str, Any]] = []
    pressure_rows: list[dict[str, Any]] = []
    try:
        runtime.wait_ready(proc)
        before = capture("ready", snapshots)
        first = send(prompts["a"], stream=False, max_tokens=1, normal=True)
        after_first = capture("after_A_first", snapshots)
        offload = gpu_to_cpu_offload_bytes(before, after_first)
        if offload != OFFLOAD_BYTES:
            raise RuntimeError(f"A offload bytes {offload}, expected {OFFLOAD_BYTES}")
        time.sleep(1.0)
        for index in range(1, 5):
            response = send(prompts[f"pressure_{index}"], stream=False, max_tokens=1, normal=False)
            pressure_rows.append({"mode": mode, "repetition": repetition, "stage": "setup", "pressure": index, "completed": True, "e2e_ms": (response["completed_monotonic_s"] - response["sent_monotonic_s"]) * 1000.0})
        capture("after_setup_pressure", snapshots)

        lease_key = "A" if mode == "LeasedPrefetch" else None
        native = native_prefetch(prompts["a"], lease_key=lease_key)
        native_correct = (
            native.get("external_tokens_loaded") == 4096
            and native.get("model_forward_count") == 0
            and native.get("generated_token_count") == 0
            and native.get("mixed_prefetch_model_step_count") == 0
            and native.get("status") == "ok"
        )
        if not native_correct:
            raise RuntimeError(f"native prefetch invariant failure: {native}")

        pressure_started = time.monotonic()
        for index in range(1, 5):
            response = send(prompts[f"pressure_{index}"], stream=False, max_tokens=1, normal=False)
            pressure_rows.append({"mode": mode, "repetition": repetition, "stage": "post_prefetch", "pressure": index, "completed": True, "e2e_ms": (response["completed_monotonic_s"] - response["sent_monotonic_s"]) * 1000.0})
        baseline = capture("before_A_resume", snapshots)
        resume = send(prompts["a"] + prompts["suffix"], stream=True, max_tokens=TOKENS, trace=True, session="A", gate=True, normal=False)
        final = capture("after_A_resume", snapshots)

        local = int(source_tokens_delta(baseline, final, "local_cache_hit"))
        external = int(source_tokens_delta(baseline, final, "external_kv_transfer"))
        log_path = OUT / "server_logs" / f"{label}.log"
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        resume_trace = parse_resume_trace(text, mode, repetition)
        lease_trace = parse_lease_trace(text, mode, repetition)
        remote_wait = any(row.get("session") == "A" and row.get("event") == "WAITING_FOR_REMOTE_KVS" for row in resume_trace)
        lease_events = [row for row in lease_trace if row.get("lease_key") == "A"]
        acquired = sum(row["event"] == "LEASE_ACQUIRED" for row in lease_events)
        released = sum(row["event"] == "LEASE_RELEASED" for row in lease_events)
        leaked = bool(lease_events and lease_events[-1].get("active_lease_count") != 0)
        acquire_us = next((float(row["operation_cpu_us"]) for row in lease_events if row["event"] == "LEASE_ACQUIRED"), 0.0)
        release_us = next((float(row["operation_cpu_us"]) for row in lease_events if row["event"] == "LEASE_RELEASED"), 0.0)
        blocks = next((int(row["num_blocks"]) for row in lease_events if row["event"] == "LEASE_ACQUIRED"), 0)
        lease_ok = mode == "UnleasedPrefetch" or (acquired == 1 and released == 1 and not leaked and lease_sequence_valid(lease_events))
        run_valid = native_correct and len(resume["token_ids"]) == TOKENS and source_conservation(local, external) and lease_ok
        run = {
            "mode": mode, "repetition": repetition, "native_correct": native_correct,
            "lease_acquired": acquired == 1, "lease_released": released == 1,
            "lease_leak": leaked, "leased_block_count": blocks,
            "pressure_validation": all(row["completed"] for row in pressure_rows),
            "local_cache_hit_tokens": local, "external_kv_transfer_tokens": external,
            "local_fraction": local / 4096.0, "external_fraction": external / 4096.0,
            "remote_wait_seen": remote_wait, "generated_token_count": len(resume["token_ids"]),
            "TTFT_ms": resume["ttft_ms"],
            "completion_ms": (resume["completed_monotonic_s"] - resume["sent_monotonic_s"]) * 1000.0,
            "lease_acquire_cpu_us": acquire_us, "lease_release_cpu_us": release_us,
            "source_conservation": source_conservation(local, external), "run_validation": run_valid,
        }
        session = {"mode": mode, "repetition": repetition, "session": "A", "pressure_start_monotonic_s": pressure_started, **{k: run[k] for k in ("local_cache_hit_tokens", "external_kv_transfer_tokens", "remote_wait_seen", "generated_token_count", "TTFT_ms", "completion_ms")}}
        native_row = {"mode": mode, "repetition": repetition, "session": "A", **{key: native.get(key) for key in ("external_tokens_loaded", "transfer_bytes", "client_e2e_ms", "elapsed_ms", "cpu_to_gpu_transfer_ms", "connector_only_step_count", "normal_requests_delayed", "model_forward_count", "generated_token_count", "mixed_prefetch_model_step_count", "status")}}
        return run, session, native_row, pressure_rows, resume_trace, lease_trace, snapshots
    finally:
        base.stop_server(proc)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "server_logs").mkdir(exist_ok=True)
    prompts = base.make_prompts()
    runs=[]; sessions=[]; natives=[]; pressures=[]; traces=[]; leases=[]; snapshots=[]
    for mode in MODES:
        if mode == "LeasedPrefetch":
            evidence = sum((row["external_kv_transfer_tokens"] > 0 or row["remote_wait_seen"]) for row in runs if row["mode"] == "UnleasedPrefetch")
            if evidence == 0:
                break
        for repetition in range(RUNS):
            print(f"RUN {mode} {repetition}", flush=True)
            run, session, native, pressure, trace, lease, snap = run_one(mode, repetition, prompts)
            runs.append(run); sessions.append(session); natives.append(native); pressures.extend(pressure); traces.extend(trace); leases.extend(lease); snapshots.extend(snap)
    write_csv(OUT / "run_metrics.csv", runs); write_csv(OUT / "session_metrics.csv", sessions)
    write_csv(OUT / "native_prefetch_metrics.csv", natives); write_csv(OUT / "pressure_metrics.csv", pressures)
    write_csv(OUT / "scheduler_trace.csv", traces); write_csv(OUT / "lease_events.csv", leases)
    (OUT / "metric_snapshots.json").write_text(json.dumps(snapshots, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
