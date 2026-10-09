"""Frozen BGv2 OnDemand diagnostic; own server, bounded duration, raw logs kept."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import threading
import time
import traceback
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--gate", type=int, default=2)
    parser.add_argument("--backfill", type=int, choices=(0, 2), default=0)
    parser.add_argument("--execution", choices=("eager", "default"), default="eager")
    parser.add_argument("--oracle", action="store_true")
    args = parser.parse_args()
    if args.gate < 1 or Path(args.name).name != args.name:
        parser.error("positive gate and plain run name required")
    from experiments.phase4d_a import runtime
    from experiments.phase4g_a import run_phase4g_a as legacy
    from experiments.phase4g_a_fix import run_phase4g_a_fix as harness
    from experiments.phase4g_c_fix.absolute_sleep import phase4g_c_fix_wait_until
    from experiments.phase4g_a.workload import SOURCE, SOURCE_SHA

    root = Path(__file__).resolve().parents[2]
    previous = [json.loads(p.read_text()) for p in
                (root / "results/autonomous_admission").glob("*/execution.json")]
    if any(item["status"] == "RUNNING" for item in previous):
        raise RuntimeError("another owned experiment is still marked RUNNING")
    spent = sum(item.get("gpu_hours_conservative", 0) for item in previous)
    if spent + 0.5 > 4:
        raise RuntimeError("remaining round budget cannot cover a 30-minute run")
    out = root / "results/autonomous_admission" / args.name
    out.mkdir(parents=True, exist_ok=False)
    runtime.SRC = Path("/media/a6000/D/ZhaoBinHe/slackkv-autonomous-20261009-source")
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA:
        raise RuntimeError("frozen workload fingerprint mismatch")
    harness.OUT = out
    harness.configure_legacy_lifecycle()
    legacy.K = args.gate
    legacy.wait_until = phase4g_c_fix_wait_until
    os.environ["SLACKKV_CAPACITY_BACKFILL_LIMIT"] = str(args.backfill)
    original_command = runtime.server_command

    def server_command(connector, gpu_bytes):
        command = original_command(connector, gpu_bytes)
        if args.execution == "default":
            command.remove("--enforce-eager")
        return command

    runtime.server_command = server_command
    original_start = runtime.start_server
    owned_server = []

    def start_server(*positional, **kwargs):
        proc, log = original_start(*positional, **kwargs)
        owned_server.append(proc)
        return proc, log

    runtime.start_server = start_server
    original_stop = runtime.stop_server

    def stop_server(proc, log, run_directory, label):
        try:
            saved = run_directory / "run.json"
            if args.oracle and saved.exists() and json.loads(saved.read_text())["valid"]:
                from experiments.autonomous_admission.oracle import run_oracle
                run_oracle(runtime, out)
        finally:
            code = original_stop(proc, log, run_directory, label)
        return code

    runtime.stop_server = stop_server
    original_send = runtime.send
    output_lock = threading.Lock()

    def send(*positional, **kwargs):
        params = dict(kwargs.get("params") or {})
        if args.trace:
            params.update(slackkv_allocator_trace=True,
                          slackkv_allocator_trace_interval_ms=100)
        kwargs["params"] = params
        response = original_send(*positional, **kwargs)
        session = params.get("slackkv_session")
        record = {"session": session, "tokens": response["tokens"]}
        if session in {"E02:turn:2", "E04:turn:1"}:
            record["prompt"] = positional[0]
        with output_lock:
            with (out / "output_tokens.jsonl").open("a") as stream:
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        return response

    runtime.send = send
    metadata = {
        "name": args.name, "gate": args.gate, "allocator_trace": args.trace,
        "backfill_limit": args.backfill,
        "execution_mode": args.execution,
        "cold_oracle": args.oracle,
        "trace_interval_ms": 100, "gpu_count": 1, "max_seconds": 1800,
        "source_sha": SOURCE_SHA, "source_directory": str(runtime.SRC),
        "kv_bytes": 2147483648, "bg_tokens": 8192,
        "server_command": runtime.server_command(True, 2147483648),
        "round_gpu_hours_before_run": spent,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "started_unix": time.time(), "status": "RUNNING",
        "purpose": "diagnosis" if args.trace else "pre-registered matched screen",
    }
    metadata["allocator_sha256"] = hashlib.sha256(
        (runtime.SRC / "vllm/v1/core/kv_cache_manager.py").read_bytes()).hexdigest()
    metadata["scheduler_sha256"] = hashlib.sha256(
        (runtime.SRC / "vllm/v1/core/sched/scheduler.py").read_bytes()).hexdigest()
    path = out / "execution.json"
    path.write_text(json.dumps(metadata, indent=2) + "\n")
    started = time.monotonic()

    def timeout(_signum, _frame):
        for proc in owned_server:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
        raise TimeoutError("experiment watchdog; 90 seconds reserved for cleanup")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(1710)
    try:
        result = legacy.run_lifecycle("A", "measured", 1)
        harness._decorate_agent_run(result, "measured_rep1_a")
        metadata["status"] = "VALID" if result["valid"] else "INVALID"
    except BaseException:
        metadata["status"] = "FAILED"
        (out / "failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        signal.alarm(0)
        metadata["finished_unix"] = time.time()
        metadata["wall_seconds"] = time.monotonic() - started
        metadata["gpu_hours_conservative"] = metadata["wall_seconds"] / 3600
        path.write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
