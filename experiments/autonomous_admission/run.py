"""Frozen BGv2 OnDemand diagnostic; own server, bounded duration, raw logs kept."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import time
import traceback
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--gate", type=int, default=2)
    parser.add_argument("--backfill", type=int, choices=(0, 2), default=0)
    args = parser.parse_args()
    if args.gate < 1 or Path(args.name).name != args.name:
        parser.error("positive gate and plain run name required")
    from experiments.phase4d_a import runtime
    from experiments.phase4g_a import run_phase4g_a as legacy
    from experiments.phase4g_a_fix import run_phase4g_a_fix as harness
    from experiments.phase4g_c_fix.absolute_sleep import phase4g_c_fix_wait_until
    from experiments.phase4g_a.workload import SOURCE, SOURCE_SHA

    root = Path(__file__).resolve().parents[2]
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
    original_send = runtime.send

    def send(*positional, **kwargs):
        params = dict(kwargs.get("params") or {})
        if args.trace:
            params.update(slackkv_allocator_trace=True,
                          slackkv_allocator_trace_interval_ms=100)
        kwargs["params"] = params
        return original_send(*positional, **kwargs)

    runtime.send = send
    metadata = {
        "name": args.name, "gate": args.gate, "allocator_trace": args.trace,
        "backfill_limit": args.backfill,
        "trace_interval_ms": 100, "gpu_count": 1, "max_seconds": 1800,
        "source_sha": SOURCE_SHA, "source_directory": str(runtime.SRC),
        "kv_bytes": 2147483648, "bg_tokens": 8192,
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
        raise TimeoutError("1800-second experiment budget")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(1800)
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
