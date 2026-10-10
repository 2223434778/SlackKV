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
    parser.add_argument("--gate-off", action="store_true")
    parser.add_argument("--backfill", type=int, choices=(0, 2), default=0)
    parser.add_argument("--execution", choices=("eager", "default"), default="eager")
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--batch-invariant", action="store_true")
    parser.add_argument("--policy", choices=("A", "D"), default="A")
    parser.add_argument("--round-directory", default="autonomous_admission",
                        choices=("autonomous_admission", "autonomous_capacity", "autonomous_fork", "autonomous_round4"))
    parser.add_argument("--kv-gib", type=int, choices=(2, 4, 8), default=2)
    parser.add_argument("--replay-inputs", type=Path)
    parser.add_argument("--continuation-credit", action="store_true")
    parser.add_argument("--runtime-source", type=Path)
    parser.add_argument("--working-set-epochs", action="store_true")
    args = parser.parse_args()
    if args.gate < 1 or Path(args.name).name != args.name:
        parser.error("positive gate and plain run name required")
    if args.gate_off and (args.policy != "A" or args.backfill or
                         args.continuation_credit or args.working_set_epochs):
        parser.error("gate-off requires plain OnDemand")
    if args.continuation_credit and (args.policy != "A" or args.backfill):
        parser.error("continuation credit requires OnDemand without capacity backfill")
    if args.working_set_epochs and (args.policy != "A" or args.backfill or args.continuation_credit):
        parser.error("working-set epochs require plain OnDemand")
    from experiments.phase4d_a import runtime
    from experiments.phase4g_a import run_phase4g_a as legacy
    from experiments.phase4g_a_fix import run_phase4g_a_fix as harness
    from experiments.phase4g_c_fix.absolute_sleep import phase4g_c_fix_wait_until
    from experiments.phase4g_a.workload import SOURCE, SOURCE_SHA

    root = Path(__file__).resolve().parents[2]
    previous = [json.loads(p.read_text()) for p in
                (root / "results" / args.round_directory).glob("*/execution.json")]
    if any(item["status"] == "RUNNING" for item in previous):
        raise RuntimeError("another owned experiment is still marked RUNNING")
    spent = sum(item.get("gpu_hours_conservative", 0) for item in previous)
    if spent + 0.5 > 4:
        raise RuntimeError("remaining round budget cannot cover a 30-minute run")
    out = root / "results" / args.round_directory / args.name
    out.mkdir(parents=True, exist_ok=False)
    runtime.SRC = Path("/media/a6000/D/ZhaoBinHe/slackkv-autonomous-20261009-source")
    if args.runtime_source:
        runtime.SRC = args.runtime_source
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA:
        raise RuntimeError("frozen workload fingerprint mismatch")
    harness.OUT = out
    harness.configure_legacy_lifecycle()
    legacy.K = args.gate
    legacy.wait_until = phase4g_c_fix_wait_until
    os.environ["SLACKKV_CAPACITY_BACKFILL_LIMIT"] = str(args.backfill)
    os.environ["VLLM_BATCH_INVARIANT"] = "1" if args.batch_invariant else "0"
    os.environ["SLACKKV_CONTINUATION_CREDIT"] = "1" if args.continuation_credit else "0"
    os.environ["SLACKKV_WORKING_SET_EPOCHS"] = "1" if args.working_set_epochs else "0"
    if args.round_directory == "autonomous_round4":
        os.environ["SLACKKV_FORK_TRACE"] = "0"
        import socket
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", 8000)) == 0:
                raise RuntimeError("port 8000 occupied; refuse to replace another service")
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
        kwargs["gpu_bytes"] = args.kv_gib * 2**30
        proc, log = original_start(*positional, **kwargs)
        owned_server.append(proc)
        return proc, log

    runtime.start_server = start_server
    if args.round_directory != "autonomous_admission":
        from experiments.autonomous_capacity.capacity import validate_pool

        def validate_runtime(log, block_size):
            actual = gpu_capacity_blocks(log, block_size)
            return validate_pool(args.kv_gib * 2**30, block_size, actual)

        from experiments.phase4d_a.experiment import gpu_capacity_blocks
        legacy.validate_runtime = validate_runtime
    original_stop = runtime.stop_server

    def stop_server(proc, log, run_directory, label):
        try:
            saved = run_directory / "run.json"
            if args.oracle and saved.exists() and json.loads(saved.read_text())["valid"]:
                from experiments.autonomous_admission.oracle import run_oracle
                run_oracle(runtime, out, log)
            elif saved.exists() and json.loads(saved.read_text())["valid"]:
                (out / "metrics_after_timed_workload.prom").write_text(runtime.metric_text())
        finally:
            code = original_stop(proc, log, run_directory, label)
        return code

    runtime.stop_server = stop_server
    original_send = runtime.send
    output_lock = threading.Lock()
    replay = {}
    if args.replay_inputs:
        import gzip
        raw = args.replay_inputs.read_bytes()
        if args.replay_inputs.suffix == ".gz":
            raw = gzip.decompress(raw)
        replay = {row["session"]: row for row in
                  (json.loads(line) for line in raw.decode().splitlines())}
        if set(replay) != {f"E{i:02}:turn:{j}" for i in range(8) for j in range(5)}:
            raise ValueError("replay requires exactly all forty agent inputs")
        episode_ids = {row["session_hash"]: row["episode_id"] for row in legacy.load_source()}

        def next_prompt(previous, target, shared, **kwargs):
            session = f'{episode_ids[kwargs["session_hash"]]}:turn:{kwargs["turn_index"]}'
            prompt = replay[session]["prompt"]
            if len(prompt) != target:
                raise ValueError("replay input length mismatch")
            return prompt

        legacy.make_next_prompt = next_prompt

    def send(*positional, **kwargs):
        params = dict(kwargs.get("params") or {})
        if args.gate_off and params.get("slackkv_trace_resume") is True:
            params["slackkv_first_token_gate"] = False
        if args.trace:
            params.update(slackkv_allocator_trace=True,
                          slackkv_allocator_trace_interval_ms=100)
        kwargs["params"] = params
        session = params.get("slackkv_session")
        if session and session.startswith("E"):
            params["slackkv_epoch_program"] = session.rsplit(":turn:", 1)[0]
            params["slackkv_epoch_continues"] = "slackkv_trace_next_boundary" in params
        if session in replay:
            if positional[0] != replay[session]["prompt"]:
                raise ValueError("constructed and transmitted replay prompts differ")
        response = original_send(*positional, **kwargs)
        record = {"session": session, "tokens": response["tokens"]}
        if args.round_directory == "autonomous_round4":
            record["effective_params"] = params
        if session in {"E02:turn:2", "E04:turn:1"}:
            record["prompt"] = positional[0]
        with output_lock:
            with (out / "output_tokens.jsonl").open("a") as stream:
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            if session and session.startswith("E"):
                with (out / "inputs.jsonl").open("a") as stream:
                    stream.write(json.dumps({"session": session, "prompt": positional[0]},
                                            separators=(",", ":")) + "\n")
        return response

    runtime.send = send
    metadata = {
        "name": args.name, "gate": args.gate, "gate_enabled": not args.gate_off,
        "allocator_trace": args.trace,
        "backfill_limit": args.backfill,
        "continuation_credit": args.continuation_credit,
        "working_set_epochs": args.working_set_epochs,
        "execution_mode": args.execution,
        "cold_oracle": args.oracle,
        "batch_invariant": args.batch_invariant,
        "policy": args.policy,
        "trace_interval_ms": 100, "gpu_count": 1, "max_seconds": 1800,
        "source_sha": SOURCE_SHA, "source_directory": str(runtime.SRC),
        "kv_bytes": args.kv_gib * 2**30, "bg_tokens": 8192,
        "server_command": runtime.server_command(True, args.kv_gib * 2**30),
        "round_directory": args.round_directory,
        "run_kind": "historical_regression" if args.round_directory == "autonomous_fork" else "historical_workload",
        "replay_inputs": str(args.replay_inputs) if args.replay_inputs else None,
        "replay_sha256": hashlib.sha256(args.replay_inputs.read_bytes()).hexdigest() if args.replay_inputs else None,
        "round_gpu_hours_before_run": spent,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "started_unix": time.time(), "status": "RUNNING",
        "purpose": "diagnosis" if args.trace else "pre-registered matched screen",
    }
    metadata["allocator_sha256"] = hashlib.sha256(
        (runtime.SRC / "vllm/v1/core/kv_cache_manager.py").read_bytes()).hexdigest()
    metadata["scheduler_sha256"] = hashlib.sha256(
        (runtime.SRC / "vllm/v1/core/sched/scheduler.py").read_bytes()).hexdigest()
    if args.round_directory == "autonomous_round4":
        import subprocess
        metadata["git_head"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        metadata["git_diff_sha256"] = hashlib.sha256(subprocess.check_output(
            ["git", "diff", "--binary"], cwd=root)).hexdigest()
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
        result = legacy.run_lifecycle(args.policy, "measured", 1)
        if args.gate_off:
            result["policy"] = "OnDemand-custom-gate-off"
        harness._decorate_agent_run(result, f"measured_rep1_{args.policy.lower()}")
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
