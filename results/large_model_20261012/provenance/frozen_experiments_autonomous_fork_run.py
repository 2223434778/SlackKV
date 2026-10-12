"""Owned-server, fixed-input fork/join screen with a continuous Round 3 ledger."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import threading
import time
import traceback
import urllib.error
import urllib.request


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def fixed_tokens(length: int, seed: int) -> list[int]:
    # Fixed ordinary IDs in the Qwen2.5 vocabulary, disjoint from special IDs.
    import random
    generator = random.Random(seed)
    return [generator.randrange(1000, 150000) for _ in range(length)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--kv-gib", choices=(4, 8), type=int, default=4)
    parser.add_argument("--gate", choices=(2, 8), type=int, default=8)
    parser.add_argument("--batch-invariant", action="store_true")
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--background-phase", choices=("cold", "decode"), default="cold")
    parser.add_argument("--source", type=Path, default=Path("/media/a6000/D/ZhaoBinHe/slackkv-autonomous-round3-source"))
    args = parser.parse_args()
    if args.oracle and not args.batch_invariant:
        parser.error("oracle requires separate batch-invariant correctness mode")
    if Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("plain unique run name required")
    root = Path(__file__).resolve().parents[2]
    ledger = root / "results/autonomous_fork"
    previous = [json.loads(p.read_text()) for p in ledger.glob("*/execution.json")]
    if any(item["status"] == "RUNNING" for item in previous):
        raise RuntimeError("owned experiment still RUNNING")
    spent = sum(item["gpu_hours_conservative"] for item in previous)
    if spent + 0.5 > 4:
        raise RuntimeError("insufficient remaining Round 3 GPU budget")
    with socket.socket() as probe:
        probe.settimeout(2)
        if probe.connect_ex(("127.0.0.1", 8000)) == 0:
            raise RuntimeError("port 8000 already occupied; refusing to disturb service")
    out = ledger / args.name
    out.mkdir(parents=True, exist_ok=False)
    metadata = dict(vars(args), source=str(args.source), coalesce=False, gpu_count=1, status="RUNNING",
                    round_gpu_hours_before_run=spent, started_unix=time.time(),
                    baseline_commit="7519bda64de6f2c48051b17f60f8e21c83be564e")
    metadata["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    metadata["scheduler_sha256"] = hashlib.sha256((args.source / "vllm/v1/core/sched/scheduler.py").read_bytes()).hexdigest()
    write_json(out / "execution.json", metadata)
    started = time.monotonic()
    owned = []

    def timeout(_signum, _frame):
        for proc in owned:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
        raise TimeoutError("1710-second watchdog; cleanup reserved")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(1710)
    try:
        from experiments.phase4d_a import runtime
        runtime.SRC = args.source
        os.environ.update(VLLM_BATCH_INVARIANT=str(int(args.batch_invariant)),
                          SLACKKV_CONTINUATION_CREDIT="0", SLACKKV_WORKING_SET_EPOCHS="0",
                          SLACKKV_CAPACITY_BACKFILL_LIMIT="0", SLACKKV_FORK_TRACE="1")
        original_command = runtime.server_command

        def command(connector, gpu_bytes):
            argv = original_command(connector, gpu_bytes)
            argv.remove("--enforce-eager")
            return argv

        runtime.server_command = command
        metadata["server_command"] = command(True, args.kv_gib * 2**30)
        write_json(out / "execution.json", metadata)
        proc, log = runtime.start_server(args.name, gpu_bytes=args.kv_gib * 2**30, out=out)
        owned.append(proc)
        try:
            _, block_size = runtime.wait_ready(proc)
            if block_size != 16:
                raise RuntimeError("unexpected cache block size")
            from experiments.phase4d_a.experiment import gpu_capacity_blocks
            from experiments.autonomous_capacity.capacity import validate_pool
            metadata["pool"] = validate_pool(args.kv_gib * 2**30, block_size, gpu_capacity_blocks(log, block_size))
            write_json(out / "execution.json", metadata)
            runtime.send(fixed_tokens(256, 919), 16, stream=True, params={"max_offload_tokens": 0})
            records = []
            episodes = []
            lock = threading.Lock()

            def send(prompt, count, session, *, marked=True, on_first=None):
                params = {"slackkv_session": session, "slackkv_trace_resume": marked,
                          "slackkv_first_token_gate": marked,
                          "slackkv_first_token_gate_limit": args.gate,
                          "max_offload_tokens": len(prompt) if marked else 0}
                response = runtime.send(prompt, count, stream=True, params=params, on_first=on_first)
                if len(response["tokens"]) != count:
                    raise RuntimeError(f"short output: {session}")
                row = dict(response, session=session, prompt_sha256=hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
                           ttft_ms=(response["first_ns"] - response["issue_ns"]) / 1e6)
                with lock:
                    records.append(row)
                    write_json(out / "requests.json", records)
                    with (out / "inputs.jsonl").open("a") as stream:
                        stream.write(json.dumps(dict(session=session, prompt=prompt), separators=(",", ":")) + "\n")
                return row

            for episode in range(3):
                ancestor = fixed_tokens(24576, 100 + episode)
                # Priming is not measured; GPU reset leaves CPU connector intact.
                runtime.send(ancestor, 1, stream=True, params={"max_offload_tokens": len(ancestor)})
                time.sleep(0.5)
                reset = urllib.request.Request("http://127.0.0.1:8000/reset_prefix_cache?reset_external=false", data=b"", method="POST")
                with urllib.request.urlopen(reset, timeout=20) as response:
                    reset_result = dict(status=response.status, body=response.read().decode())
                before = runtime.metric_text()
                (out / f"episode-{episode}-before.prom").write_text(before)
                barrier = threading.Barrier(9)
                bg_first = threading.Event()

                def child(index):
                    prefix = ancestor + fixed_tokens(256, 1000 + episode * 10 + index)
                    barrier.wait(timeout=20)
                    first = send(prefix, 128, f"F{episode}:{index}:0")
                    # Exact resource replay: fixed canonical history, not semantic output feedback.
                    time.sleep(0.5)
                    second_prompt = prefix + fixed_tokens(192, 2000 + episode * 10 + index)
                    second = send(second_prompt, 128, f"F{episode}:{index}:1")
                    return dict(index=index, issue_ns=first["issue_ns"], done_ns=second["done_ns"])

                def background():
                    if args.background_phase == "cold":
                        barrier.wait(timeout=20)
                    return send(fixed_tokens(8192, 3000 + episode), 1024, f"BG{episode}", marked=False,
                                on_first=lambda _: bg_first.set())

                with ThreadPoolExecutor(max_workers=9) as pool:
                    bg = pool.submit(background)
                    if args.background_phase == "decode" and not bg_first.wait(timeout=60):
                        raise TimeoutError("background did not enter decode")
                    futures = [pool.submit(child, index) for index in range(8)]
                    if args.background_phase == "decode":
                        barrier.wait(timeout=20)
                    children = [future.result(timeout=300) for future in futures]
                    background_result = bg.result(timeout=300)
                after = runtime.metric_text()
                (out / f"episode-{episode}-after.prom").write_text(after)
                episodes.append(dict(episode=episode, reset=reset_result, children=children,
                                     workflow_ms=(max(c["done_ns"] for c in children) - min(c["issue_ns"] for c in children)) / 1e6,
                                     bg_ms=background_result["e2e_ms"],
                                     source_accounting="Use raw SLACKKV_SOURCE_TRACE and actual migration counters; missing series are UNKNOWN"))
                write_json(out / "requests.json", records)
                write_json(out / "episodes.json", episodes)
            if len(records) != 51:
                raise RuntimeError("incomplete fixed workload")
            metadata["workflow_median_ms"] = statistics.median(e["workflow_ms"] for e in episodes)
            metadata["bg_median_ms"] = statistics.median(e["bg_ms"] for e in episodes)
            metadata["status"] = "COMPLETE"
            (out / "metrics_after.prom").write_text(runtime.metric_text())
            if args.oracle:
                from experiments.autonomous_fork.oracle import verify
                rows = verify(runtime, out, log)
                metadata["oracle_pass"] = len(rows) == 16 and all(
                    r.get("exact_match") and r.get("verified_cold") and
                    r.get("cold_token_count") == 128 for r in rows)
        finally:
            metadata["server_exit_code"] = runtime.stop_server(proc, log, out, args.name)
    except BaseException:
        metadata["status"] = "FAILED"
        (out / "failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        signal.alarm(0)
        metadata["finished_unix"] = time.time()
        metadata["wall_seconds"] = time.monotonic() - started
        metadata["gpu_hours_conservative"] = metadata["wall_seconds"] / 3600
        write_json(out / "execution.json", metadata)


if __name__ == "__main__":
    main()
