"""Owned, budgeted invariant smoke before physical retention performance screens."""
from __future__ import annotations

import hashlib
import json
import os
import signal
import socket
import time
import traceback
import urllib.request
from pathlib import Path


def main():
    from experiments.phase4d_a import runtime
    from experiments.autonomous_admission.oracle import cold_sources
    root = Path(__file__).resolve().parents[2]
    base = root / "results/autonomous_round4"
    previous = [json.loads(p.read_text()) for p in base.glob("*/execution.json")]
    if any(r["status"] == "RUNNING" for r in previous):
        raise RuntimeError("another owned experiment is running")
    spent = sum(r.get("gpu_hours_conservative", 0) for r in previous)
    if spent + .5 > 4:
        raise RuntimeError("insufficient remaining round GPU budget")
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", 8000)) == 0:
            raise RuntimeError("port occupied")
    out = base / "retention-smoke"
    out.mkdir(exist_ok=False)
    runtime.SRC = Path("/media/a6000/D/ZhaoBinHe/slackkv-round4-control-source")
    os.environ.update(SLACKKV_RETENTION_CONTROL="1", SLACKKV_RETENTION_PIN="1",
                      SLACKKV_FORK_TRACE="0", SLACKKV_RESOURCE_TRACE="0",
                      SLACKKV_CAPACITY_BACKFILL_LIMIT="0", SLACKKV_CONTINUATION_CREDIT="0",
                      SLACKKV_WORKING_SET_EPOCHS="0", VLLM_BATCH_INVARIANT="1")
    original = runtime.server_command
    def command(*args):
        cmd = original(*args)
        cmd.remove("--enforce-eager")
        return cmd
    runtime.server_command = command
    metadata = {"status": "RUNNING", "name": out.name, "gpu_count": 1,
                "round_gpu_hours_before_run": spent, "started_unix": time.time(),
                "purpose": "correctness smoke, not performance mode",
                "batch_invariant": True, "source_directory": str(runtime.SRC),
                "scheduler_sha256": hashlib.sha256((runtime.SRC / "vllm/v1/core/sched/scheduler.py").read_bytes()).hexdigest(),
                "command": runtime.server_command(True, 2 * 2**30)}
    path = out / "execution.json"
    path.write_text(json.dumps(metadata, indent=2) + "\n")
    start = time.monotonic()
    proc, log = None, None
    records = []
    def timeout(*_):
        raise TimeoutError("smoke watchdog")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(1710)
    try:
        proc, log = runtime.start_server("retention_smoke", out=out)
        runtime.wait_ready(proc)
        def params(program, continues, session):
            return {"slackkv_session": session, "slackkv_trace_resume": True,
                    "slackkv_first_token_gate": False, "slackkv_epoch_program": program,
                    "slackkv_epoch_continues": continues}
        prompt = runtime.tokenizer_ids("round4-smoke", "handoff", 0, 2048)
        warm = runtime.send(prompt, 16, stream=True, params=params("handoff", True, "smoke-warm"))
        time.sleep(.05)  # allow native asynchronous store/free to publish the idle pin
        extension = runtime.tokenizer_ids("round4-smoke", "extension", 0, 16)
        resumed_prompt = prompt + warm["tokens"] + extension
        resumed = runtime.send(resumed_prompt, 16, stream=True,
                               params=params("handoff", False, "smoke-resumed"))
        records.extend([{"session": "smoke-warm", "prompt": prompt, **warm},
                        {"session": "smoke-resumed", "prompt": resumed_prompt, **resumed}])
        payload = {"model": str(runtime.MODEL), "prompt": resumed_prompt, "max_tokens": 16,
                   "temperature": 0, "ignore_eos": True, "return_token_ids": True,
                   "cache_salt": "round4-smoke-cold-unique",
                   "kv_transfer_params": {"slackkv_session": "smoke-cold",
                                           "slackkv_trace_resume": True, "max_offload_tokens": 0}}
        req = urllib.request.Request(runtime.API, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as response:
            cold = json.load(response)
        records.append({"session": "smoke-cold", "response": cold})
        (out / "smoke_requests.json").write_text(json.dumps(records, indent=2) + "\n")
        text = log.read_text(errors="replace")
        cold_verified, sources = cold_sources(text, "smoke-cold")
        hot_sources = [json.loads(line.split("SLACKKV_SOURCE_TRACE ", 1)[1]) for line in text.splitlines()
                       if "SLACKKV_SOURCE_TRACE " in line]
        hot_sources = [r for r in hot_sources if r["session"] == "smoke-resumed"]
        assert cold_verified and resumed["tokens"] == cold["choices"][0]["token_ids"]
        assert hot_sources and max(r["local_cached_tokens"] for r in hot_sources) >= 2048
        expiry_prompt = runtime.tokenizer_ids("round4-smoke", "expiry", 0, 2048)
        expiry = runtime.send(expiry_prompt, 16, stream=True,
                              params=params("expiry", True, "smoke-expiry"))
        records.append({"session": "smoke-expiry", "prompt": expiry_prompt, **expiry})
        (out / "smoke_requests.json").write_text(json.dumps(records, indent=2) + "\n")
        time.sleep(1.25)
        text = log.read_text(errors="replace")
        expiry_seen = any('"program":"expiry"' in line and '"reason":"TTL_EXPIRED"' in line
                          for line in text.splitlines())
        assert expiry_seen, "idle expiry must happen before another HTTP request"
        cancel_payload = {"model": str(runtime.MODEL), "prompt": expiry_prompt, "max_tokens": 4096,
                          "temperature": 0, "ignore_eos": True, "return_token_ids": True, "stream": True,
                          "kv_transfer_params": params("cancel", True, "smoke-cancel")}
        req = urllib.request.Request(runtime.API, data=json.dumps(cancel_payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as response:
            for line in response:
                if line.startswith(b"data: ") and line[6:].strip() != b"[DONE]":
                    if (json.loads(line[6:]).get("choices") or [{}])[0].get("token_ids"):
                        break
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            text = log.read_text(errors="replace")
            if any('"event":"CANCEL"' in line and '"program":"cancel"' in line for line in text.splitlines()):
                break
            time.sleep(.1)
        else:
            raise AssertionError("disconnect did not reach engine cancellation")
        events = [json.loads(line.split("SLACKKV_ROUND4_RETENTION ", 1)[1]) for line in text.splitlines()
                  if "SLACKKV_ROUND4_RETENTION " in line]
        assert events and events[-1]["retained_programs"] == 0
        (out / "metrics.prom").write_text(runtime.metric_text())
        (out / "smoke.json").write_text(json.dumps({"exact_cold_match": True,
            "verified_cold": cold_verified, "cold_sources": sources, "hot_sources": hot_sources,
            "idle_expiry_before_new_request": expiry_seen, "cancel_reached_engine": True,
            "final_retained_programs": 0, "events": events, "requests": records,
            "not_covered": ["remote DMA leader/follower cancellation", "all preemption interleavings",
                            "all forty workload outputs", "semantic agent task quality"]}, indent=2) + "\n")
        metadata["status"] = "VALID"
    except BaseException:
        metadata["status"] = "FAILED"
        (out / "failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        try:
            if proc is not None:
                runtime.stop_server(proc, log, out, "retention_smoke")
        finally:
            signal.alarm(0)
            metadata["finished_unix"] = time.time()
            metadata["wall_seconds"] = time.monotonic() - start
            metadata["gpu_hours_conservative"] = metadata["wall_seconds"] / 3600
            path.write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
