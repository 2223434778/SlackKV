"""Run the small Phase 2C selective-offload A/B experiment."""
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from .helpers import counter_delta, make_completion_payload, metric_value, parse_prometheus_samples, ttft_ms

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "phase2c_fix"
MODEL = "/home/a6000/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
ENV_ROOT = "/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix"
SERVER = [
    f"{ENV_ROOT}/bin/vllm", "serve", MODEL, "--dtype", "float16", "--tensor-parallel-size", "1",
    "--enable-prefix-caching", "--kv-cache-memory-bytes", "335544320", "--kv-transfer-config",
    '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"block_size":64,"cpu_bytes_to_use":2147483648}}',
    "--max-model-len", "10000", "--host", "127.0.0.1", "--port", "8000", "--enforce-eager",
]
BASE_ENV = {
    "PATH": f"{ENV_ROOT}/bin:" + os.environ.get("PATH", ""), "CUDA_VISIBLE_DEVICES": "0",
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "LD_LIBRARY_PATH": f"{ENV_ROOT}/targets/x86_64-linux/lib:{ENV_ROOT}/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:{ENV_ROOT}/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:" + os.environ.get("LD_LIBRARY_PATH", ""),
}


def metrics_text() -> str:
    with urllib.request.urlopen("http://127.0.0.1:8000/metrics", timeout=15) as response:
        return response.read().decode()


def snapshot(stage: str) -> dict[str, Any]:
    raw = metrics_text()
    selected = []
    for line in raw.splitlines():
        if any(x in line for x in ("prefix", "kv_", "prompt_tokens", "cache", "preemption")):
            selected.append(line)
    return {"stage": stage, "captured_at": time.time(), "raw_relevant": "\n".join(selected)}


def wait_ready(proc: subprocess.Popen[str], timeout: float = 300.0) -> None:
    end = time.time() + timeout
    while time.time() < end:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode}")
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=3) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(2)
    raise TimeoutError("server health timeout")


def start_server(log_path: Path) -> subprocess.Popen[str]:
    env = os.environ.copy(); env.update(BASE_ENV)
    log = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(SERVER, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)


def stop_server(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait(timeout=10)


def request_tokens(prompt: list[int], *, keep: bool, stream: bool) -> tuple[float, float | None, str]:
    payload = make_completion_payload(MODEL, prompt, stream=stream, keep=keep)
    sent = time.time()
    request = urllib.request.Request("http://127.0.0.1:8000/v1/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    first: float | None = None; first_text = ""
    with urllib.request.urlopen(request, timeout=240) as response:
        if not stream:
            body = json.loads(response.read()); return (time.time() - sent, None, body["choices"][0].get("text", ""))
        for line in response:
            if not line.startswith(b"data: "): continue
            item = line[6:].strip()
            if item == b"[DONE]": continue
            chunk = json.loads(item); choices = chunk.get("choices") or []
            if choices and first is None:
                first = time.time(); first_text = choices[0].get("text", "")
    if first is None: raise RuntimeError("no streamed token")
    return (time.time() - sent, ttft_ms(sent, first), first_text)


def build_prompts(tokenizer: Any) -> dict[str, list[int]]:
    enc = lambda text: tokenizer.encode(text, add_special_tokens=False)
    a = enc("Session A ordinary coding context marker A. " * 900)[:4096]
    b = enc("Session B ordinary coding context marker B. " * 900)[:4096]
    pressure = enc("PRESSURE UNIQUE CONTEXT unrelated file-search output item. " * 3000)[:9500]
    pressure2 = enc("SECOND PRESSURE UNIQUE CONTEXT unrelated compiler output item. " * 3000)[:9500]
    suffix = enc("\nTool result: operation completed successfully.\nAssistant:")
    if (len(a), len(b), len(pressure), len(pressure2)) != (4096, 4096, 9500, 9500) or a == b or pressure == pressure2:
        raise AssertionError("invalid prompt construction")
    return {"a": a, "b": b, "pressure": pressure, "pressure2": pressure2, "a_resume": a + suffix, "b_resume": b + suffix}


def first_summary(before: Any, after: Any, e2e: float, text: str) -> dict[str, Any]:
    return {"e2e_ms": e2e * 1000, "offload_bytes": counter_delta(before, after, "vllm:kv_offload_total_bytes_total"), "first_text": text}


def one_repetition(rep: int, prompts: dict[str, list[int]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    proc = start_server(RESULTS / "server_logs" / f"rep{rep}.log")
    try:
        wait_ready(proc); snaps = [snapshot(f"rep{rep}_server_ready")]
        first: dict[str, Any] = {}
        for session, keep in (("A", False), ("B", True)):
            before = parse_prometheus_samples(metrics_text())
            e2e, _, text = request_tokens(prompts[session.lower()], keep=keep, stream=False)
            after = parse_prometheus_samples(metrics_text())
            first[session] = first_summary(before, after, e2e, text)
            snaps.append(snapshot(f"rep{rep}_{session}_first"))
        time.sleep(1); snaps.append(snapshot(f"rep{rep}_after_store_wait"))
        request_tokens(prompts["pressure"], keep=True, stream=False)
        request_tokens(prompts["pressure2"], keep=True, stream=False)
        snaps.append(snapshot(f"rep{rep}_after_pressure"))
        order = ["A", "B"] if rep % 2 == 0 else ["B", "A"]
        rows = []
        for position, session in enumerate(order):
            before = parse_prometheus_samples(metrics_text())
            e2e, ttft, text = request_tokens(prompts[f"{session.lower()}_resume"], keep=True, stream=True)
            after = parse_prometheus_samples(metrics_text()); snaps.append(snapshot(f"rep{rep}_{session}_resume"))
            rows.append({
                "repetition": rep, "resume_order": position, "session": session,
                "admission": "OFFLOAD" if session == "A" else "KEEP", "resume_ttft_ms": ttft,
                "resume_e2e_ms": e2e * 1000, "local_prefix_hit_tokens": counter_delta(before, after, "vllm:prompt_tokens_by_source_total", {"source": "local_cache_hit"}),
                "external_prefix_hit_tokens": counter_delta(before, after, "vllm:prompt_tokens_by_source_total", {"source": "external_kv_transfer"}),
                "recomputed_prompt_tokens": counter_delta(before, after, "vllm:prompt_tokens_by_source_total", {"source": "local_compute"}),
                "external_cache_hit_tokens": counter_delta(before, after, "vllm:external_prefix_cache_hits_total"),
                "gpu_kv_usage_before_resume": metric_value(before, "vllm:kv_cache_usage_perc"), "gpu_kv_usage_after_resume": metric_value(after, "vllm:kv_cache_usage_perc"),
                "generation_validity": "PASS" if text is not None else "FAIL",
            })
        return rows, snaps, first
    finally:
        stop_server(proc)


def control(prompts: dict[str, list[int]]) -> dict[str, Any]:
    proc = start_server(RESULTS / "server_logs" / "control.log")
    try:
        wait_ready(proc); request_tokens(prompts["a"], keep=False, stream=False); time.sleep(1)
        before = parse_prometheus_samples(metrics_text()); _, ttft, text = request_tokens(prompts["a_resume"], keep=False, stream=True); after = parse_prometheus_samples(metrics_text())
        return {"ttft_ms": ttft, "local_hit_tokens": counter_delta(before, after, "vllm:prompt_tokens_by_source_total", {"source": "local_cache_hit"}), "first_text": text}
    finally:
        stop_server(proc)


def capability_gate() -> dict[str, Any]:
    import inspect
    import torch
    import vllm
    from vllm import SamplingParams
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import OffloadingConnector, OffloadingConnectorScheduler
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if "max_offload_tokens" not in inspect.getsource(OffloadingConnectorScheduler):
        raise RuntimeError("installed OffloadingConnector lacks max_offload_tokens")
    params = SamplingParams(max_tokens=1, extra_args={"kv_transfer_params": {"max_offload_tokens": 0}})
    return {"vllm_version": vllm.__version__, "torch_version": torch.__version__, "torch_cuda": torch.version.cuda, "cuda_available": True, "gpu0": torch.cuda.get_device_name(0), "offloading_connector": OffloadingConnector.__name__, "max_offload_tokens_value": params.extra_args["kv_transfer_params"]["max_offload_tokens"]}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--seeds", default="0,1,2"); args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True); (RESULTS / "server_logs").mkdir(exist_ok=True)
    gate = capability_gate()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True); prompts = build_prompts(tokenizer)
    control_result = control(prompts); rows: list[dict[str, Any]] = []; snaps: list[dict[str, Any]] = []; first: list[dict[str, Any]] = []
    for rep in (int(x) for x in args.seeds.split(",")):
        r, s, f = one_repetition(rep, prompts); rows.extend(r); snaps.extend(s); first.append({"repetition": rep, **f})
    with (RESULTS / "resume_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (RESULTS / "metric_snapshots.json").write_text(json.dumps(snaps, indent=2), encoding="utf-8")
    (RESULTS / "cache_metrics.csv").write_text("see metric_snapshots.json for relevant raw Prometheus samples\n", encoding="utf-8")
    (RESULTS / "config.json").write_text(json.dumps({"model": MODEL, "gpu": "GPU0", "gpu_kv_capacity_bytes": 335544320, "cpu_offload_bytes": 2147483648, "prefix_tokens": 4096, "pressure_tokens_each": 9500, "pressure_requests": 2, "repetitions": [0, 1, 2], "control": control_result}, indent=2), encoding="utf-8")
    gate.update({"installation": "isolated vLLM source build, CUDA 12.6; FlashInfer 0.6.12 no-deps plus runtime dependencies", "first_turn_summaries": first})
    (RESULTS / "environment.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(json.dumps({"resume_rows": len(rows), "control": control_result, "result_dir": str(RESULTS)}, indent=2))


if __name__ == "__main__":
    main()
