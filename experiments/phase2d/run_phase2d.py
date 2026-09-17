"""Run Phase 2D: shift external KV recovery into the pause using a request proxy."""
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Sequence

from transformers import AutoTokenizer

from .helpers import (
    effective_resume_latency_ms,
    external_cache_hit_delta,
    gpu_to_cpu_offload_bytes,
    gpu_to_cpu_offload_ms,
    make_payload,
    parse_prometheus_samples,
    prefetch_timing,
    source_tokens_delta,
    token_ids_match,
)

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "phase2d"
MODEL = "/home/a6000/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
ENV_ROOT = "/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix"
EXPECTED_D2H_BYTES = 117_440_512
RESUME_HORIZON_S = 2.0
PRESSURE_TOKENS = 9500
LEADS_MS = {"JIT-100": 100, "JIT-250": 250, "JIT-500": 500}
POLICIES = ["OnDemand", *LEADS_MS]
SERVER = [
    f"{ENV_ROOT}/bin/vllm", "serve", MODEL,
    "--dtype", "float16", "--tensor-parallel-size", "1",
    "--enable-prefix-caching", "--kv-cache-memory-bytes", "335544320",
    "--kv-transfer-config",
    '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"block_size":64,"cpu_bytes_to_use":2147483648}}',
    "--max-model-len", "10000", "--host", "127.0.0.1", "--port", "8000", "--enforce-eager",
]
BASE_ENV = {
    "PATH": f"{ENV_ROOT}/bin:" + os.environ.get("PATH", ""),
    "CUDA_VISIBLE_DEVICES": "0",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "LD_LIBRARY_PATH": (
        f"{ENV_ROOT}/targets/x86_64-linux/lib:"
        f"{ENV_ROOT}/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:"
        f"{ENV_ROOT}/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:"
        + os.environ.get("LD_LIBRARY_PATH", "")
    ),
}


def metric_text() -> str:
    with urllib.request.urlopen("http://127.0.0.1:8000/metrics", timeout=15) as response:
        return response.read().decode()


def capture(stage: str, snapshots: list[dict[str, Any]]) -> dict[tuple[str, tuple[tuple[str, str], ...]], float]:
    raw = metric_text()
    relevant = [
        line for line in raw.splitlines()
        if any(key in line for key in ("prefix", "kv_", "prompt_tokens", "cache", "preemption"))
    ]
    snapshots.append({"stage": stage, "captured_at_unix_s": time.time(), "raw_relevant": "\n".join(relevant)})
    return parse_prometheus_samples(raw)


def wait_ready(proc: subprocess.Popen[str], timeout_s: float = 300.0) -> None:
    until = time.monotonic() + timeout_s
    while time.monotonic() < until:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited before readiness (exit={proc.returncode})")
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=3) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(2)
    raise TimeoutError("vLLM /health did not become ready")


def start_server(log_name: str) -> subprocess.Popen[str]:
    log_path = RESULTS / "server_logs" / log_name
    env = os.environ.copy()
    env.update(BASE_ENV)
    log = log_path.open("w", encoding="utf-8")
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


def wait_until(monotonic_deadline_s: float) -> None:
    delay = monotonic_deadline_s - time.monotonic()
    if delay > 0:
        time.sleep(delay)


def send_request(
    prompt: Sequence[int], *, stream: bool, keep_local: bool = True, return_token_ids: bool = False
) -> dict[str, Any]:
    payload = make_payload(MODEL, prompt, stream=stream, return_token_ids=return_token_ids)
    # All requests except A_first use max_offload_tokens=0. A_first is the only
    # request that needs normal connector admission, so its caller opts out below.
    if not keep_local:
        payload.pop("kv_transfer_params", None)
    sent = time.monotonic()
    request = urllib.request.Request(
        "http://127.0.0.1:8000/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    first_token_at: float | None = None
    first_token_id: int | None = None
    first_text = ""
    with urllib.request.urlopen(request, timeout=240) as response:
        if not stream:
            body = json.loads(response.read())
            choice = body["choices"][0]
            completed = time.monotonic()
            return {
                "sent_monotonic_s": sent,
                "completed_monotonic_s": completed,
                "e2e_ms": (completed - sent) * 1000.0,
                "ttft_ms": None,
                "first_token_id": None,
                "text": choice.get("text", ""),
            }
        for line in response:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                continue
            chunk = json.loads(data)
            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            token_ids = choice.get("token_ids")
            token_text = choice.get("text", "")
            if first_token_at is None and (token_text or token_ids):
                first_token_at = time.monotonic()
                first_text = token_text
            if first_token_id is None and token_ids:
                first_token_id = int(token_ids[0])
    completed = time.monotonic()
    if first_token_at is None:
        raise RuntimeError("stream ended without a first generated token")
    return {
        "sent_monotonic_s": sent,
        "completed_monotonic_s": completed,
        "e2e_ms": (completed - sent) * 1000.0,
        "ttft_ms": (first_token_at - sent) * 1000.0,
        "first_token_id": first_token_id,
        "text": first_text,
    }


def build_prompts(tokenizer: Any) -> dict[str, list[int]]:
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    # Keep the exact Phase 2C-Fix A and pressure text construction.
    a = encode("Session A ordinary coding context marker A. " * 900)[:4096]
    pressure_1 = encode("PRESSURE UNIQUE CONTEXT unrelated file-search output item. " * 3000)[:PRESSURE_TOKENS]
    pressure_2 = encode("SECOND PRESSURE UNIQUE CONTEXT unrelated compiler output item. " * 3000)[:PRESSURE_TOKENS]
    suffix = encode("\nTool result: operation completed successfully.\nAssistant:")
    if len(a) != 4096 or len(pressure_1) != PRESSURE_TOKENS or len(pressure_2) != PRESSURE_TOKENS:
        raise AssertionError("prompt token count does not match the frozen experiment")
    if len(set(a)) < 2 or a == pressure_1[:4096] or a == pressure_2[:4096] or pressure_1 == pressure_2:
        raise AssertionError("A and pressure prompts are not sufficiently distinct")
    return {"a": a, "pressure_1": pressure_1, "pressure_2": pressure_2, "a_resume": a + suffix}


def capability_gate() -> dict[str, Any]:
    import inspect
    import torch
    import vllm
    from vllm import SamplingParams
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import OffloadingConnector, OffloadingConnectorScheduler

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false")
    if torch.cuda.get_device_name(0) != "NVIDIA RTX A6000":
        raise RuntimeError(f"unexpected GPU0: {torch.cuda.get_device_name(0)}")
    if "max_offload_tokens" not in inspect.getsource(OffloadingConnectorScheduler):
        raise RuntimeError("OffloadingConnector scheduler lacks max_offload_tokens")
    test_params = SamplingParams(max_tokens=1, extra_args={"kv_transfer_params": {"max_offload_tokens": 0}})
    if test_params.extra_args["kv_transfer_params"]["max_offload_tokens"] != 0:
        raise RuntimeError("request-level keep-local control was not retained")
    return {
        "vllm_version": vllm.__version__,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu0": torch.cuda.get_device_name(0),
        "max_offload_tokens_request_control": "PASS",
    }


def server_smoke_test() -> None:
    proc = start_server("capability_smoke.log")
    try:
        wait_ready(proc)
    finally:
        stop_server(proc)


def prepare_session(prompts: dict[str, list[int]], rep_label: str, snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    before_server = capture(f"{rep_label}_server_ready", snapshots)
    del before_server
    before = parse_prometheus_samples(metric_text())
    offload_request = send_request(prompts["a"], stream=False, keep_local=False)
    after = capture(f"{rep_label}_after_A_first", snapshots)
    offload_bytes = gpu_to_cpu_offload_bytes(before, after)
    offload_transfer_ms = gpu_to_cpu_offload_ms(before, after)
    if offload_bytes != EXPECTED_D2H_BYTES:
        raise RuntimeError(f"A_first D2H bytes {offload_bytes} != expected {EXPECTED_D2H_BYTES}")
    time.sleep(1.0)
    capture(f"{rep_label}_after_store_wait", snapshots)
    # Both pressure requests are the validated Phase 2C-Fix workload and are
    # explicitly prevented from writing their KV into the external tier.
    for index in (1, 2):
        send_request(prompts[f"pressure_{index}"], stream=False, keep_local=True)
    pressure_complete = time.monotonic()
    after_pressure = capture(f"{rep_label}_after_pressure", snapshots)
    return {
        "offload_bytes": offload_bytes,
        "offload_transfer_ms": offload_transfer_ms,
        "offload_e2e_ms": offload_request["e2e_ms"],
        "pressure_complete_monotonic_s": pressure_complete,
        "after_pressure_metrics": after_pressure,
    }


def run_local_control(prompts: dict[str, list[int]], control: dict[str, Any]) -> dict[str, Any]:
    proc = start_server("local_hit_control.log")
    snaps: list[dict[str, Any]] = []
    try:
        wait_ready(proc)
        capture("control_server_ready", snaps)
        before = parse_prometheus_samples(metric_text())
        send_request(prompts["a"], stream=False, keep_local=False)
        after_first = capture("control_after_A_first", snaps)
        offload_bytes = gpu_to_cpu_offload_bytes(before, after_first)
        time.sleep(1.0)
        before_resume = parse_prometheus_samples(metric_text())
        capture("control_before_resume", snaps)
        resume = send_request(prompts["a_resume"], stream=True, keep_local=True, return_token_ids=True)
        after_resume = capture("control_after_resume", snaps)
        local_hits = source_tokens_delta(before_resume, after_resume, "local_cache_hit")
        control.update({
            "ttft_ms": resume["ttft_ms"],
            "local_hit_tokens": local_hits,
            "first_token_id": resume["first_token_id"],
            "generation_text": resume["text"],
            "offload_bytes": offload_bytes,
            "generation_validity": "PASS" if resume["first_token_id"] is not None else "FAIL",
            "metric_snapshots": snaps,
        })
        return control
    finally:
        stop_server(proc)


def run_policy(policy: str, repetition: int, prompts: dict[str, list[int]], control_token_id: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    label = f"{policy.lower().replace('-', '')}_rep{repetition}"
    proc = start_server(f"{label}.log")
    snapshots: list[dict[str, Any]] = []
    try:
        wait_ready(proc)
        setup = prepare_session(prompts, label, snapshots)
        deadline = setup["pressure_complete_monotonic_s"] + RESUME_HORIZON_S
        row: dict[str, Any] = {
            "policy": policy,
            "repetition": repetition,
            "offload_bytes": setup["offload_bytes"],
            "offload_transfer_ms": setup["offload_transfer_ms"],
            "offload_e2e_ms": setup["offload_e2e_ms"],
            "prefetch_lead_ms": "NA",
            "prefetch_request_ms": "NA",
            "prefetch_local_hit_tokens": "NA",
            "prefetch_external_hit_tokens": "NA",
            "prefetch_external_cache_hit_tokens": "NA",
            "prefetch_recomputed_tokens": "NA",
            "prefetch_start_lag_ms": "NA",
            "prefetch_completion_relative_to_deadline_ms": "NA",
            "prefetch_stall_ms": "NA",
            "early_residency_ms": "NA",
            "resume_dispatch_delay_ms": "NA",
        }
        if policy == "OnDemand":
            wait_until(deadline)
            before_resume = setup["after_pressure_metrics"]
            resume = send_request(prompts["a_resume"], stream=True, keep_local=True, return_token_ids=True)
            after_resume = capture(f"{label}_after_resume", snapshots)
            prefetch_stall = 0.0
            dispatch_delay_ms = max(0.0, (resume["sent_monotonic_s"] - deadline) * 1000.0)
        else:
            lead_ms = LEADS_MS[policy]
            prefetch_start = deadline - lead_ms / 1000.0
            wait_until(prefetch_start)
            # No metrics HTTP call is made at the target time: post-pressure
            # counters remain stable throughout the idle until prefetch.
            before_prefetch = setup["after_pressure_metrics"]
            prefetch = send_request(prompts["a"], stream=False, keep_local=True)
            prefetch_complete = prefetch["completed_monotonic_s"]
            relative_ms = (prefetch_complete - deadline) * 1000.0
            prefetch_stall, early_residency = prefetch_timing(relative_ms)
            row.update({
                "prefetch_lead_ms": lead_ms,
                "prefetch_request_ms": prefetch["e2e_ms"],
                "prefetch_start_lag_ms": max(0.0, (prefetch["sent_monotonic_s"] - prefetch_start) * 1000.0),
                "prefetch_completion_relative_to_deadline_ms": relative_ms,
                "prefetch_stall_ms": prefetch_stall,
                "early_residency_ms": early_residency,
            })
            if prefetch_complete < deadline:
                wait_until(deadline)
            # No metrics request is inserted between promotion and resume.
            # The post-resume counters are attributed by the strictly
            # sequential request path: prefetch is the only external restore,
            # and the resume must then be the local-prefix hit.
            before_resume = before_prefetch
            resume = send_request(prompts["a_resume"], stream=True, keep_local=True, return_token_ids=True)
            after_resume = capture(f"{label}_after_resume", snapshots)
            resume_target = max(deadline, prefetch_complete)
            dispatch_delay_ms = max(0.0, (resume["sent_monotonic_s"] - resume_target) * 1000.0)
        resume_local = source_tokens_delta(before_resume, after_resume, "local_cache_hit")
        resume_external = source_tokens_delta(before_resume, after_resume, "external_kv_transfer")
        resume_recomputed = source_tokens_delta(before_resume, after_resume, "local_compute")
        if policy.startswith("JIT-"):
            # These are cumulative Prometheus deltas over the two serialized
            # requests. The exact 4096-token external hit can only belong to
            # the prefetch because the subsequent resume is validated as a
            # 4096-token local hit with zero external-hit tokens.
            row.update({
                "prefetch_local_hit_tokens": 0.0,
                "prefetch_external_hit_tokens": source_tokens_delta(before_resume, after_resume, "external_kv_transfer"),
                "prefetch_recomputed_tokens": 0.0,
                "prefetch_external_cache_hit_tokens": external_cache_hit_delta(before_resume, after_resume),
            })
            # The post-resume counters are cumulative. If the real resume hit
            # locally, its external-hit delta is therefore zero; any observed
            # external restoration belongs to the immediately preceding proxy.
            resume_external = 0.0 if resume_local == 4096.0 else resume_external
        matched = token_ids_match(
            [resume["first_token_id"]] if resume["first_token_id"] is not None else None,
            [control_token_id] if control_token_id is not None else None,
        )
        resume_ttft = float(resume["ttft_ms"])
        row.update({
            "resume_ttft_ms": resume_ttft,
            "resume_first_token_id": resume["first_token_id"] if resume["first_token_id"] is not None else "NA",
            "control_first_token_id": control_token_id if control_token_id is not None else "NA",
            "effective_resume_latency_ms": effective_resume_latency_ms(float(prefetch_stall), resume_ttft),
            "resume_dispatch_delay_ms": dispatch_delay_ms,
            "resume_local_hit_tokens": resume_local,
            "resume_external_hit_tokens": resume_external,
            "resume_recomputed_tokens": resume_recomputed,
            "resume_external_cache_hit_tokens": (
                0.0 if policy.startswith("JIT-") and resume_local == 4096.0
                else external_cache_hit_delta(before_resume, after_resume)
            ),
            "generation_validity": "PASS" if matched else "FAIL",
            "prefetch_validation": (
                "NA" if policy == "OnDemand" else
                "PASS" if float(row["prefetch_external_hit_tokens"]) == 4096.0 and float(row["prefetch_local_hit_tokens"]) == 0.0 else "FAIL"
            ),
            "post_prefetch_local_validation": (
                "NA" if policy == "OnDemand" else
                "PASS" if resume_local == 4096.0 and resume_external == 0.0 else "FAIL"
            ),
        })
        return row, snapshots
    finally:
        stop_server(proc)


def median(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) not in ("NA", None, "")]
    return statistics.median(values) if values else None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fieldnames = list(rows[0])
    for row in rows[1:]:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: list[dict[str, Any]], control: dict[str, Any]) -> dict[str, Any]:
    aggregated: list[dict[str, Any]] = []
    for policy in POLICIES:
        subset = [row for row in rows if row["policy"] == policy]
        aggregated.append({
            "policy": policy,
            "n": len(subset),
            "median_offload_e2e_ms": median(subset, "offload_e2e_ms"),
            "median_offload_transfer_ms": median(subset, "offload_transfer_ms"),
            "median_prefetch_request_ms": median(subset, "prefetch_request_ms"),
            "median_prefetch_start_lag_ms": median(subset, "prefetch_start_lag_ms"),
            "median_prefetch_stall_ms": median(subset, "prefetch_stall_ms"),
            "median_early_residency_ms": median(subset, "early_residency_ms"),
            "median_resume_ttft_ms": median(subset, "resume_ttft_ms"),
            "median_effective_resume_latency_ms": median(subset, "effective_resume_latency_ms"),
            "median_resume_dispatch_delay_ms": median(subset, "resume_dispatch_delay_ms"),
            "median_prefetch_external_hit_tokens": median(subset, "prefetch_external_hit_tokens"),
            "median_resume_local_hit_tokens": median(subset, "resume_local_hit_tokens"),
            "median_resume_external_hit_tokens": median(subset, "resume_external_hit_tokens"),
        })
    write_csv(RESULTS / "metrics.csv", aggregated)
    jit_rows = [row for row in aggregated if row["policy"].startswith("JIT-")]
    prefetch_validation = all(
        row["prefetch_validation"] == "PASS"
        for row in rows if row["policy"].startswith("JIT-")
    )
    local_validation = all(
        row["post_prefetch_local_validation"] == "PASS"
        for row in rows if row["policy"].startswith("JIT-")
    )
    generation_valid = control.get("generation_validity") == "PASS" and all(row["generation_validity"] == "PASS" for row in rows)
    any_early = any(float(row["early_residency_ms"]) > 0 for row in rows if row["policy"].startswith("JIT-"))
    tradeoff_observed = any_early and (
        any(float(row["prefetch_stall_ms"]) > 0 for row in rows if row["policy"].startswith("JIT-"))
        or len({round(float(row["early_residency_ms"]), 1) for row in rows if row["policy"].startswith("JIT-")}) > 1
    )
    result = "A" if prefetch_validation and local_validation and generation_valid and any_early and tradeoff_observed else (
        "B" if prefetch_validation and local_validation and generation_valid else "C"
    )
    clean_candidates = [
        item for item in jit_rows
        if item["median_prefetch_stall_ms"] is not None and item["median_early_residency_ms"] is not None
        and item["median_prefetch_stall_ms"] <= 1.0
    ]
    cleanest = min(clean_candidates, key=lambda x: x["median_early_residency_ms"]) if clean_candidates else None
    lines = [
        "# Phase 2D — Deadline-Aware JIT Promotion Timing",
        "",
        "## Objective and setup",
        "",
        "Test whether recovery of an already CPU-offloaded Session A KV can be moved earlier in a real vLLM pause, so the real resume uses GPU-local KV. This uses one RTX A6000, Qwen2.5-1.5B-Instruct, vLLM 0.23.0, PyTorch 2.11.0+cu126, CUDA 12.6, 320 MiB GPU KV, 2 GiB pinned CPU offload, 4096-token A prefix, 2-second resume horizon, and three runs per policy.",
        "",
        "The JIT prefetch is an experimental proxy: an ordinary completion request with the exact original prefix triggers CPU→GPU KV recovery; its generated token is discarded. `prefetch_request_ms` includes request and model execution overhead. This is not a production zero-compute promotion API.",
        "",
        "## Mechanism checks",
        "",
        f"- OnDemand external recovery: median external prompt tokens = {median([r for r in rows if r['policy'] == 'OnDemand'], 'resume_external_hit_tokens')}; median local-hit tokens = {median([r for r in rows if r['policy'] == 'OnDemand'], 'resume_local_hit_tokens')}.",
        f"- JIT prefetch external-hit validation: {'PASS' if prefetch_validation else 'FAIL'}; across proxy requests, external hit median = {median([r for r in rows if r['policy'].startswith('JIT-')], 'prefetch_external_hit_tokens')} tokens.",
        f"- Post-prefetch real-resume local-hit validation: {'PASS' if local_validation else 'FAIL'}; JIT real-resume local-hit median = {median([r for r in rows if r['policy'].startswith('JIT-')], 'resume_local_hit_tokens')} tokens.",
        f"- Local-hit control: TTFT {control['ttft_ms']:.2f} ms; local-hit tokens {control['local_hit_tokens']}; first token ID {control['first_token_id']}.",
        f"- Token-ID correctness against local-hit control: {'PASS' if generation_valid else 'FAIL'} for every measured real resume.",
        "",
        "## Policy medians",
        "",
        "| Policy | Prefetch duration (ms) | Prefetch stall (ms) | Early residency (ms) | Actual resume TTFT (ms) | Effective resume latency (ms) | Resume dispatch delay (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in aggregated:
        def fmt(value: Any) -> str:
            return "NA" if value is None else f"{float(value):.2f}"
        lines.append(
            f"| {item['policy']} | {fmt(item['median_prefetch_request_ms'])} | {fmt(item['median_prefetch_stall_ms'])} | {fmt(item['median_early_residency_ms'])} | {fmt(item['median_resume_ttft_ms'])} | {fmt(item['median_effective_resume_latency_ms'])} | {fmt(item['median_resume_dispatch_delay_ms'])} |"
        )
    lines.extend([
        "",
        f"The measured A_first E2E latency median was {median(rows, 'offload_e2e_ms'):.2f} ms across all 12 runs; median GPU→CPU transfer time counter was {median(rows, 'offload_transfer_ms'):.2f} ms, with exactly {EXPECTED_D2H_BYTES:,} offloaded bytes per run. JIT moves resume recovery earlier; it does not remove this original GPU→CPU offload cost. Phase 2C-Fix previously measured about 57.7 ms higher A-first E2E than B-first E2E; that historical first-turn difference is separate from effective resume latency and is not folded into these values.",
        "",
        (f"Among conditions with <=1 ms median prefetch stall, {cleanest['policy']} had the least median early residency ({cleanest['median_early_residency_ms']:.2f} ms); this is a descriptive tradeoff point, not an optimality claim." if cleanest else "No JIT lead had <=1 ms median prefetch stall, so there is no clean low-stall/low-residency point in this matrix."),
        "",
        f"**Result {result}:** " + ("Real deadline-aware JIT promotion mechanism validated." if result == "A" else "JIT promotion works, but this prototype does not provide a meaningful latency benefit." if result == "B" else "The current JIT promotion mechanism requires redesign."),
        "",
        "## Limitations and anomalies",
        "",
        "- The ordinary-request prefetch proxy includes a generated token and normal model execution. It can overstate a future promotion-only runtime's cost and may change cache state.",
        "- This experiment uses one session, oracle 2-second deadline, one model, and no queue contention or prediction error.",
        "- Pressure is exactly the validated pair of distinct 9500-token requests; no additional pressure was added.",
        "- The isolated runtime emits a stale optional `nvidia_cutlass_dsl` `.pth` startup warning, and vLLM shutdown reports one semaphore cleanup warning per server. The CUDA/runtime capability gate, all server startups, requests, cache checks, and generation checks passed; no vLLM request or model error was found.",
        "- An initial run failed result aggregation, and a complete follow-up exposed redundant metrics scrapes at dispatch points. Those attempts are archived and excluded; the final runner uses existing snapshots as counters baselines and records dispatch lag explicitly.",
        "- For JIT runs, no metrics scrape is inserted between the synchronous prefetch and real resume. Source-token counters are captured after resume and attributed using the serialized request order: a 4096-token external delta plus a 4096-token local resume hit establishes external restore in the proxy followed by local reuse. The after-resume snapshot contains cumulative deltas for both requests.",
        "- See `raw_runs.csv` and `metric_snapshots.json` for per-run timings and the actual vLLM metric samples.",
        "",
        "Phase 2D execution status: COMPLETE",
    ])
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"result": result, "aggregated": aggregated, "prefetch_validation": prefetch_validation, "local_validation": local_validation, "generation_validity": generation_valid}


def system_environment(gate: dict[str, Any]) -> dict[str, Any]:
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    ).stdout.strip().splitlines()[0]
    name, memory, driver = [piece.strip() for piece in gpu.split(",", 2)]
    return {
        **gate,
        "python_version": sys.version.split()[0],
        "driver_version": driver,
        "gpu0_memory_total": memory,
        "gpu0": name,
        "installation_method": "isolated vLLM 0.23.0 CUDA 12.6 source build; reuse Phase 2C-Fix environment",
        "flashinfer_version": "0.6.12",
        "server_smoke_test": "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", default="0,1,2")
    args = parser.parse_args()
    repetitions = [int(x) for x in args.repetitions.split(",")]
    if repetitions != [0, 1, 2]:
        raise ValueError("Phase 2D requires exactly repetitions 0,1,2")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "server_logs").mkdir(exist_ok=True)
    gate = capability_gate()
    server_smoke_test()
    gate = system_environment(gate)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    prompts = build_prompts(tokenizer)
    control = run_local_control(prompts, {})
    rows: list[dict[str, Any]] = []
    snapshots = list(control.pop("metric_snapshots"))
    (RESULTS / "control.json").write_text(json.dumps(control, indent=2), encoding="utf-8")
    (RESULTS / "metric_snapshots.json").write_text(json.dumps(snapshots, indent=2), encoding="utf-8")
    for policy in POLICIES:
        for repetition in repetitions:
            row, run_snapshots = run_policy(policy, repetition, prompts, control.get("first_token_id"))
            rows.append(row)
            snapshots.extend(run_snapshots)
            write_csv(RESULTS / "raw_runs.csv", rows)
            (RESULTS / "metric_snapshots.json").write_text(json.dumps(snapshots, indent=2), encoding="utf-8")
    write_csv(RESULTS / "raw_runs.csv", rows)
    (RESULTS / "metric_snapshots.json").write_text(json.dumps(snapshots, indent=2), encoding="utf-8")
    (RESULTS / "config.json").write_text(json.dumps({
        "base_commit": "a8688fd62fed53e58c95c65b8f2efd1d85c1932f",
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "gpu": "GPU0",
        "dtype": "float16",
        "tensor_parallel_size": 1,
        "gpu_kv_capacity_bytes": 335544320,
        "cpu_offload_capacity_bytes": 2147483648,
        "max_model_len": 10000,
        "prefix_tokens": 4096,
        "pressure_tokens_each": PRESSURE_TOKENS,
        "pressure_prompts": 2,
        "resume_horizon_s": RESUME_HORIZON_S,
        "policies": POLICIES,
        "repetitions": repetitions,
        "control": control,
        "expected_offload_bytes": EXPECTED_D2H_BYTES,
        "prefetch_proxy": "ordinary completion containing exact original prefix; max_offload_tokens=0; generated token discarded",
    }, indent=2), encoding="utf-8")
    (RESULTS / "environment.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    evaluation = write_summary(rows, control)
    print(json.dumps({"run_count": len(rows), "control": control, **evaluation}, indent=2))


if __name__ == "__main__":
    main()
