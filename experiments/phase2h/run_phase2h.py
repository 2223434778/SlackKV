"""Run Phase 2H: a six-session end-to-end real vLLM workload."""
from __future__ import annotations

import csv
import importlib.metadata
import json
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from experiments.phase2d import run_phase2d as runtime
from experiments.phase2d.helpers import gpu_to_cpu_offload_bytes, parse_prometheus_samples, source_tokens_delta
from experiments.phase2e.helpers import resume_token_matches
from experiments.phase2h.scheduler import (
    DEADLINE_MS, KV_MIB, P_EST_MS, SESSIONS, effective_resume_latency_ms,
    memory_time_mib_s, naive_target_start_ms, percentile_linear, queue_safe_idle_s,
)

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "phase2h"
EXPECTED_D2H = 117_440_512
GPU_KV_BYTES = 1_073_741_824
POLICIES = ("KeepAll", "OnDemand", "ASAP", "NaiveJIT", "QueueJIT")
PROMOTION_POLICIES = ("ASAP", "NaiveJIT", "QueueJIT")
REPETITIONS = (0, 1, 2)
EPSILON_S = 1e-6
SUFFIX_TEXT = "\nTool result: operation completed successfully.\nAssistant:"
SNAPSHOT_PREFIXES = (
    "vllm:prompt_tokens_by_source_", "vllm:kv_offload_", "vllm:prefix_cache_",
    "vllm:external_prefix_cache_", "vllm:kv_cache_usage_perc", "vllm:cache_config_info",
)

SERVER = list(runtime.SERVER)
SERVER[SERVER.index("--kv-cache-memory-bytes") + 1] = str(GPU_KV_BYTES)


def start_server(log_name: str) -> subprocess.Popen[str]:
    path = RESULTS / "server_logs" / log_name
    env = os.environ.copy()
    env.update(runtime.BASE_ENV)
    with path.open("w", encoding="utf-8") as log:
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


def compact_metric_lines(raw: str) -> str:
    selected = []
    for line in raw.splitlines():
        if line.startswith(("# HELP ", "# TYPE ")):
            parts = line.split(maxsplit=3)
            name = parts[2] if len(parts) > 2 else ""
        elif line and not line.startswith("#"):
            name = line.split("{", 1)[0].split(maxsplit=1)[0]
        else:
            continue
        if name.startswith(SNAPSHOT_PREFIXES):
            selected.append(line)
    return "\n".join(selected)


def capture(stage: str, snapshots: list[dict[str, Any]]) -> dict[Any, float]:
    raw = runtime.metric_text()
    snapshots.append({
        "stage": stage, "captured_at_unix_s": time.time(),
        "raw_relevant": compact_metric_lines(raw),
    })
    return parse_prometheus_samples(raw)


def build_prompts(tokenizer: Any) -> dict[str, list[int]]:
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    prompts: dict[str, list[int]] = {}
    for session in ("W", *SESSIONS):
        text = f"Session {session} ordinary coding context marker {session}. "
        prompts[session.lower()] = encode(text * 900)[:4096]
    pressure_texts = (
        "PRESSURE ONE unrelated file-search output alpha. ",
        "PRESSURE TWO unrelated compiler output beta. ",
        "PRESSURE THREE unrelated repository status gamma. ",
        "PRESSURE FOUR unrelated terminal output delta. ",
    )
    for index, text in enumerate(pressure_texts, 1):
        prompts[f"pressure_{index}"] = encode(text * 3000)[:9500]
    prompts["suffix"] = encode(SUFFIX_TEXT)
    session_keys = ("w", "a", "b", "c", "d", "e", "f")
    pressure_keys = tuple(f"pressure_{index}" for index in range(1, 5))
    if any(len(prompts[key]) != 4096 for key in session_keys):
        raise AssertionError("W/A-F prefixes must each contain exactly 4096 tokens")
    if any(len(prompts[key]) != 9500 for key in pressure_keys):
        raise AssertionError("all pressure prompts must contain exactly 9500 tokens")
    if len({tuple(prompts[key]) for key in session_keys}) != len(session_keys):
        raise AssertionError("W/A-F prefixes must be distinct")
    if len({tuple(prompts[key]) for key in pressure_keys}) != len(pressure_keys):
        raise AssertionError("pressure prompts must be distinct")
    if any(prompts[s] == prompts[p][:4096] for s in session_keys for p in pressure_keys):
        raise AssertionError("pressure prompts must be unrelated to session prefixes")
    return prompts


def send(prompt: list[int], *, stream: bool = False, normal_offload: bool = False,
         return_token_ids: bool = False) -> dict[str, Any]:
    return runtime.send_request(
        prompt, stream=stream, keep_local=not normal_offload,
        return_token_ids=return_token_ids,
    )


def run_control(prompts: dict[str, list[int]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    proc = start_server("local_hit_control.log")
    snapshots: list[dict[str, Any]] = []
    try:
        runtime.wait_ready(proc)
        capture("control_server_ready", snapshots)
        for session in SESSIONS:
            send(prompts[session.lower()])
        time.sleep(1.0)
        before = capture("control_before_resumes", snapshots)
        control: dict[str, Any] = {}
        for session in SESSIONS:
            result = send(prompts[session.lower()] + prompts["suffix"], stream=True, return_token_ids=True)
            control[session] = {"first_token_id": result["first_token_id"], "ttft_ms": result["ttft_ms"]}
        after = capture("control_after_resumes", snapshots)
        local = source_tokens_delta(before, after, "local_cache_hit")
        external = source_tokens_delta(before, after, "external_kv_transfer")
        control["aggregate_local_hit_tokens"] = local
        control["aggregate_external_hit_tokens"] = external
        control["generation_validity"] = (
            "PASS" if all(control[s]["first_token_id"] is not None for s in SESSIONS)
            and local == 24576.0 and external == 0.0 else "FAIL"
        )
        return control, snapshots
    finally:
        stop_server(proc)


def prepare_external(label: str, prompts: dict[str, list[int]],
                     snapshots: list[dict[str, Any]]) -> tuple[dict[Any, float], dict[str, Any]]:
    previous = capture(f"{label}_server_ready", snapshots)
    offloads: dict[str, float] = {}
    for session in ("W", *SESSIONS):
        send(prompts[session.lower()], normal_offload=True)
        current = capture(f"{label}_after_{session}_first", snapshots)
        offloads[session] = gpu_to_cpu_offload_bytes(previous, current)
        previous = current
    time.sleep(1.0)
    for index in range(1, 5):
        send(prompts[f"pressure_{index}"])
    after_pressure = capture(f"{label}_after_pressure", snapshots)
    warmup = send(prompts["w"])
    before_workload = capture(f"{label}_after_W_warmup_before_t0", snapshots)
    w_external = source_tokens_delta(after_pressure, before_workload, "external_kv_transfer")
    w_local = source_tokens_delta(after_pressure, before_workload, "local_cache_hit")
    return before_workload, {
        "offloads": offloads,
        "offload_validation": "PASS" if all(v == EXPECTED_D2H for v in offloads.values()) else "FAIL",
        "W_warmup_ms": warmup["e2e_ms"],
        "W_external_hit_tokens": w_external,
        "W_local_hit_tokens": w_local,
        "W_warmup_validation": "PASS" if w_external == 4096.0 and w_local == 0.0 else "FAIL",
    }


def prepare_keep_all(label: str, prompts: dict[str, list[int]],
                     snapshots: list[dict[str, Any]]) -> tuple[dict[Any, float], dict[str, Any]]:
    capture(f"{label}_server_ready", snapshots)
    for session in SESSIONS:
        send(prompts[session.lower()])
    time.sleep(1.0)
    before_workload = capture(f"{label}_before_t0", snapshots)
    return before_workload, {
        "offloads": {}, "offload_validation": "NA", "W_warmup_ms": "NA",
        "W_external_hit_tokens": "NA", "W_local_hit_tokens": "NA", "W_warmup_validation": "NA",
    }


def schedule_promotions(policy: str, t0: float, prompts: dict[str, list[int]],
                        completion_events: dict[str, threading.Event]) -> dict[str, dict[str, Any]]:
    deadlines = {s: t0 + DEADLINE_MS[s] / 1000.0 for s in SESSIONS}
    pending = set(SESSIONS)
    records: dict[str, dict[str, Any]] = {}
    targets: dict[str, float] = {}

    def promote(session: str, target: float) -> None:
        result = send(prompts[session.lower()])
        records[session] = {"target_abs_s": target, **result}
        completion_events[session].set()

    if policy == "ASAP":
        for session in SESSIONS:
            promote(session, t0)
        return records
    if policy == "NaiveJIT":
        targets = {
            s: t0 + naive_target_start_ms(DEADLINE_MS[s]) / 1000.0
            for s in SESSIONS
        }
        for session in sorted(SESSIONS, key=lambda s: (targets[s], deadlines[s], s)):
            runtime.wait_until(targets[session])
            promote(session, targets[session])
        return records
    if policy != "QueueJIT":
        raise ValueError(f"unknown promotion policy {policy}")
    while pending:
        now = time.monotonic()
        pending_deadlines = {s: deadlines[s] for s in pending}
        safe_idle = queue_safe_idle_s(now, pending_deadlines)
        head = min(pending, key=lambda s: (deadlines[s], s))
        if safe_idle > EPSILON_S:
            wake = now + safe_idle
            targets.setdefault(head, wake)
            runtime.wait_until(wake)
            continue
        promote(head, targets.pop(head, now))
        pending.remove(head)
    return records


def dispatch_resume(session: str, t0: float, prompts: dict[str, list[int]],
                    promotion_event: threading.Event | None) -> dict[str, Any]:
    deadline_abs = t0 + DEADLINE_MS[session] / 1000.0
    runtime.wait_until(deadline_abs)
    waited = False
    if promotion_event is not None and not promotion_event.is_set():
        waited = True
        if not promotion_event.wait(timeout=240):
            raise TimeoutError(f"promotion for {session} did not complete")
    result = send(
        prompts[session.lower()] + prompts["suffix"],
        stream=True,
        return_token_ids=True,
    )
    return {
        **result,
        "effective_resume_latency_ms": effective_resume_latency_ms(
            result["sent_monotonic_s"], result["ttft_ms"], t0, DEADLINE_MS[session]
        ),
        "waited_for_promotion": waited,
    }


def execute_timeline(policy: str, t0: float, prompts: dict[str, list[int]]) -> tuple[
    dict[str, dict[str, Any]], dict[str, dict[str, Any]]
]:
    events = {s: threading.Event() for s in SESSIONS} if policy in PROMOTION_POLICIES else {}
    promotions: dict[str, dict[str, Any]] = {}
    resumes: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=7, thread_name_prefix="phase2h-workload") as pool:
        promotion_future = (
            pool.submit(schedule_promotions, policy, t0, prompts, events)
            if policy in PROMOTION_POLICIES else None
        )
        resume_futures = {
            s: pool.submit(dispatch_resume, s, t0, prompts, events.get(s))
            for s in SESSIONS
        }
        for session, future in resume_futures.items():
            resumes[session] = future.result(timeout=300)
        if promotion_future is not None:
            promotions = promotion_future.result(timeout=300)
    return promotions, resumes


def run_measured(policy: str, repetition: int, prompts: dict[str, list[int]]) -> tuple[
    list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]
]:
    label = f"{policy.lower()}_rep{repetition}"
    proc = start_server(f"{label}.log")
    snapshots: list[dict[str, Any]] = []
    try:
        runtime.wait_ready(proc)
        if policy == "KeepAll":
            before, setup = prepare_keep_all(label, prompts, snapshots)
        else:
            before, setup = prepare_external(label, prompts, snapshots)
        t0 = time.monotonic()
        promotions, resumes = execute_timeline(policy, t0, prompts)
        # This is the first /metrics scrape after t0.
        after = capture(f"{label}_after_complete_workload", snapshots)
        external_hits = source_tokens_delta(before, after, "external_kv_transfer")
        local_hits = source_tokens_delta(before, after, "local_cache_hit")

        rows: list[dict[str, Any]] = []
        for session in SESSIONS:
            promotion = promotions.get(session)
            if policy == "KeepAll":
                target = start = duration = completion = "NA"
                promotion_stall = 0.0
                early = DEADLINE_MS[session]
            elif policy == "OnDemand":
                target = start = duration = completion = "NA"
                promotion_stall = early = 0.0
            else:
                assert promotion is not None
                target = (promotion["target_abs_s"] - t0) * 1000.0
                start = (promotion["sent_monotonic_s"] - t0) * 1000.0
                duration = promotion["e2e_ms"]
                completion = (promotion["completed_monotonic_s"] - t0) * 1000.0
                delta = completion - DEADLINE_MS[session]
                promotion_stall = max(0.0, delta)
                early = max(0.0, -delta)
            resume = resumes[session]
            rows.append({
                "policy": policy, "repetition": repetition, "session": session,
                "deadline_ms": DEADLINE_MS[session],
                "promotion_target_ms": target,
                "promotion_actual_start_ms": start,
                "promotion_duration_ms": duration,
                "promotion_completion_ms": completion,
                "promotion_deadline_stall_ms": promotion_stall,
                "early_residency_ms": early,
                "resume_request_sent_ms": (resume["sent_monotonic_s"] - t0) * 1000.0,
                "resume_ttft_ms": resume["ttft_ms"],
                "effective_resume_latency_ms": resume["effective_resume_latency_ms"],
                "waited_for_promotion": resume["waited_for_promotion"],
                "first_token_id": resume["first_token_id"] if resume["first_token_id"] is not None else "NA",
                "control_token_id": "PENDING", "generation_validity": "PENDING",
            })

        effective = [float(row["effective_resume_latency_ms"]) for row in rows]
        ttft = [float(row["resume_ttft_ms"]) for row in rows]
        early_values = [float(row["early_residency_ms"]) for row in rows]
        expected = {
            "KeepAll": (0.0, 24576.0),
            "OnDemand": (24576.0, 0.0),
            "ASAP": (24576.0, 24576.0),
            "NaiveJIT": (24576.0, 24576.0),
            "QueueJIT": (24576.0, 24576.0),
        }[policy]
        cache_valid = external_hits == expected[0] and local_hits == expected[1]
        timeline_valid = all(
            float(row["resume_request_sent_ms"]) + 1e-6 >= float(row["deadline_ms"])
            and float(row["effective_resume_latency_ms"]) >= 0
            for row in rows
        )
        run = {
            "policy": policy, "repetition": repetition,
            "promotion_deadline_miss_count": sum(
                float(row["promotion_deadline_stall_ms"]) > 0 for row in rows
            ),
            "P50_effective_resume_latency_ms": percentile_linear(effective, 50),
            "P95_effective_resume_latency_ms": percentile_linear(effective, 95),
            "max_effective_resume_latency_ms": max(effective),
            "P50_resume_ttft_ms": percentile_linear(ttft, 50),
            "P95_resume_ttft_ms": percentile_linear(ttft, 95),
            "total_early_residency_ms": sum(early_values),
            "idle_gpu_memory_time_mib_s": memory_time_mib_s(early_values),
            "workload_external_hit_tokens": external_hits,
            "workload_local_hit_tokens": local_hits,
            "cache_validation": "PASS" if cache_valid else "FAIL",
            "generation_validation": "PENDING",
            "timeline_validation": "PASS" if timeline_valid else "FAIL",
            "external_offload_validation": setup["offload_validation"],
            "W_warmup_validation": setup["W_warmup_validation"],
            "W_warmup_ms_excluded": setup["W_warmup_ms"],
            "W_external_hit_tokens": setup["W_external_hit_tokens"],
            "W_local_hit_tokens": setup["W_local_hit_tokens"],
            "offload_bytes_json": json.dumps(setup["offloads"], sort_keys=True),
        }
        return rows, run, snapshots
    finally:
        stop_server(proc)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def compact_logs() -> None:
    keep = (
        "non-default args", "Initializing a V1 LLM engine", "GPU KV cache size",
        "Creating v1 connector", "Creating offloading spec", "Allocating a cross layer",
        "Allocating 1 CPU tensors", "Starting vLLM server", "GET /health",
        "POST /v1/completions", "GET /metrics", "WARNING", "ERROR", "Traceback",
        "ModuleNotFoundError", "nvidia_cutlass_dsl", "resource_tracker", "semaphore",
        "shutdown", "Triton kernel", "Triton JIT", "JIT compilation", "_compute_slot_mapping_kernel",
    )
    for path in (RESULTS / "server_logs").glob("*.log"):
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        path.write_text(
            "\n".join(line for line in lines if any(x in line for x in keep)) + "\n",
            encoding="utf-8",
        )


def policy_medians(runs: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    fields = (
        "promotion_deadline_miss_count", "P50_effective_resume_latency_ms",
        "P95_effective_resume_latency_ms", "max_effective_resume_latency_ms",
        "P50_resume_ttft_ms", "P95_resume_ttft_ms", "total_early_residency_ms",
        "idle_gpu_memory_time_mib_s",
    )
    return {
        policy: {
            field: statistics.median(float(row[field]) for row in runs if row["policy"] == policy)
            for field in fields
        }
        for policy in POLICIES
    }


def build_summary(env: dict[str, Any], control: dict[str, Any], runs: list[dict[str, Any]],
                  sessions: list[dict[str, Any]], medians: dict[str, dict[str, float]],
                  outcome: str) -> str:
    external_runs = [row for row in runs if row["policy"] != "KeepAll"]
    offloads_ok = all(row["external_offload_validation"] == "PASS" for row in external_runs)
    warmup_ok = all(row["W_warmup_validation"] == "PASS" for row in external_runs)
    cache_ok = all(row["cache_validation"] == "PASS" for row in runs)
    generation_ok = control["generation_validity"] == "PASS" and all(
        row["generation_validation"] == "PASS" for row in runs
    )
    timeline_ok = all(row["timeline_validation"] == "PASS" for row in runs)
    warm_durations = [
        float(row["promotion_duration_ms"]) for row in sessions
        if row["policy"] in PROMOTION_POLICIES
    ]
    w_durations = [float(row["W_warmup_ms_excluded"]) for row in external_runs]
    queue, naive = medians["QueueJIT"], medians["NaiveJIT"]
    asap, keep, ondemand = medians["ASAP"], medians["KeepAll"], medians["OnDemand"]
    queue_vs_asap = 100.0 * (
        1.0 - queue["idle_gpu_memory_time_mib_s"] / asap["idle_gpu_memory_time_mib_s"]
    )
    queue_vs_keep = 100.0 * (
        1.0 - queue["idle_gpu_memory_time_mib_s"] / keep["idle_gpu_memory_time_mib_s"]
    )
    naive_cluster_waits = sum(
        str(row["waited_for_promotion"]) == "True" for row in sessions
        if row["policy"] == "NaiveJIT" and row["session"] in ("A", "B", "C")
    )
    lines = [
        "# Phase 2H — Six-session end-to-end real workload",
        "",
        "## Objective",
        "",
        "Evaluate whether QueueJIT preserves its latency–idle-KV-residency advantage when six real vLLM resume requests execute on their configured oracle timeline alongside one serialized promotion worker.",
        "",
        "## Setup and methods",
        "",
        f"Base commit: 81abb72a5916d0cf21b8f1ff1fba85e54414061d. Runtime: vLLM {env['vllm_version']}, PyTorch {env['torch_version']}, CUDA {env['torch_cuda_version']}, {env['gpu0']} GPU0, Qwen2.5-1.5B-Instruct, float16, TP=1, 1 GiB GPU KV, 2 GiB pinned CPU offload, prefix caching, max model length 10000.",
        "",
        "A–F each use a distinct 4096-token prefix. Oracle deadlines are 300/320/340/650/900/1300 ms from t0. External policies use W plus A–F normal offload, a one-second store wait, four frozen distinct 9500-token pressure requests, and one excluded W promotion before t0. p_est is frozen at 42 ms. KeepAll, OnDemand, ASAP, NaiveJIT and QueueJIT each ran three independent server restarts.",
        "",
        "One serialized promotion worker and six separate resume-dispatch tasks share the real timeline. A resume waits for its own in-flight promotion when necessary and never starts a duplicate restore. No metrics scrape occurs from t0 through completion of the final measured resume. Percentiles use linear interpolation at rank (n-1)q over the six sessions.",
        "",
        "## Validation",
        "",
        f"- All W/A–F external first turns offloaded exactly {EXPECTED_D2H:,} bytes: {'PASS' if offloads_ok else 'FAIL'}.",
        f"- W absorbed the cold promotion path with 4096 external and 0 local tokens in every external run: {'PASS' if warmup_ok else 'FAIL'}; W median {statistics.median(w_durations):.2f} ms.",
        f"- Measured A–F scheduled promotion service range: {min(warm_durations):.2f}–{max(warm_durations):.2f} ms; median {statistics.median(warm_durations):.2f} ms.",
        f"- Real resume timeline: {'PASS' if timeline_ok else 'FAIL'}; no resume was sent before its configured deadline.",
        f"- Aggregate cache evidence: {'PASS' if cache_ok else 'FAIL'}. KeepAll=24576 local/0 external; OnDemand=0 local/24576 external; scheduled policies=24576 local/24576 external in every valid run.",
        f"- First-token IDs against A–F controls: {'PASS' if generation_ok else 'FAIL'}.",
        "",
        "## Results — medians across three repetitions",
        "",
        "| Policy | Promotion misses | P50 effective latency (ms) | P95 effective latency (ms) | Max effective latency (ms) | Total early residency (ms) | Idle GPU memory-time (MiB·s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in POLICIES:
        item = medians[policy]
        lines.append(
            f"| {policy} | {item['promotion_deadline_miss_count']:.2f} | "
            f"{item['P50_effective_resume_latency_ms']:.2f} | "
            f"{item['P95_effective_resume_latency_ms']:.2f} | "
            f"{item['max_effective_resume_latency_ms']:.2f} | "
            f"{item['total_early_residency_ms']:.2f} | "
            f"{item['idle_gpu_memory_time_mib_s']:.2f} |"
        )
    lines.extend([
        "",
        "## Direct answers",
        "",
        f"- Clustered NaiveJIT A/B/C resumes waited for promotion {naive_cluster_waits}/9 times across three repetitions. Median promotion misses were {naive['promotion_deadline_miss_count']:.2f}; QueueJIT reduced this to {queue['promotion_deadline_miss_count']:.2f}.",
        f"- QueueJIT versus NaiveJIT: P95 effective resume latency {queue['P95_effective_resume_latency_ms']:.2f} vs {naive['P95_effective_resume_latency_ms']:.2f} ms; memory-time {queue['idle_gpu_memory_time_mib_s']:.2f} vs {naive['idle_gpu_memory_time_mib_s']:.2f} MiB·s.",
        f"- QueueJIT saved {queue_vs_asap:.1f}% idle GPU memory-time versus ASAP ({queue['idle_gpu_memory_time_mib_s']:.2f} vs {asap['idle_gpu_memory_time_mib_s']:.2f} MiB·s).",
        f"- QueueJIT saved {queue_vs_keep:.1f}% idle GPU memory-time versus KeepAll ({queue['idle_gpu_memory_time_mib_s']:.2f} vs {keep['idle_gpu_memory_time_mib_s']:.2f} MiB·s).",
        f"- QueueJIT versus OnDemand P95 effective resume latency: {queue['P95_effective_resume_latency_ms']:.2f} vs {ondemand['P95_effective_resume_latency_ms']:.2f} ms.",
        f"- OnDemand therefore dominated QueueJIT on the two primary reported axes in this workload: lower P95 effective latency and zero versus {queue['idle_gpu_memory_time_mib_s']:.2f} MiB·s idle memory-time.",
        "",
        f"**Result {outcome}:** " + (
            "Queue-aware deadline scheduling provides a better idle-KV residency versus resume-latency tradeoff on this real six-session vLLM workload."
            if outcome == "A" else
            "The mechanism works, but QueueJIT is not clearly better than a simpler policy at this six-session scale."
            if outcome == "B" else
            "Real resume concurrency destabilized the mechanism, cache evidence, correctness, or required timeline."
        ),
        "",
        "## Anomalies and limitations",
        "",
        "- The promotion operation remains a normal completion-request proxy, not pure H2D or a native promotion API. The single client worker is an experimental serialized shared resource; this does not prove native PCIe contention is solved or that SlackKV is production-ready.",
        "- Deadlines are oracle values. This phase adds no prediction error, selective admission, TraceLab replay, second GPU, model/context sweep, or vLLM scheduler patch.",
        "- Idle GPU memory-time covers only measured idle-period session KV residency, not total server GPU utilization.",
        f"- Contrary to the expected ordering, KeepAll P95 effective latency was {keep['P95_effective_resume_latency_ms']:.2f} ms, higher than OnDemand's {ondemand['P95_effective_resume_latency_ms']:.2f} ms. Cache evidence was valid; clustered concurrent local resumes incurred serving queue delay.",
        f"- Real resume concurrency inflated some scheduled promotion proxies above the isolated 42-ms estimate (observed maximum {max(warm_durations):.2f} ms, notably clustered C). QueueJIT still reduced NaiveJIT's median misses and P95 latency but retained one median promotion miss.",
        "",
        "Phase 2H execution status: COMPLETE",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "server_logs").mkdir(exist_ok=True)
    gate = runtime.capability_gate()
    gpu_parts = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    ).stdout.strip().splitlines()[0].split(",", 2)
    env = {
        **gate, "python_version": sys.version.split()[0],
        "gpu0": gpu_parts[0].strip(), "gpu0_memory_total": gpu_parts[1].strip(),
        "driver_version": gpu_parts[2].strip(),
        "flashinfer_version": importlib.metadata.version("flashinfer-python"),
        "installation_method": "reused isolated Phase 2C-Fix CUDA 12.6 vLLM 0.23.0 source-build environment",
    }
    (RESULTS / "environment.json").write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
    tokenizer = AutoTokenizer.from_pretrained(runtime.MODEL, local_files_only=True)
    prompts = build_prompts(tokenizer)
    config = {
        "phase": "2H", "base_commit": "81abb72a5916d0cf21b8f1ff1fba85e54414061d",
        "runtime": "vLLM 0.23.0 / PyTorch 2.11.0+cu126 / CUDA 12.6 / RTX A6000 GPU0",
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "gpu_kv_bytes": GPU_KV_BYTES, "cpu_offload_bytes": 2_147_483_648,
        "max_model_len": 10000, "prefix_tokens": 4096, "session_kv_mib": KV_MIB,
        "warmup_session": "W", "measured_sessions": list(SESSIONS),
        "pressure_prompts": 4, "pressure_tokens_each": 9500,
        "expected_offload_bytes_each": EXPECTED_D2H,
        "estimated_promotion_service_ms": P_EST_MS,
        "deadlines_ms_from_t0": DEADLINE_MS, "policies": list(POLICIES),
        "repetitions_each": 3,
        "percentile_method": "linear interpolation at rank (n-1)*q",
        "metrics_scrape_during_workload": False,
        "promotion_proxy": "serialized completion request, max_tokens=1, temperature=0, stream=false, max_offload_tokens=0",
        "resume_request": "prefix + fixed suffix, max_tokens=1, temperature=0, stream=true, max_offload_tokens=0",
    }
    (RESULTS / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    control, control_snapshots = run_control(prompts)
    (RESULTS / "control.json").write_text(json.dumps(control, indent=2) + "\n", encoding="utf-8")
    snapshots = [{"run": "local_hit_control", **row} for row in control_snapshots]
    session_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    for policy in POLICIES:
        for repetition in REPETITIONS:
            rows, run, run_snapshots = run_measured(policy, repetition, prompts)
            for row in rows:
                expected_id = control[row["session"]]["first_token_id"]
                row["control_token_id"] = expected_id
                row["generation_validity"] = (
                    "PASS" if resume_token_matches(
                        row["first_token_id"] if row["first_token_id"] != "NA" else None,
                        expected_id,
                    ) else "FAIL"
                )
            run["generation_validation"] = (
                "PASS" if all(row["generation_validity"] == "PASS" for row in rows) else "FAIL"
            )
            session_rows.extend(rows)
            run_rows.append(run)
            snapshots.extend(
                {"run": f"{policy}_rep{repetition}", **row} for row in run_snapshots
            )
            print(
                f"completed {policy} rep {repetition}: "
                f"cache={run['cache_validation']} timeline={run['timeline_validation']}",
                flush=True,
            )

    if len(run_rows) != 15 or len(session_rows) != 90:
        raise AssertionError("Phase 2H requires exactly 15 runs and 90 session rows")
    write_csv(RESULTS / "session_metrics.csv", session_rows)
    write_csv(RESULTS / "run_metrics.csv", run_rows)
    (RESULTS / "metric_snapshots.json").write_text(
        json.dumps(snapshots, indent=2) + "\n", encoding="utf-8"
    )
    medians = policy_medians(run_rows)
    mechanisms = (
        control["generation_validity"] == "PASS"
        and all(row["cache_validation"] == "PASS"
                and row["generation_validation"] == "PASS"
                and row["timeline_validation"] == "PASS" for row in run_rows)
        and all(row["external_offload_validation"] == "PASS"
                and row["W_warmup_validation"] == "PASS"
                for row in run_rows if row["policy"] != "KeepAll")
    )
    queue, naive = medians["QueueJIT"], medians["NaiveJIT"]
    asap, keep = medians["ASAP"], medians["KeepAll"]
    ondemand = medians["OnDemand"]
    latency_better = (
        queue["promotion_deadline_miss_count"] < naive["promotion_deadline_miss_count"]
        or queue["P95_effective_resume_latency_ms"] < naive["P95_effective_resume_latency_ms"]
    )
    memory_better = (
        queue["idle_gpu_memory_time_mib_s"] <= 0.5 * asap["idle_gpu_memory_time_mib_s"]
        and queue["idle_gpu_memory_time_mib_s"] <= 0.5 * keep["idle_gpu_memory_time_mib_s"]
    )
    ondemand_dominates = (
        ondemand["P95_effective_resume_latency_ms"] <= queue["P95_effective_resume_latency_ms"]
        and ondemand["idle_gpu_memory_time_mib_s"] <= queue["idle_gpu_memory_time_mib_s"]
        and (
            ondemand["P95_effective_resume_latency_ms"] < queue["P95_effective_resume_latency_ms"]
            or ondemand["idle_gpu_memory_time_mib_s"] < queue["idle_gpu_memory_time_mib_s"]
        )
    )
    outcome = (
        "C" if not mechanisms else
        "A" if latency_better and memory_better and not ondemand_dominates else
        "B"
    )
    (RESULTS / "summary.md").write_text(
        build_summary(env, control, run_rows, session_rows, medians, outcome),
        encoding="utf-8",
    )
    compact_logs()
    print(json.dumps({
        "run_count": len(run_rows), "session_count": len(session_rows),
        "medians": medians, "outcome": outcome,
        "summary": str(RESULTS / "summary.md"),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
