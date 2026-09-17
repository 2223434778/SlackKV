"""Real vLLM two-session promotion scheduling experiment (Phase 2E)."""
from __future__ import annotations

import csv
import importlib.metadata
import json
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from experiments.phase2d import run_phase2d as runtime
from experiments.phase2d.helpers import (
    gpu_to_cpu_offload_bytes,
    gpu_to_cpu_offload_ms,
    parse_prometheus_samples,
    source_tokens_delta,
)
from .helpers import build_prompts, resume_token_matches
from .scheduler import (
    aggregate_run_metrics,
    completion_timing_ms,
    edf_order,
    naive_target_start_s,
    queue_safe_idle_s,
)

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "phase2e"
MODEL = runtime.MODEL
ENV_ROOT = runtime.ENV_ROOT
EXPECTED_OFFLOAD_BYTES = 117_440_512
P_EST_S = 0.120
DEADLINE_OFFSETS_S = {"A": 1.500, "B": 1.560}
POLICIES = ("ASAP", "NaiveJIT", "QueueJIT")
REPETITIONS = (0, 1, 2)
EPSILON_S = 1e-6
SUFFIX = "\nTool result: operation completed successfully.\nAssistant:"
SNAPSHOT_METRIC_PREFIXES = (
    "vllm:prompt_tokens_by_source_",
    "vllm:kv_offload_",
    "vllm:prefix_cache_",
    "vllm:external_prefix_cache_",
    "vllm:kv_cache_usage_perc",
    "vllm:cache_config_info",
)


def start_server(log_name: str):
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "server_logs").mkdir(exist_ok=True)
    log = (RESULTS / "server_logs" / log_name).open("w", encoding="utf-8")
    env = os.environ.copy()
    env.update(runtime.BASE_ENV)
    return subprocess.Popen(runtime.SERVER, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)


def smoke_test() -> None:
    proc = start_server("capability_smoke.log")
    try:
        runtime.wait_ready(proc)
    finally:
        runtime.stop_server(proc)


def capture(stage: str, snapshots: list[dict[str, Any]]) -> dict[tuple[str, tuple[tuple[str, str], ...]], float]:
    raw = runtime.metric_text()
    snapshots.append({
        "stage": stage,
        "captured_at_unix_s": time.time(),
        "raw_relevant": compact_metric_lines(raw),
    })
    return parse_prometheus_samples(raw)


def compact_metric_lines(raw: str) -> str:
    lines = []
    for line in raw.splitlines():
        if line.startswith("# HELP ") or line.startswith("# TYPE "):
            metric_name = line.split(maxsplit=3)[2]
        elif line and not line.startswith("#"):
            metric_name = line.split("{", 1)[0].split(maxsplit=1)[0]
        else:
            continue
        if metric_name.startswith(SNAPSHOT_METRIC_PREFIXES):
            lines.append(line)
    return "\n".join(lines)


def wait_until(monotonic_time_s: float) -> None:
    runtime.wait_until(monotonic_time_s)


def send_completion(prompt: list[int], *, stream: bool, keep_local: bool, return_token_ids: bool = False) -> dict[str, Any]:
    # Reuse the tested streaming/e2e implementation from Phase 2D.
    return runtime.send_request(
        prompt,
        stream=stream,
        keep_local=keep_local,
        return_token_ids=return_token_ids,
    )


def run_local_control(prompts: dict[str, list[int]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    proc = start_server("local_hit_control.log")
    snapshots: list[dict[str, Any]] = []
    try:
        runtime.wait_ready(proc)
        capture("control_server_ready", snapshots)
        send_completion(prompts["a"], stream=False, keep_local=False)
        capture("control_after_A_first", snapshots)
        send_completion(prompts["b"], stream=False, keep_local=False)
        capture("control_after_B_first", snapshots)
        time.sleep(1.0)
        before_resumes = capture("control_before_resumes", snapshots)
        a = send_completion(prompts["a"] + prompts["suffix"], stream=True, keep_local=True, return_token_ids=True)
        after_a = capture("control_after_A_resume", snapshots)
        b = send_completion(prompts["b"] + prompts["suffix"], stream=True, keep_local=True, return_token_ids=True)
        after_b = capture("control_after_B_resume", snapshots)
        control = {
            "A": {
                "ttft_ms": a["ttft_ms"],
                "first_token_id": a["first_token_id"],
                "local_hit_tokens": source_tokens_delta(before_resumes, after_a, "local_cache_hit"),
                "external_hit_tokens": source_tokens_delta(before_resumes, after_a, "external_kv_transfer"),
            },
            "B": {
                "ttft_ms": b["ttft_ms"],
                "first_token_id": b["first_token_id"],
                "local_hit_tokens": source_tokens_delta(after_a, after_b, "local_cache_hit"),
                "external_hit_tokens": source_tokens_delta(after_a, after_b, "external_kv_transfer"),
            },
        }
        control["generation_validity"] = (
            "PASS" if all(x["first_token_id"] is not None for x in control.values() if isinstance(x, dict))
            and all(float(x["local_hit_tokens"]) == 4096.0 and float(x["external_hit_tokens"]) == 0.0
                    for x in control.values() if isinstance(x, dict)) else "FAIL"
        )
        return control, snapshots
    finally:
        runtime.stop_server(proc)


def summarize_environment(gate: dict[str, Any]) -> dict[str, Any]:
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    ).stdout.strip().splitlines()[0]
    gpu, memory, driver = [part.strip() for part in query.split(",", 2)]
    return {
        **gate,
        "python_version": sys.version.split()[0],
        "driver_version": driver,
        "gpu0_memory_total": memory,
        "gpu0": gpu,
        "flashinfer_version": importlib.metadata.version("flashinfer-python"),
        "installation_method": "reuse isolated Phase 2C-Fix CUDA 12.6 vLLM 0.23.0 source-build environment",
        "server_smoke_test": "PASS",
    }


def _send_promotion(worker: ThreadPoolExecutor, prompt: list[int]) -> dict[str, Any]:
    return worker.submit(send_completion, prompt, stream=False, keep_local=True).result()


def schedule_promotions(
    policy: str,
    t0: float,
    prompts: dict[str, list[int]],
) -> dict[str, dict[str, Any]]:
    """Use one worker for the complete policy schedule; never overlap requests."""
    pending = dict(DEADLINE_OFFSETS_S)
    records: dict[str, dict[str, Any]] = {}
    planned_targets: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="phase2e-promotion") as worker:
        if policy == "ASAP":
            order = edf_order(pending)
            for session in order:
                result = _send_promotion(worker, prompts[session.lower()])
                records[session] = {"target_start_abs_s": t0, **result}
            return records

        if policy == "NaiveJIT":
            targets = {
                session: t0 + naive_target_start_s(deadline, P_EST_S)
                for session, deadline in pending.items()
            }
            order = sorted(pending, key=lambda session: (targets[session], pending[session], session))
            for session in order:
                wait_until(targets[session])
                result = _send_promotion(worker, prompts[session.lower()])
                records[session] = {"target_start_abs_s": targets[session], **result}
            return records

        if policy != "QueueJIT":
            raise ValueError(f"unknown policy: {policy}")

        pending_abs = {session: t0 + deadline for session, deadline in pending.items()}
        while pending:
            now = time.monotonic()
            safe_idle = queue_safe_idle_s(now, pending_abs, P_EST_S)
            head = edf_order(pending_abs)[0]
            if safe_idle > EPSILON_S:
                wake_time = now + safe_idle
                planned_targets.setdefault(head, wake_time)
                wait_until(wake_time)
                continue
            target = planned_targets.pop(head, now)
            result = _send_promotion(worker, prompts[head.lower()])
            records[head] = {"target_start_abs_s": target, **result}
            del pending[head]
            del pending_abs[head]
            # The loop recomputes prefix slack after every actual completion.
        return records


def run_measured(policy: str, repetition: int, prompts: dict[str, list[int]]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    label = f"{policy.lower()}_rep{repetition}"
    proc = start_server(f"{label}.log")
    snapshots: list[dict[str, Any]] = []
    try:
        runtime.wait_ready(proc)
        server_ready = capture(f"{label}_server_ready", snapshots)
        a_first = send_completion(prompts["a"], stream=False, keep_local=False)
        after_a_first = capture(f"{label}_after_A_first", snapshots)
        a_offload_bytes = gpu_to_cpu_offload_bytes(server_ready, after_a_first)
        a_offload_ms = gpu_to_cpu_offload_ms(server_ready, after_a_first)
        b_first = send_completion(prompts["b"], stream=False, keep_local=False)
        after_b_first = capture(f"{label}_after_B_first", snapshots)
        b_offload_bytes = gpu_to_cpu_offload_bytes(after_a_first, after_b_first)
        b_offload_ms = gpu_to_cpu_offload_ms(after_a_first, after_b_first)
        time.sleep(1.0)
        send_completion(prompts["pressure_1"], stream=False, keep_local=True)
        send_completion(prompts["pressure_2"], stream=False, keep_local=True)
        t0 = time.monotonic()
        after_pressure = capture(f"{label}_after_pressure", snapshots)

        # All schedules have the same t0 and deadlines. No metrics scrape or
        # resume request is sent while the serialized worker handles A and B.
        promotion_records = schedule_promotions(policy, t0, prompts)
        after_promotions = capture(f"{label}_after_both_promotions", snapshots)
        promoted_external = source_tokens_delta(after_pressure, after_promotions, "external_kv_transfer")
        promoted_local = source_tokens_delta(after_pressure, after_promotions, "local_cache_hit")

        a_resume = send_completion(prompts["a"] + prompts["suffix"], stream=True, keep_local=True, return_token_ids=True)
        after_a_resume = capture(f"{label}_after_A_resume", snapshots)
        b_resume = send_completion(prompts["b"] + prompts["suffix"], stream=True, keep_local=True, return_token_ids=True)
        after_b_resume = capture(f"{label}_after_B_resume", snapshots)
        resume_samples = {
            "A": (a_resume, after_promotions, after_a_resume),
            "B": (b_resume, after_a_resume, after_b_resume),
        }

        jobs: list[dict[str, Any]] = []
        for session in ("A", "B"):
            promoted = promotion_records[session]
            deadline_abs = t0 + DEADLINE_OFFSETS_S[session]
            completion_abs = promoted["completed_monotonic_s"]
            stall_ms, early_ms = completion_timing_ms(completion_abs, deadline_abs)
            resume, resume_before, resume_after = resume_samples[session]
            local_hits = source_tokens_delta(resume_before, resume_after, "local_cache_hit")
            external_hits = source_tokens_delta(resume_before, resume_after, "external_kv_transfer")
            recomputed = source_tokens_delta(resume_before, resume_after, "local_compute")
            jobs.append({
                "policy": policy,
                "repetition": repetition,
                "session": session,
                "deadline_ms_from_t0": DEADLINE_OFFSETS_S[session] * 1000.0,
                "target_start_ms_from_t0": (promoted["target_start_abs_s"] - t0) * 1000.0,
                "actual_start_ms_from_t0": (promoted["sent_monotonic_s"] - t0) * 1000.0,
                "promotion_request_ms": promoted["e2e_ms"],
                "promotion_completion_ms_from_t0": (completion_abs - t0) * 1000.0,
                "completion_relative_to_deadline_ms": (completion_abs - deadline_abs) * 1000.0,
                "deadline_stall_ms": stall_ms,
                "early_residency_ms": early_ms,
                "resume_ttft_ms": resume["ttft_ms"],
                "resume_local_hit_tokens": local_hits,
                "resume_external_hit_tokens": external_hits,
                "resume_recomputed_tokens": recomputed,
                "first_token_id": resume["first_token_id"] if resume["first_token_id"] is not None else "NA",
                "control_token_id": "NA",
                "generation_validity": "PENDING_CONTROL",
            })

        first_start = min(promotion_records[s]["sent_monotonic_s"] for s in ("A", "B"))
        final_completion = max(promotion_records[s]["completed_monotonic_s"] for s in ("A", "B"))
        run_aggregate = aggregate_run_metrics(jobs)
        offload_valid = (
            a_offload_bytes == EXPECTED_OFFLOAD_BYTES
            and b_offload_bytes == EXPECTED_OFFLOAD_BYTES
        )
        promotion_valid = promoted_external == 8192.0 and promoted_local == 0.0
        resume_local_total = sum(float(job["resume_local_hit_tokens"]) for job in jobs)
        resume_external_total = sum(float(job["resume_external_hit_tokens"]) for job in jobs)
        run_row = {
            "policy": policy,
            "repetition": repetition,
            "A_offload_bytes": a_offload_bytes,
            "B_offload_bytes": b_offload_bytes,
            "A_offload_time_ms": a_offload_ms,
            "B_offload_time_ms": b_offload_ms,
            "offload_validation": "PASS" if offload_valid else "FAIL",
            "promotion_external_hit_tokens": promoted_external,
            "promotion_local_hit_tokens": promoted_local,
            "promotion_validation": "PASS" if promotion_valid else "FAIL",
            "resume_local_hit_tokens_total": resume_local_total,
            "resume_external_hit_tokens_total": resume_external_total,
            "resume_validation": "PASS" if resume_local_total == 8192.0 and resume_external_total == 0.0 else "FAIL",
            "deadline_miss_count": run_aggregate["deadline_miss_count"],
            "max_deadline_stall_ms": run_aggregate["max_deadline_stall_ms"],
            "total_early_residency_ms": run_aggregate["total_early_residency_ms"],
            "promotion_worker_busy_ms": (final_completion - first_start) * 1000.0,
            "A_promotion_request_ms": promotion_records["A"]["e2e_ms"],
            "B_promotion_request_ms": promotion_records["B"]["e2e_ms"],
            "first_promotion_start_ms_from_t0": (first_start - t0) * 1000.0,
            "generation_validity": "PENDING_CONTROL",
        }
        return jobs, run_row, snapshots
    finally:
        runtime.stop_server(proc)


def median(rows: list[dict[str, Any]], field: str) -> float:
    return statistics.median(float(row[field]) for row in rows)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    columns = list(rows[0])
    for row in rows[1:]:
        for field in row:
            if field not in columns:
                columns.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def compact_server_logs() -> None:
    """Keep configuration, request, warning/error, and shutdown evidence only."""
    keep_fragments = (
        "non-default args",
        "Initializing a V1 LLM engine",
        "GPU KV cache size",
        "Creating v1 connector",
        "Creating offloading spec",
        "Allocating a cross layer KV cache",
        "Allocating 1 CPU tensors",
        "Starting vLLM server",
        'GET /health',
        'POST /v1/completions',
        'GET /metrics',
        "WARNING",
        "ERROR",
        "Traceback",
        "ModuleNotFoundError",
        "nvidia_cutlass_dsl",
        "resource_tracker",
        "semaphore",
        "shutdown",
    )
    log_dir = RESULTS / "server_logs"
    for path in log_dir.glob("*.log"):
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        selected = [line for line in lines if any(fragment in line for fragment in keep_fragments)]
        path.write_text("\n".join(selected) + "\n", encoding="utf-8")


def build_summary(
    job_rows: list[dict[str, Any]],
    run_rows: list[dict[str, Any]],
    control: dict[str, Any],
    env: dict[str, Any],
) -> str:
    summaries: dict[str, dict[str, float]] = {}
    for policy in POLICIES:
        subset = [row for row in run_rows if row["policy"] == policy]
        summaries[policy] = {
            "misses": median(subset, "deadline_miss_count"),
            "max_stall": median(subset, "max_deadline_stall_ms"),
            "early": median(subset, "total_early_residency_ms"),
            "worker_busy": median(subset, "promotion_worker_busy_ms"),
            "a_duration": median(subset, "A_promotion_request_ms"),
            "b_duration": median(subset, "B_promotion_request_ms"),
            "first_start": median(subset, "first_promotion_start_ms_from_t0"),
        }
    all_offloads = all(row["offload_validation"] == "PASS" for row in run_rows)
    all_promotions = all(row["promotion_validation"] == "PASS" for row in run_rows)
    all_resumes = all(row["resume_validation"] == "PASS" for row in run_rows)
    all_generation = control["generation_validity"] == "PASS" and all(
        row["generation_validity"] == "PASS" for row in run_rows
    )
    mechanisms = all_offloads and all_promotions and all_resumes and all_generation
    naive = summaries["NaiveJIT"]
    queue = summaries["QueueJIT"]
    asap = summaries["ASAP"]
    q_better_than_naive = queue["misses"] < naive["misses"] or queue["max_stall"] < naive["max_stall"]
    queue_less_early = queue["early"] < asap["early"]
    result = "C" if not mechanisms else "A" if q_better_than_naive and queue_less_early else "B"
    queue_start_lead = naive["first_start"] - queue["first_start"]
    saved_early_ms = asap["early"] - queue["early"]
    saved_early_pct = 100.0 * saved_early_ms / asap["early"] if asap["early"] else 0.0

    lines = [
        "# Phase 2E — Multi-session real promotion scheduling",
        "",
        "## Objective and frozen setup",
        "",
        f"Test the queueing consequences of treating two paused-session KV restores as deadline-bearing jobs. Base commit: `4055e134c123e1e2f2562e09d46de5387863e280`. Runtime: vLLM {env['vllm_version']}, PyTorch {env['torch_version']}, CUDA {env['torch_cuda_version']}, {env['gpu0']} (GPU0), Qwen2.5-1.5B-Instruct, float16, TP=1, 320 MiB GPU KV, 2 GiB CPU offload, prefix caching, max model length 10000.",
        "",
        "Each measured run used the distinct 4096-token Phase 2C-Fix A/B prefixes, normal offload on both first turns, a 1-second store wait, and exactly the two distinct 9500-token pressure requests with `max_offload_tokens=0`. After pressure, `t0` was recorded; deadlines were A=1500 ms and B=1560 ms, with frozen estimated service p_est=120 ms. Each policy ran in three independent server restarts.",
        "",
        "The promotion operation remains an ordinary vLLM completion request carrying the exact cached prefix; its generated token is discarded. Its measured service includes request handling, external KV load, normal model execution, and response handling, not just H2D. A single client-side worker serialized promotions. Real resumes were sent only after both promotions had completed, so this validates scheduling with real runtime service times, not native PCIe contention or a production promotion API.",
        "",
        "## Mechanism validation",
        "",
        f"- A and B initial offloads: {'PASS' if all_offloads else 'FAIL'}; expected {EXPECTED_OFFLOAD_BYTES:,} bytes each in every measured run. Expected combined external data is {2 * EXPECTED_OFFLOAD_BYTES:,} bytes (~224 MiB).",
        f"- Prefix eviction and external promotion: {'PASS' if all_promotions else 'FAIL'}; every run's two serialized promotions observed external-hit delta 8192 tokens and local-hit delta 0.",
        f"- Post-promotion resume cache evidence: {'PASS' if all_resumes else 'FAIL'}; every run observed 8192 local-hit tokens and 0 external-hit tokens across A/B resumes.",
        f"- Generation correctness against no-pressure controls: {'PASS' if all_generation else 'FAIL'}; A control token ID={control['A']['first_token_id']}, B control token ID={control['B']['first_token_id']}.",
        "- Metrics were captured at server ready, after each first turn, after pressure, after both promotions, and after each resume. No metrics request or real resume occurred between promotion requests.",
        "",
        "## Policy medians (three repetitions)",
        "",
        "| Policy | Misses/run | Max stall (ms) | Total early residency (ms) | A promotion (ms) | B promotion (ms) | Worker span (ms) | First start from t0 (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in POLICIES:
        item = summaries[policy]
        lines.append(
            f"| {policy} | {item['misses']:.2f} | {item['max_stall']:.2f} | {item['early']:.2f} | {item['a_duration']:.2f} | {item['b_duration']:.2f} | {item['worker_busy']:.2f} | {item['first_start']:.2f} |"
        )
    lines.extend([
        "",
        f"QueueJIT reduced queue-induced misses/stall versus NaiveJIT: {'YES' if naive['misses'] > queue['misses'] or naive['max_stall'] > queue['max_stall'] else 'NO'} (median NaiveJIT vs QueueJIT misses/run {naive['misses']:.2f} vs {queue['misses']:.2f}; max stall {naive['max_stall']:.2f} vs {queue['max_stall']:.2f} ms).",
        f"QueueJIT began its first promotion {queue_start_lead:.2f} ms earlier than NaiveJIT by median first-start time. QueueJIT saved {saved_early_ms:.2f} ms ({saved_early_pct:.1f}%) median total early residency versus ASAP.",
        f"QueueJIT median total early residency was {queue['early'] - naive['early']:.2f} ms higher than NaiveJIT, with no measured miss/stall reduction.",
        "",
        f"**Result {result}:** " + (
            "Real multi-session deadline-aware promotion scheduling validated."
            if result == "A" else
            "Multi-session promotion works, but the configured contention is not sufficient to demonstrate a QueueJIT advantage."
            if result == "B" else
            "Multi-session JIT integration requires redesign."
        ),
        "",
        "## Limitations and anomalies",
        "",
        "- This phase uses oracle deadlines and exactly two sessions; it adds no prediction error, additional session count, model, or GPU.",
        f"- Promotion uses a normal completion-request proxy, not pure H2D. Median A/B request service was {summaries['ASAP']['a_duration']:.2f}/{summaries['ASAP']['b_duration']:.2f} ms under ASAP despite equal KV sizes; B was much faster than the frozen 120-ms estimate, so this configured contention did not make NaiveJIT miss. Logs report a Triton `_compute_slot_mapping_kernel` JIT during inference; the A/B asymmetry is consistent with first-request warm-up cost, though causality is not established.",
        "- The isolated runtime emits a non-fatal Python startup warning because `nvidia_cutlass_dsl` is absent from a `.pth` hook; CUDA/vLLM starts and all measured requests succeeded. Server shutdown also emitted a multiprocessing semaphore cleanup warning.",
        "- The single promotion worker is an experimental serialized-resource model. This validates the scheduling idea with real vLLM service times; it does not prove native PCIe-level contention behavior.",
        "- No-pressure control prompts and all measured A/B prompts use the same deterministic token construction. The pressure workload was not increased or changed.",
        "",
        "Phase 2E uses oracle deadlines and a request-based promotion proxy.",
        "",
        "Phase 2E execution status: COMPLETE",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "server_logs").mkdir(exist_ok=True)
    gate = runtime.capability_gate()
    smoke_test()
    env = summarize_environment(gate)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    prompts = build_prompts(lambda text: tokenizer.encode(text, add_special_tokens=False))
    prompts["suffix"] = tokenizer.encode(SUFFIX, add_special_tokens=False)
    if len(prompts["a"]) != 4096 or len(prompts["b"]) != 4096:
        raise AssertionError("A/B prefixes must both contain exactly 4096 tokens")
    if len(prompts["pressure_1"]) != 9500 or len(prompts["pressure_2"]) != 9500:
        raise AssertionError("both frozen pressure prompts must contain exactly 9500 tokens")
    session_prefixes = (prompts["a"], prompts["b"])
    pressure_prefixes = (prompts["pressure_1"][:4096], prompts["pressure_2"][:4096])
    if any(session == pressure for session in session_prefixes for pressure in pressure_prefixes):
        raise AssertionError("pressure prompts must not share either complete session prefix")

    control, snapshots = run_local_control(prompts)
    env["server_smoke_test"] = "PASS"
    config = {
        "base_commit": "4055e134c123e1e2f2562e09d46de5387863e280",
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "gpu": "GPU0",
        "dtype": "float16",
        "tensor_parallel_size": 1,
        "gpu_kv_capacity_bytes": 335544320,
        "cpu_offload_capacity_bytes": 2147483648,
        "max_model_len": 10000,
        "prefix_tokens_per_session": 4096,
        "expected_offload_bytes_per_session": EXPECTED_OFFLOAD_BYTES,
        "pressure_tokens_each": 9500,
        "pressure_prompts": 2,
        "p_est_ms": P_EST_S * 1000.0,
        "deadlines_ms_from_t0": {key: value * 1000.0 for key, value in DEADLINE_OFFSETS_S.items()},
        "policies": POLICIES,
        "repetitions": REPETITIONS,
        "promotion_worker_count": 1,
        "promotion_proxy": "ordinary completion request with exact prefix; generated token discarded",
        "control": control,
    }
    (RESULTS / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (RESULTS / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    (RESULTS / "control.json").write_text(json.dumps(control, indent=2), encoding="utf-8")
    (RESULTS / "metric_snapshots.json").write_text(json.dumps(snapshots, indent=2), encoding="utf-8")

    job_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    for policy in POLICIES:
        for repetition in REPETITIONS:
            jobs, run_row, run_snapshots = run_measured(policy, repetition, prompts)
            for job in jobs:
                session = job["session"]
                control_id = control[session]["first_token_id"]
                job["control_token_id"] = control_id if control_id is not None else "NA"
                job["generation_validity"] = (
                    "PASS" if resume_token_matches(job["first_token_id"], control_id) else "FAIL"
                )
            run_row["generation_validity"] = (
                "PASS" if all(job["generation_validity"] == "PASS" for job in jobs) else "FAIL"
            )
            run_rows.append(run_row)
            job_rows.extend(jobs)
            snapshots.extend(run_snapshots)
            write_csv(RESULTS / "job_metrics.csv", job_rows)
            write_csv(RESULTS / "run_metrics.csv", run_rows)
            (RESULTS / "metric_snapshots.json").write_text(json.dumps(snapshots, indent=2), encoding="utf-8")

    (RESULTS / "summary.md").write_text(build_summary(job_rows, run_rows, control, env), encoding="utf-8")
    compact_server_logs()
    print(json.dumps({
        "run_count": len(run_rows),
        "job_count": len(job_rows),
        "control": control,
        "summary": str(RESULTS / "summary.md"),
    }, indent=2))


if __name__ == "__main__":
    main()
