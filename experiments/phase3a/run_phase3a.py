"""Run Phase 3A native KV promotion and six-session workload."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import signal
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from typing import Any

from transformers import AutoTokenizer

from experiments.phase2d import run_phase2d as runtime
from experiments.phase2d.helpers import (
    gpu_to_cpu_offload_bytes,
    parse_prometheus_samples,
    source_tokens_delta,
)
from experiments.phase2h.run_phase2h import SERVER, build_prompts
from experiments.phase3a.helpers import percentile_linear, token_matches, write_csv
from experiments.phase3a.native_client import native_prefetch
from experiments.phase3a.scheduler import (
    DEADLINE_MS,
    SESSIONS,
    effective_resume_latency_ms,
    freeze_p_native_ms,
    memory_time_mib_s,
    naive_target_start_ms,
    queue_safe_idle_s,
)

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "phase3a"
VLLM_SOURCE = Path("/media/a6000/D/ZhaoBinHe/slackkv-phase2c-fix-sourceprobe")
BASE_COMMIT = "e4199eccaf40770ce5df9093b0fdf358831be3f8"
VLLM_COMMIT = "0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665"
EXPECTED_BYTES = 117_440_512
EXPECTED_TOKENS = 4096
POLICIES = ("OnDemand", "ASAP-Native", "NaiveJIT-Native", "QueueJIT-Native")
NATIVE_POLICIES = POLICIES[1:]
REPETITIONS = (0, 1, 2)
EPSILON_S = 1e-6
OLD_PROXY_P95_MS = 65.01
OLD_PROXY_MEMORY_MIB_S = 9.23
OLD_ONDEMAND_P95_MS = 57.51
SNAPSHOT_PREFIXES = (
    "vllm:prompt_tokens_by_source_",
    "vllm:kv_offload_",
    "vllm:prefix_cache_",
    "vllm:external_prefix_cache_",
    "vllm:kv_cache_usage_perc",
    "vllm:cache_config_info",
)


def start_server(log_name: str) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env.update(runtime.BASE_ENV)
    env["VLLM_SERVER_DEV_MODE"] = "1"
    env["PYTHONPATH"] = f"{VLLM_SOURCE}:{ROOT}"
    path = RESULTS / "server_logs" / log_name
    with path.open("w", encoding="utf-8") as log:
        return subprocess.Popen(
            SERVER, env=env, stdout=log, stderr=subprocess.STDOUT, text=True
        )


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
    selected: list[str] = []
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
    snapshots.append(
        {
            "stage": stage,
            "captured_at_unix_s": time.time(),
            "raw_relevant": compact_metric_lines(raw),
        }
    )
    return parse_prometheus_samples(raw)


def send(
    prompt: list[int],
    *,
    stream: bool = False,
    normal_offload: bool = False,
    return_token_ids: bool = False,
) -> dict[str, Any]:
    return runtime.send_request(
        prompt,
        stream=stream,
        keep_local=not normal_offload,
        return_token_ids=return_token_ids,
    )


def make_prompts() -> dict[str, list[int]]:
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct", local_files_only=True
    )
    prompts = build_prompts(tokenizer)
    for index in range(1, 4):
        text = f"Calibration C{index} distinct ordinary project context. "
        prompts[f"c{index}"] = tokenizer.encode(
            text * 1000, add_special_tokens=False
        )[:4096]
    if any(len(prompts[f"c{i}"]) != 4096 for i in range(1, 4)):
        raise AssertionError("calibration prefixes must contain exactly 4096 tokens")
    keys = ("w", "a", "b", "c", "d", "e", "f", "c1", "c2", "c3")
    if len({tuple(prompts[key]) for key in keys}) != len(keys):
        raise AssertionError("all native-prefetch prefixes must be distinct")
    return prompts


def run_control(
    prompts: dict[str, list[int]], snapshots: list[dict[str, Any]]
) -> dict[str, Any]:
    proc = start_server("local_hit_control.log")
    try:
        runtime.wait_ready(proc)
        capture("control_server_ready", snapshots)
        for session in SESSIONS:
            send(prompts[session.lower()])
        time.sleep(1.0)
        before = capture("control_before_resumes", snapshots)
        result: dict[str, Any] = {}
        for session in SESSIONS:
            resume = send(
                prompts[session.lower()] + prompts["suffix"],
                stream=True,
                return_token_ids=True,
            )
            result[session] = {
                "first_token_id": resume["first_token_id"],
                "ttft_ms": resume["ttft_ms"],
            }
        after = capture("control_after_resumes", snapshots)
        result["aggregate_local_hit_tokens"] = source_tokens_delta(
            before, after, "local_cache_hit"
        )
        result["aggregate_external_hit_tokens"] = source_tokens_delta(
            before, after, "external_kv_transfer"
        )
        result["ordinary_completion_validation"] = (
            "PASS"
            if all(result[s]["first_token_id"] is not None for s in SESSIONS)
            and result["aggregate_local_hit_tokens"] == 24576.0
            and result["aggregate_external_hit_tokens"] == 0.0
            else "FAIL"
        )
        return result
    finally:
        stop_server(proc)


def apply_pressure(prompts: dict[str, list[int]]) -> None:
    for index in range(1, 5):
        send(prompts[f"pressure_{index}"])


def run_native_sanity(
    prompts: dict[str, list[int]],
    control: dict[str, Any],
    snapshots: list[dict[str, Any]],
) -> dict[str, Any]:
    proc = start_server("native_sanity.log")
    try:
        runtime.wait_ready(proc)
        before_first = capture("sanity_server_ready", snapshots)
        first = send(prompts["a"], normal_offload=True)
        time.sleep(1.0)
        after_first = capture("sanity_after_A_first", snapshots)
        offload_bytes = gpu_to_cpu_offload_bytes(before_first, after_first)
        apply_pressure(prompts)
        capture("sanity_after_pressure", snapshots)
        native = native_prefetch(prompts["a"])
        after_prefetch = capture("sanity_after_native_prefetch", snapshots)
        resume = send(
            prompts["a"] + prompts["suffix"],
            stream=True,
            return_token_ids=True,
        )
        after_resume = capture("sanity_after_A_resume", snapshots)
        local_hit = source_tokens_delta(
            after_prefetch, after_resume, "local_cache_hit"
        )
        external_hit = source_tokens_delta(
            after_prefetch, after_resume, "external_kv_transfer"
        )
        result = {
            "offload_bytes": offload_bytes,
            "A_first_e2e_ms": first["e2e_ms"],
            "external_tokens_loaded": native["external_tokens_loaded"],
            "transfer_bytes": native["transfer_bytes"],
            "native_prefetch_e2e_ms": native["elapsed_ms"],
            "native_prefetch_client_e2e_ms": native["client_e2e_ms"],
            "cpu_to_gpu_transfer_ms": native["cpu_to_gpu_transfer_ms"],
            "model_forward_count": native["model_forward_count"],
            "generated_token_count": native["generated_token_count"],
            "post_prefetch_local_hit_tokens": local_hit,
            "post_prefetch_external_hit_tokens": external_hit,
            "resume_ttft_ms": resume["ttft_ms"],
            "first_token_id": resume["first_token_id"],
            "control_token_id": control["A"]["first_token_id"],
            "token_correctness": (
                "PASS"
                if token_matches(resume["first_token_id"], control["A"]["first_token_id"])
                else "FAIL"
            ),
            "max_offload_tokens_zero_validation": (
                "PASS" if local_hit == 4096.0 and external_hit == 0.0 else "FAIL"
            ),
        }
        result["native_sanity_validation"] = (
            "PASS"
            if offload_bytes == EXPECTED_BYTES
            and native["external_tokens_loaded"] == EXPECTED_TOKENS
            and native["transfer_bytes"] == EXPECTED_BYTES
            and native["model_forward_count"] == 0
            and native["generated_token_count"] == 0
            and local_hit == EXPECTED_TOKENS
            and external_hit == 0.0
            and result["token_correctness"] == "PASS"
            else "FAIL"
        )
        return result
    finally:
        stop_server(proc)


def run_calibration(
    prompts: dict[str, list[int]], snapshots: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    proc = start_server("native_calibration.log")
    try:
        runtime.wait_ready(proc)
        previous = capture("calibration_server_ready", snapshots)
        offloads: dict[str, float] = {}
        for key in ("w", "c1", "c2", "c3"):
            send(prompts[key], normal_offload=True)
            current = capture(f"calibration_after_{key}_first", snapshots)
            offloads[key] = gpu_to_cpu_offload_bytes(previous, current)
            previous = current
        if any(value != EXPECTED_BYTES for value in offloads.values()):
            raise AssertionError(f"calibration offload mismatch: {offloads}")
        time.sleep(1.0)
        apply_pressure(prompts)
        capture("calibration_after_pressure", snapshots)
        warm = native_prefetch(prompts["w"])
        if warm["external_tokens_loaded"] != EXPECTED_TOKENS:
            raise AssertionError("native warm-up did not recover W")
        rows: list[dict[str, Any]] = []
        for index in range(1, 4):
            result = native_prefetch(prompts[f"c{index}"])
            rows.append(
                {
                    "calibration_index": index,
                    "session": f"C{index}",
                    "native_prefetch_e2e_ms": result["elapsed_ms"],
                    "native_prefetch_client_e2e_ms": result["client_e2e_ms"],
                    "cpu_to_gpu_transfer_ms": result["cpu_to_gpu_transfer_ms"],
                    "transfer_bytes": result["transfer_bytes"],
                    "external_tokens_loaded": result["external_tokens_loaded"],
                    "model_forward_count": result["model_forward_count"],
                    "generated_token_count": result["generated_token_count"],
                    "status": result["status"],
                }
            )
        capture("calibration_after_three_native_prefetches", snapshots)
        return rows
    finally:
        stop_server(proc)


def environment() -> dict[str, Any]:
    import torch
    import vllm

    return {
        "slackkv_base_commit": BASE_COMMIT,
        "vllm_version": vllm.__version__,
        "vllm_commit": VLLM_COMMIT,
        "vllm_source": str(VLLM_SOURCE),
        "python_version": os.sys.version.split()[0],
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu0": torch.cuda.get_device_name(0),
        "gpu0_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "driver_version": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).splitlines()[0],
        "transformers_version": importlib.metadata.version("transformers"),
        "installation_method": "reused isolated Phase 2C-Fix CUDA 12.6 vLLM source build",
    }


def sanity_stage() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "server_logs").mkdir(exist_ok=True)
    prompts = make_prompts()
    snapshots: list[dict[str, Any]] = []
    control = run_control(prompts, snapshots)
    sanity = run_native_sanity(prompts, control, snapshots)
    calibration = run_calibration(prompts, snapshots)
    p_native = freeze_p_native_ms(
        [float(row["native_prefetch_e2e_ms"]) for row in calibration]
    )
    config = {
        "phase": "3A",
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "gpu_kv_bytes": 1_073_741_824,
        "cpu_offload_bytes": 2_147_483_648,
        "prefix_tokens": 4096,
        "kv_mib_per_session": 112,
        "deadlines_ms": DEADLINE_MS,
        "policies": POLICIES,
        "repetitions": list(REPETITIONS),
        "p_native_ms": p_native,
        "p_native_rule": "ceil(max(three warm native-prefetch endpoint E2E times))",
        "pressure_prompt_tokens": [9500, 9500, 9500, 9500],
        "native_endpoint": "/slackkv/prefetch",
    }
    (RESULTS / "control.json").write_text(
        json.dumps(control, indent=2), encoding="utf-8"
    )
    (RESULTS / "sanity_metrics.json").write_text(
        json.dumps(sanity, indent=2), encoding="utf-8"
    )
    write_csv(RESULTS / "calibration_metrics.csv", calibration)
    (RESULTS / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    (RESULTS / "environment.json").write_text(
        json.dumps(environment(), indent=2), encoding="utf-8"
    )
    (RESULTS / "metric_snapshots.json").write_text(
        json.dumps(snapshots, indent=2), encoding="utf-8"
    )
    if control["ordinary_completion_validation"] != "PASS":
        raise AssertionError("ordinary completion regression failed")
    if sanity["native_sanity_validation"] != "PASS":
        raise AssertionError("native sanity gate failed")
    print(f"SANITY PASS p_native={p_native}ms")


def prepare_workload(
    label: str,
    prompts: dict[str, list[int]],
    snapshots: list[dict[str, Any]],
) -> tuple[dict[Any, float], dict[str, Any]]:
    previous = capture(f"{label}_server_ready", snapshots)
    offloads: dict[str, float] = {}
    for session in ("W", *SESSIONS):
        send(prompts[session.lower()], normal_offload=True)
        current = capture(f"{label}_after_{session}_first", snapshots)
        offloads[session] = gpu_to_cpu_offload_bytes(previous, current)
        previous = current
    time.sleep(1.0)
    apply_pressure(prompts)
    capture(f"{label}_after_pressure", snapshots)
    warm = native_prefetch(prompts["w"])
    before_t0 = capture(f"{label}_after_W_native_warmup_before_t0", snapshots)
    return before_t0, {
        "offloads": offloads,
        "offload_validation": (
            "PASS"
            if all(value == EXPECTED_BYTES for value in offloads.values())
            else "FAIL"
        ),
        "W_warmup": warm,
        "W_warmup_validation": (
            "PASS"
            if warm["external_tokens_loaded"] == EXPECTED_TOKENS
            and warm["model_forward_count"] == 0
            and warm["generated_token_count"] == 0
            else "FAIL"
        ),
    }


def schedule_native_promotions(
    policy: str,
    t0: float,
    prompts: dict[str, list[int]],
    events: dict[str, Event],
    p_native_ms: int,
) -> dict[str, dict[str, Any]]:
    deadlines = {
        session: t0 + DEADLINE_MS[session] / 1000.0 for session in SESSIONS
    }
    records: dict[str, dict[str, Any]] = {}

    def promote(session: str, target: float) -> None:
        result = native_prefetch(prompts[session.lower()])
        records[session] = {"target_abs_s": target, **result}
        events[session].set()

    if policy == "ASAP-Native":
        for session in SESSIONS:
            promote(session, t0)
        return records
    if policy == "NaiveJIT-Native":
        targets = {
            session: t0
            + naive_target_start_ms(DEADLINE_MS[session], p_native_ms) / 1000.0
            for session in SESSIONS
        }
        for session in sorted(
            SESSIONS, key=lambda item: (targets[item], deadlines[item], item)
        ):
            runtime.wait_until(targets[session])
            promote(session, targets[session])
        return records
    if policy != "QueueJIT-Native":
        raise ValueError(f"unexpected native policy {policy}")
    pending = set(SESSIONS)
    while pending:
        now = time.monotonic()
        pending_deadlines = {session: deadlines[session] for session in pending}
        safe_idle = queue_safe_idle_s(now, pending_deadlines, p_native_ms)
        if safe_idle > EPSILON_S:
            runtime.wait_until(now + safe_idle)
            continue
        head = min(pending, key=lambda item: (deadlines[item], item))
        promote(head, now)
        pending.remove(head)
    return records


def dispatch_resume(
    session: str,
    t0: float,
    prompts: dict[str, list[int]],
    promotion_event: Event | None,
) -> dict[str, Any]:
    deadline = t0 + DEADLINE_MS[session] / 1000.0
    runtime.wait_until(deadline)
    waited = False
    if promotion_event is not None and not promotion_event.is_set():
        waited = True
        if not promotion_event.wait(timeout=120):
            raise TimeoutError(f"native prefetch for {session} did not finish")
    result = send(
        prompts[session.lower()] + prompts["suffix"],
        stream=True,
        return_token_ids=True,
    )
    first_token_s = result["sent_monotonic_s"] + result["ttft_ms"] / 1000.0
    return {
        **result,
        "effective_resume_latency_ms": effective_resume_latency_ms(
            first_token_s, deadline
        ),
        "waited_for_prefetch": waited,
    }


def execute_timeline(
    policy: str,
    t0: float,
    prompts: dict[str, list[int]],
    p_native_ms: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    events = (
        {session: Event() for session in SESSIONS}
        if policy in NATIVE_POLICIES
        else {}
    )
    with ThreadPoolExecutor(max_workers=7, thread_name_prefix="phase3a") as pool:
        promotion_future = (
            pool.submit(
                schedule_native_promotions,
                policy,
                t0,
                prompts,
                events,
                p_native_ms,
            )
            if policy in NATIVE_POLICIES
            else None
        )
        resume_futures = {
            session: pool.submit(
                dispatch_resume, session, t0, prompts, events.get(session)
            )
            for session in SESSIONS
        }
        resumes = {
            session: future.result(timeout=180)
            for session, future in resume_futures.items()
        }
        promotions = (
            promotion_future.result(timeout=180) if promotion_future else {}
        )
    return promotions, resumes


def run_workload_once(
    policy: str,
    repetition: int,
    prompts: dict[str, list[int]],
    control: dict[str, Any],
    p_native_ms: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
]:
    label = f"{policy.lower().replace('-', '_')}_rep{repetition}"
    proc = start_server(f"{label}.log")
    snapshots: list[dict[str, Any]] = []
    try:
        runtime.wait_ready(proc)
        before, setup = prepare_workload(label, prompts, snapshots)
        t0 = time.monotonic()
        promotions, resumes = execute_timeline(policy, t0, prompts, p_native_ms)
        # First metric scrape after t0 and all six resumes.
        after = capture(f"{label}_after_final_resume", snapshots)
        official_external = source_tokens_delta(
            before, after, "external_kv_transfer"
        )
        official_local = source_tokens_delta(before, after, "local_cache_hit")
        native_rows: list[dict[str, Any]] = []
        session_rows: list[dict[str, Any]] = []
        for session in SESSIONS:
            deadline = DEADLINE_MS[session]
            promotion = promotions.get(session)
            if promotion is None:
                completion = None
                early = 0.0
            else:
                completion = (
                    promotion["completed_monotonic_s"] - t0
                ) * 1000.0
                delta = completion - deadline
                early = max(0.0, -delta)
                native_rows.append(
                    {
                        "policy": policy,
                        "repetition": repetition,
                        "session": session,
                        "target_ms": (
                            promotion["target_abs_s"] - t0
                        )
                        * 1000.0,
                        "actual_start_ms": (
                            promotion["sent_monotonic_s"] - t0
                        )
                        * 1000.0,
                        "completion_ms": completion,
                        "native_prefetch_e2e_ms": promotion["elapsed_ms"],
                        "native_prefetch_client_e2e_ms": promotion[
                            "client_e2e_ms"
                        ],
                        "cpu_to_gpu_transfer_ms": promotion[
                            "cpu_to_gpu_transfer_ms"
                        ],
                        "transfer_bytes": promotion["transfer_bytes"],
                        "model_forward_count": promotion[
                            "model_forward_count"
                        ],
                        "generated_token_count": promotion[
                            "generated_token_count"
                        ],
                        "deadline_stall_ms": max(0.0, delta),
                        "early_residency_ms": early,
                        "status": promotion["status"],
                    }
                )
            resume = resumes[session]
            session_rows.append(
                {
                    "policy": policy,
                    "repetition": repetition,
                    "session": session,
                    "deadline_ms": deadline,
                    "prefetch_completed_before_deadline": (
                        "NA" if completion is None else completion <= deadline
                    ),
                    "waited_for_prefetch": resume["waited_for_prefetch"],
                    "resume_request_sent_ms": (
                        resume["sent_monotonic_s"] - t0
                    )
                    * 1000.0,
                    "resume_ttft_ms": resume["ttft_ms"],
                    "effective_resume_latency_ms": resume[
                        "effective_resume_latency_ms"
                    ],
                    # No intrusive per-request scrape is allowed on this timeline.
                    "local_hit_tokens": "NA",
                    "external_hit_tokens": "NA",
                    "first_token_id": resume["first_token_id"],
                    "control_token_id": control[session]["first_token_id"],
                    "generation_validity": (
                        "PASS"
                        if token_matches(
                            resume["first_token_id"],
                            control[session]["first_token_id"],
                        )
                        else "FAIL"
                    ),
                }
            )
        effective = [
            float(row["effective_resume_latency_ms"]) for row in session_rows
        ]
        early_values = [
            float(row["early_residency_ms"]) for row in native_rows
        ]
        native_external_tokens = sum(
            EXPECTED_TOKENS
            for row in native_rows
            if row["transfer_bytes"] == EXPECTED_BYTES
        )
        expected_official = (
            (24576.0, 0.0)
            if policy == "OnDemand"
            else (0.0, 24576.0)
        )
        native_valid = policy == "OnDemand" or (
            len(native_rows) == 6
            and native_external_tokens == 24576
            and all(
                row["model_forward_count"] == 0
                and row["generated_token_count"] == 0
                and row["transfer_bytes"] == EXPECTED_BYTES
                for row in native_rows
            )
        )
        cache_valid = (
            official_external == expected_official[0]
            and official_local == expected_official[1]
            and native_valid
        )
        run = {
            "policy": policy,
            "repetition": repetition,
            "native_prefetch_deadline_miss_count": sum(
                float(row["deadline_stall_ms"]) > 0 for row in native_rows
            ),
            "P50_effective_resume_latency_ms": percentile_linear(effective, 50),
            "P95_effective_resume_latency_ms": percentile_linear(effective, 95),
            "max_effective_resume_latency_ms": max(effective),
            "total_early_residency_ms": sum(early_values),
            "idle_gpu_memory_time_mib_s": memory_time_mib_s(early_values),
            "median_native_prefetch_ms": (
                statistics.median(
                    float(row["native_prefetch_e2e_ms"]) for row in native_rows
                )
                if native_rows
                else "NA"
            ),
            "max_native_prefetch_ms": (
                max(
                    float(row["native_prefetch_e2e_ms"]) for row in native_rows
                )
                if native_rows
                else "NA"
            ),
            "median_cpu_to_gpu_transfer_ms": (
                statistics.median(
                    float(row["cpu_to_gpu_transfer_ms"]) for row in native_rows
                )
                if native_rows
                else "NA"
            ),
            "native_prefetch_model_forward_total": sum(
                int(row["model_forward_count"]) for row in native_rows
            ),
            "native_prefetch_generated_tokens_total": sum(
                int(row["generated_token_count"]) for row in native_rows
            ),
            "native_external_tokens_loaded": native_external_tokens,
            "official_external_hit_tokens": official_external,
            "official_local_hit_tokens": official_local,
            "cache_validation": "PASS" if cache_valid else "FAIL",
            "generation_validation": (
                "PASS"
                if all(
                    row["generation_validity"] == "PASS"
                    for row in session_rows
                )
                else "FAIL"
            ),
            "timeline_validation": (
                "PASS"
                if all(
                    float(row["resume_request_sent_ms"]) + 1e-6
                    >= float(row["deadline_ms"])
                    for row in session_rows
                )
                else "FAIL"
            ),
            "offload_validation": setup["offload_validation"],
            "W_warmup_validation": setup["W_warmup_validation"],
            "W_warmup_e2e_ms": setup["W_warmup"]["elapsed_ms"],
            "offload_bytes_json": json.dumps(
                setup["offloads"], sort_keys=True
            ),
        }
        return native_rows, session_rows, run, snapshots
    finally:
        stop_server(proc)


def policy_medians(
    runs: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    fields = (
        "native_prefetch_deadline_miss_count",
        "P50_effective_resume_latency_ms",
        "P95_effective_resume_latency_ms",
        "max_effective_resume_latency_ms",
        "total_early_residency_ms",
        "idle_gpu_memory_time_mib_s",
    )
    return {
        policy: {
            field: statistics.median(
                float(row[field])
                for row in runs
                if row["policy"] == policy
            )
            for field in fields
        }
        for policy in POLICIES
    }


def build_summary(
    sanity: dict[str, Any],
    calibration: list[dict[str, Any]],
    config: dict[str, Any],
    runs: list[dict[str, Any]],
    medians: dict[str, dict[str, float]],
    result: str,
) -> str:
    ondemand = medians["OnDemand"]
    asap = medians["ASAP-Native"]
    naive = medians["NaiveJIT-Native"]
    queue = medians["QueueJIT-Native"]
    memory_saving = 100.0 * (
        1.0
        - queue["idle_gpu_memory_time_mib_s"]
        / asap["idle_gpu_memory_time_mib_s"]
    )
    warm = [float(row["native_prefetch_e2e_ms"]) for row in calibration]
    all_valid = all(
        row["cache_validation"] == "PASS"
        and row["generation_validation"] == "PASS"
        and row["timeline_validation"] == "PASS"
        and row["offload_validation"] == "PASS"
        and row["W_warmup_validation"] == "PASS"
        for row in runs
    )
    conclusion = {
        "A": "Native QueueJIT establishes a latency–memory tradeoff superior to OnDemand on this workload.",
        "B": "Native promotion removes proxy overhead, but OnDemand remains competitive at this KV size and hardware regime.",
        "C": "Native promotion integration is not correct.",
        "D": "Native promotion works in isolation but requires concurrency/lifecycle fixes.",
    }[result]
    return f"""# Phase 3A — Native KV promotion

## Objective

Replace the completion-request promotion proxy with a research-only native CPU-external-to-GPU KV load that executes zero transformer forward and generates zero tokens, then repeat the frozen six-session workload.

## Setup

SlackKV base `{BASE_COMMIT}`; pinned vLLM `{VLLM_COMMIT}` (0.23.0), PyTorch 2.11.0+cu126, CUDA 12.6, RTX A6000 GPU0, Qwen2.5-1.5B-Instruct, float16, TP=1, 1 GiB GPU KV, 2 GiB CPU offload, prefix caching, max model length 10000. A–F are distinct 4096-token prefixes with oracle deadlines 300/320/340/650/900/1300 ms. Each policy has three independent server restarts.

The local research endpoint routes the normal prompt identity through OffloadingConnector lookup/allocation/load metadata. Its load-only engine step enters `kv_connector_no_forward`; after transfer completion the scheduler caches the loaded blocks, finishes the special request without sampling, and frees its request ownership while leaving prefix-cache blocks reusable.

## Native sanity

- External tokens loaded: {sanity['external_tokens_loaded']}
- Transfer bytes: {sanity['transfer_bytes']:,}
- Native endpoint E2E: {sanity['native_prefetch_e2e_ms']:.2f} ms
- CPU→GPU transfer counter: {sanity['cpu_to_gpu_transfer_ms']:.2f} ms
- Model forwards / generated tokens: {sanity['model_forward_count']} / {sanity['generated_token_count']}
- Subsequent resume local/external hit tokens: {sanity['post_prefetch_local_hit_tokens']:.0f} / {sanity['post_prefetch_external_hit_tokens']:.0f}
- First-token correctness: {sanity['token_correctness']}
- Sanity gate: {sanity['native_sanity_validation']}

## Calibration

Three warm native endpoint E2E times: {warm[0]:.2f}, {warm[1]:.2f}, {warm[2]:.2f} ms. Frozen `p_native` is {config['p_native_ms']} ms (`ceil(max)`), set before the workload.

## Six-session results — medians across three repetitions

| Policy | Native deadline misses | P50 effective latency (ms) | P95 effective latency (ms) | Max effective latency (ms) | Idle GPU memory-time (MiB·s) |
|---|---:|---:|---:|---:|---:|
| OnDemand | {ondemand['native_prefetch_deadline_miss_count']:.2f} | {ondemand['P50_effective_resume_latency_ms']:.2f} | {ondemand['P95_effective_resume_latency_ms']:.2f} | {ondemand['max_effective_resume_latency_ms']:.2f} | {ondemand['idle_gpu_memory_time_mib_s']:.2f} |
| ASAP-Native | {asap['native_prefetch_deadline_miss_count']:.2f} | {asap['P50_effective_resume_latency_ms']:.2f} | {asap['P95_effective_resume_latency_ms']:.2f} | {asap['max_effective_resume_latency_ms']:.2f} | {asap['idle_gpu_memory_time_mib_s']:.2f} |
| NaiveJIT-Native | {naive['native_prefetch_deadline_miss_count']:.2f} | {naive['P50_effective_resume_latency_ms']:.2f} | {naive['P95_effective_resume_latency_ms']:.2f} | {naive['max_effective_resume_latency_ms']:.2f} | {naive['idle_gpu_memory_time_mib_s']:.2f} |
| QueueJIT-Native | {queue['native_prefetch_deadline_miss_count']:.2f} | {queue['P50_effective_resume_latency_ms']:.2f} | {queue['P95_effective_resume_latency_ms']:.2f} | {queue['max_effective_resume_latency_ms']:.2f} | {queue['idle_gpu_memory_time_mib_s']:.2f} |

## Direct comparison

- Phase 2H QueueJIT-Proxy reference: P95 {OLD_PROXY_P95_MS:.2f} ms, memory-time {OLD_PROXY_MEMORY_MIB_S:.2f} MiB·s.
- Phase 2H OnDemand reference: P95 {OLD_ONDEMAND_P95_MS:.2f} ms, memory-time 0.
- QueueJIT-Native versus current OnDemand: P95 {queue['P95_effective_resume_latency_ms']:.2f} vs {ondemand['P95_effective_resume_latency_ms']:.2f} ms; memory-time {queue['idle_gpu_memory_time_mib_s']:.2f} vs 0 MiB·s.
- QueueJIT-Native versus old QueueJIT-Proxy: P95 change {queue['P95_effective_resume_latency_ms'] - OLD_PROXY_P95_MS:+.2f} ms; memory-time change {queue['idle_gpu_memory_time_mib_s'] - OLD_PROXY_MEMORY_MIB_S:+.2f} MiB·s.
- QueueJIT-Native saves {memory_saving:.1f}% idle memory-time versus ASAP-Native.
- All workload cache/generation/timeline/offload validations: {'PASS' if all_valid else 'FAIL'}.

## Anomalies and limitations

The endpoint is research-only and local. It preserves normal OffloadingConnector allocation, bookkeeping, transfer, completion, and prefix-cache lifecycle, but is not a production API. Deadlines remain oracle values. The client-side promotion worker remains serialized. No prediction error, TraceLab replay, second GPU, model/context sweep, or Phase 3B work is included.

## Conclusion

**Result {result}:** {conclusion}

Phase 3A execution status: COMPLETE
"""


def compact_logs() -> None:
    keep = (
        "non-default args",
        "Initializing a V1 LLM engine",
        "GPU KV cache size",
        "Creating v1 connector",
        "Creating offloading spec",
        "Allocating a cross layer",
        "Allocating 1 CPU tensors",
        "Starting vLLM server",
        "POST /slackkv/prefetch",
        "POST /v1/completions",
        "SLACKKV_NATIVE",
        "WARNING",
        "ERROR",
        "Traceback",
        "ModuleNotFoundError",
        "shutdown",
    )
    for path in (RESULTS / "server_logs").glob("*.log"):
        lines = path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        path.write_text(
            "\n".join(
                line
                for line in lines
                if any(marker in line for marker in keep)
            )
            + "\n",
            encoding="utf-8",
        )


def workload_stage() -> None:
    config = json.loads(
        (RESULTS / "config.json").read_text(encoding="utf-8")
    )
    sanity = json.loads(
        (RESULTS / "sanity_metrics.json").read_text(encoding="utf-8")
    )
    with (RESULTS / "calibration_metrics.csv").open(encoding="utf-8") as stream:
        calibration = list(csv.DictReader(stream))
    control = json.loads(
        (RESULTS / "control.json").read_text(encoding="utf-8")
    )
    snapshots = json.loads(
        (RESULTS / "metric_snapshots.json").read_text(encoding="utf-8")
    )
    if sanity["native_sanity_validation"] != "PASS":
        raise AssertionError("Part A gate did not pass")
    prompts = make_prompts()
    native_rows: list[dict[str, Any]] = []
    session_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    for policy in POLICIES:
        for repetition in REPETITIONS:
            print(f"RUN {policy} repetition={repetition}", flush=True)
            native, sessions, run, run_snapshots = run_workload_once(
                policy,
                repetition,
                prompts,
                control,
                int(config["p_native_ms"]),
            )
            native_rows.extend(native)
            session_rows.extend(sessions)
            run_rows.append(run)
            snapshots.extend(run_snapshots)
    write_csv(RESULTS / "native_prefetch_metrics.csv", native_rows)
    write_csv(RESULTS / "session_metrics.csv", session_rows)
    write_csv(RESULTS / "run_metrics.csv", run_rows)
    (RESULTS / "metric_snapshots.json").write_text(
        json.dumps(snapshots, indent=2), encoding="utf-8"
    )
    medians = policy_medians(run_rows)
    all_valid = all(
        row["cache_validation"] == "PASS"
        and row["generation_validation"] == "PASS"
        and row["timeline_validation"] == "PASS"
        and row["offload_validation"] == "PASS"
        and row["W_warmup_validation"] == "PASS"
        and int(row["native_prefetch_model_forward_total"]) == 0
        and int(row["native_prefetch_generated_tokens_total"]) == 0
        for row in run_rows
    )
    queue = medians["QueueJIT-Native"]
    ondemand = medians["OnDemand"]
    asap = medians["ASAP-Native"]
    if not all_valid:
        result = "D"
    elif (
        queue["P95_effective_resume_latency_ms"]
        < ondemand["P95_effective_resume_latency_ms"]
        and queue["idle_gpu_memory_time_mib_s"]
        < asap["idle_gpu_memory_time_mib_s"]
    ):
        result = "A"
    else:
        result = "B"
    (RESULTS / "summary.md").write_text(
        build_summary(
            sanity, calibration, config, run_rows, medians, result
        ),
        encoding="utf-8",
    )
    compact_logs()
    print(f"WORKLOAD COMPLETE Result {result}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("sanity", "workload"), required=True
    )
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "server_logs").mkdir(exist_ok=True)
    if args.stage == "sanity":
        sanity_stage()
    else:
        workload_stage()


if __name__ == "__main__":
    main()
