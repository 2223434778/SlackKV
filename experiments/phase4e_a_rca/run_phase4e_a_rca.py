from __future__ import annotations

import hashlib
import csv
import json
import os
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from experiments.phase4d_a import runtime
from experiments.phase4d_a.experiment import materialize_and_evict, write_csv, write_json
from experiments.phase4d_a.run_policy import stream_resume
from experiments.phase4d_b.capacity_client import capacity_prefetch
from experiments.phase4d_b.experiment import reconstructed_data, validate_runtime
from experiments.phase4e_a_fix.run_phase4e_a_fix import warmup, start_background_512
from experiments.phase4e_a_rca.analysis import (
    CONDITIONS, EXPECTED_BLOCKS, GPU_BLOCKS, WORKLOAD_SHA, condition_blocks,
    counters, final_result, first_e04_block, parse_structured_log, r1_result,
    u3_nonprogress, verify_jobs,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/phase4e_a_rca"
START_COMMIT = "e829bee1fb6d45dcdfb7bdce395c5a10154cc892"
WAITING_MARKER = "SLACKKV_WAITING_TRACE"
RESUME_MARKER = "SLACKKV_RESUME_TRACE"
LEASE_MARKER = "SLACKKV_LEASE_TRACE"
CAPACITY_MARKER = "SLACKKV_CAPACITY_TRACE"
WATCHDOG_S = 20.0
U3_OBSERVE_S = 12.0


def append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    previous: list[dict[str, Any]] = []
    if path.exists():
        import csv
        with path.open(newline="", encoding="utf-8") as stream:
            previous = list(csv.DictReader(stream))
    write_csv(path, previous + rows)


def live_events(log: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        "waiting": parse_structured_log(log, WAITING_MARKER),
        "resume": parse_structured_log(log, RESUME_MARKER),
        "lease": parse_structured_log(log, LEASE_MARKER),
        "capacity": parse_structured_log(log, CAPACITY_MARKER),
    }


def wait_until(target_s: float) -> None:
    while (remaining := target_s - time.monotonic()) > 0:
        time.sleep(min(0.05, remaining))


def validate_prefetch(
    episode: str, result: dict[str, Any], job: dict[str, Any],
    leased: bool, log: Path,
) -> None:
    if result.get("status") != "ok":
        raise RuntimeError(f"RUN_INVALID_PRECONDITION:{episode}:{result.get('status')}")
    if int(result.get("external_tokens_loaded", -1)) != int(job["required_prefix_tokens"]):
        raise RuntimeError(f"RUN_INVALID_PRECONDITION:{episode}:loaded tokens")
    if int(result.get("transfer_bytes", 0)) <= 0:
        raise RuntimeError(f"RUN_INVALID_PRECONDITION:{episode}:transfer bytes")
    if any(int(result.get(field, -1)) != 0 for field in (
        "model_forward_count", "generated_token_count", "mixed_prefetch_model_step_count"
    )):
        raise RuntimeError(f"RUN_INVALID_PRECONDITION:{episode}:model work")
    times = result.get("phase3m_timestamps") or {}
    if "T6" not in times or "T7" not in times:
        raise RuntimeError(f"RUN_INVALID_PRECONDITION:{episode}:T6/T7 missing")
    if leased:
        end = time.monotonic() + 3
        while time.monotonic() < end:
            events = live_events(log)["lease"]
            for name in ("LEASE_ACQUIRED", "LEASE_HELD_AFTER_PREFETCH_FREE"):
                if not any(event.get("event") == name
                           and event.get("lease_key") == job["lease_key"]
                           and int(event.get("num_blocks", -1)) == int(job["required_blocks"])
                           for event in events):
                    break
            else:
                return
            time.sleep(0.02)
        raise RuntimeError(f"RUN_INVALID_PRECONDITION:{episode}:lease events missing")


def compact_log(log: Path, label: str) -> None:
    if not log.exists():
        return
    markers = (
        WAITING_MARKER, RESUME_MARKER, LEASE_MARKER, CAPACITY_MARKER,
        "GPU KV cache size", "ERROR", "OOM", "WARNING",
    )
    selected = [
        line[:2500] for line in log.read_text(encoding="utf-8", errors="replace").splitlines()
        if any(marker in line for marker in markers)
    ]
    target = OUT / "server_logs"
    target.mkdir(parents=True, exist_ok=True)
    (target / f"{label}_compact.txt").write_text(
        "\n".join(selected[-1500:]) + "\n", encoding="utf-8"
    )
    log.unlink()


def run_condition(
    stage: str,
    condition: str,
    repetition: int,
    jobs: list[dict[str, Any]],
    data: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    selected, leased = CONDITIONS[condition]
    label = f"{stage}_{condition}_rep{repetition}"
    run_dir = OUT / "runs" / label
    run_dir.mkdir(parents=True, exist_ok=True)
    proc, log = runtime.start_server(label, out=run_dir)
    resume_pool: ThreadPoolExecutor | None = None
    xpool: ThreadPoolExecutor | None = None
    futures: dict[str, Any] = {}
    t0_s: float | None = None
    precondition: list[dict[str, Any]] = []
    x_first_ns: int | None = None
    x_generated_count: int | str = "NA"
    checkpoints: dict[str, dict[str, int]] = {}
    try:
        _, block_size = runtime.wait_ready(proc)
        capacity = validate_runtime(log, block_size)
        if capacity["total_gpu_kv_blocks"] != GPU_BLOCKS:
            raise RuntimeError("EXPERIMENT_INVALID_ENVIRONMENT:GPU blocks")
        warmup(label)
        prepared = materialize_and_evict(proc, log, label, jobs, data, run_dir)
        if not prepared["pressure"]["capacity_exceeded"] or any(
            not row["validation"] for row in prepared["materialization"]
        ):
            raise RuntimeError("RUN_INVALID_PRECONDITION:materialization/pressure")

        if stage == "R2":
            xpool, xfuture, xmeta = start_background_512()
            x_first_ns = int(xmeta["first_ns"])

        # Promotion set is fixed and processed strictly sequentially before resume_t0.
        for episode in selected:
            job = data[episode]["job"]
            result = capacity_prefetch(
                data[episode]["prefix"],
                required_blocks=int(job["required_blocks"]),
                lease_key=job["lease_key"] if leased else None,
                timeout_s=5,
            )
            validate_prefetch(episode, result, job, leased, log)
            precondition.append({
                "episode_id": episode, "lease_key": job["lease_key"] if leased else "NA",
                "required_blocks": int(job["required_blocks"]),
                "status": result["status"],
                "external_tokens_loaded": result["external_tokens_loaded"],
                "transfer_bytes": result["transfer_bytes"],
                "T6_monotonic_s": result["phase3m_timestamps"]["T6"],
                "T7_monotonic_s": result["phase3m_timestamps"]["T7"],
                "client_issue_ns": result["client_issue_ns"],
                "client_return_ns": result["client_return_ns"],
            })
        if stage == "R2" and xfuture.done():
            raise RuntimeError("RUN_INVALID_PRECONDITION:X ended before resume phase")
        write_csv(run_dir / "sequential_precondition.csv", precondition)
        t0_s = time.monotonic()
        states = {
            job["episode_id"]: {"resume_started_ns": None, "first_token_ns": None,
                                "leased": leased and job["episode_id"] in selected}
            for job in jobs
        }
        state_lock = threading.Lock()
        resume_pool = ThreadPoolExecutor(max_workers=len(jobs))
        for episode, entry in data.items():
            futures[episode] = resume_pool.submit(
                stream_resume, entry["job"]["lease_key"], entry, t0_s,
                float(entry["job"]["deadline_ms"]), True, states, state_lock,
            )

        if condition == "U3_UNLEASED_HEAD_LEASE3":
            wait_until(t0_s + 5)
            checkpoints["5s"] = counters(live_events(log)["resume"])
            wait_until(t0_s + U3_OBSERVE_S)
            checkpoints["12s"] = counters(live_events(log)["resume"])
            if checkpoints["12s"]["resume_admitted"] > 0:
                wait(list(futures.values()), timeout=max(0, t0_s + WATCHDOG_S - time.monotonic()))
        else:
            wait(list(futures.values()), timeout=WATCHDOG_S)
        if stage == "R2" and condition != "U3_UNLEASED_HEAD_LEASE3":
            x_generated_count = len(xfuture.result(timeout=WATCHDOG_S)["tokens"])
            if x_generated_count != 512:
                raise RuntimeError("RUN_INVALID_BACKGROUND:X output count")

        snapshot = live_events(log)
        counts = counters(snapshot["resume"])
        waiting = snapshot["waiting"]
        lease = snapshot["lease"]
        first_arrival = next((event.get("session") for event in snapshot["resume"]
                              if event.get("event") == "RESUME_ARRIVED"), "NA")
        block = first_e04_block(waiting)
        first_attempt = next((event for event in waiting if event.get("event") == "WAITING_ATTEMPT"
                              and event.get("session") == "E04:turn:1"), None)
        alive = proc.poll() is None
        raw = log.read_text(encoding="utf-8", errors="replace")
        oom = "cuda out of memory" in raw.lower() or "outofmemoryerror" in raw.lower()
        server_error = " ERROR " in raw
        releases = sum(event.get("event") == "LEASE_RELEASED" for event in lease)
        max_pre_first = max(
            (int(event.get("num_slackkv_pre_first_token") or 0)
             for event in snapshot["resume"]), default=0
        )
        if max_pre_first > 2:
            raise RuntimeError("K2_VALIDATION_FAIL")
        ended = time.monotonic()
        if condition == "U3_UNLEASED_HEAD_LEASE3" and u3_nonprogress(
            alive=alive, oom=oom, server_error=server_error,
            waiting=waiting, resume=snapshot["resume"], lease=lease,
            final_elapsed_s=ended - t0_s,
        ) and first_arrival == "E04:turn:1":
            classification = "LEASE_HOL_NONPROGRESS"
        elif all(future.done() and future.exception() is None for future in futures.values()):
            responses = [future.result() for future in futures.values()]
            e04_admitted = any(
                event.get("session") == "E04:turn:1"
                and event.get("event") == "RESUME_ADMITTED"
                for event in snapshot["resume"]
            )
            e04_handoff = all(
                any(event.get("lease_key") == "E04:turn:1"
                    and event.get("event") == required for event in lease)
                for required in ("LEASE_RELEASE_READY", "LEASE_RELEASED")
            )
            control_ok = e04_admitted and (
                condition != "S3_HEAD_LEASED" or e04_handoff
            ) and (
                condition != "N3_EVICTABLE_PREFETCH3"
                or not any(event.get("event") == "LEASE_ACQUIRED" for event in lease)
            )
            classification = "PROGRESS" if (
                alive and not oom and not server_error
                and
                counts["finished"] == len(jobs)
                and all(response["generated_token_count"] == int(data[response["episode_id"]]["job"]["resume_osl"]) for response in responses)
                and control_ok
            ) else "INVALID_RESPONSE"
        else:
            classification = "OTHER_NONPROGRESS"

        if stage == "R2" and condition == "U3_UNLEASED_HEAD_LEASE3":
            if xfuture.done():
                x_generated_count = len(xfuture.result()["tokens"])
            else:
                x_generated_count = "still_active_at_final_snapshot"

        row = {
            "stage": stage, "condition": condition, "repetition": repetition,
            "X_enabled": stage == "R2", "selected_prefetches": "|".join(selected),
            "selected_leases": "|".join(selected) if leased else "",
            "leased_blocks_before_resume": condition_blocks(condition) if leased else 0,
            "free_blocks_near_first_resume": first_attempt.get("free_blocks") if first_attempt else "NA",
            "first_resume_session": first_arrival,
            "resume_arrived_5s": checkpoints.get("5s", {}).get("resume_arrived", "NA"),
            "resume_admitted_5s": checkpoints.get("5s", {}).get("resume_admitted", "NA"),
            "first_tokens_5s": checkpoints.get("5s", {}).get("first_tokens", "NA"),
            "finished_5s": checkpoints.get("5s", {}).get("finished", "NA"),
            "resume_arrived_final": counts["resume_arrived"],
            "resume_admitted_final": counts["resume_admitted"],
            "first_tokens_final": counts["first_tokens"],
            "finished_final": counts["finished"],
            "E04_allocate_none_count": sum(event.get("event") == "WAITING_ALLOCATE_NONE"
                                           and event.get("session") == "E04:turn:1" for event in waiting),
            "E04_first_block_free_blocks": block.get("free_blocks") if block else "NA",
            "E04_first_block_will_break": block.get("will_break_waiting_loop") if block else "NA",
            "E04_first_block_matching_lease": block.get("matching_lease") if block else "NA",
            "lease_acquired_count": sum(event.get("event") == "LEASE_ACQUIRED" for event in lease),
            "lease_released_count": releases,
            "K2_max_pre_first": max_pre_first,
            "server_alive": alive, "OOM": oom, "server_error": server_error,
            "X_first_before_prefetch": x_first_ns is not None if stage == "R2" else "NA",
            "X_generated_count_at_snapshot": x_generated_count,
            "elapsed_observation_s": ended - t0_s,
            "classification": classification,
        }
        append_csv(OUT / "liveness_runs.csv", [row])
        append_csv(OUT / "waiting_trace.csv", [{
            "stage": stage, "condition": condition, "repetition": repetition,
            "timestamp_ms": (float(event["timestamp_monotonic"]) - t0_s) * 1000,
            **event,
        } for event in waiting])
        append_csv(OUT / "lease_events.csv", [{
            "stage": stage, "condition": condition, "repetition": repetition,
            "timestamp_ms": (float(event["timestamp_monotonic"]) - t0_s) * 1000,
            **event,
        } for event in lease])
        append_csv(OUT / "resume_trace.csv", [{
            "stage": stage, "condition": condition, "repetition": repetition,
            "timestamp_ms": (float(event["timestamp_monotonic"]) - t0_s) * 1000,
            **event,
        } for event in snapshot["resume"]])
        append_csv(OUT / "capacity_trace.csv", [{
            "stage": stage, "condition": condition, "repetition": repetition,
            "timestamp_ms": (float(event["timestamp_monotonic"]) - t0_s) * 1000,
            **event,
        } for event in snapshot["capacity"]])
        return row
    finally:
        if proc.poll() is None:
            runtime.stop_server(proc, log, run_dir, label)
        if resume_pool is not None:
            resume_pool.shutdown(wait=False, cancel_futures=True)
        if xpool is not None:
            xpool.shutdown(wait=False, cancel_futures=True)
        compact_log(log, label)


def run_stage(stage: str, plan: list[tuple[str, int]], jobs, data) -> list[dict[str, Any]]:
    existing: dict[tuple[str, str, int], dict[str, Any]] = {}
    path = OUT / "liveness_runs.csv"
    if path.exists():
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                key = (row["stage"], row["condition"], int(row["repetition"]))
                if key in existing:
                    raise RuntimeError(f"duplicate run checkpoint: {key}")
                existing[key] = row
    result = []
    for name, repetition in plan:
        key = (stage, name, repetition)
        result.append(existing[key] if key in existing else run_condition(stage, name, repetition, jobs, data))
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = verify_jobs()
    data = reconstructed_data(jobs)
    start_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if start_commit != START_COMMIT:
        raise RuntimeError(f"unexpected start commit: {start_commit}")
    write_json(OUT / "config.json", {
        "phase": "4E-A-RCA", "start_commit": start_commit,
        "workload_sha256": WORKLOAD_SHA, "GPU_KV_blocks": GPU_BLOCKS,
        "K": 2, "GPU_KV_bytes": 2147483648,
        "CPU_external_KV_bytes": 8589934592,
        "R1_conditions": CONDITIONS,
        "R1_plan": [(name, rep) for name in CONDITIONS for rep in (1, 2)],
        "R2_plan": [(name, rep) for name in ("U3_UNLEASED_HEAD_LEASE3", "S3_HEAD_LEASED") for rep in (1, 2, 3)],
        "U3_checkpoints_s": [5, U3_OBSERVE_S], "progress_watchdog_s": WATCHDOG_S,
    })
    write_json(OUT / "environment.json", {
        "python": subprocess.check_output([str(runtime.ENV_ROOT / "bin/python"), "--version"], text=True).strip(),
        "GPU": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader", "-i", "0"], text=True).strip(),
        "model": str(runtime.MODEL), "dtype": "float16", "TP": 1,
        "vLLM_version": "0.23.0", "pinned_vLLM_commit": "0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665",
        "CUDA_VISIBLE_DEVICES": "0",
    })
    (OUT / "commands.sh").write_text(
        "PYTHONPATH=/media/a6000/D/ZhaoBinHe/slackkv-phase2c-fix-sourceprobe:$PWD "
        "/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python "
        "-m experiments.phase4e_a_rca.run_phase4e_a_rca\n",
        encoding="utf-8",
    )
    r1: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        r1.extend(run_stage("R1", [(condition, repetition) for repetition in (1, 2)], jobs, data))
        if condition == "U3_UNLEASED_HEAD_LEASE3":
            u3 = [row["classification"] for row in r1]
            if u3 != ["LEASE_HOL_NONPROGRESS", "LEASE_HOL_NONPROGRESS"]:
                status = ("RESULT_C_HYPOTHESIS_NOT_SUPPORTED" if u3 == ["PROGRESS", "PROGRESS"]
                          else "RESULT_B_ROOT_CAUSE_PARTIAL")
                write_json(OUT / "stage_gate.json", {"R1": status, "R2_started": False})
                return
    grouped = {
        name: [row["classification"] for row in r1 if row["condition"] == name]
        for name in CONDITIONS
    }
    r1_status = r1_result(grouped)
    write_json(OUT / "stage_gate.json", {"R1": r1_status, "R2_started": r1_status == "R1_EXPECTED_PATTERN"})
    if r1_status != "R1_EXPECTED_PATTERN":
        return
    r2 = run_stage(
        "R2",
        [(name, rep) for name in ("U3_UNLEASED_HEAD_LEASE3", "S3_HEAD_LEASED") for rep in (1, 2, 3)],
        jobs, data,
    )
    write_json(OUT / "stage_gate.json", {
        "R1": r1_status, "R2_started": True,
        "R2_U3": [row["classification"] for row in r2 if row["condition"] == "U3_UNLEASED_HEAD_LEASE3"],
        "R2_S3": [row["classification"] for row in r2 if row["condition"] == "S3_HEAD_LEASED"],
        "final_result": final_result(
            r1_status,
            [row["classification"] for row in r2 if row["condition"] == "U3_UNLEASED_HEAD_LEASE3"],
            [row["classification"] for row in r2 if row["condition"] == "S3_HEAD_LEASED"],
        ),
    })


if __name__ == "__main__":
    main()
