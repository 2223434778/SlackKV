from __future__ import annotations

import csv
import hashlib
import json
import os
import signal
import statistics
import subprocess
import threading
import time
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from experiments.phase2d.helpers import source_tokens_delta
from experiments.phase3f.scheduler import latest_feasible_start, merge_blackouts
from experiments.phase4d_a import runtime
from experiments.phase4d_a.experiment import materialize_and_evict, source_snapshot, write_csv, write_json, x_prompt
from experiments.phase4d_a.helpers import ReservationLedger, BYTES_PER_TOKEN
from experiments.phase4d_a.run_policy import parse_trace, stream_resume
from experiments.phase4d_a_fix.release_fix import LeaseTraceReader, RuntimeReleaseQueue, parse_lease_trace
from experiments.phase4d_b.capacity import capacity_invariant, parse_capacity_trace, validate_capacity_rejection
from experiments.phase4d_b.capacity_client import capacity_prefetch
from experiments.phase4d_b.experiment import frozen_jobs, reconstructed_data, validate_runtime
from experiments.phase4f_a.logic import progress_blocks, protected_episode, psa_trace_valid, verified_jobs
from experiments.phase4f_a.progress_client import progress_prefetch
from experiments.phase4e_a_rca.analysis import parse_structured_log
from experiments.phase4f_b.metrics import (
    WILLIAMS_ORDER, MIB, MINIMUM_H2D_BYTES, POLICIES, POLICY_BY_NAME,
    combined_decode_tok_s, deadline_latency, h2d_bytes, lease_memory_time_mib_s,
    measured_background_valid, nearest_rank, server_ttft, smoke_protocol_pass,
    source_conserved, x_output_valid, x_tail_margin_ms, x_tpot_ms,
    X_PROMPT_TOKENS, X_OUTPUT_TOKENS,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/phase4f_b"
SRC = runtime.SRC
START_COMMIT = "012dd5f20efabe2d5132540153be7f6991b81145"
WORKLOAD_SHA = "9b381336a70e73b0efbddbb79d2c88608c19663247e819c1e7a364efd2971105"
VLLM_DIFF_SHA = "e33055ec22f1d89c67342e71470134fdef3fda9dbbb5b7ac1e056e71676aab14"
P_SAFE = 102.0
GUARD = 110.0
BUDGET = 3510
MAX_PREFETCH_SECONDS = 5.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def current_vllm_diff_sha() -> str:
    return hashlib.sha256(subprocess.check_output(["git", "diff"], cwd=SRC)).hexdigest()


def verify_frozen(jobs: list[dict[str, Any]]) -> None:
    if sha256(ROOT / "results/phase4c/hybrid_workload.csv") != WORKLOAD_SHA:
        raise RuntimeError("EXPERIMENT_INVALID_WORKLOAD:Phase4C SHA")
    if current_vllm_diff_sha() != VLLM_DIFF_SHA:
        raise RuntimeError("PERFORMANCE_MATRIX_INVALID:vLLM source diff changed")
    if len(jobs) != 8 or sum(int(j["required_prefix_tokens"]) for j in jobs) != 168016:
        raise RuntimeError("EXPERIMENT_INVALID_WORKLOAD:jobs")
    if sum(int(j["required_blocks"]) for j in jobs) != 10501 or MINIMUM_H2D_BYTES / MIB != 4594.1875:
        raise RuntimeError("EXPERIMENT_INVALID_WORKLOAD:blocks/H2D")


def append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    previous: list[dict[str, Any]] = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as stream:
            previous = list(csv.DictReader(stream))
    write_csv(path, previous + rows)


def resource_snapshot() -> dict[str, Any]:
    command = ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used", "--format=csv,noheader,nounits", "-i", "0"]
    raw = subprocess.check_output(command, text=True).strip()
    mem_available = "NA"
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            mem_available = int(line.split()[1])
            break
    values = [x.strip() for x in raw.split(",")]
    return dict(zip(("gpu_temperature_C", "gpu_power_W", "gpu_sm_clock_MHz", "gpu_memory_clock_MHz", "gpu_memory_used_MiB"), values)) | {"MemAvailable_kB": mem_available, "timestamp_monotonic_s": time.monotonic()}


def warmup(label: str) -> None:
    prompt = runtime.tokenizer_ids("phase4e-a-warmup", label, 0, 20480)
    response = runtime.send(prompt, 16, params={"max_offload_tokens": 0})
    if len(response["tokens"]) != 16:
        raise RuntimeError("warmup output count mismatch")


def start_background_512():
    first: dict[str, int] = {}
    ready = threading.Event()

    def on_first(timestamp_ns: int) -> None:
        first["first_ns"] = timestamp_ns
        ready.set()

    prompt = x_prompt()
    if len(prompt) != X_PROMPT_TOKENS:
        raise RuntimeError("background X prompt length mismatch")
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(runtime.send, prompt, X_OUTPUT_TOKENS, stream=True,
                         params={"max_offload_tokens": 0}, on_first=on_first)
    if not ready.wait(120):
        pool.shutdown(wait=False, cancel_futures=True)
        raise TimeoutError("background X produced no first token")
    return pool, future, first


def run_x_only(repetition: int) -> dict[str, Any]:
    verify_frozen(verified_jobs())
    label = f"x_only_{repetition}"
    run_dir = OUT / "runs" / label
    run_dir.mkdir(parents=True, exist_ok=True)
    proc, log = runtime.start_server(label, out=run_dir)
    try:
        _, block = runtime.wait_ready(proc)
        validate_runtime(log, block)
        warmup(label)
        prompt = x_prompt()
        if len(prompt) != X_PROMPT_TOKENS:
            raise RuntimeError("background X prompt length mismatch")
        x = runtime.send(prompt, X_OUTPUT_TOKENS, stream=True, params={"max_offload_tokens": 0})
        duration = (x["done_ns"] - x["first_ns"]) / 1e6
        count = len(x["tokens"])
        row = {"repetition": repetition, "generated_tokens": count, "X_decode_ms": duration,
               "X_TPOT_ms": x_tpot_ms(duration, count) if x_output_valid(count) else "NA",
               "valid": x_output_valid(count)}
        append_csv(OUT / "x_only_runs.csv", [row])
        if not row["valid"]:
            raise RuntimeError("X-only token count mismatch")
        return row
    finally:
        runtime.stop_server(proc, log, run_dir, label)
        compact_log(log, run_dir, label)


def compact_log(path: Path, run_dir: Path, label: str, failure: bool = False) -> None:
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    markers = ("ERROR", "OOM", "GPU KV cache size", "SLACKKV_CAPACITY_TRACE", "SLACKKV_LEASE_TRACE", "SLACKKV_RESUME_TRACE")
    selected = [line[:2000] for line in lines if any(marker in line for marker in markers)]
    target = (OUT / "failures" / label if failure else OUT / "server_logs")
    target.mkdir(parents=True, exist_ok=True)
    (target / f"{label}_compact.log").write_text("\n".join(selected[-1500:]) + "\n", encoding="utf-8")
    if not failure:
        path.unlink()


def jit_target(now_ms: float, pending: set[str], deadlines: dict[str, float], blackouts: list[tuple[float, float]]) -> float:
    return latest_feasible_start(now_ms, [(episode, deadlines[episode]) for episode in pending], blackouts, P_SAFE)


def schedule_promotions(
    policy, jobs: list[dict[str, Any]], data: dict[str, dict[str, Any]],
    states: dict[str, dict[str, Any]], state_lock: threading.Lock,
    t0_s: float, log: Path,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], ReservationLedger | None]:
    if not policy.jit_enabled:
        return {}, [], [], None
    ledger = ReservationLedger(BUDGET) if policy.logical_reservation_enabled else None
    key_by_episode = {j["episode_id"]: j["lease_key"] for j in jobs}
    block_by_episode = {j["episode_id"]: int(j["required_blocks"]) for j in jobs}
    release_queue = RuntimeReleaseQueue(ledger, key_by_episode, block_by_episode, t0_s) if ledger else None
    lease_reader = LeaseTraceReader(log) if release_queue else None
    deadlines = {j["episode_id"]: float(j["deadline_ms"]) for j in jobs}
    blackouts = merge_blackouts(deadlines, GUARD)
    pending = set(deadlines)
    promotions: dict[str, dict[str, Any]] = {}
    decisions: list[dict[str, Any]] = []
    releases: list[dict[str, Any]] = []
    pool = ThreadPoolExecutor(max_workers=8)

    def drain() -> None:
        if release_queue is None or lease_reader is None:
            return
        for episode, state in states.items():
            if state.get("first_token_ns") is not None and episode in ledger.active:
                release_queue.enqueue_first_token(episode, int(state["first_token_ns"]))
        release_queue.observe_lease_events(lease_reader.read_events())
        for row in release_queue.drain(time.monotonic_ns()):
            releases.append(row)
            decisions.append({"episode_id": row["episode_id"], "decision": row["event"], "reserved_blocks_after": row["reserved_blocks_after"]})

    def harvest() -> None:
        for episode, item in promotions.items():
            future: Future = item["future"]
            if item.get("harvested") or not future.done():
                continue
            result = future.result()
            status = result.get("status")
            if status not in {"ok", "capacity_rejected", "progress_rejected"}:
                raise RuntimeError(f"unexpected promotion status {episode}:{status}")
            item["result"] = result
            item["harvested"] = True
            if status in {"capacity_rejected", "progress_rejected"}:
                if status == "capacity_rejected" and not validate_capacity_rejection(result):
                    raise RuntimeError("invalid capacity rejection")
                if ledger:
                    before = ledger.reserved_blocks
                    ledger.release(episode)
                    decisions.append({"episode_id": episode, "decision": status.upper() + "_RELEASE", "reserved_blocks_before": before, "reserved_blocks_after": ledger.reserved_blocks, "timestamp_monotonic_s": time.monotonic()})
                with state_lock:
                    states[episode]["leased"] = False
                    states[episode]["promoted"] = False
            else:
                expected = int(data[episode]["job"]["required_prefix_tokens"])
                if int(result.get("external_tokens_loaded", -1)) != expected or int(result.get("transfer_bytes", 0)) <= 0:
                    raise RuntimeError(f"native prefetch token/transfer mismatch:{episode}")
                if any(int(result.get(key, -1)) != 0 for key in ("model_forward_count", "generated_token_count", "mixed_prefetch_model_step_count")):
                    raise RuntimeError(f"native prefetch performed model work:{episode}")
                if not {"T0", "T1", "T6", "T7"}.issubset(result.get("phase3m_timestamps") or {}):
                    raise RuntimeError(f"native timestamps incomplete:{episode}")
                with state_lock:
                    states[episode]["leased"] = policy.physical_lease_enabled
                    states[episode]["promoted"] = True
            decisions.append({"episode_id": episode, "decision": "PROMOTION_" + status.upper(), "timestamp_monotonic_s": time.monotonic()})

    try:
        while pending:
            harvest()
            drain()
            now_ms = (time.monotonic() - t0_s) * 1000
            episode = min(pending, key=lambda e: (deadlines[e], e))
            target = jit_target(now_ms, pending, deadlines, blackouts)
            if target > now_ms + 0.5:
                time.sleep(min(0.01, (target - now_ms) / 1000))
                continue
            with state_lock:
                stale = states[episode].get("resume_started_ns") is not None
                reserved = not stale and (ledger.reserve(episode, block_by_episode[episode]) if ledger else True)
                protected = protected_episode(jobs, states) if policy.psa_enabled else None
                reserved_at_ns = time.monotonic_ns() if ledger and reserved else None
            if stale or (policy.psa_enabled and protected is None):
                if ledger and reserved:
                    ledger.release(episode)
                decisions.append({"episode_id": episode, "decision": "STALE_PROMOTION_CANCELLED", "target_start_ms": target})
            elif not reserved:
                    decisions.append({"episode_id": episode, "decision": "RESERVATION_SKIP", "target_start_ms": target, "required_blocks": block_by_episode[episode], "reserved_blocks": ledger.reserved_blocks})
            else:
                lease_key = key_by_episode[episode] if policy.physical_lease_enabled else None
                item: dict[str, Any] = {"episode_id": episode, "lease_key": lease_key, "target_start_ms": target, "required_blocks": block_by_episode[episode], "submitted_s": time.monotonic(), "harvested": False, "protected_episode": protected}
                with state_lock:
                    if policy.psa_enabled:
                        protected_job = data[protected]["job"]
                        item["future"] = pool.submit(progress_prefetch, data[episode]["prefix"],
                            required_blocks=block_by_episode[episode], lease_key=lease_key,
                            protected_lease_key=key_by_episode[protected],
                            protected_progress_blocks=progress_blocks(int(protected_job["turn1_isl"])),
                            timeout_s=MAX_PREFETCH_SECONDS)
                    else:
                        item["future"] = pool.submit(capacity_prefetch, data[episode]["prefix"], required_blocks=block_by_episode[episode], lease_key=lease_key, timeout_s=MAX_PREFETCH_SECONDS)
                promotions[episode] = item
                decisions.append({"episode_id": episode, "decision": "PROMOTION_ATTEMPT", "target_start_ms": target, "required_blocks": block_by_episode[episode], "lease_key": lease_key or "NA", "protected_episode": protected or "NA", "reserved_blocks": ledger.reserved_blocks if ledger else "NA", "timestamp_monotonic_ns": reserved_at_ns or time.monotonic_ns()})
            pending.remove(episode)
        while any(not item.get("harvested") for item in promotions.values()):
            harvest()
            drain()
            if any(not item.get("harvested") and time.monotonic() - item["submitted_s"] > MAX_PREFETCH_SECONDS for item in promotions.values()):
                raise RuntimeError("PREFETCH_NONPROGRESS")
            time.sleep(0.01)
        return promotions, decisions, releases, ledger
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _parse_run(log: Path, t0_ns: int):
    text = log.read_text(encoding="utf-8", errors="replace")
    trace, _ = parse_trace(log, t0_ns)
    leases = parse_lease_trace(text)
    capacities = parse_capacity_trace(text)
    return trace, leases, capacities, text


def run_policy(policy, stage: str, repetition: int, position: int, jobs, data, x_baseline: float) -> dict[str, Any]:
    label = f"{stage}_rep{repetition}_pos{position}_{policy.name.replace('+', 'plus')}"
    run_dir = OUT / "runs" / label
    run_dir.mkdir(parents=True, exist_ok=True)
    verify_frozen(jobs)
    proc, log = runtime.start_server(label, out=run_dir)
    completed = False
    try:
        _, block = runtime.wait_ready(proc)
        capacity = validate_runtime(log, block)
        warmup(label)
        preparation = materialize_and_evict(proc, log, label, jobs, data, run_dir)
        if not preparation["pressure"]["capacity_exceeded"] or any(not row["validation"] for row in preparation["materialization"]):
            raise RuntimeError("RUN_INVALID_PREPARATION")
        resources_start = resource_snapshot()
        source_before = source_snapshot()
        states = {j["episode_id"]: {"resume_started_ns": None, "first_token_ns": None, "leased": False, "promoted": False} for j in jobs}
        lock = threading.Lock()
        xpool, xfuture, xmeta = start_background_512()
        t0_ns = int(xmeta["first_ns"])
        t0_s = t0_ns / 1e9
        resume_pool = ThreadPoolExecutor(max_workers=8)
        futures = {episode: resume_pool.submit(stream_resume, entry["job"]["lease_key"], entry, t0_s, float(entry["job"]["deadline_ms"]), True, states, lock) for episode, entry in data.items()}
        promotions, decisions, releases, ledger = schedule_promotions(policy, jobs, data, states, lock, t0_s, log)
        responses = {}
        for episode, future in futures.items():
            responses[episode] = future.result(timeout=30)
        resume_pool.shutdown(wait=True)
        if ledger:
            reader = LeaseTraceReader(log)
            queue = RuntimeReleaseQueue(ledger, {j["episode_id"]: j["lease_key"] for j in jobs}, {j["episode_id"]: int(j["required_blocks"]) for j in jobs}, t0_s)
            end = time.monotonic() + 5
            while ledger.active and time.monotonic() < end:
                for episode in list(ledger.active):
                    ns = states[episode].get("first_token_ns")
                    if ns is not None:
                        queue.enqueue_first_token(episode, int(ns))
                queue.observe_lease_events(reader.read_events())
                releases.extend(queue.drain(time.monotonic_ns()))
                if ledger.active:
                    time.sleep(0.02)
        # The source snapshot is outside the deadline-to-first-token critical path.
        source_after = source_snapshot()
        local = int(round(source_tokens_delta(source_before, source_after, "local_cache_hit")))
        external = int(round(source_tokens_delta(source_before, source_after, "external_kv_transfer")))
        x = xfuture.result(timeout=30)
        xpool.shutdown(wait=True)
        resources_end = resource_snapshot()
        server_alive_before_shutdown = proc.poll() is None
        if server_alive_before_shutdown:
            try:
                with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2) as health:
                    server_alive_before_shutdown = health.status == 200
            except OSError:
                server_alive_before_shutdown = False
        runtime.stop_server(proc, log, run_dir, label)
        trace, leases, capacities, raw = _parse_run(log, t0_ns)
        psa_events = [event for event in capacities if event.get("progress_guard") is True]
        psa_violations = sum(not psa_trace_valid(event) for event in psa_events)
        by_episode: dict[str, dict[str, dict[str, Any]]] = {}
        for event in trace:
            episode = str(event.get("session") or "").split(":turn:", 1)[0]
            if episode in data:
                by_episode.setdefault(episode, {}).setdefault(str(event["event"]), event)
        sessions = []
        for job in jobs:
            episode = job["episode_id"]
            response = responses[episode]
            events = by_episode.get(episode, {})
            for required in ("RESUME_ARRIVED", "RESUME_ADMITTED", "FIRST_TOKEN", "REQUEST_FINISHED"):
                if required not in events:
                    raise RuntimeError(f"missing real scheduler trace:{episode}:{required}")
            arrival = float(events["RESUME_ARRIVED"]["timestamp_ms"])
            first = float(events["FIRST_TOKEN"]["timestamp_ms"])
            finish = float(events["REQUEST_FINISHED"]["timestamp_ms"])
            sessions.append({"repetition": repetition, "policy": policy.name, "episode_id": episode, "deadline_ms": float(job["deadline_ms"]), "required_blocks": int(job["required_blocks"]), "required_prefix_tokens": int(job["required_prefix_tokens"]), "OSL": int(job["resume_osl"]), "resume_arrival_ms": arrival, "first_token_ms": first, "request_finished_ms": finish, "deadline_resume_latency_ms": deadline_latency(first, float(job["deadline_ms"])), "server_TTFT_ms": server_ttft(first, arrival), "completion_from_deadline_ms": finish - float(job["deadline_ms"]), "completion_from_arrival_ms": finish - arrival, "generated_tokens": response["generated_token_count"], "output_sha256": response["output_sha256"], "expected_output_sha256": data[episode]["expected_output_sha256"], "identity_match": response["output_sha256"] == data[episode]["expected_output_sha256"], "promotion_attempted": episode in promotions, "capacity_decision": next((c["decision"] for c in capacities if c.get("lease_key") == (job["lease_key"] if policy.physical_lease_enabled else None) and c.get("required_blocks") == job["required_blocks"]), "NA"), "reservation_decision": next((d["decision"] for d in decisions if d.get("episode_id") == episode and d.get("decision") in ("RESERVATION_SKIP", "PROMOTION_ATTEMPT")), "NA"), "promotion_status": promotions[episode]["result"]["status"] if episode in promotions else "NA", "T6_ms": "NA", "promotion_usefulness": "NA", "local_source_tokens": "NA", "external_source_tokens": "NA"})
        lease_by_key: dict[str, dict[str, dict[str, Any]]] = {}
        for event in leases:
            lease_by_key.setdefault(str(event["lease_key"]), {})[str(event["event"])] = event
        prefetch_rows = []
        native_bytes = 0
        memory_time = 0.0
        lead_exposure = 0.0
        for episode, item in promotions.items():
            result = item["result"]
            job = data[episode]["job"]
            event = by_episode[episode]
            arrival_raw = float(event["RESUME_ARRIVED"]["timestamp_monotonic_raw"])
            remote = "WAITING_FOR_REMOTE_KVS" in event
            timestamps = result.get("phase3m_timestamps") or {}
            t6_raw = float(timestamps["T6"]) if "T6" in timestamps else None
            status = result["status"]
            if status == "ok":
                native_bytes += int(result["transfer_bytes"])
            classification = ("CAPACITY_REJECTED" if status == "capacity_rejected" else
                              "PROGRESS_REJECTED" if status == "progress_rejected" else
                              "PROMOTION_READY_MISS" if t6_raw is None or t6_raw >= arrival_raw else
                              "PROMOTION_REMOTE_WAIT" if remote else "PROMOTION_USEFUL")
            t6_ms = (t6_raw - t0_s) * 1000 if t6_raw is not None else None
            session = next(row for row in sessions if row["episode_id"] == episode)
            session["T6_ms"] = t6_ms if t6_ms is not None else "NA"
            session["promotion_usefulness"] = classification
            size_mib = int(job["required_prefix_tokens"]) * BYTES_PER_TOKEN / MIB
            if status == "ok" and t6_ms is not None:
                lead_exposure += size_mib * max(0.0, float(event["RESUME_ARRIVED"]["timestamp_ms"]) - t6_ms) / 1000
            if status == "ok" and policy.physical_lease_enabled:
                lifecycle = lease_by_key.get(job["lease_key"], {})
                acquired = lifecycle.get("LEASE_ACQUIRED")
                released = lifecycle.get("LEASE_RELEASED")
                if acquired is None or released is None or t6_ms is None:
                    raise RuntimeError(f"lease lifecycle incomplete:{episode}")
                memory_time += lease_memory_time_mib_s(int(job["required_prefix_tokens"]),
                    (float(acquired["timestamp_monotonic"]) - t0_s) * 1000,
                    (float(released["timestamp_monotonic"]) - t0_s) * 1000)
            prefetch_rows.append({"repetition": repetition, "policy": policy.name, "episode_id": episode, "lease_key": item["lease_key"] or "NA", "target_start_ms": item["target_start_ms"], "required_blocks": item["required_blocks"], "status": status, "classification": classification, "issue_monotonic": result["client_issue_ns"] / 1e9, "completion_monotonic": result["client_return_ns"] / 1e9, "T0": timestamps.get("T0", "NA"), "T1": timestamps.get("T1", "NA"), "T6": timestamps.get("T6", "NA"), "T7": timestamps.get("T7", "NA"), "client_issue_ns": result["client_issue_ns"], "client_return_ns": result["client_return_ns"], "client_response_ms": result["client_response_ms"], "T6_ms": t6_ms if t6_ms is not None else "NA", "issue_to_T6_ms": (t6_raw - result["client_issue_ns"] / 1e9) * 1000 if t6_raw is not None else "NA", "issue_to_T7_ms": (float(timestamps["T7"]) - result["client_issue_ns"] / 1e9) * 1000 if "T7" in timestamps else "NA", "transfer_bytes": result.get("transfer_bytes", 0), "external_tokens_loaded": result.get("external_tokens_loaded", 0), "model_forward_count": result.get("model_forward_count", 0), "generated_token_count": result.get("generated_token_count", 0), "mixed_prefetch_model_step_count": result.get("mixed_prefetch_model_step_count", 0)})
        native, external_bytes, total, duplicate = h2d_bytes(native_bytes, external)
        x_duration = (x["done_ns"] - x["first_ns"]) / 1e6
        x_count = len(x["tokens"])
        latencies = [float(row["deadline_resume_latency_ms"]) for row in sessions]
        ttfts = [float(row["server_TTFT_ms"]) for row in sessions]
        max_pre = max([int(row.get("num_slackkv_pre_first_token") or 0) for row in trace] + [0])
        capacity_ok = all(psa_trace_valid(event) if event.get("progress_guard") is True
                          else capacity_invariant(event) for event in capacities)
        x_finish_ms = (x["done_ns"] - t0_ns) / 1e6
        last_agent_first_ms = max(float(row["first_token_ms"]) for row in sessions)
        tail_margin = x_tail_margin_ms(x_finish_ms, last_agent_first_ms)
        x_active = measured_background_valid(x_count, tail_margin)
        lease_leak = sum(event["event"] == "LEASE_ACQUIRED" for event in leases) - sum(event["event"] == "LEASE_RELEASED" for event in leases)
        lease_peak = max_lease_blocks(leases)
        unsafe_releases = sum(float(row["logical_minus_physical_ms"]) < 0 for row in releases)
        oom = any(marker in raw.lower() for marker in ("cuda out of memory", "outofmemoryerror"))
        service_values = [float(row["issue_to_T6_ms"]) for row in prefetch_rows if row["status"] == "ok" and row["issue_to_T6_ms"] != "NA"]
        service_overruns = sum(value > P_SAFE for value in service_values)
        rejected_keys = {data[episode]["job"]["lease_key"] for episode, item in promotions.items() if item["result"]["status"] in {"capacity_rejected", "progress_rejected"}}
        reject_lifecycle_ok = all(str(event.get("lease_key")) not in rejected_keys for event in leases)
        valid = (len(sessions) == 8 and all(row["generated_tokens"] == row["OSL"] for row in sessions)
            and source_conserved(local, external) and max_pre <= 2 and x_active and capacity_ok
            and lease_leak == 0 and (ledger is None or not ledger.active)
            and all(row["client_response_ms"] < 5000 for row in prefetch_rows)
            and len(capacities) == len(promotions) and reject_lifecycle_ok and psa_violations == 0
            and lease_peak <= (BUDGET if policy.logical_reservation_enabled else capacity["total_gpu_kv_blocks"])
            and (ledger is None or ledger.peak_blocks <= BUDGET)
            and unsafe_releases == 0 and server_alive_before_shutdown and not oom)
        if policy.name == "OnDemand-K2":
            valid &= not promotions and not leases and not capacities and local == 0 and external == 168016 and total == MINIMUM_H2D_BYTES
        if not policy.physical_lease_enabled:
            valid &= not leases
        if not policy.psa_enabled:
            valid &= not psa_events
        else:
            valid &= len(psa_events) == len(promotions)
        if not policy.logical_reservation_enabled:
            valid &= not decisions or all(d["decision"] not in ("RESERVATION_SKIP", "CAPACITY_REJECT_RELEASE", "RUNTIME_RELEASE_AFTER_PHYSICAL_LEASE") for d in decisions)
        reserve_start = {d["episode_id"]: int(d["timestamp_monotonic_ns"])
                         for d in decisions if d["decision"] == "PROMOTION_ATTEMPT" and ledger}
        reserve_end = {}
        for d in decisions:
            if d["decision"] in {"CAPACITY_REJECTED_RELEASE", "PROGRESS_REJECTED_RELEASE"}:
                reserve_end[d["episode_id"]] = int(float(d["timestamp_monotonic_s"]) * 1e9)
        for d in releases:
            reserve_end[d["episode_id"]] = int((t0_s + float(d["logical_release_ms"]) / 1000) * 1e9)
        reservation_block_time = sum(block_by * (reserve_end[episode] - start) / 1e9
                                     for episode, start in reserve_start.items()
                                     if episode in reserve_end
                                     for block_by in [int(data[episode]["job"]["required_blocks"])])
        if ledger and set(reserve_start) != set(reserve_end):
            valid = False
        active_lease_count = 0
        peak_lease_count = 0
        for event in sorted(leases, key=lambda e: float(e["timestamp_monotonic"])):
            if event["event"] == "LEASE_ACQUIRED":
                active_lease_count += 1
            elif event["event"] == "LEASE_RELEASED":
                active_lease_count -= 1
            peak_lease_count = max(peak_lease_count, active_lease_count)
        run = {
            "repetition": repetition, "run_order_position": position, "policy": policy.name,
            "valid": valid, "resume_count": len(sessions),
            "deadline_latency_P50": nearest_rank(latencies, 50),
            "deadline_latency_P95": nearest_rank(latencies, 95),
            "deadline_latency_max": max(latencies),
            "TTFT_P50": nearest_rank(ttfts, 50), "TTFT_P95": nearest_rank(ttfts, 95),
            "TTFT_max": max(ttfts),
            "local_source_tokens": local, "external_source_tokens": external,
            "local_source_fraction": local / 168016,
            "promotion_candidates": 8 if policy.jit_enabled else 0,
            "promotion_attempts": len(promotions),
            "capacity_admits": sum(e["decision"] == "ADMIT" for e in capacities),
            "capacity_rejects": sum(e["decision"] in {"REJECT", "CAPACITY_REJECT"} for e in capacities),
            "progress_rejects": sum(e["decision"] == "PROGRESS_REJECT" for e in capacities),
            "reservation_skips": sum(d["decision"] == "RESERVATION_SKIP" for d in decisions),
            "promotion_useful": sum(r["classification"] == "PROMOTION_USEFUL" for r in prefetch_rows),
            "promotion_ready_miss": sum(r["classification"] == "PROMOTION_READY_MISS" for r in prefetch_rows),
            "promotion_remote_wait": sum(r["classification"] == "PROMOTION_REMOTE_WAIT" for r in prefetch_rows),
            "successful_promotions": sum(r["status"] == "ok" for r in prefetch_rows),
            "useful_rate": (sum(r["classification"] == "PROMOTION_USEFUL" for r in prefetch_rows) /
                            sum(r["status"] == "ok" for r in prefetch_rows)) if any(r["status"] == "ok" for r in prefetch_rows) else "NA",
            "stale_cancellation": sum(d["decision"] == "STALE_PROMOTION_CANCELLED" for d in decisions),
            "promotion_H2D_MiB": native / MIB, "resume_H2D_MiB": external_bytes / MIB,
            "total_H2D_MiB": total / MIB, "duplicate_H2D_MiB": duplicate / MIB,
            "lease_memory_time_MiB_s": memory_time if policy.physical_lease_enabled else "NA",
            "peak_lease_count": peak_lease_count if policy.physical_lease_enabled else "NA",
            "reservation_block_time": reservation_block_time if ledger else "NA",
            "promotion_lead_exposure_MiB_s": lead_exposure if policy.jit_enabled else "NA",
            "peak_reserved_blocks": ledger.peak_blocks if ledger else "NA",
            "peak_leased_blocks": lease_peak if policy.physical_lease_enabled else "NA",
            "X_generated_tokens": x_count, "X_decode_ms": x_duration,
            "X_TPOT_ms": x_tpot_ms(x_duration, x_count) if x_output_valid(x_count) else "NA",
            "X_tail_margin_ms": tail_margin, "X_slowdown": x_duration / x_baseline,
            "combined_tok_s": combined_decode_tok_s(
                [int(row["OSL"]) for row in sessions],
                [float(row["first_token_ms"]) for row in sessions],
                [float(row["request_finished_ms"]) for row in sessions],
                x_count, 0.0, x_finish_ms,
            ) if x_output_valid(x_count) else "NA",
            "server_crash": not server_alive_before_shutdown, "OOM": oom,
            "logical_leaks": len(ledger.active) if ledger else 0, "lease_leaks": lease_leak,
            "unsafe_releases": unsafe_releases, "K2_max_pre_first": max_pre,
            "PSA_invariant_violation": psa_violations,
            "X_active_through_last_first": x_active,
            "promotion_T6_min_ms": min(service_values) if service_values else "NA",
            "promotion_T6_median_ms": statistics.median(service_values) if service_values else "NA",
            "promotion_T6_P95_ms": nearest_rank(service_values, 95) if service_values else "NA",
            "promotion_T6_max_ms": max(service_values) if service_values else "NA",
            "service_overrun_count": service_overruns,
            "service_overrun_rate": service_overruns / len(service_values) if service_values else "NA",
        }
        run.update({
            "position": position, "deadline_P50_ms": run["deadline_latency_P50"],
            "deadline_P95_ms": run["deadline_latency_P95"],
            "deadline_max_ms": run["deadline_latency_max"],
            "TTFT_P50_ms": run["TTFT_P50"], "TTFT_P95_ms": run["TTFT_P95"],
            "TTFT_max_ms": run["TTFT_max"],
            "promotion_admits": run["capacity_admits"], "crash": run["server_crash"],
            "lease_leak": run["lease_leaks"], "logical_leak": run["logical_leaks"],
        })
        append_csv(OUT / ("semantic_smoke.csv" if stage == "smoke" else "run_metrics.csv"), [run])
        append_csv(OUT / "session_metrics.csv", [{"stage": stage, **row} for row in sessions])
        append_csv(OUT / "prefetch_metrics.csv", [{"stage": stage, **row} for row in prefetch_rows])
        append_csv(OUT / "capacity_trace.csv", [{"stage": stage, "repetition": repetition, "policy": policy.name, **row} for row in capacities])
        append_csv(OUT / "progress_admission.csv", [{"stage": stage, "repetition": repetition,
            "policy": policy.name, **row} for row in psa_events])
        append_csv(OUT / "resource_costs.csv", [{"stage": stage, "repetition": repetition, "policy": policy.name,
            "lease_memory_time_MiB_s": run["lease_memory_time_MiB_s"],
            "peak_leased_blocks": run["peak_leased_blocks"],
            "peak_lease_count": run["peak_lease_count"],
            "reservation_block_time": run["reservation_block_time"],
            "peak_reserved_blocks": run["peak_reserved_blocks"]}])
        append_csv(OUT / "reservation_events.csv", [{"stage": stage, "repetition": repetition, "policy": policy.name, **row} for row in decisions + releases])
        append_csv(OUT / "lease_events.csv", [{"stage": stage, "repetition": repetition, "policy": policy.name, **row} for row in leases])
        append_csv(OUT / "scheduler_trace.csv", [{"stage": stage, "repetition": repetition, "policy": policy.name, **row} for row in trace])
        append_csv(OUT / "h2d_accounting.csv", [{"stage": stage, "repetition": repetition, "policy": policy.name, "promotion_bytes": native, "resume_external_bytes": external_bytes, "total_bytes": total, "minimum_bytes": MINIMUM_H2D_BYTES, "duplicate_bytes": duplicate}])
        append_csv(OUT / "resource_metrics.csv", [{"stage": stage, "repetition": repetition, "policy": policy.name, "point": "before", **resources_start}, {"stage": stage, "repetition": repetition, "policy": policy.name, "point": "after", **resources_end}])
        compact_log(log, run_dir, label)
        if not valid:
            raise RuntimeError(f"RUN_INVALID:{label}:{run}")
        completed = True
        return run
    finally:
        if proc.poll() is None:
            runtime.stop_server(proc, log, run_dir, label)
        if not completed:
            compact_log(log, run_dir, label, failure=True)


def max_lease_blocks(events: list[dict[str, Any]]) -> int:
    active: dict[str, int] = {}
    peak = 0
    for event in sorted(events, key=lambda e: float(e["timestamp_monotonic"])):
        if event["event"] == "LEASE_ACQUIRED":
            active[event["lease_key"]] = int(event["num_blocks"])
        elif event["event"] == "LEASE_RELEASED":
            active.pop(event["lease_key"], None)
        peak = max(peak, sum(active.values()))
    return peak


def run_experiment() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = verified_jobs()
    verify_frozen(jobs)
    data = reconstructed_data(jobs)
    write_json(OUT / "config.json", {"phase": "4F-B", "K": 2, "P_SAFE_ms": P_SAFE, "resume_guard_ms": GUARD, "reservation_budget_blocks": BUDGET, "gpu_kv_bytes": 2147483648, "cpu_external_bytes": 8589934592, "warmup_prompt_tokens": 20480, "warmup_output_tokens": 16, "X_prompt_tokens": X_PROMPT_TOKENS, "X_output_tokens": X_OUTPUT_TOKENS, "workload_sha256": WORKLOAD_SHA})
    write_json(OUT / "policy_specs.json", [policy.__dict__ for policy in POLICIES])
    write_json(OUT / "run_order.json", {"repetitions": WILLIAMS_ORDER})
    write_json(OUT / "source_manifest.json", {"start_commit": START_COMMIT, "actual_start_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "workload_sha256": WORKLOAD_SHA, "phase4f_a_patch_sha256": sha256(ROOT / "results/phase4f_a/vllm_phase4f_a_delta.patch"), "vllm_upstream_commit": "0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665", "vllm_diff_sha256": current_vllm_diff_sha(), "vllm_status_short": subprocess.check_output(["git", "status", "--short"], cwd=SRC, text=True)})
    write_json(OUT / "environment.json", {"python": subprocess.check_output([str(runtime.ENV_ROOT / "bin/python"), "--version"], text=True).strip(), "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader", "-i", "0"], text=True).strip(), "vllm_version": "0.23.0", "model": str(runtime.MODEL), "dtype": "float16", "TP": 1, "max_model_len": 32768, "gpu_blocks_expected": 4681, "block_size": 16})
    (OUT / "commands.sh").write_text("PYTHONPATH=/media/a6000/D/ZhaoBinHe/slackkv-phase2c-fix-sourceprobe:$PWD /media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python -m experiments.phase4f_b.run_phase4f_b\n", encoding="utf-8")
    x_rows = [run_x_only(rep) for rep in range(1, 5)]
    baseline = statistics.median(float(row["X_decode_ms"]) for row in x_rows)
    if len(x_rows) != 4 or not all(row["valid"] for row in x_rows):
        raise RuntimeError("EXPERIMENT_INVALID_MATRIX:X-only")
    smoke_rows = []
    for position, policy in enumerate(POLICIES, start=1):
        smoke_rows.append(run_policy(policy, "smoke", 0, position, jobs, data, baseline))
    margins = {row["policy"]: float(row["X_tail_margin_ms"]) for row in smoke_rows}
    if not all(row["valid"] for row in smoke_rows) or not smoke_protocol_pass(list(margins.values())):
        raise RuntimeError(f"PERFORMANCE_MATRIX_INVALID:Stage A:{margins}")
    for repetition, names in enumerate(WILLIAMS_ORDER, start=1):
        for position, name in enumerate(names, start=1):
            run_policy(POLICY_BY_NAME[name], "measured", repetition, position, jobs, data, baseline)
    x_rows.extend(run_x_only(rep) for rep in range(5, 9))
    if len(x_rows) != 8 or not all(row["valid"] for row in x_rows):
        raise RuntimeError("EXPERIMENT_INVALID_MATRIX:X-only post")
    from experiments.phase4f_b.finalize_phase4f_b import finalize
    finalize()


def main() -> None:
    try:
        run_experiment()
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        failure = {"phase": "4F-B", "final_result": "EXPERIMENT_INVALID_MATRIX",
                   "error_type": type(exc).__name__, "error": str(exc),
                   "X_only_completed": 0, "Stage_A_completed": 0, "Stage_B_completed": 0}
        for key, filename in (("X_only_completed", "x_only_runs.csv"),
                              ("Stage_A_completed", "semantic_smoke.csv"),
                              ("Stage_B_completed", "run_metrics.csv")):
            if (OUT / filename).exists():
                with (OUT / filename).open(newline="", encoding="utf-8") as stream:
                    failure[key] = sum(1 for _ in csv.DictReader(stream))
        write_json(OUT / "failure_report.json", failure)
        (OUT / "summary.md").write_text(
            "# Phase 4F-B — incomplete frozen matrix\n\n"
            "**EXPERIMENT_INVALID_MATRIX.** The first failed validity gate stopped the ordered matrix; "
            "no partial performance aggregation or policy winner is reported.\n\n"
            f"Failure: `{type(exc).__name__}: {exc}`.\n\n"
            f"Completed X-only: {failure['X_only_completed']}/8; Stage A: {failure['Stage_A_completed']}/4; "
            f"Stage B: {failure['Stage_B_completed']}/32. See `failure_report.json` and compact server logs.\n",
            encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
