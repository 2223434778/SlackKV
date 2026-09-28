"""Causal Phase 4G-A lifecycle smoke; no boundary pre-materialization."""
from __future__ import annotations

import csv
import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from experiments.phase4c.synthetic_tokens import longest_full_block_match, make_next_prompt, token_hash
from experiments.phase4d_a import runtime
from experiments.phase4d_a.experiment import x_prompt
from experiments.phase4d_a.run_policy import wait_until
from experiments.phase4d_b.experiment import validate_runtime
from experiments.phase4d_b.capacity import capacity_invariant, parse_capacity_trace
from experiments.phase4g_a.controller import DynamicController
from experiments.phase4f_a.logic import psa_trace_valid
from experiments.phase4g_a.workload import OUT, build_manifest, load_source, nearest_rank

X_OUTPUT_TOKENS = 2048
X_PROMPT_TOKENS = 9500
K = 2


def source_events(path: Path) -> list[dict[str, Any]]:
    events = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "SLACKKV_OFFLOAD_TRACE " in line:
            events.append(json.loads(line.split("SLACKKV_OFFLOAD_TRACE ", 1)[1]))
    return events


def start_background():
    ready = threading.Event()
    first: dict[str, int] = {}
    prompt = x_prompt()
    if len(prompt) != X_PROMPT_TOKENS:
        raise RuntimeError("X prompt length mismatch")

    def on_first(ns: int) -> None:
        first["ns"] = ns
        ready.set()

    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(runtime.send, prompt, X_OUTPUT_TOKENS, stream=True,
                         params={"max_offload_tokens": 0}, on_first=on_first)
    if not ready.wait(120):
        pool.shutdown(wait=False, cancel_futures=True)
        raise TimeoutError("background X first token timeout")
    return pool, future, first["ns"]


def run_lifecycle(policy: str, stage: str = "smoke", repetition: int = 0) -> dict[str, Any]:
    if policy not in {"A", "D"}:
        raise ValueError("Phase 4G-A compares only A and D")
    if stage not in {"smoke", "measured"}:
        raise ValueError("invalid Phase 4G-A stage")
    source = load_source()
    manifest = build_manifest(source)
    turns_by_episode: dict[str, list[dict[str, str]]] = {}
    for row in source:
        turns_by_episode.setdefault(row["episode_id"], []).append(row)
    boundaries = {(row["episode_id"], row["boundary_index"]): row for row in manifest}
    label = f"{stage}_rep{repetition}_{policy.lower()}"
    run_dir = OUT / "runs" / label
    run_dir.mkdir(parents=True, exist_ok=True)
    proc, log = runtime.start_server(label, out=run_dir)
    lock = threading.Lock()
    sent_prompts: list[list[int]] = []
    turn_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    try:
        _, block_size = runtime.wait_ready(proc)
        validate_runtime(log, block_size)
        warmup_prompt = runtime.tokenizer_ids("phase4g-a-warmup", "unique", 0, 20480)
        if len(runtime.send(warmup_prompt, 16, params={"max_offload_tokens": 0})["tokens"]) != 16:
            raise RuntimeError("warmup output mismatch")
        x_pool, x_future, x_first_ns = start_background()
        controller = DynamicController(manifest, log, x_first_ns) if policy == "D" else None
        sessions_done = threading.Event()
        controller_thread = threading.Thread(target=controller.run, args=(sessions_done,), daemon=True) if controller else None
        if controller_thread:
            controller_thread.start()

        def worker(episode_id: str) -> None:
            episode_turns = sorted(turns_by_episode[episode_id], key=lambda row: int(row["replay_turn_index"]))
            prior_served: list[int] | None = None
            prior_finish_ns: int | None = None
            for index, row in enumerate(episode_turns):
                target = int(row["source_isl"])
                if index == 0:
                    prompt = runtime.tokenizer_ids(row["config"], row["session_hash"], 0, target)
                    expected_shared = 0
                else:
                    assert prior_served is not None and prior_finish_ns is not None
                    boundary = boundaries[episode_id, index - 1]
                    expected_shared = int(boundary["required_prefix_tokens"])
                    prompt = make_next_prompt(
                        prior_served, target, expected_shared,
                        phase="Phase4C", config=row["config"], session_hash=row["session_hash"],
                        turn_index=index, vocab_size=len(runtime.TOKENIZER),
                        special_ids=runtime.tokenizer_specials(),
                    )
                with lock:
                    actual_shared = longest_full_block_match(prompt, sent_prompts)
                    if actual_shared > expected_shared:
                        raise RuntimeError(f"EXPERIMENT_INVALID_LIFECYCLE: accidental longer prefix {episode_id}:{index}:{actual_shared}>{expected_shared}")
                    sent_prompts.append(prompt)
                if len(prompt) != target:
                    raise RuntimeError("prompt length mismatch")
                if index:
                    pause_ms = float(boundaries[episode_id, index - 1]["proxy_pause_ms"])
                    boundary_id = f"{episode_id}:b{index - 1}"
                    registered_ns = time.monotonic_ns()
                    deadline_ns = (controller.register(boundary_id, prompt[:expected_shared], prior_finish_ns)
                                   if controller else prior_finish_ns + round(pause_ms * 1e6))
                    if controller:
                        registered_ns = controller.registered[boundary_id]["registered_ns"]
                    wait_until(deadline_ns / 1e9)
                    actual_issue_ns = time.monotonic_ns()
                    with lock:
                        boundary_rows.append({"episode_id": episode_id, "boundary_index": index - 1,
                                              "boundary_id": f"{episode_id}:b{index - 1}",
                                              "previous_finish_ns": prior_finish_ns,
                                              "registered_ns": registered_ns,
                                              "target_resume_issue_ns": deadline_ns,
                                              "actual_issue_ns": actual_issue_ns,
                                              "pause_error_ms": (actual_issue_ns - deadline_ns) / 1e6,
                                              "proxy_pause_ms": pause_ms,
                                              "required_prefix_tokens": expected_shared})
                params: dict[str, Any] = {"slackkv_session": f"{episode_id}:turn:{index}"}
                if index:
                    params.update({"slackkv_trace_resume": True,
                                   "slackkv_first_token_gate": True,
                                   "slackkv_first_token_gate_limit": K})
                    if controller:
                        controller.mark_resume_started(boundary_id, actual_issue_ns)
                        with controller.lock:
                            if controller.registered[boundary_id]["decision"] == "OK":
                                params["max_offload_tokens"] = 0
                if index < 4:
                    next_boundary = boundaries[episode_id, index]
                    params.update({"slackkv_trace_next_boundary": next_boundary["boundary_id"],
                                   "slackkv_trace_source_required_tokens": int(next_boundary["required_prefix_tokens"])})
                def on_first(ns: int) -> None:
                    if controller and index:
                        controller.mark_first_token(boundary_id, ns)
                response = runtime.send(prompt, int(row["source_osl"]), stream=True,
                                        params=params, on_first=on_first)
                if len(response["tokens"]) != int(row["source_osl"]):
                    raise RuntimeError(f"output count mismatch:{episode_id}:{index}")
                prior_served = prompt + response["tokens"]
                prior_finish_ns = response["done_ns"]
                with lock:
                    turn_rows.append({"episode_id": episode_id, "turn_index": index,
                                      "prompt_tokens": len(prompt), "output_tokens": len(response["tokens"]),
                                      "prompt_sha256": token_hash(prompt),
                                      "output_sha256": token_hash(response["tokens"]),
                                      "issue_ns": response["issue_ns"], "first_ns": response["first_ns"],
                                      "finish_ns": prior_finish_ns})

        try:
            with ThreadPoolExecutor(max_workers=8) as workers:
                futures = [workers.submit(worker, episode) for episode in sorted(turns_by_episode)]
                for future in futures:
                    future.result(timeout=300)
        finally:
            sessions_done.set()
        if controller_thread:
            controller_thread.join(timeout=15)
            if controller_thread.is_alive():
                raise TimeoutError("dynamic controller did not drain")
            if controller.error:
                raise controller.error
        x = x_future.result(timeout=300)
        x_pool.shutdown(wait=True)
        if len(x["tokens"]) != X_OUTPUT_TOKENS:
            raise RuntimeError("X output count mismatch")
        last_agent_first = max(row["first_ns"] for row in turn_rows)
        x_tail_margin_ms = (x["done_ns"] - last_agent_first) / 1e6
        pause_errors = [abs(row["pause_error_ms"]) for row in boundary_rows]
        events = source_events(log)
        raw_log = log.read_text(encoding="utf-8", errors="replace")
        resume_trace = [json.loads(line.split("SLACKKV_RESUME_TRACE ", 1)[1])
                        for line in raw_log.splitlines() if "SLACKKV_RESUME_TRACE " in line]
        source_trace = [json.loads(line.split("SLACKKV_SOURCE_TRACE ", 1)[1])
                        for line in raw_log.splitlines() if "SLACKKV_SOURCE_TRACE " in line]
        capacities = parse_capacity_trace(raw_log)
        max_pre_first = max((int(event.get("num_slackkv_pre_first_token") or 0)
                             for event in resume_trace), default=0)
        trace_counts = {event: sum(row.get("event") == event for row in resume_trace)
                        for event in ("RESUME_ARRIVED", "FIRST_TOKEN", "REQUEST_FINISHED")}
        capacity_valid = all(psa_trace_valid(row) if row.get("progress_guard") is True
                             else capacity_invariant(row) for row in capacities)
        server_alive = proc.poll() is None
        oom = "out of memory" in raw_log.lower() or "outofmemoryerror" in raw_log.lower()
        lease_leaks = 0
        if controller:
            lease_leaks = (sum(row.get("event") == "LEASE_ACQUIRED" for row in controller.lease_events)
                           - sum(row.get("event") == "LEASE_RELEASED" for row in controller.lease_events))
        valid = (
            len(turn_rows) == 40 and len(boundary_rows) == 32
            and len({(row["episode_id"], row["turn_index"]) for row in turn_rows}) == 40
            and all(row["previous_finish_ns"] <= row["registered_ns"] <= row["actual_issue_ns"]
                    for row in boundary_rows)
            and all(row["target_resume_issue_ns"] - row["previous_finish_ns"] == round(row["proxy_pause_ms"] * 1e6)
                    for row in boundary_rows)
            and nearest_rank(pause_errors, .95) <= 10 and max(pause_errors) <= 30
            and (x_tail_margin_ms >= 3000 if stage == "smoke" else x_tail_margin_ms > 0)
            and max_pre_first <= K and server_alive and not oom
            and all(trace_counts[event] == 32 for event in trace_counts)
            and len({row["session"] for row in source_trace}) == 32
            and capacity_valid and lease_leaks == 0
            and (controller is None or (not controller.ledger.active and controller.release_queue.unsafe_release_count == 0))
        )
        result = {
            "stage": stage, "repetition": repetition,
            "policy": "OnDemand-K2" if policy == "A" else "FullSlackKV+PSA1-K2", "turns_complete": len(turn_rows),
            "boundaries": len(boundary_rows), "pre_materialized_boundary_count": 0,
            "X_tokens": len(x["tokens"]), "X_tail_margin_ms": x_tail_margin_ms,
            "pause_abs_P95_ms": nearest_rank(pause_errors, .95),
            "pause_abs_max_ms": max(pause_errors),
            "source_ready_count": sum(event["event"] == "SOURCE_READY" for event in events),
            "source_not_ready_count": sum(event["event"] == "SOURCE_NOT_READY" for event in events),
            "source_trace_event_count": len(source_trace),
            "source_trace_unique_resume_count": len({row["session"] for row in source_trace}),
            "source_events": events,
            "promotion_attempts": len(controller.promotions) if controller else 0,
            "logical_leaks": len(controller.ledger.active) if controller else 0,
            "lease_event_count": len(controller.lease_events) if controller else 0,
            "decisions": controller.decisions if controller else [],
            "valid": valid, "max_pre_first": max_pre_first,
            "resume_trace_counts": trace_counts, "capacity_valid": capacity_valid,
            "lease_leaks": lease_leaks, "server_alive": server_alive, "OOM": oom,
            "X_first_ns": x["first_ns"], "X_finish_ns": x["done_ns"],
            "X_decode_ms": (x["done_ns"] - x["first_ns"]) / 1e6,
            "X_TPOT_ms": (x["done_ns"] - x["first_ns"]) / 1e6 / (len(x["tokens"]) - 1),
            "turn_rows": turn_rows, "boundary_rows": boundary_rows,
            "source_trace": source_trace, "capacity_trace": capacities,
            "lease_events": controller.lease_events if controller else [],
            "release_events": controller.releases if controller else [],
            "peak_reserved_blocks": controller.ledger.peak_blocks if controller else 0,
            "promotions": ({key: {field: value for field, value in item.items() if field != "future"}
                            for key, item in controller.promotions.items()} if controller else {}),
            "resume_trace": [row for row in resume_trace if row.get("event") in {
                "RESUME_ARRIVED", "RESUME_ADMITTED", "FIRST_TOKEN", "REQUEST_FINISHED",
                "WAITING_FOR_REMOTE_KVS", "REMOTE_KV_READY", "FIRST_MODEL_SCHEDULED"}],
        }
        if stage == "smoke":
            (OUT / f"lifecycle_smoke_{policy.lower()}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        (run_dir / "run.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: value for key, value in result.items() if key not in {
            "source_events", "decisions", "turn_rows", "boundary_rows", "source_trace",
            "capacity_trace", "lease_events", "release_events", "promotions", "resume_trace"}}))
        if not valid:
            raise RuntimeError(f"EXPERIMENT_INVALID_{'LIFECYCLE' if stage == 'smoke' else 'MATRIX'}:{label}")
        return result
    finally:
        runtime.stop_server(proc, log, run_dir, label)
        if log.exists():
            lines = [line[:1200] for line in log.read_text(encoding="utf-8", errors="replace").splitlines()
                     if any(marker in line for marker in ("SLACKKV_", "ERROR", "OOM", "GPU KV cache size"))]
            (OUT / "server_logs").mkdir(parents=True, exist_ok=True)
            (OUT / "server_logs" / f"{label}_compact.log").write_text("\n".join(lines[-100:]) + "\n", encoding="utf-8")


if __name__ == "__main__":
    import sys
    run_lifecycle(sys.argv[1] if len(sys.argv) > 1 else "A")
