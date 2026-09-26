from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from experiments.phase3f.native_client import native_prefetch
from experiments.phase4d_a import runtime
from experiments.phase4d_a_fix.release_fix import (
    EXPECTED_GPU_BLOCKS,
    LeaseTraceReader,
    parse_lease_trace,
    smallest_prefix_groups,
)
from experiments.phase4d_b.capacity import capacity_invariant, parse_capacity_trace, validate_capacity_rejection
from experiments.phase4d_b.capacity_client import capacity_prefetch
from experiments.phase4d_b.experiment import (
    OUT,
    RESERVATION_BUDGET,
    reconstructed_data,
    frozen_jobs,
    validate_runtime,
)
from experiments.phase4d_a.experiment import materialize_and_evict, write_csv, write_json

MIN_AVAILABLE_BYTES = 16 * 1024**3
WATCHDOG_SECONDS = 5.0


def available_memory_bytes() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable missing")


def check_native_result(result: dict[str, Any], expected_tokens: int) -> None:
    timestamps = result.get("phase3m_timestamps") or {}
    if result.get("status") != "ok":
        raise RuntimeError(f"setup native promotion failed: {result!r}")
    if int(result.get("external_tokens_loaded", -1)) != expected_tokens:
        raise RuntimeError("native promotion external token count mismatch")
    if int(result.get("transfer_bytes", 0)) <= 0:
        raise RuntimeError("native promotion transferred no bytes")
    if not all(key in timestamps for key in ("T0", "T6", "T7")):
        raise RuntimeError("native promotion missing T0/T6/T7")
    for name in ("model_forward_count", "generated_token_count", "mixed_prefetch_model_step_count"):
        if int(result.get(name, -1)) != 0:
            raise RuntimeError(f"native prefetch invariant failed: {name}")


def run_case(side: str, repetition: int, jobs: list[dict[str, Any]], data: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    group = smallest_prefix_groups(jobs)
    count = 2 if side == "safe" else 3
    candidate = group["ordered_jobs"][count]
    setup_jobs = group["ordered_jobs"][: count + (1 if side == "unsafe" else 1)]
    if side == "safe":
        setup_jobs = group["ordered_jobs"][:3]
        owners = group["ordered_jobs"][:2]
    else:
        setup_jobs = group["ordered_jobs"][:4]
        owners = group["ordered_jobs"][:3]
    label = f"g1_{side}_rep{repetition}"
    if available_memory_bytes() < MIN_AVAILABLE_BYTES:
        raise RuntimeError("EXPERIMENT_INVALID_ENVIRONMENT:MemAvailable below 16 GiB")
    proc, log = runtime.start_server(label, connector=True, out=OUT)
    cap_rows: list[dict[str, Any]] = []
    lease_rows: list[dict[str, Any]] = []
    try:
        _, block_size = runtime.wait_ready(proc)
        runtime_info = validate_runtime(log, block_size)
        subset_data = {row["episode_id"]: data[row["episode_id"]] for row in setup_jobs}
        prep = materialize_and_evict(proc, log, label, setup_jobs, subset_data, OUT)
        for job in owners:
            result = native_prefetch(
                subset_data[job["episode_id"]]["prefix"],
                lease_key=job["lease_key"],
                timeout_s=WATCHDOG_SECONDS,
            )
            check_native_result(result, int(job["required_prefix_tokens"]))
        guarded = data[candidate["episode_id"]]
        response_start = time.monotonic()
        response = capacity_prefetch(
            guarded["prefix"],
            required_blocks=int(candidate["required_blocks"]),
            lease_key=candidate["lease_key"],
            timeout_s=WATCHDOG_SECONDS,
        )
        response_ms = (time.monotonic() - response_start) * 1000
        if side == "safe":
            check_native_result(response, int(candidate["required_prefix_tokens"]))
            expected_decision = "ADMIT"
        else:
            if not validate_capacity_rejection(response):
                raise RuntimeError(f"invalid capacity rejection: {response!r}")
            if response_ms >= 500:
                raise RuntimeError(f"capacity rejection exceeded 500 ms: {response_ms:.3f}")
            expected_decision = "REJECT"

        raw_log = log.read_text(encoding="utf-8", errors="replace")
        capacity_events = parse_capacity_trace(raw_log)
        matching = [
            event for event in capacity_events
            if event.get("lease_key") == candidate["lease_key"]
        ]
        if len(matching) != 1:
            raise RuntimeError(f"expected one capacity trace for {candidate['episode_id']}: {matching!r}")
        event = matching[0]
        if not capacity_invariant(event) or event.get("decision") != expected_decision:
            raise RuntimeError(f"capacity invariant/decision failure: {event!r}")
        if event.get("total_blocks") != EXPECTED_GPU_BLOCKS:
            raise RuntimeError(f"capacity trace total mismatch: {event!r}")
        if int(event.get("required_blocks", -1)) != int(candidate["required_blocks"]):
            raise RuntimeError(f"capacity required mismatch: {event!r}")

        lease_events = parse_lease_trace(raw_log)
        for lease_event in lease_events:
            lease_rows.append({"stage": "G1", "side": side, "repetition": repetition, **lease_event})
        for owner in owners:
            owner_acquires = [
                row for row in lease_events
                if row.get("lease_key") == owner["lease_key"]
                and row.get("event") == "LEASE_ACQUIRED"
            ]
            if len(owner_acquires) != 1 or int(owner_acquires[0].get("num_blocks", -1)) != int(owner["required_blocks"]):
                raise RuntimeError(f"owner lease acquire mismatch for {owner['episode_id']}")
        rejected_acquires = [
            row for row in lease_events
            if row.get("lease_key") == candidate["lease_key"]
            and row.get("event") == "LEASE_ACQUIRED"
        ]
        if side == "unsafe" and rejected_acquires:
            raise RuntimeError("rejected request acquired a physical lease")

        cap_row = {
            "stage": "G1",
            "side": side,
            "repetition": repetition,
            "episode_id": candidate["episode_id"],
            "request_id": event.get("request_id"),
            "lease_key": event.get("lease_key"),
            "required_blocks": event.get("required_blocks"),
            "total_blocks": event.get("total_blocks"),
            "free_blocks_before": event.get("free_blocks_before"),
            "used_blocks_before": event.get("used_blocks_before"),
            "running_unique_blocks": event.get("running_unique_blocks"),
            "waiting_unique_blocks": event.get("waiting_unique_blocks"),
            "lease_unique_blocks": event.get("lease_unique_blocks"),
            "num_running": event.get("num_running"),
            "num_waiting": event.get("num_waiting"),
            "decision": event.get("decision"),
            "endpoint_status": response.get("status"),
            "response_ms": response_ms,
            "external_tokens_loaded": response.get("external_tokens_loaded"),
            "transfer_bytes": response.get("transfer_bytes"),
            "model_forward_count": response.get("model_forward_count"),
            "generated_token_count": response.get("generated_token_count"),
            "candidate_lease_acquires": len(rejected_acquires),
            "rejected_request_lease_acquires": len(rejected_acquires) if side == "unsafe" else "NA",
        }
        cap_rows.append(cap_row)
        result = {
            "stage": "G1",
            "side": side,
            "repetition": repetition,
            "episode_ids": "|".join(row["episode_id"] for row in setup_jobs),
            "leased_episode_ids": "|".join(row["episode_id"] for row in owners),
            "candidate_episode": candidate["episode_id"],
            "total_required_blocks": sum(int(row["required_blocks"]) for row in setup_jobs),
            "gpu_total_blocks": runtime_info["total_gpu_kv_blocks"],
            "materialization_valid": all(row["validation"] for row in prep["materialization"]),
            "owner_promotions": len(owners),
            "owner_promotions_valid": True,
            "required_blocks": int(candidate["required_blocks"]),
            "free_blocks_before": event.get("free_blocks_before"),
            "decision": event.get("decision"),
            "endpoint_status": response.get("status"),
            "response_ms": response_ms,
            "transfer_bytes": response.get("transfer_bytes"),
            "candidate_lease_acquires": len(rejected_acquires),
            "rejected_request_lease_acquires": len(rejected_acquires) if expected_decision == "REJECT" else "NA",
            "pass": True,
        }
        return result, cap_rows, lease_rows
    finally:
        runtime.stop_server(proc, log, OUT, label)
        try:
            text = log.read_text(encoding="utf-8", errors="replace")
            compact = [
                line[:1000]
                for line in text.splitlines()
                if "SLACKKV_CAPACITY_TRACE" in line or "SLACKKV_LEASE_TRACE" in line
            ]
            (OUT / "server_logs" / f"{label}_capacity_lease.log").write_text(
                "\n".join(compact[-300:]) + "\n", encoding="utf-8"
            )
            log.unlink(missing_ok=True)
        except OSError:
            pass


def run_g1() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = frozen_jobs()
    groups = smallest_prefix_groups(jobs)
    expected = ["E04", "E05", "E06", "E00"]
    if groups["episode_ids"] != expected or [groups[f"N{i}"] for i in range(1, 5)] != [1271, 2545, 3821, 5098]:
        raise RuntimeError("frozen mechanical prefix ordering changed")
    data = reconstructed_data(jobs)
    run_rows: list[dict[str, Any]] = []
    cap_rows: list[dict[str, Any]] = []
    lease_rows: list[dict[str, Any]] = []
    for side in ("safe", "unsafe"):
        for repetition in (1, 2):
            row, capacities, leases = run_case(side, repetition, jobs, data)
            run_rows.append(row)
            cap_rows.extend(capacities)
            lease_rows.extend(leases)
            write_csv(OUT / "guard_micro_runs.csv", run_rows)
            write_csv(OUT / "capacity_trace.csv", cap_rows)
            write_csv(OUT / "lease_events.csv", lease_rows)
    safe = [row for row in run_rows if row["side"] == "safe"]
    unsafe = [row for row in run_rows if row["side"] == "unsafe"]
    report = {
        "safe_pass": len(safe) == 2 and all(row["pass"] and row["decision"] == "ADMIT" and row["endpoint_status"] == "ok" for row in safe),
        "unsafe_pass": len(unsafe) == 2 and all(row["pass"] and row["decision"] == "REJECT" and row["endpoint_status"] == "capacity_rejected" and row["response_ms"] < 500 and row["transfer_bytes"] == 0 and row["rejected_request_lease_acquires"] == 0 for row in unsafe),
        "safe_runs": safe,
        "unsafe_runs": unsafe,
        "physical_capacity_crossing": groups,
    }
    write_json(OUT / "guard_micro_trace.json", report)
    if not report["safe_pass"] or not report["unsafe_pass"]:
        write_json(OUT / "g1_gate.json", {"pass": False, **report})
        return report
    write_json(OUT / "g1_gate.json", {"pass": True, **report})
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["g1"], required=True)
    args = parser.parse_args()
    if args.mode != "g1":
        raise ValueError("Phase 4D-B G1 runner does not implement performance stages")
    result = run_g1()
    if not result["safe_pass"] or not result["unsafe_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
