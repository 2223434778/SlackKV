from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.phase4d_b.experiment import frozen_jobs

GPU_BLOCKS = 4681
WORKLOAD_SHA = "9b381336a70e73b0efbddbb79d2c88608c19663247e819c1e7a364efd2971105"
STAGES = ("R1", "R2")
CONDITIONS = {
    "U3_UNLEASED_HEAD_LEASE3": (("E00", "E01", "E02"), True),
    "H2_HEADROOM_LEASE2": (("E00", "E01"), True),
    "N3_EVICTABLE_PREFETCH3": (("E00", "E01", "E02"), False),
    "S3_HEAD_LEASED": (("E04", "E00", "E05"), True),
}
EXPECTED_BLOCKS = {
    "E04": 1271, "E05": 1274, "E06": 1276, "E00": 1277,
    "E03": 1306, "E02": 1323, "E01": 1330, "E07": 1444,
}


def verify_jobs() -> list[dict[str, Any]]:
    jobs = frozen_jobs()
    actual = {job["episode_id"]: int(job["required_blocks"]) for job in jobs}
    if actual != EXPECTED_BLOCKS:
        raise RuntimeError(f"EXPERIMENT_INVALID_ENVIRONMENT:frozen blocks {actual}")
    if min(jobs, key=lambda job: float(job["deadline_ms"]))["episode_id"] != "E04":
        raise RuntimeError("EXPERIMENT_INVALID_ENVIRONMENT:E04 is not earliest")
    if not math.isclose(float(next(job for job in jobs if job["episode_id"] == "E04")["deadline_ms"]), 120.0):
        raise RuntimeError("EXPERIMENT_INVALID_ENVIRONMENT:E04 deadline")
    return jobs


def condition_blocks(name: str) -> int:
    return sum(EXPECTED_BLOCKS[episode] for episode in CONDITIONS[name][0])


def should_emit_blocked(blocked_count: int) -> bool:
    return blocked_count > 0 and blocked_count & (blocked_count - 1) == 0


def parse_structured_log(log: Path, marker: str) -> list[dict[str, Any]]:
    if not log.exists():
        return []
    out: list[dict[str, Any]] = []
    prefix = marker + " "
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if prefix not in line:
            continue
        try:
            event = json.loads(line.split(prefix, 1)[1])
        except (ValueError, IndexError):
            continue
        if isinstance(event, dict):
            out.append(event)
    return out


def lease_unique_blocks(leases: dict[str, list[int]]) -> int:
    return len({block_id for blocks in leases.values() for block_id in blocks})


def counters(events: list[dict[str, Any]]) -> dict[str, int]:
    count = Counter(event.get("event") for event in events)
    return {
        "resume_arrived": count["RESUME_ARRIVED"],
        "resume_admitted": count["RESUME_ADMITTED"],
        "first_tokens": count["FIRST_TOKEN"],
        "finished": count["REQUEST_FINISHED"],
    }


def first_e04_block(waiting: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next(
        (event for event in waiting
         if event.get("event") == "WAITING_ALLOCATE_NONE"
         and event.get("session") == "E04:turn:1"),
        None,
    )


def u3_nonprogress(
    *,
    alive: bool,
    oom: bool,
    server_error: bool,
    waiting: list[dict[str, Any]],
    resume: list[dict[str, Any]],
    lease: list[dict[str, Any]],
    final_elapsed_s: float,
) -> bool:
    block = first_e04_block(waiting)
    counts = counters(resume)
    acquired = {event.get("lease_key") for event in lease if event.get("event") == "LEASE_ACQUIRED"}
    released = [event for event in lease if event.get("event") == "LEASE_RELEASED"]
    return (
        final_elapsed_s >= 12
        and alive and not oom and not server_error
        and block is not None
        and block.get("matching_lease") is False
        and block.get("will_break_waiting_loop") is True
        and int(block["free_blocks"]) < EXPECTED_BLOCKS["E04"]
        and {"E00:turn:1", "E01:turn:1", "E02:turn:1"} <= acquired
        and not released
        and counts["resume_admitted"] == 0
        and counts["first_tokens"] == 0
        and counts["finished"] == 0
    )


def r1_result(by_condition: dict[str, list[str]]) -> str:
    u3 = by_condition.get("U3_UNLEASED_HEAD_LEASE3", [])
    if len(u3) != 2:
        return "INCOMPLETE_R1"
    if u3 == ["PROGRESS", "PROGRESS"]:
        return "RESULT_C_HYPOTHESIS_NOT_SUPPORTED"
    if u3 != ["LEASE_HOL_NONPROGRESS", "LEASE_HOL_NONPROGRESS"]:
        return "RESULT_B_ROOT_CAUSE_PARTIAL"
    if any(by_condition.get(name) != ["PROGRESS", "PROGRESS"]
           for name in ("H2_HEADROOM_LEASE2", "N3_EVICTABLE_PREFETCH3", "S3_HEAD_LEASED")):
        return "RESULT_B_ROOT_CAUSE_PARTIAL"
    return "R1_EXPECTED_PATTERN"


def final_result(r1: str, r2_u3: list[str], r2_s3: list[str]) -> str:
    if r1 != "R1_EXPECTED_PATTERN":
        return r1
    if r2_u3 == ["LEASE_HOL_NONPROGRESS"] * 3 and r2_s3 == ["PROGRESS"] * 3:
        return "RESULT_A_LEASE_HOL_CIRCULAR_WAIT_CONFIRMED"
    if r2_u3 == ["PROGRESS"] * 3:
        return "RESULT_C_HYPOTHESIS_NOT_SUPPORTED"
    return "RESULT_B_ROOT_CAUSE_PARTIAL"
