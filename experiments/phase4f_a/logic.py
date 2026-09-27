from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from experiments.phase4d_b.experiment import frozen_jobs

GPU_BLOCKS = 4681
BUDGET_BLOCKS = 3510
P_SAFE_MS = 102
K = 2
BLOCK_SIZE = 16
WORKLOAD_SHA = "9b381336a70e73b0efbddbb79d2c88608c19663247e819c1e7a364efd2971105"
EXPECTED_PROGRESS = {
    "E04": 1278, "E00": 1431, "E05": 1292, "E02": 1328,
    "E01": 1337, "E06": 1321, "E07": 1450, "E03": 1359,
}


def progress_blocks(turn1_isl: int, block_size: int = BLOCK_SIZE) -> int:
    if turn1_isl <= 0 or block_size <= 0:
        raise ValueError("invalid prompt/block size")
    return math.ceil((turn1_isl + 1) / block_size)


def verified_jobs() -> list[dict[str, Any]]:
    jobs = frozen_jobs()
    actual = {job["episode_id"]: progress_blocks(int(job["turn1_isl"])) for job in jobs}
    if actual != EXPECTED_PROGRESS:
        raise RuntimeError(f"frozen progress footprint mismatch: {actual}")
    if sum(int(job["required_blocks"]) for job in jobs) != 10501:
        raise RuntimeError("frozen blocks mismatch")
    return jobs


def protected_episode(
    jobs: list[dict[str, Any]], states: Mapping[str, Mapping[str, Any]]
) -> str | None:
    unfinished = [job for job in jobs
                  if states[job["episode_id"]].get("first_token_ns") is None]
    if not unfinished:
        return None
    return min(unfinished, key=lambda job: (float(job["deadline_ms"]), job["episode_id"]))["episode_id"]


def psa_decision(free: int, required: int, progress: int, existing_lease: int,
                 matching_candidate: bool) -> tuple[str, int, int]:
    hypothetical = max(existing_lease, required if matching_candidate else 0)
    reserve = max(0, progress - hypothetical)
    if free < required:
        return "CAPACITY_REJECT", hypothetical, reserve
    if free - required < reserve:
        return "PROGRESS_REJECT", hypothetical, reserve
    return "ADMIT", hypothetical, reserve


def psa_trace_valid(event: Mapping[str, Any]) -> bool:
    try:
        free = int(event["free_blocks_before"])
        required = int(event["candidate_required_blocks"])
        after = int(event["free_blocks_after_candidate"])
        reserve = int(event["progress_reserve_blocks"])
        decision = str(event["decision"])
    except (KeyError, TypeError, ValueError):
        return False
    if after != free - required or reserve < 0:
        return False
    return ((decision == "ADMIT" and free >= required and after >= reserve)
            or (decision == "PROGRESS_REJECT" and free >= required and after < reserve)
            or (decision == "CAPACITY_REJECT" and free < required))


def rejected_response_valid(response: Mapping[str, Any], status: str) -> bool:
    return (response.get("status") == status
            and all(int(response.get(field, -1)) == 0 for field in (
                "external_tokens_loaded", "transfer_bytes", "model_forward_count",
                "generated_token_count", "mixed_prefetch_model_step_count",
            )))
