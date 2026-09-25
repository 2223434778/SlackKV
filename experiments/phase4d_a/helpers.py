from __future__ import annotations

import csv
import hashlib
import itertools
import math
from pathlib import Path
from typing import Any

BYTES_PER_TOKEN = 28_672
BLOCK_TOKENS = 16
GPU_KV_BYTES = 2_147_483_648
CPU_KV_BYTES = 8_589_934_592
HANDOFF_GUARD_MS = 100.0
RESERVATION_END_GUARD_MS = 250.0
RESERVATION_FRACTION = 0.75
K2 = 2
X_PROMPT_TOKENS = 9_500
X_OUTPUT_TOKENS = 256
FROZEN_SHA256 = "9b381336a70e73b0efbddbb79d2c88608c19663247e819c1e7a364efd2971105"
FROZEN_EXPECTED = {
    "E00": (171.0, 20432, 1277, 22885, 2453, 84),
    "E01": (237.0, 21280, 1330, 21388, 108, 47),
    "E02": (219.0, 21168, 1323, 21235, 67, 53),
    "E03": (2037.9375, 20896, 1306, 21728, 832, 42),
    "E04": (120.0, 20336, 1271, 20434, 98, 98),
    "E05": (186.0, 20384, 1274, 20661, 277, 148),
    "E06": (686.0, 20416, 1276, 21135, 719, 63),
    "E07": (1175.0, 23104, 1444, 23195, 91, 53),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_boundary_jobs(path: Path) -> list[dict[str, Any]]:
    by_episode: dict[str, list[dict[str, str]]] = {}
    for row in csv.DictReader(path.open(encoding="utf-8")):
        by_episode.setdefault(row["episode_id"], []).append(row)
    jobs = []
    for episode, turns in sorted(by_episode.items()):
        turns.sort(key=lambda row: int(row["replay_turn_index"]))
        if len(turns) != 5 or int(turns[0]["replay_turn_index"]) != 0 or int(turns[1]["replay_turn_index"]) != 1:
            raise AssertionError(f"{episode}: expected five ordered turns")
        t0, t1 = turns[0], turns[1]
        job = {
            "episode_id": episode, "config": t0["config"], "session_hash": t0["session_hash"],
            "boundary_index": 0, "proxy_pause_ms": float(t0["proxy_pause_ms"]),
            "deadline_ms": float(t0["proxy_pause_ms"]),
            "required_prefix_tokens": int(t1["target_cached_tokens"]),
            "required_blocks": int(t1["target_cached_blocks"]),
            "turn1_isl": int(t1["source_isl"]), "turn1_isl_new": int(t1["source_isl_new"]),
            "resume_osl": int(t1["source_osl"]), "turn0_isl": int(t0["source_isl"]),
            "turn0_osl": int(t0["source_osl"]), "turn1_seed": t1["synthetic_seed"],
            "lease_key": f"{episode}:turn:1",
        }
        if job["required_prefix_tokens"] % BLOCK_TOKENS:
            raise AssertionError("frozen reusable prefix must be block aligned")
        if job["required_blocks"] != job["required_prefix_tokens"] // BLOCK_TOKENS:
            raise AssertionError("frozen block count disagrees with target tokens")
        expected = FROZEN_EXPECTED[episode]
        observed = (job["proxy_pause_ms"], job["required_prefix_tokens"], job["required_blocks"],
                    job["turn1_isl"], job["turn1_isl_new"], job["resume_osl"])
        if not math.isclose(observed[0], expected[0], abs_tol=1e-9) or observed[1:] != expected[1:]:
            raise AssertionError(f"frozen workload changed for {episode}: {observed!r}")
        jobs.append(job)
    if len(jobs) != 8:
        raise AssertionError(f"expected 8 boundary-0 jobs, got {len(jobs)}")
    if sum(j["required_prefix_tokens"] for j in jobs) != 168016:
        raise AssertionError("frozen cached token sum mismatch")
    if sum(j["required_blocks"] for j in jobs) != 10501:
        raise AssertionError("frozen block sum mismatch")
    return jobs


def minimum_h2d_bytes(jobs: list[dict[str, Any]]) -> int:
    return sum(int(j["required_prefix_tokens"]) * BYTES_PER_TOKEN for j in jobs)


def minimum_h2d_mib(jobs: list[dict[str, Any]]) -> float:
    return minimum_h2d_bytes(jobs) / (1024.0 ** 2)


def duplicate_h2d_bytes(native_bytes: int, external_tokens: int, jobs: list[dict[str, Any]]) -> int:
    total = int(native_bytes) + int(external_tokens) * BYTES_PER_TOKEN
    return max(0, total - minimum_h2d_bytes(jobs))


def reservation_budget(total_gpu_kv_blocks: int, jobs: list[dict[str, Any]]) -> int:
    if total_gpu_kv_blocks <= 0:
        raise ValueError("total GPU KV blocks must be positive")
    budget = math.floor(RESERVATION_FRACTION * total_gpu_kv_blocks)
    if max(j["required_blocks"] for j in jobs) >= budget:
        raise ValueError("mechanical reservation budget cannot admit the largest frozen job")
    return budget


def maximum_simultaneous_jobs(jobs: list[dict[str, Any]], budget: int) -> int:
    sizes = [int(j["required_blocks"]) for j in jobs]
    for count in range(len(sizes), 0, -1):
        if any(sum(combo) <= budget for combo in itertools.combinations(sizes, count)):
            return count
    return 0


def pressure_request_count(total_gpu_kv_blocks: int) -> int:
    full_blocks = X_PROMPT_TOKENS // BLOCK_TOKENS
    return math.ceil(total_gpu_kv_blocks / full_blocks) + 1


def frozen_service_budget(max_t0_t6_ms: float) -> int:
    if max_t0_t6_ms <= 0:
        raise ValueError("positive T0→T6 measurements are required")
    return math.ceil(1.2 * max_t0_t6_ms)
def block_size_valid(actual: int) -> bool:
    return actual == BLOCK_TOKENS


def source_conserved(local_tokens: int, external_tokens: int, expected: int = 168016) -> bool:
    return local_tokens >= 0 and external_tokens >= 0 and local_tokens + external_tokens == expected


def gate_valid(max_pre_first: int, limit: int = K2) -> bool:
    return 0 <= max_pre_first <= limit


def promotion_cancelled(resume_started: bool, committed: bool) -> bool:
    return resume_started and not committed


def physical_before_logical_release(physical_release_ns: int, logical_release_ns: int) -> bool:
    return physical_release_ns <= logical_release_ns


def prefix_memory_time_mib_s(tokens: int, handoff_ms: float, t6_ms: float) -> float:
    return tokens * BYTES_PER_TOKEN / (1024.0 ** 2) * max(0.0, handoff_ms - t6_ms) / 1000.0


class ReservationLedger:
    def __init__(self, budget_blocks: int) -> None:
        if budget_blocks <= 0:
            raise ValueError("budget must be positive")
        self.budget_blocks = budget_blocks
        self.active: dict[str, int] = {}
        self.peak_blocks = 0

    @property
    def reserved_blocks(self) -> int:
        return sum(self.active.values())

    def reserve(self, key: str, blocks: int) -> bool:
        if key in self.active:
            raise AssertionError(f"duplicate reservation {key}")
        if blocks <= 0 or self.reserved_blocks + blocks > self.budget_blocks:
            return False
        self.active[key] = blocks
        self.peak_blocks = max(self.peak_blocks, self.reserved_blocks)
        assert self.reserved_blocks <= self.budget_blocks
        return True

    def release(self, key: str) -> int:
        if key not in self.active:
            raise KeyError(key)
        return self.active.pop(key)


def result_classification(*, setup_valid: bool, calibration_valid: bool, mechanism_valid: bool,
                          all_runs_complete: bool, benefit_valid: bool) -> str:
    if not setup_valid:
        return "EXPERIMENT_INVALID_SETUP"
    if not calibration_valid:
        return "EXPERIMENT_INVALID_CALIBRATION"
    if not mechanism_valid:
        return "RESULT_C_HYBRID_MECHANISM_INVALID"
    if not all_runs_complete:
        return "INCOMPLETE"
    return "RESULT_A_HYBRID_MECHANISM_WITH_BENEFIT" if benefit_valid else "RESULT_B_HYBRID_MECHANISM_VALID_TRADEOFF"
