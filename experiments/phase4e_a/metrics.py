from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from experiments.phase4d_a.helpers import BYTES_PER_TOKEN

EXPECTED_SOURCE_TOKENS = 168_016
MINIMUM_H2D_BYTES = EXPECTED_SOURCE_TOKENS * BYTES_PER_TOKEN
MIB = 1024 * 1024


@dataclass(frozen=True)
class PolicySpec:
    name: str
    jit_enabled: bool
    capacity_guard_enabled: bool
    physical_lease_enabled: bool
    logical_reservation_enabled: bool
    K2_enabled: bool = True


POLICIES = (
    PolicySpec("OnDemand-K2", False, False, False, False),
    PolicySpec("GuardedJIT-Oracle-K2", True, True, False, False),
    PolicySpec("GuardedJIT+Lease-Oracle-K2", True, True, True, False),
    PolicySpec("FullSlackKV-Oracle-K2", True, True, True, True),
)
POLICY_BY_NAME = {policy.name: policy for policy in POLICIES}
LATIN_SQUARE = tuple(tuple(POLICIES[(repetition + position) % 4].name for position in range(4)) for repetition in range(4))


def nearest_rank(values: list[float], percentile: float) -> float:
    if not values or not 0 <= percentile <= 100:
        raise ValueError("nonempty values and percentile in [0,100] required")
    ordered = sorted(float(x) for x in values)
    return ordered[max(0, math.ceil(percentile / 100 * len(ordered)) - 1)]


def deadline_latency(first_token_ms: float, deadline_ms: float) -> float:
    return max(0.0, first_token_ms - deadline_ms)


def server_ttft(first_token_ms: float, arrival_ms: float) -> float:
    value = first_token_ms - arrival_ms
    if value < 0:
        raise ValueError("first token before scheduler arrival")
    return value


def h2d_bytes(native_transfer_bytes: int, external_source_tokens: int) -> tuple[int, int, int, int]:
    if native_transfer_bytes < 0 or external_source_tokens < 0:
        raise ValueError("negative transfer accounting")
    external = external_source_tokens * BYTES_PER_TOKEN
    total = native_transfer_bytes + external
    return native_transfer_bytes, external, total, max(0, total - MINIMUM_H2D_BYTES)


def source_conserved(local_tokens: int, external_tokens: int) -> bool:
    return local_tokens >= 0 and external_tokens >= 0 and local_tokens + external_tokens == EXPECTED_SOURCE_TOKENS


def lease_memory_time_mib_s(tokens: int, t6_ms: float, release_ms: float) -> float:
    if release_ms < t6_ms:
        raise ValueError("lease released before T6")
    return tokens * BYTES_PER_TOKEN / MIB * (release_ms - t6_ms) / 1000


def combined_decode_tok_s(agent_output_counts: list[int], first_ms: list[float], finish_ms: list[float], x_first_ms: float, x_finish_ms: float) -> float:
    """Phase 3U-FIX workload-level definition: all post-first tokens / common span."""
    if len(agent_output_counts) != len(first_ms) or len(first_ms) != len(finish_ms):
        raise ValueError("agent arrays differ in length")
    post_first = sum(count - 1 for count in agent_output_counts) + 255
    span_ms = max(*finish_ms, x_finish_ms) - min(*first_ms, x_first_ms)
    if span_ms <= 0:
        raise ValueError("nonpositive combined decode span")
    return post_first * 1000 / span_ms


def hypothesis_flags(a: dict[str, float], d: dict[str, float]) -> dict[str, bool]:
    return {
        "H1": d["deadline_latency_P95"] <= 0.90 * a["deadline_latency_P95"],
        "H2": d["total_H2D_MiB"] <= 1.05 * a["total_H2D_MiB"],
        "H3": d["X_slowdown"] <= a["X_slowdown"] + 0.05,
    }


def classify(stage_a_valid: int, stage_b_valid: int, flags: dict[str, bool]) -> str:
    if stage_a_valid != 4 or stage_b_valid != 16:
        return "EXPERIMENT_INVALID_MATRIX"
    return "RESULT_A_FULL_SLACKKV_DIRECTIONAL_BENEFIT" if all(flags.values()) else "RESULT_B_VALID_MATRIX_MIXED_TRADEOFF"


def median_range(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("empty policy metric")
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}
