"""Small, testable deterministic timing and fidelity helpers."""
from __future__ import annotations

import math
import random
from typing import Sequence


def quantile(values: Sequence[float], q: float) -> float:
    if not values or not 0 <= q <= 1:
        raise ValueError("values must be nonempty and q in [0, 1]")
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return ordered[lo] if lo == hi else ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def trace_tool_proxy_quantiles(pool: Sequence[float], count: int = 32, seed: int = 20260925) -> list[float]:
    if count <= 0 or not pool:
        raise ValueError("pause pool and count must be positive")
    ordered = sorted(pool)
    values = [quantile(ordered, (index + 0.5) / count) for index in range(count)]
    random.Random(seed).shuffle(values)
    return values


def actual_pause_ms(previous_response_end_ns: int, actual_issue_ns: int) -> float:
    return (actual_issue_ns - previous_response_end_ns) / 1_000_000.0


def pause_error_ms(actual_ms: float, proxy_ms: float) -> float:
    return actual_ms - proxy_ms


def block_alignment(source_cached: int, block_size: int = 16) -> tuple[int, int]:
    if source_cached < 0 or block_size <= 0:
        raise ValueError("invalid cache size")
    blocks = source_cached // block_size
    return blocks, source_cached - blocks * block_size


def result_classification(*, source_ok: bool, env_ok: bool, eligible_pro: int, eligible_verified: int,
                          calibration_ok: bool, smoke_ok: bool, full_ok: bool) -> str:
    if not source_ok:
        return "EXPERIMENT_INVALID_SOURCE"
    if not env_ok:
        return "EXPERIMENT_INVALID_ENVIRONMENT"
    if eligible_pro < 4 or eligible_verified < 4:
        return "RESULT_B_NATIVE_FIT_SPARSE"
    if not calibration_ok:
        return "RESULT_C_PREFIX_MEASUREMENT_INVALID"
    if not smoke_ok or not full_ok:
        return "RESULT_D_REPLAY_FIDELITY_FAIL"
    return "RESULT_A_HYBRID_REPLAY_CORRECT"
