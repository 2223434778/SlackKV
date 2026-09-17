"""Pure timing and accounting helpers for Phase 2H."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

SESSIONS = ("A", "B", "C", "D", "E", "F")
DEADLINE_MS = {
    "A": 300.0,
    "B": 320.0,
    "C": 340.0,
    "D": 650.0,
    "E": 900.0,
    "F": 1300.0,
}
P_EST_MS = 42.0
KV_MIB = 112.0


def naive_target_start_ms(deadline_ms: float, service_ms: float = P_EST_MS) -> float:
    if not math.isfinite(deadline_ms) or not math.isfinite(service_ms) or service_ms <= 0:
        raise ValueError("deadline and positive service estimate must be finite")
    return deadline_ms - service_ms


def queue_safe_idle_s(
    now_s: float,
    deadlines_s: Mapping[str, float],
    service_s: float = P_EST_MS / 1000.0,
) -> float:
    if not math.isfinite(now_s) or not math.isfinite(service_s) or service_s <= 0:
        raise ValueError("time and positive service estimate must be finite")
    if not deadlines_s or any(not math.isfinite(value) for value in deadlines_s.values()):
        raise ValueError("at least one finite deadline is required")
    ordered = sorted(deadlines_s, key=lambda session: (deadlines_s[session], session))
    return min(
        deadlines_s[session] - now_s - (index + 1) * service_s
        for index, session in enumerate(ordered)
    )


def effective_resume_latency_ms(
    request_sent_s: float,
    ttft_ms: float,
    t0_s: float,
    deadline_ms: float,
) -> float:
    values = (request_sent_s, ttft_ms, t0_s, deadline_ms)
    if any(not math.isfinite(value) for value in values) or ttft_ms < 0 or deadline_ms < 0:
        raise ValueError("timestamps and non-negative durations must be finite")
    return (request_sent_s - t0_s) * 1000.0 + ttft_ms - deadline_ms


def memory_time_mib_s(early_residency_ms: Sequence[float], kv_mib: float = KV_MIB) -> float:
    if not math.isfinite(kv_mib) or kv_mib <= 0:
        raise ValueError("kv_mib must be finite and positive")
    if any(not math.isfinite(value) or value < 0 for value in early_residency_ms):
        raise ValueError("early residency must be finite and non-negative")
    return sum(early_residency_ms) * kv_mib / 1000.0


def percentile_linear(values: Sequence[float], percentile: float) -> float:
    """Linear interpolation at rank (n-1)*q, matching NumPy's default."""
    if not values or not 0 <= percentile <= 100:
        raise ValueError("non-empty values and percentile in [0, 100] required")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in ordered):
        raise ValueError("percentile values must be finite")
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
