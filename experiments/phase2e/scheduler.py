"""Small, deterministic scheduling primitives for the Phase 2E worker."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def _positive_service_time(service_time_s: float) -> None:
    if not math.isfinite(service_time_s) or service_time_s <= 0:
        raise ValueError("service_time_s must be finite and positive")


def naive_target_start_s(deadline_s: float, service_time_s: float) -> float:
    _positive_service_time(service_time_s)
    if not math.isfinite(deadline_s):
        raise ValueError("deadline_s must be finite")
    return deadline_s - service_time_s


def edf_order(deadlines_s: Mapping[str, float]) -> list[str]:
    if not deadlines_s:
        return []
    if any(not math.isfinite(deadline) for deadline in deadlines_s.values()):
        raise ValueError("deadlines must be finite")
    return sorted(deadlines_s, key=lambda session: (deadlines_s[session], session))


def queue_safe_idle_s(now_s: float, deadlines_s: Mapping[str, float], service_time_s: float) -> float:
    """Minimum prefix slack for EDF jobs currently pending on one worker."""
    _positive_service_time(service_time_s)
    if not math.isfinite(now_s):
        raise ValueError("now_s must be finite")
    ordered = edf_order(deadlines_s)
    if not ordered:
        raise ValueError("queue_safe_idle_s requires at least one pending job")
    return min(
        deadlines_s[session] - now_s - (index + 1) * service_time_s
        for index, session in enumerate(ordered)
    )


def completion_timing_ms(completion_s: float, deadline_s: float) -> tuple[float, float]:
    """Return (deadline stall, early residency) in milliseconds."""
    if not math.isfinite(completion_s) or not math.isfinite(deadline_s):
        raise ValueError("timestamps must be finite")
    delta_ms = (completion_s - deadline_s) * 1000.0
    return max(0.0, delta_ms), max(0.0, -delta_ms)


def aggregate_run_metrics(jobs: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    if not jobs:
        raise ValueError("cannot aggregate an empty job list")
    stalls = [float(job["deadline_stall_ms"]) for job in jobs]
    early = [float(job["early_residency_ms"]) for job in jobs]
    if any(not math.isfinite(value) or value < 0 for value in (*stalls, *early)):
        raise ValueError("stall and residency values must be finite and non-negative")
    return {
        "deadline_miss_count": sum(stall > 0 for stall in stalls),
        "max_deadline_stall_ms": max(stalls),
        "total_early_residency_ms": sum(early),
    }
