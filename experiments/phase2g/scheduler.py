"""Frozen timing arithmetic for the Phase 2G single-worker experiment."""
from __future__ import annotations

import math
import statistics
from collections.abc import Mapping

P_EST_MS = 42.0
DEADLINE_MS = {"A": 350.0, "B": 370.0}


def naive_target_start_ms(deadline_ms: float, service_ms: float = P_EST_MS) -> float:
    if not math.isfinite(deadline_ms) or not math.isfinite(service_ms) or service_ms <= 0:
        raise ValueError("deadline and positive service estimate must be finite")
    return deadline_ms - service_ms


def queue_safe_idle_s(now_s: float, deadlines_s: Mapping[str, float], service_s: float = P_EST_MS / 1000.0) -> float:
    if not math.isfinite(now_s) or not math.isfinite(service_s) or service_s <= 0:
        raise ValueError("time and positive service estimate must be finite")
    if not deadlines_s or any(not math.isfinite(x) for x in deadlines_s.values()):
        raise ValueError("at least one finite deadline is required")
    ordered = sorted(deadlines_s, key=lambda session: (deadlines_s[session], session))
    return min(deadlines_s[s] - now_s - (i + 1) * service_s for i, s in enumerate(ordered))


def completion_timing_ms(completion_s: float, deadline_s: float) -> tuple[float, float]:
    if not math.isfinite(completion_s) or not math.isfinite(deadline_s):
        raise ValueError("timestamps must be finite")
    delta = (completion_s - deadline_s) * 1000.0
    return max(0.0, delta), max(0.0, -delta)


def warm_state_valid(promotion_durations_ms: list[float], threshold_ms: float = 60.0) -> bool:
    if not promotion_durations_ms or any(not math.isfinite(x) or x < 0 for x in promotion_durations_ms):
        return False
    return statistics.median(promotion_durations_ms) <= threshold_ms
