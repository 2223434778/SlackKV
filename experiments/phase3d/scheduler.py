"""Frozen Phase 3D native-promotion scheduling helpers."""
from __future__ import annotations

import math

SESSIONS = ("A", "B", "C", "D", "E", "F")
DEADLINE_MS = {"A": 300.0, "B": 320.0, "C": 340.0, "D": 650.0, "E": 900.0, "F": 1300.0}
KV_MIB = 112.0
P_SAFE_MS = 30.0
RESUME_GUARD_MS = 110.0


def freeze_p_native_ms(warm_times_ms: list[float]) -> int:
    if len(warm_times_ms) != 3 or any(value <= 0 for value in warm_times_ms):
        raise ValueError("exactly three positive warm times are required")
    return math.ceil(max(warm_times_ms))


def naive_target_start_ms(deadline_ms: float, p_native_ms: float) -> float:
    return deadline_ms - p_native_ms


def queue_safe_idle_s(
    now_s: float, deadlines_s: dict[str, float], p_native_ms: float
) -> float:
    cumulative_s = 0.0
    slacks: list[float] = []
    for session in sorted(deadlines_s, key=lambda item: (deadlines_s[item], item)):
        cumulative_s += p_native_ms / 1000.0
        slacks.append(deadlines_s[session] - now_s - cumulative_s)
    if not slacks:
        raise ValueError("QueueJIT requires at least one pending job")
    return min(slacks)


def merge_blackouts(deadlines_ms: dict[str, float], guard_ms: float) -> list[tuple[float, float]]:
    intervals = sorted((d, d + guard_ms) for d in deadlines_ms.values())
    merged: list[tuple[float, float]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def blackout_conflict(start_ms: float, end_ms: float, intervals: list[tuple[float, float]]) -> tuple[bool, tuple[float, float] | None]:
    for start, end in intervals:
        if start_ms < end and end_ms > start:
            return True, (start, end)
    return False, None


def feasible(start_ms: float, jobs: list[tuple[str, float]], intervals: list[tuple[float, float]], service_ms: float = P_SAFE_MS) -> bool:
    now = start_ms
    for _, deadline in sorted(jobs, key=lambda x: (x[1], x[0])):
        while True:
            active = next(((s, e) for s, e in intervals if s <= now < e), None)
            if active:
                now = active[1]
                continue
            nxt = next(((s, e) for s, e in intervals if s > now), None)
            if nxt and now + service_ms > nxt[0]:
                now = nxt[1]
                continue
            if now + service_ms > deadline:
                return False
            now += service_ms
            break
    return True


def latest_feasible_start(now_ms: float, jobs: list[tuple[str, float]], intervals: list[tuple[float, float]], service_ms: float = P_SAFE_MS) -> float:
    if not jobs or not feasible(now_ms, jobs, intervals, service_ms):
        return now_ms
    low, high = now_ms, min(d for _, d in jobs)
    while high - low > 0.5:
        mid = (low + high) / 2.0
        if feasible(mid, jobs, intervals, service_ms):
            low = mid
        else:
            high = mid
    return low


def effective_resume_latency_ms(
    first_token_s: float, deadline_s: float
) -> float:
    return max(0.0, (first_token_s - deadline_s) * 1000.0)


def memory_time_mib_s(early_residency_ms: list[float]) -> float:
    if any(value < 0 for value in early_residency_ms):
        raise ValueError("early residency cannot be negative")
    return KV_MIB * sum(early_residency_ms) / 1000.0
