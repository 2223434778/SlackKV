"""Frozen Phase 3C native-promotion scheduling helpers."""
from __future__ import annotations

import math

SESSIONS = ("A", "B", "C", "D", "E", "F")
DEADLINE_MS = {"A": 300.0, "B": 320.0, "C": 340.0, "D": 650.0, "E": 900.0, "F": 1300.0}
KV_MIB = 112.0


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


def effective_resume_latency_ms(
    first_token_s: float, deadline_s: float
) -> float:
    return max(0.0, (first_token_s - deadline_s) * 1000.0)


def memory_time_mib_s(early_residency_ms: list[float]) -> float:
    if any(value < 0 for value in early_residency_ms):
        raise ValueError("early residency cannot be negative")
    return KV_MIB * sum(early_residency_ms) / 1000.0
