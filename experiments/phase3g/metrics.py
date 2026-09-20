from __future__ import annotations


def aligned_event_ms(timestamp_monotonic: float, t0_monotonic: float) -> float:
    return (timestamp_monotonic - t0_monotonic) * 1000.0


def gpu_ready_residency_ms(admission_ms: float, promotion_completion_ms: float) -> float:
    return max(0.0, admission_ms - promotion_completion_ms)


def gate_wait_ms(first_blocked_ms: float | None, admission_ms: float) -> float:
    if first_blocked_ms is None:
        return 0.0
    return max(0.0, admission_ms - first_blocked_ms)


def decode_overlap_ms(a_first: float, a_finish: float,
                      b_first: float, b_finish: float) -> float:
    return max(0.0, min(a_finish, b_finish) - max(a_first, b_first))


def interval_max_concurrency(intervals: list[tuple[float, float]]) -> int:
    events = sorted([(a, 1) for a, _ in intervals] + [(b, -1) for _, b in intervals])
    current = maximum = 0
    for _, delta in events:
        current += delta
        maximum = max(maximum, current)
    return maximum
