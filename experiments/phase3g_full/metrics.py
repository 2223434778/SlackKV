from __future__ import annotations

import hashlib
import json


def percentile_linear(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires values")
    rank = (len(ordered) - 1) * percentile / 100.0
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def overlap(left: tuple[float, float], right: tuple[float, float]) -> float:
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def interval_concurrency(intervals: list[tuple[float, float]]) -> tuple[int, float]:
    events = sorted([(start, 1) for start, _ in intervals] + [(end, -1) for _, end in intervals])
    current = maximum = 0
    area = 0.0
    previous = events[0][0] if events else 0.0
    for timestamp, delta in events:
        area += (timestamp - previous) * current
        current += delta
        maximum = max(maximum, current)
        previous = timestamp
    return maximum, area


def sequence_hash(token_ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(token_ids, separators=(",", ":")).encode()).hexdigest()


def gate_wait_ms(first_blocked_ms: float | None, admission_ms: float) -> float:
    return 0.0 if first_blocked_ms is None else max(0.0, admission_ms - first_blocked_ms)


def gpu_ready_residency_ms(promotion_completion_ms: float | None, admission_ms: float) -> float:
    return 0.0 if promotion_completion_ms is None else max(0.0, admission_ms - promotion_completion_ms)


def decode_throughput_tok_s(post_first_tokens: int, first_times_ms: list[float], finish_times_ms: list[float]) -> float:
    span_s = (max(finish_times_ms) - min(first_times_ms)) / 1000.0
    return post_first_tokens / span_s


def same_step_pairs(steps: list[int]) -> int:
    return sum(steps[i] == steps[j] for i in range(len(steps)) for j in range(i + 1, len(steps)))
