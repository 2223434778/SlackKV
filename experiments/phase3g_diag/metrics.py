from __future__ import annotations

import hashlib
import json
from typing import Iterable


def matching_prefix_length(left: list[int], right: list[int]) -> int:
    n = 0
    for a, b in zip(left, right):
        if a != b:
            break
        n += 1
    return n


def first_divergence_position(left: list[int], right: list[int]) -> int | None:
    n = matching_prefix_length(left, right)
    return n + 1 if n < min(len(left), len(right)) or len(left) != len(right) else None


def sequence_hash(tokens: list[int]) -> str:
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def unique_sequence_counts(sequences: Iterable[list[int]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for seq in sequences:
        h = sequence_hash(seq)
        out[h] = out.get(h, 0) + 1
    return out


def overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def interval_concurrency(intervals: list[tuple[float, float]]) -> tuple[int, float]:
    events = sorted([(s, 1) for s, _ in intervals] + [(e, -1) for _, e in intervals])
    cur = maximum = 0
    area = 0.0
    previous = events[0][0] if events else 0.0
    for ts, delta in events:
        area += (ts - previous) * cur
        cur += delta
        maximum = max(maximum, cur)
        previous = ts
    return maximum, area


def structural_schedule(rows: dict[str, dict]) -> bool:
    try:
        a, b, c = (rows[x] for x in ("A", "B", "C"))
        return (
            a["first_token_step"] < b["admission_step"] < a["finish_step"]
            and b["first_token_step"] < c["admission_step"] < b["finish_step"]
            and a["admission_step"] < b["admission_step"] < c["admission_step"]
        )
    except (KeyError, TypeError):
        return False
