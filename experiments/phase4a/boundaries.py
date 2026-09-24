"""Pure validation helpers for a future trace-observed boundary table."""

from __future__ import annotations

import math


def consecutive_rounds(indices: list[int]) -> bool:
    return bool(indices) and all(value == indices[0] + offset for offset, value in enumerate(indices))


def valid_timing(decision_ms: float | None, resume_ms: float | None) -> bool:
    return decision_ms is not None and resume_ms is not None and math.isfinite(decision_ms) and math.isfinite(resume_ms) and resume_ms >= decision_ms


def context_replayable(previous_reusable_tokens: int | None, next_input_tokens: int | None) -> bool:
    return previous_reusable_tokens is not None and next_input_tokens is not None and next_input_tokens >= previous_reusable_tokens


def tool_signature(names: list[str]) -> str:
    counts: dict[str, int] = {}
    for name in names:
        normalized = name.strip().lower()
        counts[normalized] = counts.get(normalized, 0) + 1
    return "+".join(name + (f"x{count}" if count > 1 else "") for name, count in sorted(counts.items()))
