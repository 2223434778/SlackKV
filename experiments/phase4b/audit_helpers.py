"""Pure offline helpers for the Phase 4B public-trace audit."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Mapping

UPSTREAM_COMMIT = "83d8d1280b406b0484693017900763a0b0ad8d19"
DATASET_REVISION = "d56ae0d0bc80f3663a7a61a38c4ee60c26ce7b67"
TIMING_FIELDS = ("request_time", "e2e_ms")
TOKEN_FIELDS = ("isl", "isl_new", "osl")


def source_revision_is_pinned(actual: str, expected: str = UPSTREAM_COMMIT) -> bool:
    return actual == expected


def seconds_to_ms(seconds: float) -> float:
    return seconds * 1000.0


def reconstructed_slack_ms(
    previous_request_time_s: float,
    previous_e2e_ms: float,
    current_request_time_s: float,
) -> float:
    """Return raw interarrival minus prior E2E; negative values are preserved."""
    return seconds_to_ms(current_request_time_s - previous_request_time_s) - previous_e2e_ms


def cached_input_tokens(isl: int, isl_new: int) -> int:
    if isl < 0 or isl_new < 0 or isl_new > isl:
        raise ValueError("token counts must satisfy 0 <= isl_new <= isl")
    return isl - isl_new


def reusable_tokens(
    previous_isl: int, previous_osl: int, next_isl: int, next_isl_new: int
) -> int:
    next_cached = cached_input_tokens(next_isl, next_isl_new)
    return min(previous_isl + previous_osl, next_cached)


def required_blocks(tokens: int, block_size: int = 16) -> int:
    if tokens < 0 or block_size <= 0:
        raise ValueError("tokens must be nonnegative and block_size positive")
    return math.ceil(tokens / block_size)


def consecutive_replayable_runs(
    rows: Iterable[Mapping[str, Any]], required_fields: tuple[str, ...] = TOKEN_FIELDS
) -> dict[str, list[list[Mapping[str, Any]]]]:
    """Group rows by session, preserve turn order, and split at invalid boundaries."""
    by_session: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_session[str(row["session_id"])].append(row)

    result: dict[str, list[list[Mapping[str, Any]]]] = {}
    for session, session_rows in by_session.items():
        ordered = sorted(session_rows, key=lambda row: row["turn"])
        runs: list[list[Mapping[str, Any]]] = []
        current: list[Mapping[str, Any]] = []
        last_turn: int | None = None
        for row in ordered:
            turn = int(row["turn"])
            valid = all(row.get(field) is not None for field in required_fields)
            if last_turn is not None and turn != last_turn + 1:
                if current:
                    runs.append(current)
                current = []
            if not valid:
                if current:
                    runs.append(current)
                current = []
            else:
                current.append(row)
            last_turn = turn
        if current:
            runs.append(current)
        result[session] = runs
    return result


def timing_fields_available(schema_fields: Iterable[str]) -> bool:
    fields = set(schema_fields)
    return all(field in fields for field in TIMING_FIELDS)
