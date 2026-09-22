from __future__ import annotations

import json
from typing import Any, Iterable


EXPECTED_PREFIX_TOKENS = 4096


def source_conservation(local_tokens: int, external_tokens: int) -> bool:
    return local_tokens + external_tokens == EXPECTED_PREFIX_TOKENS


def parse_structured_events(text: str, marker: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if marker not in line:
            continue
        rows.append(json.loads(line.split(marker, 1)[1]))
    return rows


def unique_blocks(groups: Iterable[Iterable[Any]]) -> list[Any]:
    blocks: dict[int, Any] = {}
    for group in groups:
        for block in group:
            if not block.is_null:
                blocks.setdefault(block.block_id, block)
    return list(blocks.values())


def lease_sequence_valid(events: list[dict[str, Any]]) -> bool:
    names = [event["event"] for event in events]
    expected = [
        "LEASE_ACQUIRED",
        "LEASE_HELD_AFTER_PREFETCH_FREE",
        "LEASE_RELEASE_READY",
        "LEASE_RELEASED",
    ]
    return names == expected and events[-1].get("active_lease_count") == 0
