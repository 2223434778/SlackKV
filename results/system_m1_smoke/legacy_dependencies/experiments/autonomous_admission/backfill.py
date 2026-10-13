"""Bound extra admissions behind a capacity-blocked head across scheduler steps."""
from __future__ import annotations


class BoundedBackfill:
    def __init__(self, limit: int = 0):
        if limit < 0:
            raise ValueError("negative bypass limit")
        self.limit = limit
        self.counts: dict[str, int] = {}
        self.blocked_this_step: list[str] = []
        self.stop_this_step = False

    def begin_step(self, live_ids: set[str]) -> None:
        self.counts = {key: value for key, value in self.counts.items() if key in live_ids}
        self.blocked_this_step = []
        self.stop_this_step = False

    def can_skip(self, request_id: str) -> bool:
        return self.counts.get(request_id, 0) < self.limit

    def blocked(self, request_id: str) -> None:
        self.counts.setdefault(request_id, 0)
        self.blocked_this_step.append(request_id)

    def admitted(self, request_id: str) -> list[str]:
        self.counts.pop(request_id, None)
        heads = [key for key in self.blocked_this_step if key != request_id]
        for key in heads:
            self.counts[key] += 1
        # Admit at most one backfill per tick, then retry the original head.
        self.stop_this_step = bool(heads)
        return heads
