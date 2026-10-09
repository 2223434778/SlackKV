"""Soft two-call working-set epochs: admission hints, never physical KV leases."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Epoch:
    blocks: int
    busy: set[str] = field(default_factory=set)
    completed: int = 0
    idle_deadline: float = float("inf")


class WorkingSetEpochs:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.entries: dict[str, Epoch] = {}

    @staticmethod
    def program(request) -> str | None:
        return (request.kv_transfer_params or {}).get("slackkv_epoch_program")

    @staticmethod
    def blocks(request, max_model_len: int) -> int:
        tokens = min(request.num_prompt_tokens + request.max_tokens, max_model_len)
        return (tokens + 15) // 16

    def expire(self, now: float) -> None:
        self.entries = {key: entry for key, entry in self.entries.items()
                        if entry.busy or entry.idle_deadline > now}

    def allowed(self, request, *, now: float, pool_blocks: int,
                background_blocks: int, max_model_len: int) -> bool:
        if not self.enabled:
            return True
        self.expire(now)
        key = self.program(request)
        if key is None or key in self.entries or not self.entries:
            return True
        return (sum(entry.blocks for entry in self.entries.values())
                + self.blocks(request, max_model_len) <= pool_blocks - 1 - background_blocks)

    def admitted(self, request, *, max_model_len: int) -> bool:
        key = self.program(request)
        if not self.enabled or key is None:
            return False
        is_new = key not in self.entries
        entry = self.entries.setdefault(key, Epoch(self.blocks(request, max_model_len)))
        entry.blocks = max(entry.blocks, self.blocks(request, max_model_len))
        entry.busy.add(request.request_id)
        entry.idle_deadline = float("inf")
        return is_new

    def finished(self, request, *, now: float, cancelled: bool) -> None:
        if not self.enabled:
            return
        key = self.program(request)
        entry = self.entries.get(key)
        if entry is None or request.request_id not in entry.busy:
            return
        entry.busy.remove(request.request_id)
        entry.completed += 1
        continues = (request.kv_transfer_params or {}).get("slackkv_epoch_continues") is True
        if cancelled or not continues or (entry.completed >= 2 and not entry.busy):
            del self.entries[key]
        elif not entry.busy:
            entry.idle_deadline = now + 1.0
