"""Mechanism-matched fixed-TTL/program-order control, not a novel algorithm."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass


@dataclass
class Retained:
    blocks: list
    deadline: float


class RetentionControl:
    def __init__(self, pool, logger, *, enabled: bool, pin: bool, ttl: float = 1.0):
        self.pool, self.logger, self.enabled, self.pin, self.ttl = pool, logger, enabled, pin, ttl
        self.entries: dict[str, Retained] = {}
        self.program_arrival: dict[str, float] = {}
        self.request_arrival: dict[str, float] = {}
        self.latest_request: dict[str, str | None] = {}
        self.outstanding: dict[str, set[str]] = {}

    @staticmethod
    def program(request):
        return (request.kv_transfer_params or {}).get("slackkv_epoch_program")

    def emit(self, event, key, blocks, reason):
        self.logger.info("SLACKKV_ROUND4_RETENTION %s", json.dumps({
            "timestamp_monotonic": time.monotonic(), "event": event, "program": key,
            "block_ids": [b.block_id for b in blocks], "reason": reason,
            "ref_counts": [b.ref_cnt for b in blocks],
            "retained_programs": len(self.entries),
            "retained_unique_blocks": len({b.block_id for e in self.entries.values() for b in e.blocks}),
        }, separators=(",", ":")))

    def arrived(self, request, now):
        if not self.enabled:
            return
        self.request_arrival[request.request_id] = now
        key = self.program(request)
        if key is not None:
            self.latest_request[key] = request.request_id
            self.outstanding.setdefault(key, set()).add(request.request_id)
        if key is not None and key not in self.program_arrival:
            self.program_arrival[key] = request.arrival_time
            self.emit("PROGRAM_ARRIVED", key, [], "FIRST_ENGINE_ARRIVAL")

    def release(self, key, reason):
        entry = self.entries.pop(key, None)
        if entry is None:
            return False
        if any(b.ref_cnt < 1 for b in entry.blocks):
            raise AssertionError("retention reference lost before release")
        self.pool.free_blocks(entry.blocks)
        self.emit("RELEASE", key, entry.blocks, reason)
        return True

    def expire(self, now):
        for key, entry in list(self.entries.items()):
            if entry.deadline <= now:
                self.release(key, "TTL_EXPIRED")

    def delay(self, now):
        if not self.entries:
            return None
        return max(.001, min(e.deadline for e in self.entries.values()) - now)

    def reorder(self, queue, now):
        if not self.enabled or not queue:
            return
        def key(r):
            age = now - self.request_arrival.get(r.request_id, now)
            program = self.program(r)
            if age >= self.ttl:
                return (0, r.arrival_time)
            if program in self.entries:
                return (1, self.program_arrival[program])
            return (2, self.program_arrival.get(program, r.arrival_time))
        requests = list(queue)
        ordered = sorted(requests, key=key)
        if ordered != requests:
            queue.remove_requests(requests)
            for r in ordered:
                queue.add_request(r)

    def admitted(self, request):
        key = self.program(request)
        entry = self.entries.get(key)
        if entry is not None:
            # The new prompt may reuse only part of the old context. Allocation
            # has already touched matching blocks; unmatched retained blocks can
            # return to the evictable cache with no request reference.
            self.release(key, "REAL_REQUEST_ALLOCATED")

    def cancelled(self, request):
        if not self.enabled:
            return
        key = self.program(request)
        if self.latest_request.get(key) == request.request_id:
            self.release(key, "CANCEL_OR_ERROR")
            self.latest_request[key] = None
            self.program_arrival.pop(key, None)
        self.emit("CANCEL", key, [], "CANCEL_OR_ERROR")
        self.request_arrival.pop(request.request_id, None)

    def finished(self, request, blocks, now):
        if not self.enabled:
            return
        self.request_arrival.pop(request.request_id, None)
        key = self.program(request)
        if key is None:
            return
        self.outstanding.get(key, set()).discard(request.request_id)
        if self.latest_request.get(key) != request.request_id:
            # Native D2H may defer an older request's actual free until after a
            # newer turn arrives or finishes. Never pin or release that newer state.
            if not self.outstanding.get(key) and key not in self.program_arrival:
                self.outstanding.pop(key, None)
                self.latest_request.pop(key, None)
            return
        self.release(key, "FINISH_OR_CANCEL")
        continues = (request.kv_transfer_params or {}).get("slackkv_epoch_continues") is True
        normal = request.status.name in {"FINISHED_STOPPED", "FINISHED_LENGTH_CAPPED"}
        if not normal or not continues:
            self.program_arrival.pop(key, None)
            if not self.outstanding.get(key):
                self.outstanding.pop(key, None)
                self.latest_request.pop(key, None)
            return
        if not self.pin:
            return
        unique = {b.block_id: b for b in blocks if not b.is_null and b.block_hash is not None}
        if unique:
            owned = list(unique.values())
            if any(b.ref_cnt < 1 for b in owned):
                raise AssertionError("pin attempted after real ownership freed")
            self.pool.touch(owned)
            self.entries[key] = Retained(owned, now + self.ttl)
            self.emit("ACQUIRE", key, owned, "NORMAL_CONTINUING_TURN")

    def reclaim(self, reason):
        if not self.entries:
            return False
        for key in list(self.entries):
            self.release(key, reason)
        return True

    def resolve_blocked(self, request, waiting):
        if not self.entries:
            return False
        if self.program(request) is None:
            return self.reclaim("BACKGROUND_DEMAND_PRESSURE")
        if any(self.program(r) in self.entries for r in waiting):
            # A mature cold head can outrank a returning retained consumer.
            # Remove idle references immediately instead of relying on TTL to
            # eventually break a Lease/HOL cycle. Native DMA references remain.
            return self.reclaim("QUEUED_CONSUMER_HOL_FALLBACK")
        return False


def control(scheduler):
    import os
    ctl = getattr(scheduler, "_round4_retention", None)
    if ctl is None:
        from vllm.logger import init_logger
        ctl = scheduler._round4_retention = RetentionControl(
            scheduler.kv_cache_manager.block_pool,
            init_logger("vllm.round4.retention"),
            enabled=os.environ.get("SLACKKV_RETENTION_CONTROL") == "1",
            pin=os.environ.get("SLACKKV_RETENTION_PIN") == "1")
    return ctl
