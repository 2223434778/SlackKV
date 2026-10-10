"""Opt-in observed ownership snapshots; no scheduling or refcount mutation."""
from __future__ import annotations

import itertools
import json
import time


def release_witness(required: int, available: int, owners: dict[str, list[int]],
                    ref_counts: dict[int, int]) -> list[str] | None:
    """Smallest completion set that frees enough physical blocks in this snapshot.

    Counts references outside the supplied owners as protected. This deliberately
    excludes changes to in-flight reservations and future decode growth.
    """
    if required <= available:
        return []
    counts = {block: sum(block in ids for ids in owners.values()) for block in ref_counts}
    if any(ref_counts[b] < counts[b] for b in counts):
        raise ValueError("snapshot ownership exceeds actual references")
    for size in range(1, len(owners) + 1):
        for keys in itertools.combinations(sorted(owners), size):
            selected = {block: sum(block in owners[k] for k in keys) for block in counts}
            freed = sum(selected[b] > 0 and selected[b] == ref_counts[b] for b in counts)
            if available + freed >= required:
                return list(keys)
    return None


def snapshot(scheduler, request, event: str, logger) -> None:
    now = time.monotonic_ns()
    samples = getattr(scheduler, "_round4_resource_samples", None)
    if samples is None:
        samples = scheduler._round4_resource_samples = {}
    key = (request.request_id, event)
    if event == "ALLOCATE_NONE" and now - samples.get(key, 0) < 100_000_000:
        return
    samples[key] = now
    started = time.thread_time_ns()
    pool = scheduler.kv_cache_manager.block_pool
    requests = []
    for r in scheduler.requests.values():
        ids = sorted(scheduler._slackkv_capacity_owned_block_ids([r]))
        requests.append({"request_id": r.request_id,
                         "session": (r.kv_transfer_params or {}).get("slackkv_session"),
                         "status": r.status.name, "block_ids": ids,
                         "output_tokens": r.num_output_tokens, "max_tokens": r.max_tokens,
                         "computed_tokens": r.num_computed_tokens,
                         "num_preemptions": r.num_preemptions})
    busy = {b.block_id: b.ref_cnt for b in pool.blocks if not b.is_null and b.ref_cnt > 0}
    idle = [b.block_id for b in pool.blocks if not b.is_null and b.ref_cnt == 0
            and b.block_hash is not None]
    check = (request.kv_transfer_params or {}).get("_round4_last_allocator_check")
    logger.info("SLACKKV_ROUND4_RESOURCES %s", json.dumps({
        "event": event, "timestamp_monotonic_ns": now,
        "step_id": scheduler._slackkv_scheduler_step_id,
        "target": request.request_id,
        "target_session": (request.kv_transfer_params or {}).get("slackkv_session"),
        "last_allocator_check": check,
        "pool_free_blocks": pool.get_num_free_blocks(), "busy_ref_counts": busy,
        "idle_evictable_block_ids": idle, "requests": requests,
        "running_ids": [r.request_id for r in scheduler.running],
        "inflight_prefill_ids": [r.request_id for r in scheduler._inflight_prefills],
        "instrumentation_cpu_ns": time.thread_time_ns() - started,
        "meaning": "sampled physical ownership; not continuous causality or owned memory-time",
    }, separators=(",", ":")))
