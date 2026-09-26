from __future__ import annotations

from concurrent.futures import Future
from typing import Any

from experiments.phase4d_a.helpers import ReservationLedger


def harvest_completed_promotions(
    promotions: dict[str, dict[str, Any]],
    states: dict[str, dict[str, Any]],
    ledger: ReservationLedger,
    events: list[dict[str, Any]],
    *,
    now_monotonic: float,
) -> list[dict[str, Any]]:
    harvested: list[dict[str, Any]] = []
    for episode_id, item in promotions.items():
        future: Future = item["future"]
        if not future.done() or item.get("harvested"):
            continue
        try:
            result = future.result()
        except Exception as exc:
            raise RuntimeError(f"native promotion failed for {episode_id}") from exc
        status = result.get("status")
        if status == "capacity_rejected":
            if episode_id not in ledger.active:
                raise AssertionError(f"capacity reject reservation missing for {episode_id}")
            before = ledger.reserved_blocks
            released = ledger.release(episode_id)
            after = ledger.reserved_blocks
            state = states[episode_id]
            state["promoted"] = False
            state["leased"] = False
            state["capacity_rejected"] = True
            row = {
                "event": "CAPACITY_REJECT_RELEASE",
                "episode_id": episode_id,
                "lease_key": item["lease_key"],
                "required_blocks": item["required_blocks"],
                "free_blocks_at_reject": result.get("free_blocks"),
                "reserved_blocks_before": before,
                "reserved_blocks_after": after,
                "released_blocks": released,
                "timestamp_monotonic": now_monotonic,
            }
            events.append(row)
            item["result"] = result
            item["harvested"] = True
            harvested.append(row)
            continue
        if status != "ok":
            raise RuntimeError(f"unsupported native promotion status for {episode_id}: {status!r}")
        states[episode_id]["physically_admitted"] = True
        states[episode_id]["leased"] = True
        item["result"] = result
        item["harvested"] = True
        row = {
            "event": "PROMOTION_PHYSICALLY_ADMITTED",
            "episode_id": episode_id,
            "lease_key": item["lease_key"],
            "required_blocks": item["required_blocks"],
            "timestamp_monotonic": now_monotonic,
        }
        events.append(row)
        harvested.append(row)
    return harvested
