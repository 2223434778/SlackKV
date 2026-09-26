from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

from experiments.phase4d_a.helpers import ReservationLedger, physical_before_logical_release

LEASE_PREFIX = "SLACKKV_LEASE_TRACE "
FROZEN_P_SAFE_MS = 102
EXPECTED_GPU_BLOCKS = 4681
EXPECTED_WORKLOAD_SHA256 = "9b381336a70e73b0efbddbb79d2c88608c19663247e819c1e7a364efd2971105"
WATCHDOG_SECONDS = 20


def parse_lease_trace(text: str) -> list[dict[str, Any]]:
    """Parse structured lease events; never match a logical ID by substring."""
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        marker = line.find(LEASE_PREFIX)
        if marker < 0:
            continue
        try:
            event = json.loads(line[marker + len(LEASE_PREFIX):])
            if not isinstance(event.get("event"), str) or not isinstance(event.get("lease_key"), str):
                continue
            event["timestamp_monotonic"] = float(event["timestamp_monotonic"])
            event["num_blocks"] = int(event["num_blocks"])
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            continue
        events.append(event)
    return events


class LeaseTraceReader:
    """Incrementally parse newly appended lease records from a live server log."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0
        self.partial_line = b""

    def read_events(self) -> list[dict[str, Any]]:
        try:
            with self.path.open("rb") as stream:
                stream.seek(self.offset)
                chunk = stream.read()
                self.offset = stream.tell()
        except FileNotFoundError:
            return []
        if not chunk:
            return []
        lines = (self.partial_line + chunk).split(b"\n")
        self.partial_line = lines.pop()
        text = b"\n".join(lines).decode("utf-8", errors="replace")
        return parse_lease_trace(text)


class RuntimeReleaseQueue:
    """Reclaims controller reservations only after exact physical release evidence."""

    def __init__(
        self,
        ledger: ReservationLedger,
        episode_to_lease_key: dict[str, str],
        required_blocks: dict[str, int],
        t0_monotonic_s: float,
    ) -> None:
        if set(episode_to_lease_key) != set(required_blocks):
            raise ValueError("every episode needs one lease key and block requirement")
        self.ledger = ledger
        self.episode_to_lease_key = dict(episode_to_lease_key)
        self.required_blocks = dict(required_blocks)
        self.t0_monotonic_s = t0_monotonic_s
        self.release_queue: list[str] = []
        self.first_token_ns: dict[str, int] = {}
        self.physical_releases: dict[str, dict[str, Any]] = {}
        self.logical_released: set[str] = set()
        self.events: list[dict[str, Any]] = []
        self.unsafe_release_count = 0

    def observe_lease_events(self, events: Iterable[dict[str, Any]]) -> None:
        for event in events:
            if event.get("event") == "LEASE_RELEASED":
                key = str(event.get("lease_key", ""))
                if key in self.episode_to_lease_key.values():
                    self.physical_releases[key] = event

    def enqueue_first_token(self, episode_id: str, timestamp_ns: int) -> None:
        if episode_id not in self.episode_to_lease_key:
            raise KeyError(episode_id)
        if episode_id in self.logical_released:
            return
        if episode_id not in self.first_token_ns:
            self.first_token_ns[episode_id] = int(timestamp_ns)
            self.release_queue.append(episode_id)

    def drain(self, logical_release_ns: int) -> list[dict[str, Any]]:
        drained: list[dict[str, Any]] = []
        still_waiting: list[str] = []
        for episode_id in self.release_queue:
            if episode_id in self.logical_released:
                continue
            lease_key = self.episode_to_lease_key[episode_id]
            physical = self.physical_releases.get(lease_key)
            if physical is None:
                still_waiting.append(episode_id)
                continue
            physical_ns = int(float(physical["timestamp_monotonic"]) * 1e9)
            logical_ns = int(logical_release_ns)
            if not physical_before_logical_release(physical_ns, logical_ns):
                self.unsafe_release_count += 1
                still_waiting.append(episode_id)
                continue
            before = self.ledger.reserved_blocks
            if episode_id not in self.ledger.active:
                raise AssertionError(f"logical reservation missing for {episode_id}")
            blocks = self.ledger.release(episode_id)
            after = self.ledger.reserved_blocks
            row = {
                "episode_id": episode_id,
                "lease_key": lease_key,
                "required_blocks": self.required_blocks[episode_id],
                "first_token_ms": (self.first_token_ns[episode_id] / 1e9 - self.t0_monotonic_s) * 1000,
                "physical_release_ms": (physical_ns / 1e9 - self.t0_monotonic_s) * 1000,
                "logical_release_ms": (logical_ns / 1e9 - self.t0_monotonic_s) * 1000,
                "logical_minus_physical_ms": (logical_ns - physical_ns) / 1e6,
                "reserved_blocks_before": before,
                "reserved_blocks_after": after,
                "released_blocks": blocks,
                "event": "RUNTIME_RELEASE_AFTER_PHYSICAL_LEASE",
            }
            if blocks != self.required_blocks[episode_id]:
                raise AssertionError("released reservation block count mismatch")
            self.logical_released.add(episode_id)
            self.events.append(row)
            drained.append(row)
        self.release_queue = still_waiting
        return drained

    @property
    def logical_leaks(self) -> int:
        return len(self.ledger.active)

    @property
    def pending_release_count(self) -> int:
        return len(self.release_queue)


def release_order_valid(physical_ns: int, logical_ns: int) -> bool:
    return physical_before_logical_release(int(physical_ns), int(logical_ns))


def smallest_prefix_groups(jobs: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(jobs, key=lambda row: (int(row["required_blocks"]), str(row["episode_id"])))
    if len(ordered) < 4:
        raise ValueError("at least four frozen jobs are required")
    counts = [sum(int(row["required_blocks"]) for row in ordered[:n]) for n in range(1, 5)]
    return {"ordered_jobs": ordered[:4], "episode_ids": [r["episode_id"] for r in ordered[:4]],
            "N1": counts[0], "N2": counts[1], "N3": counts[2], "N4": counts[3]}


def issue_spread_ms(issue_times_ns: Iterable[int]) -> float:
    values = [int(value) for value in issue_times_ns]
    if not values:
        raise ValueError("issue timestamps cannot be empty")
    return (max(values) - min(values)) / 1e6


def issue_spread_valid(spread_ms: float, limit_ms: float = 20.0) -> bool:
    return math.isfinite(spread_ms) and 0 <= spread_ms <= limit_ms


def watchdog_classification(
    *, elapsed_s: float, pending: int, server_alive: bool, oom: bool, server_crash: bool
) -> str:
    if oom or server_crash or not server_alive:
        return "INVALID_SERVER_FAILURE"
    if elapsed_s >= WATCHDOG_SECONDS and pending > 0:
        return "NON_PROGRESS"
    if pending == 0:
        return "COMPLETE"
    return "IN_PROGRESS"


def stalled_for_five_seconds(last_completion_elapsed_ms: float | None, elapsed_ms: float, pending: int) -> bool:
    last = 0.0 if last_completion_elapsed_ms is None else float(last_completion_elapsed_ms)
    return pending > 0 and elapsed_ms - last >= 5000.0


def service_envelope(values_ms: Iterable[float], budget_ms: float = FROZEN_P_SAFE_MS) -> dict[str, Any]:
    values = sorted(float(value) for value in values_ms)
    if not values:
        raise ValueError("service measurements cannot be empty")
    rank = (len(values) - 1) * 0.95
    lo = math.floor(rank)
    hi = math.ceil(rank)
    p95 = values[lo] + (values[hi] - values[lo]) * (rank - lo)
    overrun = sum(value > budget_ms for value in values)
    return {
        "promotion_count": len(values),
        "overrun_count": overrun,
        "min_issue_to_T6_ms": values[0],
        "median_issue_to_T6_ms": values[len(values) // 2] if len(values) % 2 else (values[len(values)//2-1] + values[len(values)//2]) / 2,
        "P95_issue_to_T6_ms": p95,
        "max_issue_to_T6_ms": values[-1],
        "max_minus_budget_ms": max(0.0, values[-1] - budget_ms),
        "budget_ms": budget_ms,
        "status": "P_SAFE_102_INTEGRATION_ENVELOPE_PASS" if overrun == 0 else "P_SAFE_102_INTEGRATION_ENVELOPE_FAIL",
    }


def validate_stage_name(stage: str) -> None:
    if stage not in {"release-fix", "liveness"}:
        raise ValueError("Phase 4D-A-FIX cannot run the Stage B policy matrix")
