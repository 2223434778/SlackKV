"""C0/C1 controller variants; C1 consumes only exact full-ready rechecks."""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from experiments.phase3f.scheduler import latest_feasible_start, merge_blackouts
from experiments.phase4d_a.helpers import ReservationLedger
from experiments.phase4f_a.logic import progress_blocks
from experiments.phase4g_a.controller import (
    MAX_PREFETCH_SECONDS,
    P_SAFE_MS,
    RESUME_GUARD_MS,
    DynamicController as BaseController,
)
from experiments.phase4g_c.analysis import (
    protected_self_reserve, psa1_reserve, selection_visibility, source_recheck_event,
)
from experiments.phase4g_c.client import (
    progress_prefetch_rca,
    read_only_selection_snapshot,
)


class JsonTail:
    def __init__(self, path: Path, marker: str):
        self.path, self.marker, self.offset, self.partial = path, marker, 0, b""

    def read(self) -> list[dict[str, Any]]:
        try:
            with self.path.open("rb") as stream:
                stream.seek(self.offset)
                chunk = stream.read()
                self.offset = stream.tell()
        except FileNotFoundError:
            return []
        if not chunk:
            return []
        lines = (self.partial + chunk).split(b"\n")
        self.partial = lines.pop()
        events = []
        for line in lines:
            if self.marker.encode() not in line:
                continue
            try:
                text = line.decode("utf-8", errors="replace")
                events.append(json.loads(text.split(self.marker, 1)[1]))
            except (ValueError, IndexError):
                continue
        return events


class SourceRecheckController(BaseController):
    def __init__(self, *args, condition: str, repetition: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.condition = condition
        self.repetition = repetition
        self.probe_reader = JsonTail(self.reader.path, "SLACKKV_READY_PROBE ")
        self.source_reader = JsonTail(self.reader.path, "SLACKKV_OFFLOAD_TRACE ")
        self.initial_source: dict[str, dict[str, Any]] = {}
        self.latest_exact: dict[str, dict[str, Any]] = {}
        self.rechecked: set[str] = set()
        self.pending_rechecks: dict[str, dict[str, Any]] = {}
        self.source_rechecks: list[dict[str, Any]] = []
        self.selection_rows: list[dict[str, Any]] = []
        self.selection_by_candidate: dict[str, dict[str, Any]] = {}
        self.selection_counter = 0
        self.recheck_behavior_updates = 0

    def _consume_recheck(self, boundary_id: str, transition: dict[str, Any],
                         sample: dict[str, Any]) -> None:
        transition.update({
            "condition": self.condition, "repetition": self.repetition,
            "consumed_by_controller": self.condition == "C1",
        })
        self.source_rechecks.append(transition)
        self.rechecked.add(boundary_id)
        if self.condition == "C1":
            visible = {**sample, "event": "SOURCE_READY",
                       "timestamp_monotonic": int(sample["timestamp_monotonic_ns"]) / 1e9}
            self.offload_events[boundary_id] = visible
            if boundary_id in self.registered:
                self.registered[boundary_id]["source_ready_ns"] = int(sample["timestamp_monotonic_ns"])
            self.recheck_behavior_updates += 1

    def register(self, boundary_id: str, prefix: list[int], previous_finish_ns: int) -> int:
        deadline_ns = super().register(boundary_id, prefix, previous_finish_ns)
        with self.lock:
            pending = self.pending_rechecks.pop(boundary_id, None)
            if pending is not None and boundary_id not in self.rechecked:
                transition = dict(pending["transition"])
                transition["time_to_deadline_ms"] = (
                    deadline_ns - int(transition["timestamp_ns"])
                ) / 1e6
                self._consume_recheck(boundary_id, transition, pending["sample"])
        return deadline_ns

    def _observe_traces(self) -> None:
        super()._observe_traces()
        source_events = self.source_reader.read()
        probe_events = self.probe_reader.read()
        with self.lock:
            for event in source_events:
                boundary_id = event.get("boundary_id")
                if not isinstance(boundary_id, str):
                    continue
                if event.get("event") in {"SOURCE_READY", "SOURCE_NOT_READY"}:
                    self.initial_source.setdefault(boundary_id, dict(event))
            for sample in probe_events:
                if sample.get("event") != "SAMPLE":
                    continue
                boundary_id = sample.get("boundary_id")
                if not isinstance(boundary_id, str) or boundary_id not in self.manifest:
                    continue
                self.latest_exact[boundary_id] = dict(sample)
                if boundary_id in self.rechecked:
                    continue
                initial = self.initial_source.get(boundary_id)
                if initial is None:
                    continue
                transition = source_recheck_event(
                    sample, str(initial.get("event")),
                    round(float(initial.get("timestamp_monotonic", 0)) * 1e9),
                    int(self.registered.get(boundary_id, {}).get("deadline_ns", 0)),
                )
                if transition is None:
                    continue
                if boundary_id not in self.registered:
                    self.pending_rechecks[boundary_id] = {
                        "transition": dict(transition), "sample": dict(sample),
                    }
                    continue
                self._consume_recheck(boundary_id, transition, sample)

    def _candidate(self):
        selected = super()._candidate()
        if selected is None:
            return None
        candidate_id, state, protected_id = selected
        self.selection_counter += 1
        selection_ns = time.monotonic_ns()
        candidate = self.registered[candidate_id]
        protected = self.registered[protected_id]
        candidate_row = self.manifest[candidate_id]
        protected_row = self.manifest[protected_id]
        candidate_blocks = int(candidate_row["required_blocks"])
        protected_blocks = int(protected_row["required_blocks"])
        sample = self.latest_exact.get(protected_id, {})
        progress_need = progress_blocks(int(protected_row["resume_isl"]))
        selection_id = f"{self.condition}-r{self.repetition}-s{self.selection_counter:03d}"
        with self.lock:
            active_states = {
                item_id: item_state for item_id, item_state in self.registered.items()
                if item_state["decision"] == "REGISTERED"
                and item_state["resume_started_ns"] is None
            }
            ready_for_self = {
                item_id for item_id in active_states if item_id in self.offload_events
            }
        snapshot_rca = {
            "selection_id": selection_id,
            "selection_timestamp_ns": selection_ns,
            "candidate_id": candidate_id,
            "protected_id": protected_id,
            "candidate_matches_protected": candidate_id == protected_id,
            "candidate_controller_visible_ready": candidate_id in self.offload_events,
            "protected_controller_visible_ready": protected_id in self.offload_events,
            "candidate_prefetch_tokens": int(candidate_row["required_prefix_tokens"]),
            "candidate_source_tokens": int(candidate_row["required_prefix_tokens"]),
            "protected_lease_key": str(protected_row["lease_key"]),
            "protected_progress_blocks": progress_need,
            "protected_full_blocks": protected_blocks,
            "protected_source_tokens": int(protected_row["required_prefix_tokens"]),
            "protected_isl": int(protected_row["resume_isl"]),
            "protected_osl": int(protected_row["resume_osl"]),
            "protected_full_sequence_tokens": int(protected_row["resume_isl"])
                + int(protected_row["resume_osl"]),
            "protected_first_token_tokens": int(protected_row["resume_isl"]) + 1,
            "psa1_reserve_blocks": None,
        }
        snapshot_response = read_only_selection_snapshot(rca_snapshot=snapshot_rca)
        snapshot = snapshot_response["snapshot"]
        candidate_exact = snapshot.get("candidate_source_exact", {})
        protected_exact = snapshot.get("protected_source_exact", {})
        protected_shadow = snapshot.get("protected_self_allocator_preflight", {})
        ownership = snapshot.get("protected_local_ownership", {})
        existing_lease = int(ownership.get("current_protected_lease_blocks", 0))
        exact_ready = bool(protected_exact.get("source_fully_ready", False))
        if exact_ready:
            ready_for_self.add(protected_id)
        else:
            ready_for_self.discard(protected_id)
        actual_h = psa1_reserve(progress_need, existing_lease,
                                candidate_id == protected_id, candidate_blocks)
        self_h = protected_self_reserve(progress_need, existing_lease, protected_blocks)
        now_ms = (selection_ns - self.origin_ns) / 1e6
        deadline_ms = {
            item_id: (int(item_state["deadline_ns"]) - self.origin_ns) / 1e6
            for item_id, item_state in active_states.items()
        }
        self_start = latest_feasible_start(
            now_ms,
            [(item_id, deadline_ms[item_id]) for item_id in ready_for_self if item_id in deadline_ms],
            merge_blackouts(deadline_ms, RESUME_GUARD_MS),
            P_SAFE_MS,
        ) if ready_for_self else now_ms
        protected_self_timing_permits = now_ms + .5 >= self_start
        row = {
            "condition": self.condition, "repetition": self.repetition,
            "selection_id": selection_id, "timestamp_ns": selection_ns,
            "snapshot_timestamp_ns": int(float(snapshot.get("timestamp_monotonic", 0)) * 1e9),
            "snapshot_roundtrip_ms": float(snapshot_response["client_roundtrip_ms"]),
            "selection_to_scheduler_ms": snapshot.get("selection_to_scheduler_ms"),
            "selection_total_blocks": snapshot.get("total_blocks"),
            "selection_free_blocks": snapshot.get("free_blocks_before"),
            "selection_used_blocks": snapshot.get("used_blocks_before"),
            "selection_candidate_allocator_required": snapshot.get(
                "candidate_allocator_preflight", {}
            ).get("allocator_required_blocks"),
            "selection_candidate_allocator_available": snapshot.get(
                "candidate_allocator_preflight", {}
            ).get("allocator_available_blocks"),
            "selection_candidate_allocator_fit": snapshot.get(
                "candidate_allocator_preflight", {}
            ).get("allocator_fit"),
            "selection_protected_self_allocator_required": protected_shadow.get(
                "allocator_required_blocks"
            ),
            "selection_protected_self_allocator_available": protected_shadow.get(
                "allocator_available_blocks"
            ),
            "selection_protected_owned_running": ownership.get("owned_by_running_request"),
            "selection_protected_owned_waiting": ownership.get("owned_by_waiting_request"),
            "selection_protected_leased": ownership.get("current_protected_lease_blocks"),
            "selection_protected_evictable": ownership.get("cached_but_free_evictable"),
            "selection_protected_other_refcounted": ownership.get("other_refcounted"),
            "candidate_id": candidate_id, "protected_id": protected_id,
            "initial_source_state": self.initial_source.get(protected_id, {}).get("event"),
            "candidate_deadline_ms": (int(candidate["deadline_ns"]) - self.origin_ns) / 1e6,
            "protected_deadline_ms": (int(protected["deadline_ns"]) - self.origin_ns) / 1e6,
            "candidate_source_ready": candidate_id in self.offload_events,
            "candidate_exact_source_ready": bool(candidate_exact.get("source_fully_ready", False)),
            "protected_exact_source_ready": exact_ready,
            "protected_controller_visible_ready": protected_id in self.offload_events,
            **selection_visibility(
                candidate_id, protected_id,
                protected_id in self.offload_events, exact_ready,
            ),
            "source_sample_timestamp_ns": sample.get("timestamp_monotonic_ns"),
            "source_sample_age_ms": ((selection_ns - int(sample["timestamp_monotonic_ns"])) / 1e6
                                     if sample.get("timestamp_monotonic_ns") else None),
            "candidate_matches_protected": candidate_id == protected_id,
            "candidate_required_blocks": candidate_blocks,
            "protected_required_blocks": protected_blocks,
            "protected_progress_blocks": progress_need,
            "protected_existing_lease_blocks": existing_lease,
            "PSA1_reserve_blocks": actual_h,
            "protected_self_H": self_h,
            "delta_H": actual_h - self_h,
            "ledger_reserved_blocks": self.ledger.reserved_blocks,
            "candidate_logical_fit": self.ledger.reserved_blocks + candidate_blocks <= self.ledger.budget_blocks,
            "self_logical_fit": self.ledger.reserved_blocks + protected_blocks <= self.ledger.budget_blocks,
            "protected_self_timing_permits": protected_self_timing_permits,
            "selection_exact_allocator_fit": snapshot.get("candidate_allocator_preflight", {}).get("allocator_fit"),
            "protected_self_allocator_fit": protected_shadow.get("allocator_fit"),
            "protected_self_allocator_required": protected_shadow.get("allocator_required_blocks"),
            "protected_self_allocator_available": protected_shadow.get("allocator_available_blocks"),
            "protected_local_cached_blocks": protected_exact.get("local_cached_blocks"),
            "protected_external_ready_blocks": protected_exact.get("external_ready_blocks"),
            "protected_missing_blocks": max(0, protected_blocks - int(protected_exact.get("covered_blocks", 0))),
            "selection_snapshot": snapshot,
        }
        self.selection_rows.append(row)
        self.selection_by_candidate[candidate_id] = row
        return selected

    def _attempt(self, key: str, state: dict[str, Any], protected: str,
                 pool: ThreadPoolExecutor) -> None:
        row = self.manifest[key]
        with self.lock:
            current = self.registered[key]
            if current["resume_started_ns"] is not None or time.monotonic_ns() >= current["deadline_ns"]:
                current["decision"] = "STALE_PROMOTION_CANCELLED"
                self.decisions.append({"boundary_id": key, "event": "STALE_PROMOTION_CANCELLED",
                                       "timestamp_ns": time.monotonic_ns()})
                return
            if not self.ledger.reserve(key, int(row["required_blocks"])):
                current["decision"] = "RESERVATION_SKIP"
                self.decisions.append({"boundary_id": key, "event": "RESERVATION_SKIP",
                                       "timestamp_ns": time.monotonic_ns()})
                return
            current["decision"] = "PROMOTION_PENDING"
            selected = self.selection_by_candidate.get(key)
            if selected is None or selected["protected_id"] != protected:
                raise RuntimeError("selection snapshot identity changed before _attempt")
            protected_row = self.manifest[protected]
            rca = {
                "selection_id": selected["selection_id"],
                "selection_timestamp_ns": int(selected["timestamp_ns"]),
                "selection_snapshot_timestamp_ns": int(selected["snapshot_timestamp_ns"]),
                "candidate_id": key, "protected_id": protected,
                "candidate_matches_protected": key == protected,
                "candidate_controller_visible_ready": bool(selected["candidate_source_ready"]),
                "protected_controller_visible_ready": bool(selected["protected_controller_visible_ready"]),
                "protected_self_timing_permits": bool(selected["protected_self_timing_permits"]),
                "ledger_reserved_blocks_at_selection": int(selected["ledger_reserved_blocks"]),
                "ledger_reserved_blocks_at_attempt": int(self.ledger.reserved_blocks),
                "candidate_prefetch_tokens": int(row["required_prefix_tokens"]),
                "candidate_source_tokens": int(row["required_prefix_tokens"]),
                "protected_progress_blocks": int(selected["protected_progress_blocks"]),
                "protected_full_blocks": int(protected_row["required_blocks"]),
                "protected_source_tokens": int(protected_row["required_prefix_tokens"]),
                "protected_isl": int(protected_row["resume_isl"]),
                "protected_osl": int(protected_row["resume_osl"]),
                "protected_full_sequence_tokens": int(
                    protected_row["resume_isl"]
                ) + int(protected_row["resume_osl"]),
                "protected_first_token_tokens": int(protected_row["resume_isl"]) + 1,
                "psa1_reserve_blocks": int(selected["PSA1_reserve_blocks"]),
            }
            future = pool.submit(
                progress_prefetch_rca, state["prefix"],
                required_blocks=int(row["required_blocks"]),
                lease_key=str(row["lease_key"]),
                protected_lease_key=str(protected_row["lease_key"]),
                protected_progress_blocks=int(selected["protected_progress_blocks"]),
                rca_snapshot=rca, timeout_s=MAX_PREFETCH_SECONDS,
            )
            self.promotions[key] = {"future": future, "submitted_ns": time.monotonic_ns(),
                                    "protected_boundary_id": protected, "selection_id": selected["selection_id"],
                                    "harvested": False}
            self.decisions.append({"boundary_id": key, "event": "PROMOTION_ATTEMPT",
                                   "protected_boundary_id": protected,
                                   "selection_id": selected["selection_id"],
                                   "timestamp_ns": time.monotonic_ns(),
                                   "reserved_blocks": self.ledger.reserved_blocks})
