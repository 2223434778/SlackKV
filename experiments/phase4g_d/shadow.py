"""Read-only Phase 4G-D shadow observations on the unchanged C1 controller."""
from __future__ import annotations

import threading
import time
from typing import Any

from experiments.phase3f.scheduler import latest_feasible_start, merge_blackouts
from experiments.phase4f_a.logic import progress_blocks
from experiments.phase4g_c.analysis import protected_self_reserve, psa1_reserve
from experiments.phase4g_c.client import read_only_selection_snapshot
from experiments.phase4g_c.controller import SourceRecheckController
from experiments.phase4g_d import analysis


class ProtectedHeadShadowController(SourceRecheckController):
    """Add observations only; inherited candidate/admission behavior is unchanged."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.frontier_shadow_rows: list[dict[str, Any]] = []
        self.retry_timeline: list[dict[str, Any]] = []
        self.poison_episodes: list[dict[str, Any]] = []
        self.protected_state_timeline: list[dict[str, Any]] = []
        self.shadow_probe_errors: list[dict[str, Any]] = []
        self.shadow_cpu_ns = 0
        self.shadow_client_cpu_ns = 0
        self.shadow_analysis_cpu_ns = 0
        self.shadow_monitor_cpu_ns = 0
        self.actual_retry_count = 0
        self.actual_frontier_policy_changes = 0
        self.shadow_monitor_started = False
        self.shadow_monitor_exited = False
        self.shadow_monitor_loop_count = 0
        self.shadow_monitor_probe_attempt_count = 0
        self.shadow_monitor_resume_skip_count = 0
        self.shadow_monitor_fatal_error: str | None = None
        self._shadow_monitor_started_event = threading.Event()
        self._shadow_wake_event = threading.Event()
        self.shadow_monitor_schedule_events: list[dict[str, Any]] = []
        self._actual_attempted: set[str] = set()
        self._terminal_timestamp_ns: dict[str, int] = {}
        self._shadow_lock = threading.Lock()
        self._shadow_stop = threading.Event()
        self._active_poison: dict[str, Any] | None = None
        self._shadow_frontier_id: str | None = None
        self._shadow_frontier_entered_ns: int | None = None
        self._last_state_signature: tuple[Any, ...] | None = None
        self._shadow_probe_index = 0

    def register(self, boundary_id: str, prefix: list[int], previous_finish_ns: int) -> int:
        deadline_ns = super().register(boundary_id, prefix, previous_finish_ns)
        state = self.registered[boundary_id]
        self.protected_state_timeline.append({
            "timestamp_ns": int(state["registered_ns"]), "event": "REGISTERED",
            "boundary_id": boundary_id, "decision": "REGISTERED",
            "source_visible": False, "source_exact_ready": None,
            "resume_started": False, "first_token": False,
            "lease_blocks": None, "why_frontier_selected": "earliest_deadline_unfinished",
        })
        cpu_started_ns = time.thread_time_ns()
        self._sync_poison_episode(int(state["registered_ns"]))
        self.shadow_client_cpu_ns += max(0, time.thread_time_ns() - cpu_started_ns)
        self._shadow_wake_event.set()
        return deadline_ns

    def mark_resume_started(self, boundary_id: str, timestamp_ns: int) -> None:
        super().mark_resume_started(boundary_id, timestamp_ns)
        self.protected_state_timeline.append({
            "timestamp_ns": int(timestamp_ns), "event": "RESUME_STARTED",
            "boundary_id": boundary_id, "decision": self.registered[boundary_id]["decision"],
            "source_visible": boundary_id in self.offload_events,
            "source_exact_ready": self.latest_exact.get(boundary_id, {}).get("source_fully_ready"),
            "resume_started": True, "first_token": False,
            "lease_blocks": None, "why_frontier_selected": "resume_arrived_unfinished",
        })
        cpu_started_ns = time.thread_time_ns()
        self._sync_poison_episode(int(timestamp_ns))
        self.shadow_client_cpu_ns += max(0, time.thread_time_ns() - cpu_started_ns)
        self._shadow_wake_event.set()

    def mark_first_token(self, boundary_id: str, timestamp_ns: int) -> None:
        super().mark_first_token(boundary_id, timestamp_ns)
        self.protected_state_timeline.append({
            "timestamp_ns": int(timestamp_ns), "event": "FIRST_TOKEN",
            "boundary_id": boundary_id, "decision": self.registered[boundary_id]["decision"],
            "source_visible": boundary_id in self.offload_events,
            "source_exact_ready": self.latest_exact.get(boundary_id, {}).get("source_fully_ready"),
            "resume_started": self.registered[boundary_id]["resume_started_ns"] is not None,
            "first_token": True, "lease_blocks": None,
            "why_frontier_selected": "first_token_ends_unfinished_frontier",
        })
        cpu_started_ns = time.thread_time_ns()
        self._sync_poison_episode(int(timestamp_ns))
        self.shadow_client_cpu_ns += max(0, time.thread_time_ns() - cpu_started_ns)
        self._shadow_wake_event.set()

    def _decision_timestamp(self, boundary_id: str, decision: str) -> int:
        normalized = analysis.normalize_decision(decision)
        for row in reversed(self.decisions):
            event = analysis.normalize_decision(row.get("event"))
            if row.get("boundary_id") == boundary_id and (
                event == normalized or event.endswith(normalized)
            ):
                return int(row.get("timestamp_ns", time.monotonic_ns()))
        return int(self.registered[boundary_id].get("registered_ns", time.monotonic_ns()))

    def _frontier_locked(self) -> str | None:
        unfinished = [(key, state) for key, state in self.registered.items()
                      if state["first_token_ns"] is None]
        if not unfinished:
            return None
        return min(unfinished, key=lambda pair: (pair[1]["deadline_ns"], pair[0]))[0]

    def _sync_poison_episode(self, now_ns: int) -> None:
        with self.lock:
            frontier = self._frontier_locked()
            frontier_state = dict(self.registered[frontier]) if frontier else None
            states = {key: dict(value) for key, value in self.registered.items()}
        with self._shadow_lock:
            if frontier != self._shadow_frontier_id:
                self._shadow_frontier_id = frontier
                self._shadow_frontier_entered_ns = now_ns if frontier is not None else None
            frontier_entered_ns = self._shadow_frontier_entered_ns
            active = self._active_poison
            if active is not None:
                old_id = str(active["protected_id"])
                old_state = states.get(old_id, {})
                end_ns = None
                end_reason = None
                if old_state.get("first_token_ns") is not None:
                    end_ns = int(old_state["first_token_ns"])
                    end_reason = "FIRST_TOKEN"
                elif frontier != old_id:
                    if frontier and int(states[frontier]["deadline_ns"]) < int(old_state.get("deadline_ns", 0)):
                        end_ns = int(states[frontier].get("registered_ns", now_ns))
                    else:
                        end_ns = now_ns
                    end_reason = "EARLIER_UNFINISHED_FRONTIER" if frontier else "NO_UNFINISHED_FRONTIER"
                if end_ns is not None:
                    active.update(analysis.finish_poison_episode(
                        active, end_timestamp_ns=end_ns, end_reason=end_reason,
                        first_token_ns=old_state.get("first_token_ns"),
                    ))
                    self._active_poison = None
                    active = None
            if active is None and frontier and frontier_state is not None:
                decision = str(frontier_state.get("decision", ""))
                if analysis.is_protected_poison_episode(
                    protected_id=frontier, boundary_id=frontier,
                    first_token_ns=frontier_state.get("first_token_ns"), decision=decision,
                ):
                    terminal_ns = self._decision_timestamp(frontier, decision)
                    episode_start_ns = analysis.poison_episode_start_ns(
                        terminal_ns, frontier_entered_ns,
                    )
                    active = {
                        "run_id": f"C1-r{self.repetition}",
                        "protected_id": frontier,
                        "terminal_reason": analysis.normalize_decision(decision),
                        "terminal_timestamp_ns": terminal_ns,
                        "frontier_entered_ns": frontier_entered_ns,
                        "episode_start_timestamp_ns": episode_start_ns,
                        "first_token_ns": None,
                        "end_timestamp_ns": None,
                        "end_reason": None,
                        "poison_duration_ms": None,
                    }
                    self.poison_episodes.append(active)
                    self._active_poison = active

    def _record_protected_state(self, timestamp_ns: int | None = None) -> None:
        now_ns = int(timestamp_ns or time.monotonic_ns())
        with self.lock:
            frontier = self._frontier_locked()
            state = dict(self.registered[frontier]) if frontier else None
            exact = self.latest_exact.get(frontier, {}) if frontier else {}
            visible = frontier in self.offload_events if frontier else False
            lease_count = None
            for event in reversed(self.lease_events):
                if event.get("lease_key") == (self.manifest.get(frontier, {}).get("lease_key") if frontier else None):
                    if event.get("event") == "LEASE_RELEASED":
                        lease_count = 0
                        break
                    if event.get("event") == "LEASE_ACQUIRED":
                        lease_count = event.get("num_blocks", event.get("block_count"))
                        break
            signature = (frontier, state.get("decision") if state else None,
                         visible, exact.get("source_fully_ready"),
                         state.get("resume_started_ns") is not None if state else None,
                         state.get("first_token_ns") is not None if state else None,
                         lease_count)
            if signature == self._last_state_signature:
                return
            self._last_state_signature = signature
            self.protected_state_timeline.append({
                "timestamp_ns": now_ns, "event": "PROTECTED_STATE_CHANGE",
                "boundary_id": frontier,
                "decision": state.get("decision") if state else None,
                "source_visible": visible,
                "source_exact_ready": exact.get("source_fully_ready"),
                "resume_started": state.get("resume_started_ns") is not None if state else None,
                "first_token": state.get("first_token_ns") is not None if state else None,
                "lease_blocks": lease_count,
                "why_frontier_selected": "earliest_deadline_among_first_token_unfinished",
            })

    def _lease_blocks_at(self, boundary_id: str, timestamp_ns: int) -> int | None:
        lease_key = str(self.manifest[boundary_id]["lease_key"])
        active: int | None = 0
        for event in sorted(self.lease_events, key=lambda row: int(
                row.get("timestamp_monotonic_ns") or round(float(row.get("timestamp_monotonic", 0)) * 1e9))):
            event_ns = int(event.get("timestamp_monotonic_ns") or round(
                float(event.get("timestamp_monotonic", 0)) * 1e9
            ))
            if event_ns > timestamp_ns or event.get("lease_key") != lease_key:
                continue
            if event.get("event") == "LEASE_ACQUIRED":
                count = event.get("num_blocks", event.get("block_count"))
                ids = event.get("block_ids")
                active = int(count) if count is not None else (len(set(ids)) if isinstance(ids, list) else None)
            elif event.get("event") == "LEASE_RELEASED":
                active = 0
        return active

    def _selection_frontiers(self, selected: tuple[Any, ...]) -> None:
        candidate_id, _, protected_id = selected
        row = self.selection_rows[-1]
        snapshot = row.get("selection_snapshot") or {}
        timestamp_ns = int(row.get("timestamp_ns", time.monotonic_ns()))
        free = analysis.optional_int(snapshot.get("free_blocks_before"))
        candidate_blocks = int(row["candidate_required_blocks"])
        exact_ready = {key: value.get("source_fully_ready")
                       for key, value in self.latest_exact.items()}
        exact_ready[candidate_id] = row.get("candidate_exact_source_ready")
        exact_ready[protected_id] = row.get("protected_exact_source_ready")
        with self.lock:
            states = {key: dict(value) for key, value in self.registered.items()}
        frontiers = analysis.choose_frontiers(states, exact_ready, candidate_id)
        snapshot_f0 = frontiers["F0"]
        # The controller's returned protected identity is the authoritative F0
        # for this selection; retain later-state drift as a diagnostic only.
        frontiers["F0"] = protected_id
        protected_ownership = snapshot.get("protected_local_ownership") or {}
        frontier_rows: dict[str, dict[str, Any]] = {}
        values: dict[str, Any] = {}
        for label, frontier_id in frontiers.items():
            if frontier_id is None or frontier_id not in self.manifest:
                values[f"{label}_id"] = frontier_id
                values[f"H_{label}"] = None
                values[f"PSA_fit_{label}"] = None
                frontier_rows[label] = {}
                continue
            manifest_row = self.manifest[frontier_id]
            existing = (analysis.optional_int(protected_ownership.get("current_protected_lease_blocks"))
                        if frontier_id == protected_id else self._lease_blocks_at(frontier_id, timestamp_ns))
            if existing is None:
                existing = 0 if self._lease_blocks_at(frontier_id, timestamp_ns) == 0 else None
            progress = progress_blocks(int(manifest_row["resume_isl"]))
            need = psa1_reserve(progress, int(existing or 0),
                                candidate_id == frontier_id, candidate_blocks)
            fit = analysis.psa_fit(free, candidate_blocks, need)
            state = states[frontier_id]
            is_terminal = analysis.is_nonresident_terminal(state.get("decision"))
            item = {
                "frontier_id": frontier_id,
                "frontier_state": state.get("decision"),
                "frontier_is_terminal": is_terminal,
                "frontier_source_exact_ready": exact_ready.get(frontier_id),
                "frontier_progress_blocks": progress,
                "frontier_existing_lease_blocks": existing,
                "H": need,
                "PSA_fit": fit,
            }
            frontier_rows[label] = item
            values[f"{label}_id"] = frontier_id
            values[f"{label}_state"] = state.get("decision")
            if label == "F0":
                values["F0_is_terminal"] = is_terminal
                values["F0_terminal_reason"] = state.get("decision") if is_terminal else None
            values[f"candidate_equals_{label}"] = candidate_id == frontier_id
            values[f"H_{label}"] = need
            values[f"PSA_fit_{label}"] = fit
        values["selection_id"] = row.get("selection_id")
        values["condition"] = self.condition
        values["repetition"] = self.repetition
        values["timestamp_ns"] = timestamp_ns
        values["candidate_id"] = candidate_id
        values["F0_shadow_reselection_differs"] = snapshot_f0 != protected_id
        values["F0_is_terminal"] = values.get("F0_is_terminal", False)
        values["F0_terminal_reason"] = values.get("F0_terminal_reason")
        values["candidate_required_blocks"] = candidate_blocks
        values["pool_free_blocks"] = free
        values["allocator_available_blocks"] = row.get("selection_protected_self_allocator_available")
        values["protected_allocator_required_blocks"] = row.get("selection_protected_self_allocator_required")
        values["candidate_source_ready"] = row.get("candidate_exact_source_ready")
        values["candidate_logical_fit"] = row.get("candidate_logical_fit")
        values["timing_gate_open"] = True
        values["actual_H"] = row.get("PSA1_reserve_blocks")
        values["candidate_matches_protected"] = candidate_id == protected_id
        values["terminal_head_protection"] = bool(
            values["F0_is_terminal"] and candidate_id != values.get("F0_id")
        )
        values.update({
            f"{label}_existing_lease_blocks": item.get("frontier_existing_lease_blocks")
            for label, item in frontier_rows.items() if item
        })
        self.frontier_shadow_rows.append(values)

    def _candidate(self):
        selected = super()._candidate()
        now_ns = time.monotonic_ns()
        cpu_started_ns = time.thread_time_ns()
        self._sync_poison_episode(now_ns)
        self.shadow_analysis_cpu_ns += max(0, time.thread_time_ns() - cpu_started_ns)
        self._record_protected_state(now_ns)
        if selected is not None:
            cpu_started_ns = time.thread_time_ns()
            self._selection_frontiers(selected)
            self.shadow_analysis_cpu_ns += max(0, time.thread_time_ns() - cpu_started_ns)
        return selected

    def _attempt(self, key: str, state: dict[str, Any], protected: str, pool) -> None:
        if key in self._actual_attempted:
            self.actual_retry_count += 1
        self._actual_attempted.add(key)
        before = self.registered[key].get("decision")
        super()._attempt(key, state, protected, pool)
        after = self.registered[key].get("decision")
        if after != before:
            self._record_state_decision(key, after)

    def _record_state_decision(self, boundary_id: str, decision: Any) -> None:
        state = self.registered.get(boundary_id, {})
        timestamp_ns = self._decision_timestamp(boundary_id, str(decision))
        if analysis.is_nonresident_terminal(decision):
            self._terminal_timestamp_ns[boundary_id] = timestamp_ns
        self.protected_state_timeline.append({
            "timestamp_ns": timestamp_ns,
            "event": "DECISION", "boundary_id": boundary_id,
            "decision": str(decision),
            "source_visible": boundary_id in self.offload_events,
            "source_exact_ready": self.latest_exact.get(boundary_id, {}).get("source_fully_ready"),
            "resume_started": state.get("resume_started_ns") is not None,
            "first_token": state.get("first_token_ns") is not None,
            "lease_blocks": None,
            "why_frontier_selected": "decision_state_only_no_frontier_mutation",
        })
        cpu_started_ns = time.thread_time_ns()
        self._sync_poison_episode(timestamp_ns)
        self.shadow_analysis_cpu_ns += max(0, time.thread_time_ns() - cpu_started_ns)
        self._shadow_wake_event.set()

    def _harvest(self) -> None:
        before = {key: state.get("decision") for key, state in self.registered.items()}
        super()._harvest()
        for key, state in self.registered.items():
            if before.get(key) != state.get("decision"):
                self._record_state_decision(key, state.get("decision"))

    def run(self, all_sessions_done: threading.Event) -> None:
        self._shadow_stop.clear()
        self._shadow_wake_event.clear()
        self._shadow_monitor_started_event.clear()
        monitor = threading.Thread(target=self._shadow_monitor, name="phase4g-d-shadow", daemon=True)
        monitor.start()
        try:
            if not self._shadow_monitor_started_event.wait(timeout=1.0):
                self.shadow_monitor_fatal_error = "shadow monitor thread did not start"
            super().run(all_sessions_done)
        finally:
            self._shadow_stop.set()
            self._shadow_wake_event.set()
            monitor.join(timeout=4)
            if monitor.is_alive():
                self.shadow_monitor_fatal_error = "shadow monitor did not stop"
            self._sync_poison_episode(time.monotonic_ns())
            self._close_open_episode(time.monotonic_ns(), "RUN_END")

    def _close_open_episode(self, timestamp_ns: int, reason: str) -> None:
        with self._shadow_lock:
            active = self._active_poison
            if active is None:
                return
            state = self.registered.get(str(active["protected_id"]), {})
            if state.get("first_token_ns") is not None:
                timestamp_ns = int(state["first_token_ns"])
                reason = "FIRST_TOKEN"
            active.update(analysis.finish_poison_episode(
                active, end_timestamp_ns=timestamp_ns, end_reason=reason,
                first_token_ns=state.get("first_token_ns"),
            ))
            self._active_poison = None

    def _timing_gate_open(self, boundary_id: str, snapshot_ns: int) -> bool:
        now_ms = (snapshot_ns - self.origin_ns) / 1e6
        with self.lock:
            active = {
                key: dict(state) for key, state in self.registered.items()
                if state.get("first_token_ns") is None
                and state.get("resume_started_ns") is None
                and state.get("decision") == "REGISTERED"
            }
            active[boundary_id] = dict(self.registered[boundary_id])
            visible_ready = set(self.offload_events)
            deadlines = {
                key: (int(state["deadline_ns"]) - self.origin_ns) / 1e6
                for key, state in active.items()
            }
            ready = [key for key in active if key in visible_ready]
        if boundary_id not in ready:
            ready.append(boundary_id)
        start = latest_feasible_start(
            now_ms,
            [(key, deadlines[key]) for key in ready],
            merge_blackouts(deadlines, 110),
            102,
        )
        return now_ms + .5 >= start

    def _snapshot_for_head(self, boundary_id: str) -> dict[str, Any]:
        row = self.manifest[boundary_id]
        self._shadow_probe_index += 1
        selection_id = f"retry-shadow-r{self.repetition}-n{self._shadow_probe_index}"
        issue_ns = time.monotonic_ns()
        with self.lock:
            reserved_before = int(self.ledger.reserved_blocks)
            state_before = dict(self.registered[boundary_id])
            candidate_visible = boundary_id in self.offload_events
        progress = progress_blocks(int(row["resume_isl"]))
        rca = {
            "selection_id": selection_id,
            "selection_timestamp_ns": issue_ns,
            "candidate_id": boundary_id,
            "protected_id": boundary_id,
            "candidate_matches_protected": True,
            "candidate_controller_visible_ready": candidate_visible,
            "protected_controller_visible_ready": candidate_visible,
            "candidate_prefetch_tokens": int(row["required_prefix_tokens"]),
            "candidate_source_tokens": int(row["required_prefix_tokens"]),
            "protected_lease_key": str(row["lease_key"]),
            "protected_progress_blocks": progress,
            "protected_full_blocks": int(row["required_blocks"]),
            "protected_source_tokens": int(row["required_prefix_tokens"]),
            "protected_isl": int(row["resume_isl"]),
            "protected_osl": int(row["resume_osl"]),
            "protected_full_sequence_tokens": int(row["resume_isl"]) + int(row["resume_osl"]),
            "protected_first_token_tokens": int(row["resume_isl"]) + 1,
            "psa1_reserve_blocks": protected_self_reserve(
                progress, 0, int(row["required_blocks"]),
            ),
        }
        thread_cpu_before = time.thread_time_ns()
        request_issue_ns = time.monotonic_ns()
        response = read_only_selection_snapshot(rca_snapshot=rca, timeout_s=3.0)
        thread_cpu_after = time.thread_time_ns()
        snapshot = response["snapshot"]
        snapshot_ns = round(float(snapshot.get("timestamp_monotonic", 0)) * 1e9)
        if snapshot_ns <= 0:
            snapshot_ns = int(response["client_return_ns"])
        source = snapshot.get("protected_source_exact") or {}
        preflight = snapshot.get("protected_self_allocator_preflight") or {}
        ownership = snapshot.get("protected_local_ownership") or {}
        required = analysis.optional_int(preflight.get("allocator_required_blocks"))
        available = analysis.optional_int(preflight.get("allocator_available_blocks"))
        allocator_fit = preflight.get("allocator_fit")
        if required is None or available is None or not isinstance(allocator_fit, bool):
            allocator_fit = None
        free = analysis.optional_int(preflight.get("pool_free_blocks"))
        if free is None:
            free = analysis.optional_int(snapshot.get("free_blocks_before"))
        reserved_after = int(self.ledger.reserved_blocks)
        reserved_stable = reserved_before == reserved_after
        logical_fit = (reserved_before + int(row["required_blocks"]) <= self.ledger.budget_blocks
                       if reserved_stable else None)
        state_after = self.registered[boundary_id]
        resume_ns = [int(value) for value in (state_before.get("resume_started_ns"),
                                               state_after.get("resume_started_ns"))
                     if value is not None]
        first_token_ns = [int(value) for value in (state_before.get("first_token_ns"),
                                                    state_after.get("first_token_ns"))
                          if value is not None]
        resume_started = bool(resume_ns and min(resume_ns) <= snapshot_ns)
        first_token_seen = bool(first_token_ns and min(first_token_ns) <= snapshot_ns)
        with self.lock:
            current_frontier = self._frontier_locked()
        existing = analysis.optional_int(ownership.get("current_protected_lease_blocks"))
        if existing is None:
            existing = 0 if self._lease_blocks_at(boundary_id, snapshot_ns) == 0 else None
        progress_need = progress_blocks(int(row["resume_isl"]))
        self_h = (protected_self_reserve(progress_need, int(existing or 0),
                                         int(row["required_blocks"]))
                  if existing is not None else None)
        self_psa_fit = analysis.psa_fit(free, int(row["required_blocks"]), self_h)
        source_ready = source.get("source_fully_ready")
        timing_open = self._timing_gate_open(boundary_id, snapshot_ns)
        sample = {
            "run_id": f"C1-r{self.repetition}",
            "timestamp_ns": snapshot_ns,
            "request_issue_ns": request_issue_ns,
            "request_return_ns": int(response["client_return_ns"]),
            "snapshot_roundtrip_ms": float(response["client_roundtrip_ms"]),
            "probe_interval_ms": None,
            "protected_id": boundary_id,
            "is_current_protected_frontier": current_frontier == boundary_id,
            "terminal_timestamp_ns": self._terminal_timestamp_ns.get(boundary_id),
            "terminal_reason": self.registered[boundary_id].get("decision"),
            "source_full_ready": source_ready,
            "resume_started": resume_started,
            "first_token_seen": first_token_seen,
            "logical_reserved_blocks": reserved_before,
            "logical_reservation_fit": logical_fit,
            "timing_gate_open": timing_open,
            "self_psa_H": self_h,
            "self_psa_fit": self_psa_fit,
            "pool_free_blocks": free,
            "allocator_required_blocks": required,
            "allocator_available_blocks": available,
            "allocator_fit": allocator_fit,
            "legacy_physical_fit": (free >= int(row["required_blocks"]) if free is not None else None),
            "full_required_blocks": int(row["required_blocks"]),
            "protected_local_cached_blocks": source.get("local_cached_blocks"),
            "protected_external_ready_blocks": source.get("external_ready_blocks"),
            "protected_missing_blocks": source.get("external_ready_needed_blocks"),
            "existing_lease_blocks": existing,
            "retry_actionable": False,
            "shadow_server_cpu_ns": int(snapshot.get("capacity_trace_cpu_ns", 0))
                + int(snapshot.get("source_probe_cpu_ns", 0))
                + int(snapshot.get("allocator_shadow_cpu_ns", 0)),
            "shadow_client_cpu_ns": max(0, thread_cpu_after - thread_cpu_before),
            "snapshot_event_timestamp_ns": snapshot_ns,
        }
        sample["retry_actionable"] = analysis.retry_actionable(sample)
        return sample

    def _shadow_monitor(self) -> None:
        next_sample_ns: dict[str, int] = {}
        last_sample_ns: dict[str, int] = {}
        schedule_events: dict[str, dict[str, Any]] = {}
        self.shadow_monitor_started = True
        self._shadow_monitor_started_event.set()
        try:
            while not self._shadow_stop.is_set():
                self.shadow_monitor_loop_count += 1
                loop_cpu_start_ns = time.thread_time_ns()
                boundary_id: str | None = None
                try:
                    now_ns = time.monotonic_ns()
                    self._sync_poison_episode(now_ns)
                    self._record_protected_state(now_ns)
                    # Clear before reading state so a concurrent decision/resume
                    # either appears in the snapshot below or wakes this wait.
                    self._shadow_wake_event.clear()
                    with self.lock:
                        states = {key: dict(value) for key, value in self.registered.items()}
                    with self._shadow_lock:
                        terminal_times = dict(self._terminal_timestamp_ns)
                    for boundary_id, state in states.items():
                        decision = state.get("decision")
                        if not analysis.is_nonresident_terminal(decision):
                            continue
                        terminal_ns = terminal_times.get(boundary_id)
                        if terminal_ns is None:
                            terminal_ns = self._decision_timestamp(boundary_id, str(decision))
                            terminal_times[boundary_id] = terminal_ns
                            with self._shadow_lock:
                                self._terminal_timestamp_ns[boundary_id] = terminal_ns
                        event = schedule_events.get(boundary_id)
                        if event is None:
                            target_ns = analysis.first_retry_probe_deadline_ns(terminal_ns)
                            event = {
                                "protected_id": boundary_id,
                                "terminal_timestamp_ns": terminal_ns,
                                "episode_start_timestamp_ns": terminal_ns,
                                "monitor_first_seen_ns": now_ns,
                                "first_probe_target_ns": target_ns,
                                "outcome": "WAITING_FOR_FIRST_PROBE",
                                "resume_started_ns": None,
                                "first_token_ns": None,
                                "first_probe_attempt_ns": None,
                            }
                            schedule_events[boundary_id] = event
                            self.shadow_monitor_schedule_events.append(event)
                            next_sample_ns[boundary_id] = target_ns
                        resume_ns = state.get("resume_started_ns")
                        first_token_ns = state.get("first_token_ns")
                        if resume_ns is not None or first_token_ns is not None:
                            next_sample_ns.pop(boundary_id, None)
                            self.shadow_monitor_resume_skip_count += 1
                            event["resume_started_ns"] = resume_ns
                            event["first_token_ns"] = first_token_ns
                            if event["outcome"] == "WAITING_FOR_FIRST_PROBE":
                                event["outcome"] = "RESUME_BEFORE_FIRST_PROBE"
                            elif event["outcome"] == "FIRST_PROBE_ATTEMPTED":
                                event["outcome"] = "PROBING_STOPPED_AT_RESUME_OR_TOKEN"
                    due = [(target, boundary_id) for boundary_id, target in next_sample_ns.items()]
                    if not due:
                        self._shadow_wake_event.wait(.02)
                        continue
                    target_ns, boundary_id = min(due)
                    if now_ns < target_ns:
                        self._shadow_wake_event.wait(min(.02, (target_ns - now_ns) / 1e9))
                        continue
                    event = schedule_events[boundary_id]
                    if event["outcome"] == "WAITING_FOR_FIRST_PROBE":
                        event["outcome"] = "FIRST_PROBE_ATTEMPTED"
                        event["first_probe_attempt_ns"] = now_ns
                    self.shadow_monitor_probe_attempt_count += 1
                    sample = self._snapshot_for_head(boundary_id)
                    sample["probe_target_ns"] = target_ns
                    previous_ns = last_sample_ns.get(boundary_id)
                    if previous_ns is not None:
                        sample["probe_interval_ms"] = (sample["timestamp_ns"] - previous_ns) / 1e6
                    last_sample_ns[boundary_id] = int(sample["timestamp_ns"])
                    self.retry_timeline.append(sample)
                    self.shadow_cpu_ns += int(sample["shadow_server_cpu_ns"])
                    self.shadow_client_cpu_ns += int(sample["shadow_client_cpu_ns"])
                    next_sample_ns[boundary_id] = max(
                        target_ns + analysis.RECHECK_INTERVAL_MS * 1_000_000,
                        int(sample["timestamp_ns"]) + analysis.RECHECK_INTERVAL_MS * 1_000_000,
                    )
                except BaseException as exc:
                    self.shadow_probe_errors.append({
                        "timestamp_ns": time.monotonic_ns(),
                        "protected_id": boundary_id,
                        "error": repr(exc),
                    })
                    if boundary_id is not None and boundary_id in next_sample_ns:
                        next_sample_ns[boundary_id] = time.monotonic_ns() + analysis.RECHECK_INTERVAL_MS * 1_000_000
                    self._shadow_stop.wait(.001)
                finally:
                    self.shadow_monitor_cpu_ns += max(0, time.thread_time_ns() - loop_cpu_start_ns)
        except BaseException as exc:
            self.shadow_monitor_fatal_error = repr(exc)
            self.shadow_probe_errors.append({
                "timestamp_ns": time.monotonic_ns(), "protected_id": None,
                "error": repr(exc), "fatal": True,
            })
        finally:
            self.shadow_monitor_exited = True
