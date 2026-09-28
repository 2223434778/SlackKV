"""Event-driven, frozen FullSlackKV controller for registered boundaries only."""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from experiments.phase3f.scheduler import latest_feasible_start, merge_blackouts
from experiments.phase4d_a.helpers import ReservationLedger
from experiments.phase4d_a_fix.release_fix import RuntimeReleaseQueue, parse_lease_trace
from experiments.phase4f_a.logic import progress_blocks
from experiments.phase4f_a.progress_client import progress_prefetch

P_SAFE_MS = 102.0
RESUME_GUARD_MS = 110.0
BUDGET_BLOCKS = 3510
MAX_PREFETCH_SECONDS = 5.0


class StructuredTraceReader:
    """Read only newly appended server-log lines; preserve partial lines."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0
        self.partial = b""

    def read(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        try:
            with self.path.open("rb") as stream:
                stream.seek(self.offset)
                chunk = stream.read()
                self.offset = stream.tell()
        except FileNotFoundError:
            return [], []
        if not chunk:
            return [], []
        lines = (self.partial + chunk).split(b"\n")
        self.partial = lines.pop()
        text = b"\n".join(lines).decode("utf-8", errors="replace")
        offload = []
        for line in text.splitlines():
            if "SLACKKV_OFFLOAD_TRACE " not in line:
                continue
            try:
                event = json.loads(line.split("SLACKKV_OFFLOAD_TRACE ", 1)[1])
            except (ValueError, IndexError):
                continue
            offload.append(event)
        return offload, parse_lease_trace(text)


class DynamicController:
    def __init__(self, manifest: list[dict[str, object]], server_log: Path, origin_ns: int) -> None:
        self.manifest = {str(row["boundary_id"]): row for row in manifest}
        self.origin_ns = origin_ns
        self.reader = StructuredTraceReader(server_log)
        self.ledger = ReservationLedger(BUDGET_BLOCKS)
        keys = {key: str(row["lease_key"]) for key, row in self.manifest.items()}
        blocks = {key: int(row["required_blocks"]) for key, row in self.manifest.items()}
        self.release_queue = RuntimeReleaseQueue(self.ledger, keys, blocks, origin_ns / 1e9)
        self.lock = threading.Lock()
        self.registered: dict[str, dict[str, Any]] = {}
        self.offload_events: dict[str, dict[str, Any]] = {}
        self.promotions: dict[str, dict[str, Any]] = {}
        self.decisions: list[dict[str, Any]] = []
        self.releases: list[dict[str, Any]] = []
        self.lease_events: list[dict[str, Any]] = []
        self.error: BaseException | None = None

    def register(self, boundary_id: str, prefix: list[int], previous_finish_ns: int) -> int:
        row = self.manifest[boundary_id]
        deadline_ns = previous_finish_ns + round(float(row["proxy_pause_ms"]) * 1e6)
        if len(prefix) != int(row["required_prefix_tokens"]):
            raise ValueError("dynamic prefix length mismatch")
        registered_ns = time.monotonic_ns()
        if registered_ns < previous_finish_ns:
            raise RuntimeError("boundary registered before previous turn finish")
        with self.lock:
            if boundary_id in self.registered:
                raise RuntimeError("duplicate dynamic boundary registration")
            self.registered[boundary_id] = {
                "boundary_id": boundary_id, "prefix": prefix,
                "deadline_ns": deadline_ns, "registered_ns": registered_ns,
                "previous_finish_ns": previous_finish_ns,
                "source_ready_ns": None, "resume_started_ns": None,
                "first_token_ns": None, "decision": "REGISTERED",
            }
        return deadline_ns

    def mark_resume_started(self, boundary_id: str, timestamp_ns: int) -> None:
        with self.lock:
            self.registered[boundary_id]["resume_started_ns"] = timestamp_ns

    def mark_first_token(self, boundary_id: str, timestamp_ns: int) -> None:
        with self.lock:
            self.registered[boundary_id]["first_token_ns"] = timestamp_ns

    def _observe_traces(self) -> None:
        offload, lease = self.reader.read()
        self.lease_events.extend(lease)
        self.release_queue.observe_lease_events(lease)
        with self.lock:
            for event in offload:
                key = event.get("boundary_id")
                if not isinstance(key, str) or key not in self.manifest:
                    continue
                if event.get("event") == "SOURCE_READY":
                    self.offload_events[key] = event
                    if key in self.registered:
                        self.registered[key]["source_ready_ns"] = round(float(event["timestamp_monotonic"]) * 1e9)

    def _drain_releases(self) -> None:
        with self.lock:
            active_first = [(key, self.registered[key]["first_token_ns"])
                            for key in self.ledger.active if key in self.registered]
        for key, first_ns in active_first:
            if first_ns is not None:
                self.release_queue.enqueue_first_token(key, int(first_ns))
        self.releases.extend(self.release_queue.drain(time.monotonic_ns()))

    def _harvest(self) -> None:
        for key, item in list(self.promotions.items()):
            future: Future = item["future"]
            if item.get("harvested") or not future.done():
                continue
            response = future.result()
            status = response.get("status")
            if status not in {"ok", "capacity_rejected", "progress_rejected"}:
                raise RuntimeError(f"unexpected native status:{key}:{status}")
            item["response"] = response
            item["harvested"] = True
            with self.lock:
                state = self.registered[key]
                state["decision"] = str(status).upper()
            if status == "ok":
                if int(response.get("external_tokens_loaded", -1)) < 0 or int(response.get("transfer_bytes", -1)) < 0:
                    raise RuntimeError(f"invalid native counters:{key}")
                if any(int(response.get(field, -1)) != 0 for field in (
                    "model_forward_count", "generated_token_count", "mixed_prefetch_model_step_count"
                )):
                    raise RuntimeError(f"native promotion performed model work:{key}")
            else:
                self.ledger.release(key)
            self.decisions.append({"boundary_id": key, "event": "PROMOTION_" + str(status).upper(),
                                   "timestamp_ns": time.monotonic_ns()})

    def _candidate(self) -> tuple[str, dict[str, Any], str] | None:
        with self.lock:
            active = [(key, dict(state)) for key, state in self.registered.items()
                      if state["decision"] == "REGISTERED" and state["resume_started_ns"] is None]
            frontier = [(key, state) for key, state in self.registered.items()
                        if state["first_token_ns"] is None]
        if not active or not frontier:
            return None
        protected = min(frontier, key=lambda pair: (pair[1]["deadline_ns"], pair[0]))[0]
        ready = [(key, state) for key, state in active if key in self.offload_events]
        if not ready:
            return None
        key, state = min(ready, key=lambda pair: (pair[1]["deadline_ns"], pair[0]))
        now_ms = (time.monotonic_ns() - self.origin_ns) / 1e6
        deadlines = {candidate: (entry["deadline_ns"] - self.origin_ns) / 1e6
                     for candidate, entry in active}
        blackout = merge_blackouts(deadlines, RESUME_GUARD_MS)
        target = latest_feasible_start(now_ms, [(candidate, deadlines[candidate]) for candidate, _ in ready],
                                       blackout, P_SAFE_MS)
        if now_ms + .5 < target:
            return None
        return key, state, protected

    def _attempt(self, key: str, state: dict[str, Any], protected: str, pool: ThreadPoolExecutor) -> None:
        row = self.manifest[key]
        with self.lock:
            current = self.registered[key]
            if current["resume_started_ns"] is not None or time.monotonic_ns() >= current["deadline_ns"]:
                current["decision"] = "STALE_PROMOTION_CANCELLED"
                self.decisions.append({"boundary_id": key, "event": "STALE_PROMOTION_CANCELLED",
                                       "timestamp_ns": time.monotonic_ns()})
                return
            reserved = self.ledger.reserve(key, int(row["required_blocks"]))
            if not reserved:
                current["decision"] = "RESERVATION_SKIP"
                self.decisions.append({"boundary_id": key, "event": "RESERVATION_SKIP",
                                       "timestamp_ns": time.monotonic_ns()})
                return
            current["decision"] = "PROMOTION_PENDING"
            protected_row = self.manifest[protected]
            future = pool.submit(progress_prefetch, state["prefix"],
                required_blocks=int(row["required_blocks"]), lease_key=str(row["lease_key"]),
                protected_lease_key=str(protected_row["lease_key"]),
                protected_progress_blocks=progress_blocks(int(protected_row["resume_isl"])),
                timeout_s=MAX_PREFETCH_SECONDS)
            self.promotions[key] = {"future": future, "submitted_ns": time.monotonic_ns(),
                                    "protected_boundary_id": protected, "harvested": False}
            self.decisions.append({"boundary_id": key, "event": "PROMOTION_ATTEMPT",
                                   "protected_boundary_id": protected, "timestamp_ns": time.monotonic_ns(),
                                   "reserved_blocks": self.ledger.reserved_blocks})

    def run(self, all_sessions_done: threading.Event) -> None:
        try:
            with ThreadPoolExecutor(max_workers=8) as pool:
                while True:
                    self._observe_traces()
                    self._harvest()
                    self._drain_releases()
                    now = time.monotonic_ns()
                    with self.lock:
                        for key, state in self.registered.items():
                            if state["decision"] != "REGISTERED":
                                continue
                            if key in self.offload_events:
                                state["source_ready_ns"] = round(float(self.offload_events[key]["timestamp_monotonic"]) * 1e9)
                            if state["resume_started_ns"] is not None or now >= state["deadline_ns"]:
                                state["decision"] = "SOURCE_LATE_SKIP" if state["source_ready_ns"] is None else "STALE_PROMOTION_CANCELLED"
                                self.decisions.append({"boundary_id": key, "event": state["decision"],
                                                       "timestamp_ns": now})
                    candidate = self._candidate()
                    if candidate is not None:
                        self._attempt(*candidate, pool)
                    if any(not item.get("harvested") and (now - item["submitted_ns"]) / 1e9 > MAX_PREFETCH_SECONDS
                           for item in self.promotions.values()):
                        raise TimeoutError("PREFETCH_NONPROGRESS")
                    if all_sessions_done.is_set() and all(item.get("harvested") for item in self.promotions.values()) and not self.ledger.active:
                        break
                    time.sleep(.01)
        except BaseException as exc:
            self.error = exc
            all_sessions_done.set()
