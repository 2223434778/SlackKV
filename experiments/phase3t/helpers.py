from __future__ import annotations

from dataclasses import dataclass

SESSION_LEASE_BLOCKS = 256
LEASE_BUDGET_BLOCKS = 512
HANDOFF_GUARD_MS = 100.0
GATE_LIMIT = 2


@dataclass(frozen=True)
class Reservation:
    session: str
    blocks: int
    start_ms: float
    end_ms: float


class ReservationLedger:
    def __init__(self) -> None:
        self.active: list[Reservation] = []

    @property
    def reserved_blocks(self) -> int:
        return sum(item.blocks for item in self.active)

    def expire(self, now_ms: float) -> list[tuple[Reservation, int, int]]:
        expired=[]
        for item in list(self.active):
            if item.end_ms <= now_ms:
                before=self.reserved_blocks; self.active.remove(item)
                expired.append((item,before,self.reserved_blocks))
        return expired

    def decide(self) -> str:
        return "RESERVATION_PROMOTE" if self.reserved_blocks + SESSION_LEASE_BLOCKS <= LEASE_BUDGET_BLOCKS else "RESERVATION_SKIP"

    def grant(self, session: str, target_ms: float, deadline_ms: float) -> Reservation:
        if self.decide() != "RESERVATION_PROMOTE":
            raise AssertionError("reservation budget full")
        item=Reservation(session,SESSION_LEASE_BLOCKS,target_ms,deadline_ms+HANDOFF_GUARD_MS)
        self.active.append(item)
        if self.reserved_blocks > LEASE_BUDGET_BLOCKS:
            raise AssertionError("reservation budget exceeded")
        return item


def classify_promoted_outcome(decision: str, t6_ms, arrival_ms: float, remote_wait: bool, lease_ok: bool) -> str:
    if decision == "RESERVATION_SKIP": return "SKIPPED_ONDEMAND"
    if not lease_ok: return "LEASE_ERROR"
    if float(t6_ms) > arrival_ms: return "PROMOTED_INTERNAL_READY_MISS"
    if remote_wait: return "PROMOTED_REMOTE_WAIT"
    return "PROMOTED_LOCAL"


def promotion_useful_rate(useful: int, promoted: int) -> float:
    return useful / promoted if promoted else 0.0


def wasted_promotion_mib(remote_waits: int) -> int:
    return remote_waits * 112


def source_conservation(local: int, external: int) -> bool:
    return local + external == 49152
