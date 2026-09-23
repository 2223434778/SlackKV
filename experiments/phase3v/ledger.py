from __future__ import annotations

from dataclasses import dataclass
import math

BLOCK_SIZE_TOKENS = 16
LEASE_BUDGET_BLOCKS = 512
HANDOFF_GUARD_MS = 100.0
KV_BYTES_PER_TOKEN = 117_440_512 / 4096
KV_MIB_PER_BLOCK = 112.0 / 256.0
SESSIONS = tuple("ABCDEFGHIJKL")
DEADLINES_MS = {
    "A": 300.0, "B": 320.0, "C": 340.0, "D": 360.0,
    "E": 900.0, "F": 920.0, "G": 940.0, "H": 960.0,
    "I": 1500.0, "J": 1520.0, "K": 1540.0, "L": 1560.0,
}
UNIFORM_TOKENS = {session: 4096 for session in SESSIONS}
HETEROGENEOUS_TOKENS = {
    "A": 1024, "B": 1024, "C": 1024, "D": 4096,
    "E": 1024, "F": 1024, "G": 1024, "H": 4096,
    "I": 1024, "J": 1024, "K": 1024, "L": 4096,
}
POLICIES = (
    "OnDemand-K2",
    "NaiveAsync40-K2",
    "FixedCount2-Async40-K2",
    "DynamicBlock512-Async40-K2",
)


@dataclass(frozen=True)
class Reservation:
    session: str
    blocks: int
    start_ms: float
    end_ms: float


def required_blocks(prefix_tokens: int, block_size_tokens: int = BLOCK_SIZE_TOKENS) -> int:
    if prefix_tokens <= 0 or block_size_tokens <= 0:
        raise ValueError("token counts and block size must be positive")
    return math.ceil(prefix_tokens / block_size_tokens)


def required_mib(prefix_tokens: int) -> float:
    return prefix_tokens * KV_BYTES_PER_TOKEN / (1024.0**2)


class BlockReservationLedger:
    def __init__(self, budget_blocks: int = LEASE_BUDGET_BLOCKS) -> None:
        if budget_blocks <= 0:
            raise ValueError("reservation budget must be positive")
        self.budget_blocks = budget_blocks
        self.active: list[Reservation] = []

    @property
    def reserved_blocks(self) -> int:
        return sum(item.blocks for item in self.active)

    def expire(self, now_ms: float) -> list[tuple[Reservation, int, int]]:
        expired: list[tuple[Reservation, int, int]] = []
        for item in list(self.active):
            if item.end_ms <= now_ms:
                before = self.reserved_blocks
                self.active.remove(item)
                expired.append((item, before, self.reserved_blocks))
        return expired

    def can_admit(self, required: int) -> bool:
        return required > 0 and self.reserved_blocks + required <= self.budget_blocks

    def grant(self, session: str, blocks: int, start_ms: float, deadline_ms: float) -> Reservation:
        if any(item.session == session for item in self.active):
            raise AssertionError(f"duplicate live reservation for {session}")
        if not self.can_admit(blocks):
            raise AssertionError("reservation block budget full")
        item = Reservation(session, blocks, start_ms, deadline_ms + HANDOFF_GUARD_MS)
        self.active.append(item)
        assert self.reserved_blocks <= self.budget_blocks
        return item


class FixedCount2Ledger(BlockReservationLedger):
    """Count-based baseline; block demand is recorded but not used for admission."""

    def can_admit(self, required: int) -> bool:
        return (
            required > 0
            and len(self.active) < 2
            and self.reserved_blocks + required <= self.budget_blocks
        )


def fixed_count_decision(active_count: int, reserved_blocks: int, required: int, budget: int = LEASE_BUDGET_BLOCKS) -> bool:
    """The fixed-count baseline deliberately ignores whether free block budget remains."""
    return active_count < 2 and reserved_blocks + required <= budget


def heterogeneous_source_tokens() -> int:
    return sum(HETEROGENEOUS_TOKENS.values())


def minimum_h2d_bytes(token_sizes: dict[str, int]) -> int:
    return int(sum(token_sizes.values()) * KV_BYTES_PER_TOKEN)


def h2d_accounting(native_transfer_bytes: int, external_tokens: int, token_sizes: dict[str, int]) -> dict[str, float]:
    native = native_transfer_bytes / (1024.0**2)
    external = external_tokens * KV_BYTES_PER_TOKEN / (1024.0**2)
    minimum = minimum_h2d_bytes(token_sizes) / (1024.0**2)
    total = native + external
    return {
        "native_promotion_mib": native,
        "resume_external_mib": external,
        "total_KV_H2D_mib": total,
        "minimum_required_H2D_mib": minimum,
        "duplicate_KV_H2D_mib": max(0.0, total - minimum),
    }


def reservation_policy_decision(policy: str, ledger: BlockReservationLedger, blocks: int) -> tuple[str, str]:
    if policy == "OnDemand-K2":
        return "ONDEMAND", "ONDEMAND"
    if policy == "NaiveAsync40-K2":
        return "PROMOTE", "NAIVE_PROMOTE"
    if policy == "FixedCount2-Async40-K2":
        if len(ledger.active) >= 2:
            return "RESERVATION_SKIP", "FIXED_COUNT_FULL"
        if not ledger.can_admit(blocks):
            return "RESERVATION_SKIP", "BLOCK_BUDGET_FULL"
        return "RESERVATION_PROMOTE", "FIXED_COUNT_AVAILABLE"
    if policy == "DynamicBlock512-Async40-K2":
        if ledger.can_admit(blocks):
            return "RESERVATION_PROMOTE", "BLOCK_BUDGET_AVAILABLE"
        return "RESERVATION_SKIP", "BLOCK_BUDGET_FULL"
    raise ValueError(f"unknown policy: {policy}")


def expected_promotions(workload: str, policy: str) -> tuple[int, int]:
    sizes = UNIFORM_TOKENS if workload == "uniform" else HETEROGENEOUS_TOKENS
    if policy == "OnDemand-K2":
        return (0, 12)
    if policy == "NaiveAsync40-K2":
        return (12, 0)
    if workload == "uniform":
        return (6, 6)
    if policy == "FixedCount2-Async40-K2":
        return (6, 6)
    if policy == "DynamicBlock512-Async40-K2":
        return (12, 0)
    raise ValueError((workload, policy, sizes))
