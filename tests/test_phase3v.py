from experiments.phase3v.ledger import (
    BlockReservationLedger,
    DEADLINES_MS,
    FixedCount2Ledger,
    HETEROGENEOUS_TOKENS,
    LEASE_BUDGET_BLOCKS,
    POLICIES,
    UNIFORM_TOKENS,
    expected_promotions,
    h2d_accounting,
    heterogeneous_source_tokens,
    minimum_h2d_bytes,
    required_blocks as block_demand,
    reservation_policy_decision,
)


def test_frozen_workload():
    assert tuple(DEADLINES_MS) == tuple("ABCDEFGHIJKL")
    assert len(POLICIES) == 4
    assert len(UNIFORM_TOKENS) == len(HETEROGENEOUS_TOKENS) == 12
    assert set(DEADLINES_MS.values()) == {300, 320, 340, 360, 900, 920, 940, 960, 1500, 1520, 1540, 1560}


def test_required_blocks():
    assert [block_demand(n) for n in (1024, 2048, 4096)] == [64, 128, 256]


def test_dynamic_block_budget_admission_and_expiry():
    ledger = BlockReservationLedger()
    assert ledger.can_admit(64)
    for session, blocks in (("A", 64), ("B", 64), ("C", 64), ("D", 256)):
        ledger.grant(session, blocks, 100, 300)
        assert ledger.reserved_blocks <= LEASE_BUDGET_BLOCKS
    assert ledger.reserved_blocks == 448
    assert ledger.can_admit(64)  # 448 + 64 exactly fills the stated <=512 budget.
    ledger.grant("E", 64, 100, 300)
    assert ledger.reserved_blocks == 512
    assert not ledger.can_admit(1)
    expired = ledger.expire(400)
    assert len(expired) == 5
    assert ledger.reserved_blocks == 0


def test_fixed_count_rule_remains_count_based():
    ledger = FixedCount2Ledger()
    assert reservation_policy_decision("FixedCount2-Async40-K2", ledger, 64)[0] == "RESERVATION_PROMOTE"
    ledger.grant("A", 64, 100, 300)
    assert reservation_policy_decision("FixedCount2-Async40-K2", ledger, 64)[0] == "RESERVATION_PROMOTE"
    ledger.grant("B", 64, 110, 320)
    assert ledger.reserved_blocks == 128
    assert reservation_policy_decision("FixedCount2-Async40-K2", ledger, 64) == ("RESERVATION_SKIP", "FIXED_COUNT_FULL")


def test_workload_expected_promotion_counts():
    assert expected_promotions("uniform", "FixedCount2-Async40-K2") == (6, 6)
    assert expected_promotions("uniform", "DynamicBlock512-Async40-K2") == (6, 6)
    assert expected_promotions("heterogeneous", "FixedCount2-Async40-K2") == (6, 6)
    assert expected_promotions("heterogeneous", "DynamicBlock512-Async40-K2") == (12, 0)


def test_heterogeneous_source_and_h2d_minimum():
    assert heterogeneous_source_tokens() == 21_504
    assert minimum_h2d_bytes(HETEROGENEOUS_TOKENS) / (1024**2) == 588
    fixed = h2d_accounting(168 * 1024**2, 15_360, HETEROGENEOUS_TOKENS)
    dynamic = h2d_accounting(588 * 1024**2, 0, HETEROGENEOUS_TOKENS)
    assert fixed["total_KV_H2D_mib"] == 588
    assert dynamic["total_KV_H2D_mib"] == 588
    assert fixed["duplicate_KV_H2D_mib"] == dynamic["duplicate_KV_H2D_mib"] == 0
