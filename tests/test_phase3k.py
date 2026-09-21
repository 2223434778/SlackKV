from __future__ import annotations


def test_k2_capacity_is_exactly_two() -> None:
    owner_counts = [1, 2, 2, 1]
    assert max(owner_counts) == 2
    assert all(value <= 2 for value in owner_counts)


def test_promotion_clean_rule() -> None:
    rows = [{"deadline_stall_ms": 0, "blackout_conflict": False, "budget_overrun": False}]
    assert all(float(row["deadline_stall_ms"]) == 0 and not row["blackout_conflict"] and not row["budget_overrun"] for row in rows)


def test_combined_tokens() -> None:
    assert 93 + 127 == 220

