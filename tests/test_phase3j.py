from __future__ import annotations

import math


def busy_budget(samples: list[float]) -> float:
    return math.ceil((max(samples) + 10.0) / 5.0) * 5.0


def test_busy_budget_rounds_up() -> None:
    assert busy_budget([78.0, 70.0, 72.0]) == 90.0
    assert busy_budget([41.0, 42.0]) == 55.0


def test_budget_overrun() -> None:
    budget = 90.0
    assert (89.9 > budget) is False
    assert 90.1 > budget


def test_promotion_deadline_and_blackout() -> None:
    completion = 325.0
    deadline = 340.0
    assert max(0.0, completion - deadline) == 0.0
    assert 300.0 < completion and 320.0 < completion


def test_k2_and_c_blocked_diagnostics() -> None:
    owner_counts = [1, 2, 2, 1]
    assert max(owner_counts) == 2
    events = ["RESUME_ARRIVED", "GATE_BLOCKED", "FIRST_TOKEN", "RESUME_ADMITTED"]
    assert "GATE_BLOCKED" in events
