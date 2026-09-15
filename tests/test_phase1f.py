import pytest

from experiments.phase1f.run_phase1f import (
    actual_stall_s,
    adaptive_margin_s,
    strategy_margin_s,
)


def test_zero_k_has_zero_adaptive_margin():
    assert adaptive_margin_s(0.0, 0.5, 4.0) == 0.0


def test_higher_sigma_increases_adaptive_margin():
    assert adaptive_margin_s(1.0, 0.5, 4.0) > adaptive_margin_s(1.0, 0.1, 4.0)


def test_higher_k_increases_adaptive_margin():
    assert adaptive_margin_s(1.0, 0.25, 4.0) > adaptive_margin_s(0.5, 0.25, 4.0)


def test_actual_stall_uses_actual_deadline():
    assert actual_stall_s(2.0, 2.5) == 0.0
    assert actual_stall_s(2.6, 2.5) == pytest.approx(0.1)
