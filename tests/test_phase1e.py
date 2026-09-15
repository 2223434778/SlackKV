import pytest

from experiments.phase1e.run_phase1e import actual_stall_s, safe_predicted_deadline


def test_zero_margin_reproduces_phase1d_predicted_deadline_when_feasible():
    assert safe_predicted_deadline(10.0, 2.0, 0.2, 0.3, 0.0) == pytest.approx(12.0)


def test_increasing_margin_never_exceeds_predicted_deadline():
    predicted_deadline = 12.0
    assert safe_predicted_deadline(10.0, 2.0, 0.2, 0.3, 0.25) <= predicted_deadline


def test_larger_margin_cannot_delay_isolated_h2d_start():
    earlier = safe_predicted_deadline(0.0, 2.0, 0.2, 0.5, 0.1) - 0.5
    later = safe_predicted_deadline(0.0, 2.0, 0.2, 0.5, 0.5) - 0.5
    assert later <= earlier


def test_actual_stall_uses_actual_not_safe_deadline():
    assert actual_stall_s(2.0, 2.5) == 0.0
    assert actual_stall_s(2.6, 2.5) == pytest.approx(0.1)
