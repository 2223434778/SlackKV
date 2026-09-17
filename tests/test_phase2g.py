import pytest

from experiments.phase2g.scheduler import (
    DEADLINE_MS,
    P_EST_MS,
    completion_timing_ms,
    naive_target_start_ms,
    queue_safe_idle_s,
    warm_state_valid,
)


def test_frozen_naive_targets_use_42ms_service():
    assert P_EST_MS == 42.0
    assert naive_target_start_ms(DEADLINE_MS["A"]) == pytest.approx(308.0)
    assert naive_target_start_ms(DEADLINE_MS["B"]) == pytest.approx(328.0)


def test_queuejit_uses_cumulative_edf_service():
    deadlines = {"A": 0.350, "B": 0.370}
    assert queue_safe_idle_s(0.0, deadlines) == pytest.approx(0.286)
    assert queue_safe_idle_s(0.326, {"B": 0.370}) == pytest.approx(0.002)


def test_deadline_stall_and_early_residency():
    assert completion_timing_ms(0.390, 0.370) == pytest.approx((20.0, 0.0))
    assert completion_timing_ms(0.330, 0.350) == pytest.approx((0.0, 20.0))


def test_warm_state_threshold_is_median_based():
    assert warm_state_valid([38.5, 39.5, 110.0])
    assert not warm_state_valid([38.5, 61.0, 110.0])
    assert not warm_state_valid([])
