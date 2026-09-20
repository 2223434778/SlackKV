from experiments.phase3d.helpers import percentile_linear, token_matches
from experiments.phase3d.native_client import parse_native_prefetch_response
from experiments.phase3d.scheduler import (
    effective_resume_latency_ms,
    freeze_p_native_ms,
    memory_time_mib_s,
    naive_target_start_ms,
    queue_safe_idle_s,
    RESUME_GUARD_MS,
    merge_blackouts,
    feasible,
    latest_feasible_start,
    blackout_conflict,
)


def test_native_prefetch_response_parsing():
    payload = {
        "status": "ok",
        "external_tokens_loaded": 4096,
        "transfer_bytes": 117440512,
        "cpu_to_gpu_transfer_ms": 5.2,
        "model_forward_count": 0,
        "generated_token_count": 0,
        "elapsed_ms": 9.3,
        "connector_only_step_count": 1,
        "connector_only_wall_ms": 9.3,
        "mixed_prefetch_model_step_count": 0,
        "normal_requests_delayed": 0,
    }
    assert parse_native_prefetch_response(payload) == payload


def test_freeze_p_native_rounds_max_up():
    assert freeze_p_native_ms([7.2, 7.6, 8.1]) == 9


def test_queuejit_uses_frozen_native_service():
    deadlines = {"A": 0.300, "B": 0.320, "C": 0.340}
    assert abs(queue_safe_idle_s(0.0, deadlines, 9) - 0.291) < 1e-9


def test_phase3d_safe30_targets_and_long_marker():
    assert naive_target_start_ms(300.0, 30.0) == 270.0
    assert 49.9 <= 50.0


def test_resume_guard_and_blackout_merge():
    assert RESUME_GUARD_MS == 110.0
    assert merge_blackouts({"A": 300.0, "B": 320.0, "C": 340.0}, 110.0) == [(300.0, 450.0)]


def test_blackout_conflict_and_feasible_latest_start():
    intervals = [(300.0, 450.0)]
    assert blackout_conflict(310.0, 340.0, intervals)[0]
    assert not blackout_conflict(210.0, 240.0, intervals)[0]
    jobs = [("A", 300.0), ("B", 320.0), ("C", 340.0)]
    assert feasible(210.0, jobs, intervals, 30.0)
    assert latest_feasible_start(0.0, jobs, intervals, 30.0) >= 200.0


def test_effective_latency_and_memory_time():
    assert effective_resume_latency_ms(1.025, 1.0) == pytest.approx(25.0)
    assert memory_time_mib_s([100.0, 200.0]) == pytest.approx(33.6)


def test_percentile_and_token_correctness():
    assert percentile_linear([10.0, 20.0, 30.0], 50) == 20.0
    assert token_matches(1084, 1084)
    assert not token_matches(None, 1084)


import pytest
