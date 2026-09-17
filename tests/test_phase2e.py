import pytest

from experiments.phase2e.helpers import request_payload, resume_token_matches
from experiments.phase2e.scheduler import (
    aggregate_run_metrics,
    completion_timing_ms,
    edf_order,
    naive_target_start_s,
    queue_safe_idle_s,
)


def test_naive_jit_target_start():
    assert naive_target_start_s(1.5, 0.12) == pytest.approx(1.38)


def test_queuejit_safe_idle_uses_cumulative_edf_demand():
    deadlines = {"A": 1.5, "B": 1.56}
    assert queue_safe_idle_s(0.0, deadlines, 0.12) == pytest.approx(1.32)
    assert queue_safe_idle_s(1.434, {"B": 1.56}, 0.12) == pytest.approx(0.006)


def test_two_job_edf_ordering():
    assert edf_order({"B": 1.56, "A": 1.5}) == ["A", "B"]


def test_deadline_stall_and_early_residency():
    assert completion_timing_ms(1.61, 1.56) == pytest.approx((50.0, 0.0))
    assert completion_timing_ms(1.43, 1.5) == pytest.approx((0.0, 70.0))


def test_total_early_residency_aggregation():
    aggregate = aggregate_run_metrics([
        {"deadline_stall_ms": 0, "early_residency_ms": 60},
        {"deadline_stall_ms": 5, "early_residency_ms": 12},
    ])
    assert aggregate == {
        "deadline_miss_count": 1,
        "max_deadline_stall_ms": 5.0,
        "total_early_residency_ms": 72.0,
    }


def test_token_id_correctness_and_promotion_request_control():
    assert resume_token_matches(1084, 1084)
    assert not resume_token_matches(1084, 1085)
    assert not resume_token_matches(None, 1084)
    payload = request_payload("model", [1, 2, 3], stream=False, keep_local=True)
    assert payload["kv_transfer_params"] == {"max_offload_tokens": 0}
