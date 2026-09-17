from experiments.phase3a.helpers import percentile_linear, token_matches
from experiments.phase3a.native_client import parse_native_prefetch_response
from experiments.phase3a.scheduler import (
    effective_resume_latency_ms,
    freeze_p_native_ms,
    memory_time_mib_s,
    queue_safe_idle_s,
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
    }
    assert parse_native_prefetch_response(payload) == payload


def test_freeze_p_native_rounds_max_up():
    assert freeze_p_native_ms([7.2, 7.6, 8.1]) == 9


def test_queuejit_uses_frozen_native_service():
    deadlines = {"A": 0.300, "B": 0.320, "C": 0.340}
    assert abs(queue_safe_idle_s(0.0, deadlines, 9) - 0.291) < 1e-9


def test_effective_latency_and_memory_time():
    assert effective_resume_latency_ms(1.025, 1.0) == pytest.approx(25.0)
    assert memory_time_mib_s([100.0, 200.0]) == pytest.approx(33.6)


def test_percentile_and_token_correctness():
    assert percentile_linear([10.0, 20.0, 30.0], 50) == 20.0
    assert token_matches(1084, 1084)
    assert not token_matches(None, 1084)


import pytest
