import pytest

from experiments.phase2h.scheduler import (
    DEADLINE_MS,
    KV_MIB,
    P_EST_MS,
    SESSIONS,
    effective_resume_latency_ms,
    memory_time_mib_s,
    naive_target_start_ms,
    percentile_linear,
    queue_safe_idle_s,
)


def test_six_session_deadline_configuration_is_frozen():
    assert SESSIONS == ("A", "B", "C", "D", "E", "F")
    assert DEADLINE_MS == {
        "A": 300.0, "B": 320.0, "C": 340.0,
        "D": 650.0, "E": 900.0, "F": 1300.0,
    }


def test_naive_targets_use_42ms_estimate():
    assert P_EST_MS == 42.0
    assert naive_target_start_ms(DEADLINE_MS["A"]) == pytest.approx(258.0)
    assert naive_target_start_ms(DEADLINE_MS["F"]) == pytest.approx(1258.0)


def test_queuejit_initial_safe_idle_uses_all_prefixes():
    deadlines_s = {session: deadline / 1000.0 for session, deadline in DEADLINE_MS.items()}
    assert queue_safe_idle_s(0.0, deadlines_s) == pytest.approx(0.214)


def test_effective_latency_includes_dispatch_wait_and_ttft():
    assert effective_resume_latency_ms(10.350, 45.0, 10.0, 300.0) == pytest.approx(95.0)
    assert effective_resume_latency_ms(10.300, 25.0, 10.0, 300.0) == pytest.approx(25.0)


def test_memory_time_and_six_sample_percentile():
    assert KV_MIB == 112.0
    assert memory_time_mib_s([300, 320, 340, 650, 900, 1300]) == pytest.approx(426.72)
    assert percentile_linear([10, 20, 30, 40, 50, 60], 50) == pytest.approx(35.0)
    assert percentile_linear([10, 20, 30, 40, 50, 60], 95) == pytest.approx(57.5)
