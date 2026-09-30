from __future__ import annotations

import errno
import inspect
from pathlib import Path

import pytest

from experiments.phase4g_c_fix import absolute_sleep, metrics


def test_timespec_ns_conversion():
    value = absolute_sleep.timespec_from_ns(1_234_567_890)
    assert value.tv_sec == 1
    assert value.tv_nsec == 234_567_890


def test_past_absolute_deadline_returns_without_syscall():
    calls = []
    result = absolute_sleep.absolute_sleep_until_ns(
        100, now_ns=lambda: 101, syscall=lambda *args: calls.append(args) or 0,
    )
    assert (result.return_code, result.eintr_retry_count) == (0, 0)
    assert calls == []


def test_eintr_retries_same_absolute_target():
    calls = []

    def fake_sleep(clock_id, flags, request):
        calls.append((clock_id, flags, request.tv_sec, request.tv_nsec))
        return [errno.EINTR, errno.EINTR, 0][len(calls) - 1]

    result = absolute_sleep.absolute_sleep_until_ns(
        2_345_678_901, now_ns=lambda: 1, syscall=fake_sleep,
    )
    assert result.eintr_retry_count == 2
    assert calls == [
        (absolute_sleep.CLOCK_MONOTONIC, absolute_sleep.TIMER_ABSTIME, 2, 345_678_901),
    ] * 3


def test_fatal_native_sleep_error_raises():
    with pytest.raises(absolute_sleep.TimingBackendError) as error:
        absolute_sleep.absolute_sleep_until_ns(
            200, now_ns=lambda: 100, syscall=lambda *_: errno.EINVAL,
        )
    assert error.value.return_code == errno.EINVAL


def test_pause_waiter_has_no_relative_sleep_or_busy_spin():
    source = inspect.getsource(absolute_sleep.phase4g_c_fix_wait_until)
    assert "absolute_sleep_until_ns" in source
    assert "time.sleep" not in source
    assert "while " not in source
    assert "monotonic() <" not in source


def test_backend_and_switch_interval_are_frozen():
    assert absolute_sleep.BACKEND == "CLOCK_NANOSLEEP_ABS"
    assert metrics.SWITCH_INTERVAL_S == 0.0005
    assert metrics.PAUSE_P95_GATE_MS == 10
    assert metrics.PAUSE_MAX_GATE_MS == 30


def test_stage_t_order_is_frozen_and_balanced():
    assert metrics.STAGE_T_ORDER == [
        (1, "C0"), (2, "C1"), (3, "C1"),
        (4, "C0"), (5, "C0"), (6, "C1"),
    ]
    assert sum(condition == "C0" for _, condition in metrics.STAGE_T_ORDER) == 3
    assert sum(condition == "C1" for _, condition in metrics.STAGE_T_ORDER) == 3


def test_stage_t_requires_six_runs_and_192_boundaries():
    runs = [{"repetition": r, "condition": c, "complete": True}
            for r, c in metrics.STAGE_T_ORDER]
    rows = [{"pause_error_ms": 0.0} for _ in range(192)]
    assert metrics.validate_stage_t(runs, rows)
    assert not metrics.validate_stage_t(runs, rows[:-1])
    assert not metrics.validate_stage_t(runs[:-1], rows)


def test_calibration_is_not_a_measured_result_set():
    calibration = [{"stage": "timing_calibration", "valid": True} for _ in range(6)]
    assert not metrics.validate_measured_matrix(calibration, "/new/phase4g_c_fix")


def test_measured_matrix_requires_fresh_order_root_and_full_counts():
    rows = []
    for repetition, condition in metrics.MEASURED_ORDER:
        rows.append({
            "repetition": repetition, "condition": condition, "valid": True,
            "artifact_root": "/new/phase4g_c_fix",
            "turns_complete": 40, "boundaries": 32,
        })
    assert metrics.validate_measured_matrix(rows, "/new/phase4g_c_fix")
    assert not metrics.validate_measured_matrix(rows[:-1], "/new/phase4g_c_fix")
    mixed = [dict(row) for row in rows]
    mixed[0]["artifact_root"] = "/old/phase4g_c/runs"
    assert not metrics.validate_measured_matrix(mixed, "/new/phase4g_c_fix")


def test_pause_gate_uses_absolute_error_and_frozen_limits():
    rows = [{"pause_error_ms": -9.0}, {"pause_error_ms": 10.0}]
    assert metrics.timing_gate(rows) == {
        "p95_abs_ms": 10.0, "max_abs_ms": 10.0, "pass": True,
    }
    assert not metrics.timing_gate([{"pause_error_ms": 10.01}])["pass"]
    assert not metrics.timing_gate([{"pause_error_ms": -30.01}])["pass"]


def test_timing_decomposition_quantiles_keep_pause_error_absolute_gate():
    rows = [
        {"sleep_overshoot_ms": 1.0, "dispatch_delay_ms": 2.0, "pause_error_ms": 3.0},
        {"sleep_overshoot_ms": 2.0, "dispatch_delay_ms": 3.0, "pause_error_ms": 5.0},
    ]
    summary = metrics.summarize_timing(rows)
    assert summary["sleep_overshoot_P95_ms"] == 2.0
    assert summary["dispatch_delay_P95_ms"] == 3.0
    assert summary["pause_error_abs_P95_ms"] == 5.0


def test_timing_root_cause_classification():
    host = [{"sleep_overshoot_ms": 12.0, "dispatch_delay_ms": 1.0}] * 20
    client = [{"sleep_overshoot_ms": 1.0, "dispatch_delay_ms": 12.0}] * 20
    assert metrics.timing_root_cause(host, False) == "HOST_WAKEUP_JITTER"
    assert metrics.timing_root_cause(client, False) == "CLIENT_DISPATCH_JITTER"
    assert metrics.timing_root_cause(host, True) == "RESOLVED"


def test_frozen_source_fingerprints():
    assert metrics.START_COMMIT == "c10a6c7ac69e41b444d00497c5d258da6bfc71b0"
    assert metrics.WORKLOAD_SHA256 == (
        "9b381336a70e73b0efbddbb79d2c88608c19663247e819c1e7a364efd2971105"
    )
    assert metrics.VLLM_DIFF_SHA256 == (
        "bfd789e6849715196435f18a57ed9bbefae7b4b9d3a78d69115614f27d1542e8"
    )
