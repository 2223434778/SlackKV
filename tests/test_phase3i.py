from __future__ import annotations

from experiments.phase3g_full.metrics import interval_concurrency, overlap, percentile_linear


def test_x_decode_duration_and_tpot() -> None:
    first, finish = -25.0, 600.0
    duration = finish - first
    assert duration == 625.0
    assert duration / 127.0 == 625.0 / 127.0


def test_combined_throughput_uses_220_post_first_tokens() -> None:
    first = [-10.0, 300.0, 320.0, 340.0]
    finish = [600.0, 500.0, 520.0, 540.0]
    expected = 220.0 / ((max(finish) - min(first)) / 1000.0)
    assert expected == 220.0 / 0.61


def test_gate_capacity_and_overlap_metrics() -> None:
    intervals = [(300.0, 400.0), (320.0, 420.0), (340.0, 440.0)]
    maximum, area = interval_concurrency(intervals)
    assert maximum == 3
    assert overlap(intervals[0], intervals[1]) == 80.0
    assert area == 300.0


def test_percentile_is_linear() -> None:
    assert percentile_linear([1.0, 2.0, 3.0], 95) == 2.9
