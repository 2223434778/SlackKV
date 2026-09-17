import pytest

from experiments.phase2f.analysis import (
    crossover_orders,
    group_statistics,
    identity_differences_ms,
    position_labels,
    position_ratio,
    transfer_counter_delta,
    valid_cache_evidence,
)


def test_crossover_schedule_is_frozen_ab_ba():
    assert crossover_orders() == ["AB", "BA", "AB", "BA", "AB", "BA"]


def test_position_labels_follow_order():
    assert position_labels("AB") == [
        {"session": "A", "position": 1}, {"session": "B", "position": 2}
    ]
    assert position_labels("BA") == [
        {"session": "B", "position": 1}, {"session": "A", "position": 2}
    ]


def test_group_medians_and_range():
    rows = [
        {"run_validity": "PASS", "session": s, "position": p, "promotion_request_ms": t}
        for s, p, t in (("A", 1, 110), ("B", 1, 100), ("A", 2, 40), ("B", 2, 50))
    ]
    groups = group_statistics(rows)
    assert groups["A_first"] == {"count": 1, "median_ms": 110, "min_ms": 110, "max_ms": 110}
    assert groups["all_second"] == {"count": 2, "median_ms": 45, "min_ms": 40, "max_ms": 50}


def test_position_ratio_and_session_identity_differences():
    rows = [
        {"run_validity": "PASS", "session": s, "position": p, "promotion_request_ms": t}
        for s, p, t in (("A", 1, 120), ("B", 1, 100), ("A", 2, 50), ("B", 2, 40))
    ]
    groups = group_statistics(rows)
    assert position_ratio(groups) == pytest.approx(110 / 45)
    assert identity_differences_ms(groups) == pytest.approx((20, 10))


def test_invalid_cache_evidence_is_detected_and_not_aggregated():
    assert valid_cache_evidence(8192, 0)
    assert not valid_cache_evidence(8191, 0)
    assert not valid_cache_evidence(8192, 1)
    with pytest.raises(ValueError, match="cache-valid"):
        group_statistics([{
            "run_validity": "INVALID", "session": "A", "position": 1,
            "promotion_request_ms": 100,
        }])


def test_transfer_counter_ignores_prometheus_created_series():
    labels = (("transfer_type", "CPU_to_GPU"),)
    before = {
        ("vllm:kv_offload_total_time_total", labels): 1.0,
        ("vllm:kv_offload_total_time_created", labels): 100.0,
        ("vllm:kv_offload_total_bytes_total", labels): 10.0,
        ("vllm:kv_offload_total_bytes_created", labels): 100.0,
    }
    after = {
        ("vllm:kv_offload_total_time_total", labels): 1.006,
        ("vllm:kv_offload_total_time_created", labels): 200.0,
        ("vllm:kv_offload_total_bytes_total", labels): 110.0,
        ("vllm:kv_offload_total_bytes_created", labels): 200.0,
    }
    timing = transfer_counter_delta(before, after, time_counter=True)
    byte_count = transfer_counter_delta(before, after, time_counter=False)
    assert timing is not None
    assert timing["delta_ms"] == pytest.approx(6.0)
    assert byte_count is not None
    assert byte_count["delta"] == pytest.approx(100.0)
