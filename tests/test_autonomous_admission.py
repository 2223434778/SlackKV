from experiments.autonomous_admission.analyze import analyze, nearest_rank
from experiments.autonomous_admission.backfill import BoundedBackfill


def test_capacity_head_stops_after_two_successful_bypasses():
    policy = BoundedBackfill(2)
    for candidate in ("small1", "small2"):
        policy.begin_step({"head", "small1", "small2"})
        assert policy.can_skip("head")
        policy.blocked("head")
        assert policy.admitted(candidate) == ["head"]
        assert policy.stop_this_step
    policy.begin_step({"head"})
    assert not policy.can_skip("head")
    assert policy.admitted("head") == []
    assert "head" not in policy.counts


def test_failed_scan_does_not_use_allowance_and_cancellation_cleans_up():
    policy = BoundedBackfill(2)
    policy.begin_step({"head"})
    policy.blocked("head")
    policy.begin_step({"head"})
    assert policy.counts["head"] == 0
    policy.begin_step(set())
    assert not policy.counts


def test_default_policy_preserves_strict_capacity_head():
    assert not BoundedBackfill().can_skip("head")


def test_tail_rank_is_not_maximum():
    assert nearest_rank(list(range(1, 33)), .95) == 31


def test_failures_after_admission_and_priming_are_not_attributed():
    result = {
        "valid": True,
        "BG_total_wall_ms": 30, "BG_weighted_TPOT_ms": 2,
        "BG_tail_request_finish_ms": 10,
        "BG_total_generated_tokens": 8192, "X_finish_ns": 200, "X_first_ns": 0,
        "source_trace": [{"external_cached_tokens": 16}],
        "turn_rows": [{"episode_id": "E0", "turn_index": i,
                       "issue_ns": i * 100, "first_ns": i * 100 + 40,
                       "output_tokens": 1,
                       "finish_ns": i * 100 + 60} for i in range(2)],
        "resume_trace": [
            {"session": "E0:turn:1", "event": "RESUME_ARRIVED", "timestamp_monotonic": 100 / 1e9},
            {"session": "E0:turn:1", "event": "RESUME_ADMITTED", "timestamp_monotonic": 130 / 1e9}],
    }
    samples = [{"session": "E0:turn:1", "timestamp_monotonic_ns": t,
                "allocator_fit": False, "gate": "STEP",
                "allocator_required_blocks": 10, "allocator_available_blocks": 3,
                "inflight_reserved_blocks": 2} for t in (90, 110, 135)]
    report = analyze(result, samples)
    assert report["resumes_with_observed_capacity_failure"] == 1
    assert report["requests"][0]["observation_status"] == "UNKNOWN"
    assert report["requests"][1]["pre_admission_failed_samples"] == 1
    assert report["requests"][1]["max_observed_deficit_blocks"] == 7
