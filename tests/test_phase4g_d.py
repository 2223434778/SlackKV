from __future__ import annotations

import csv
import threading
import time
from pathlib import Path

import pytest

from experiments.phase4g_d import analysis
from experiments.phase4g_d import run_phase4g_d as runner

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_start_commit():
    assert analysis.START_COMMIT == "7bada3599076fa4fbf8e0d73120c37dbe6285ca0"


def test_frozen_workload_sha_and_file():
    import hashlib
    path = ROOT / "results/phase4c/hybrid_workload.csv"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == analysis.WORKLOAD_SHA256


def test_frozen_runtime_constants():
    assert (analysis.GPU_BLOCKS, analysis.P_SAFE_MS, analysis.RESUME_GUARD_MS,
            analysis.K2, analysis.RESERVATION_BUDGET_BLOCKS) == (4681, 102, 110, 2, 3510)


def test_recheck_interval_is_unchanged():
    assert analysis.RECHECK_INTERVAL_MS == 20


def test_allocator_missing_required_is_unknown():
    result = analysis.classify_attempt_nullable("REJECT", None, 10)
    assert set(result.values()) == {"UNKNOWN"}


def test_allocator_missing_available_is_unknown():
    result = analysis.classify_attempt_nullable("ADMIT", 10, None)
    assert set(result.values()) == {"UNKNOWN"}


def test_allocator_empty_csv_value_is_none():
    assert analysis.optional_int("") is None


def test_allocator_unknown_never_becomes_zero():
    assert analysis.classify_attempt_nullable("REJECT", "", "") == {
        "allocator_fit": "UNKNOWN", "agreement": "UNKNOWN",
        "false_admit": "UNKNOWN", "false_reject": "UNKNOWN",
    }


def test_allocator_known_fit_and_false_reject():
    result = analysis.classify_attempt_nullable("REJECT", 8, 9)
    assert result == {"allocator_fit": True, "agreement": False,
                      "false_admit": False, "false_reject": True}


def test_allocator_known_reject_and_false_admit():
    result = analysis.classify_attempt_nullable("ADMIT", 10, 9)
    assert result == {"allocator_fit": False, "agreement": False,
                      "false_admit": True, "false_reject": False}


def test_allocator_known_agreement_fit():
    assert analysis.classify_attempt_nullable("ADMIT", 9, 10)["agreement"] is True


def test_allocator_known_agreement_reject():
    assert analysis.classify_attempt_nullable("REJECT", 10, 9)["agreement"] is True


def test_previous_phase_reanalysis_counts_are_frozen():
    report = analysis.reanalyze_allocator_csv(
        ROOT / "results/phase4g_c_fix/allocator_guard_comparison.csv"
    )
    assert (report["total_guard_observations"], report["exact_known"],
            report["exact_missing"], report["known_agreement"],
            report["known_mismatch"], report["known_false_admit"],
            report["known_false_reject"]) == (123, 54, 69, 54, 0, 0, 0)


def test_previous_result_is_explicitly_withdrawn():
    report = analysis.reanalyze_allocator_csv(
        ROOT / "results/phase4g_c_fix/allocator_guard_comparison.csv"
    )
    assert "not supported" in report["previous_result_reassessment"]


def test_terminal_capacity_reject():
    assert analysis.is_nonresident_terminal("CAPACITY_REJECTED")


def test_terminal_progress_reject():
    assert analysis.is_nonresident_terminal("PROGRESS_REJECTED")


def test_terminal_reservation_skip():
    assert analysis.is_nonresident_terminal("RESERVATION_SKIP")


def test_terminal_stale_cancel():
    assert analysis.is_nonresident_terminal("STALE_PROMOTION_CANCELLED")


def test_promotion_ok_is_not_poison_terminal():
    assert not analysis.is_nonresident_terminal("PROMOTION_OK")
    assert not analysis.is_nonresident_terminal("OK")


def test_terminal_protected_head_is_poison_until_first_token_even_after_resume_starts():
    assert analysis.is_protected_poison_episode(
        protected_id="E04:b0", boundary_id="E04:b0", first_token_ns=None,
        decision="PROGRESS_REJECTED",
    )


def test_poison_episode_ends_at_first_token():
    assert not analysis.is_protected_poison_episode(
        protected_id="E04:b0", boundary_id="E04:b0", first_token_ns=123,
        decision="PROGRESS_REJECTED",
    )


def test_nonfrontier_terminal_does_not_define_poison_episode():
    assert not analysis.is_protected_poison_episode(
        protected_id="E04:b0", boundary_id="E00:b0", first_token_ns=None,
        decision="CAPACITY_REJECTED",
    )


def test_finished_poison_episode_persists_exact_first_token_timestamp():
    result = analysis.finish_poison_episode(
        {"terminal_timestamp_ns": 100, "protected_id": "E04:b0"},
        end_timestamp_ns=2_000_100, end_reason="FIRST_TOKEN", first_token_ns=2_000_100,
    )
    assert result["first_token_ns"] == 2_000_100
    assert result["poison_duration_ms"] == 2.0


def test_censored_poison_episode_keeps_first_token_unknown():
    result = analysis.finish_poison_episode(
        {"terminal_timestamp_ns": 100}, end_timestamp_ns=200,
        end_reason="RUN_END", first_token_ns=None,
    )
    assert result["first_token_ns"] is None


def test_empty_retry_timeline_retains_machine_readable_schema(tmp_path):
    path = tmp_path / "protected_retry_timeline.csv"
    runner.write_csv(path, [], runner.RETRY_TIMELINE_FIELDS)
    with path.open(newline="", encoding="utf-8") as stream:
        header = next(csv.reader(stream))
    assert {"timestamp_ns", "protected_id", "resume_started", "allocator_fit",
            "retry_actionable"}.issubset(header)


def test_parent_promotion_event_normalization():
    assert analysis.normalize_decision("PROMOTION_PROGRESS_REJECTED") == "PROGRESS_REJECTED"


def test_psa1_self_formula():
    assert analysis.psa1_need(1300, 10, True, 1270) == 30


def test_psa1_nonmatching_candidate_formula():
    assert analysis.psa1_need(1300, 10, False, 1270) == 1290


def test_psa_fit_uses_candidate_footprint():
    assert analysis.psa_fit(1500, 1200, 200) is True
    assert analysis.psa_fit(1399, 1200, 200) is False


def test_psa_fit_unknown_propagates():
    assert analysis.psa_fit(None, 1200, 200) is None
    assert analysis.psa_fit(1500, 1200, None) is None


def test_f0_is_earliest_all_unfinished():
    states = {
        "late": {"deadline_ns": 20, "decision": "REGISTERED", "first_token_ns": None},
        "head": {"deadline_ns": 10, "decision": "PROGRESS_REJECTED", "first_token_ns": None},
    }
    assert analysis.choose_frontiers(states, {}, "late")["F0"] == "head"


def test_f1_excludes_terminal_head():
    states = {
        "terminal": {"deadline_ns": 1, "decision": "CAPACITY_REJECTED",
                     "first_token_ns": None, "resume_started_ns": None},
        "active": {"deadline_ns": 2, "decision": "REGISTERED",
                    "first_token_ns": None, "resume_started_ns": None},
    }
    assert analysis.choose_frontiers(states, {}, None)["F1"] == "active"


def test_f2_requires_exact_full_source_readiness():
    states = {"a": {"deadline_ns": 1, "decision": "REGISTERED",
                     "first_token_ns": None, "resume_started_ns": None}}
    assert analysis.choose_frontiers(states, {"a": False}, None)["F2"] is None
    assert analysis.choose_frontiers(states, {"a": True}, None)["F2"] == "a"


def test_f3_is_the_frozen_candidate():
    result = analysis.choose_frontiers({}, {}, "candidate")
    assert result["F3"] == "candidate"


def test_f4_is_earliest_arrived_unfinished():
    states = {
        "not_arrived": {"deadline_ns": 1, "decision": "REGISTERED",
                        "first_token_ns": None, "resume_started_ns": None},
        "arrived": {"deadline_ns": 2, "decision": "REGISTERED",
                    "first_token_ns": None, "resume_started_ns": 3},
    }
    assert analysis.choose_frontiers(states, {}, None)["F4"] == "arrived"


def test_f4_none_when_no_resume_arrived():
    assert analysis.choose_frontiers({}, {}, None)["F4"] is None


def test_retry_actionable_requires_every_exact_gate():
    sample = {"source_full_ready": True, "resume_started": False,
              "first_token_seen": False, "logical_reservation_fit": True,
              "timing_gate_open": True, "self_psa_fit": True, "allocator_fit": True}
    assert analysis.retry_actionable(sample)
    sample["allocator_fit"] = "UNKNOWN"
    assert not analysis.retry_actionable(sample)


def test_retry_actionable_rejects_started_resume():
    sample = {"source_full_ready": True, "resume_started": True,
              "first_token_seen": False, "logical_reservation_fit": True,
              "timing_gate_open": True, "self_psa_fit": True, "allocator_fit": True}
    assert not analysis.retry_actionable(sample)


def test_retry_window_requires_20ms_continuity():
    samples = [{"timestamp_ns": 0, "retry_actionable": True},
               {"timestamp_ns": 19_000_000, "retry_actionable": True}]
    assert analysis.actionable_windows(samples) == []


def test_retry_window_accepts_20ms_continuity():
    samples = [{"timestamp_ns": 0, "retry_actionable": True},
               {"timestamp_ns": 20_000_000, "retry_actionable": True}]
    assert len(analysis.actionable_windows(samples)) == 1


def test_retry_window_breaks_on_long_sample_gap():
    samples = [{"timestamp_ns": 0, "retry_actionable": True},
               {"timestamp_ns": 50_000_000, "retry_actionable": True}]
    assert analysis.actionable_windows(samples) == []


def test_recoverability_unknown_allocator_is_not_fit():
    assert not analysis.retry_actionable({"allocator_fit": "UNKNOWN"})


def test_poison_type_retryable_precedes_other_causes():
    assert analysis.classify_poison_type(retryable=True, frontier_recoverable=True,
        allocator_known_fit_fraction=0.0, f3_psa_fail_fraction=1.0) == "Retryable head"


def test_poison_type_frontier_only():
    assert analysis.classify_poison_type(retryable=False, frontier_recoverable=True,
        allocator_known_fit_fraction=0.0, f3_psa_fail_fraction=0.0) == "Frontier-only"


def test_poison_type_capacity_bound():
    assert analysis.classify_poison_type(retryable=False, frontier_recoverable=False,
        allocator_known_fit_fraction=0.2, f3_psa_fail_fraction=0.0) == "Capacity-bound"


def test_poison_type_psa_intrinsic():
    assert analysis.classify_poison_type(retryable=False, frontier_recoverable=False,
        allocator_known_fit_fraction=0.8, f3_psa_fail_fraction=0.75) == "PSA-intrinsic"


def test_poison_type_unknown_falls_back_to_mixed():
    assert analysis.classify_poison_type(retryable=False, frontier_recoverable=False,
        allocator_known_fit_fraction=None, f3_psa_fail_fraction=None) == "Mixed/unknown"


def test_eligible_allocator_coverage_excludes_unknown_from_fit_denominator():
    rows = [{"allocator_required_blocks": 3, "allocator_available_blocks": 4},
            {"allocator_required_blocks": "", "allocator_available_blocks": 0}]
    assert analysis.exact_truth_coverage(rows) == {
        "eligible_selections": 2, "exact_known": 1, "exact_unknown": 1,
        "coverage_fraction": .5,
    }


def test_protected_need_summary_keeps_missing_allocator_values_unknown():
    needs = [
        {"selection_id": "s1", "Need_PSA1": 100, "Need_allocator_now": "",
         "allocator_available_now": "", "local_cached_blocks": 10,
         "leased_blocks": 0, "owned_blocks": 10, "evictable_cached_blocks": 0},
        {"selection_id": "s2", "Need_PSA1": 20, "Need_allocator_now": 15,
         "allocator_available_now": 20, "local_cached_blocks": 0,
         "leased_blocks": 0, "owned_blocks": 0, "evictable_cached_blocks": 0},
        {"selection_id": "s3", "Need_PSA1": 10, "Need_allocator_now": "NA",
         "allocator_available_now": 3, "local_cached_blocks": 3,
         "leased_blocks": "NA", "owned_blocks": "NA", "evictable_cached_blocks": 2},
    ]
    selections = [
        {"selection_id": "s1", "candidate_matches_protected": False},
        {"selection_id": "s2", "candidate_matches_protected": True},
        {"selection_id": "s3", "candidate_matches_protected": False},
    ]
    result = analysis.summarize_protected_need(needs, selections)
    assert result["all"]["selection_count"] == 3
    assert result["all"]["allocator_exact_known_count"] == 1
    assert result["all"]["allocator_exact_unknown_count"] == 2
    assert result["all"]["need_psa1_all"]["median"] == 20
    assert result["all"]["need_allocator_now"]["median"] == 15
    assert result["all"]["need_gap_psa1_minus_allocator"]["median"] == 5
    assert result["all"]["need_ratio_psa1_over_max_1_allocator"]["median"] == 20 / 15
    assert result["candidate_matches_protected"]["selection_count"] == 1
    assert result["candidate_differs_from_protected"]["allocator_exact_unknown_count"] == 2
    assert result["local_cached_blocks_positive"]["ownership"]["owned_blocks"]["count"] == 1
    assert result["interpretation_guardrails"]["missing_allocator_values_excluded_not_zero_filled"]


def test_source_accounting_accepts_only_the_known_prompt_only_shortfall():
    result = analysis.source_accounting(
        local_tokens=0, external_tokens=704960, expected_tokens=705728,
        explained_shortfall_tokens=768, unexplained_shortfall_tokens=0,
        unique_resume_traces=32,
    )
    assert result["valid"] is True
    assert result["raw_local_plus_external_equals_expected"] is False


def test_source_accounting_rejects_unexplained_shortfall():
    result = analysis.source_accounting(
        local_tokens=0, external_tokens=704960, expected_tokens=705728,
        explained_shortfall_tokens=768, unexplained_shortfall_tokens=1,
        unique_resume_traces=32,
    )
    assert result["valid"] is False


def test_source_accounting_rejects_missing_resume_trace():
    result = analysis.source_accounting(
        local_tokens=0, external_tokens=704960, expected_tokens=705728,
        explained_shortfall_tokens=768, unexplained_shortfall_tokens=0,
        unique_resume_traces=31,
    )
    assert result["valid"] is False


def test_result_retry_threshold_requires_count_and_fraction():
    assert analysis.classify_result(poison_episode_count=10, retry_recoverable_count=8,
        frontier_blocked_count=10, frontier_recoverable_count=0,
        relevant_allocator_known_count=10, relevant_allocator_reject_count=0,
        allocator_fit_f3_psa_fail_count=0) == "RESULT_A_RETRYABLE_PROTECTED_HEAD_DOMINANT"


def test_result_frontier_threshold():
    assert analysis.classify_result(poison_episode_count=10, retry_recoverable_count=1,
        frontier_blocked_count=10, frontier_recoverable_count=6,
        relevant_allocator_known_count=10, relevant_allocator_reject_count=0,
        allocator_fit_f3_psa_fail_count=0) == "RESULT_B_PROTECTED_FRONTIER_POISONING_DOMINANT"


def test_result_capacity_threshold():
    assert analysis.classify_result(poison_episode_count=10, retry_recoverable_count=0,
        frontier_blocked_count=10, frontier_recoverable_count=0,
        relevant_allocator_known_count=10, relevant_allocator_reject_count=6,
        allocator_fit_f3_psa_fail_count=0) == "RESULT_C_CAPACITY_DOMINANT"


def test_result_psa_intrinsic_threshold():
    assert analysis.classify_result(poison_episode_count=10, retry_recoverable_count=0,
        frontier_blocked_count=10, frontier_recoverable_count=0,
        relevant_allocator_known_count=10, relevant_allocator_reject_count=0,
        allocator_fit_f3_psa_fail_count=6) == "RESULT_D_PSA_INTRINSIC"


def test_result_mixed_default():
    assert analysis.classify_result(poison_episode_count=0, retry_recoverable_count=0,
        frontier_blocked_count=0, frontier_recoverable_count=0,
        relevant_allocator_known_count=0, relevant_allocator_reject_count=0,
        allocator_fit_f3_psa_fail_count=0) == "RESULT_E_MIXED_HEAD_CAPACITY_PSA"


def test_timing_gate_frozen():
    assert analysis.timing_gate([1.0, 2.0])
    assert not analysis.timing_gate([1.0, 31.0])


def test_runner_wires_the_absolute_waiter_used_by_capture():
    source = (ROOT / "experiments/phase4g_d/run_phase4g_d.py").read_text(encoding="utf-8")
    assert "base.phase4g_c_wait_until = absolute_sleep.phase4g_c_fix_wait_until" in source
    assert "from experiments.phase4g_c_fix import absolute_sleep" in source


def test_no_algorithm_change_is_part_of_phase_spec():
    runner = (ROOT / "experiments/phase4g_d/run_phase4g_d.py")
    if runner.exists():
        source = runner.read_text(encoding="utf-8")
        assert "actual_retry_count" in source
        assert "actual_frontier_policy_changes" in source


def test_offline_csv_reader_preserves_unknown_and_booleans():
    assert runner._parse_csv_value("") is None
    assert runner._parse_csv_value("UNKNOWN") == "UNKNOWN"
    assert runner._parse_csv_value("False") is False
    assert runner._parse_csv_value("0") == 0


def test_persisted_matrix_finalizer_requires_all_four_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "OUT", tmp_path)
    (tmp_path / "run_metrics.csv").write_text(
        "run_id,stage,valid,turns_complete,boundaries\n"
        "r1,measured,True,40,32\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="four distinct measured runs"):
        runner.finalize_existing_results()


def test_figure_writer_uses_local_pillow_renderer(tmp_path):
    path = tmp_path / "timeline.png"
    runner._write_figure(path, [{
        "elapsed_ms": 0.0, "protected_id": "E04:b0", "candidate_id": "E00:b0",
        "H_current": 1200, "pool_free_blocks": 1600,
        "allocator_available_blocks": 1400, "retry_actionable": False,
    }, {
        "elapsed_ms": 25.0, "protected_id": "E04:b0", "candidate_id": "E01:b0",
        "H_current": 1200, "pool_free_blocks": 1700,
        "allocator_available_blocks": 1500, "retry_actionable": True,
    }])
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_final_aggregate_does_not_read_observability_as_mechanism_result():
    source = (ROOT / "experiments/phase4g_d/run_phase4g_d.py").read_text(encoding="utf-8")
    assert 'json.loads((OUT / "allocator_observability_summary.json")' not in source
    assert 'metrics_summary = _write_aggregates(' in source


def test_retry_probe_coverage_requires_sample_for_long_valid_window():
    from types import SimpleNamespace
    controller = SimpleNamespace(
        poison_episodes=[{"protected_id": "E04:b0", "terminal_timestamp_ns": 100,
                          "end_timestamp_ns": 80_000_100}],
        registered={"E04:b0": {"resume_started_ns": 80_000_100,
                                "first_token_ns": 200_000_000}},
        retry_timeline=[{"protected_id": "E04:b0", "timestamp_ns": 20_000_100}],
    )
    result = runner._retry_probe_coverage(controller)
    assert result["pre_resume_or_frontier_windows_ge_20ms"] == 1
    assert result["windows_with_exact_probe_sample"] == 1
    assert result["coverage_pass"] is True


def test_retry_probe_coverage_fails_unobserved_long_window():
    from types import SimpleNamespace
    controller = SimpleNamespace(
        poison_episodes=[{"protected_id": "E04:b0", "terminal_timestamp_ns": 100,
                          "end_timestamp_ns": 80_000_100}],
        registered={"E04:b0": {"resume_started_ns": 80_000_100,
                                "first_token_ns": 200_000_000}},
        retry_timeline=[],
    )
    result = runner._retry_probe_coverage(controller)
    assert result["coverage_pass"] is False
    assert result["uncovered_protected_ids"] == ["E04:b0"]


def test_first_retry_probe_is_anchored_to_terminal_plus_20ms():
    assert analysis.first_retry_probe_deadline_ns(100) == 20_000_100


def test_34ms_pre_resume_window_requires_an_exact_probe_sample():
    from types import SimpleNamespace
    controller = SimpleNamespace(
        poison_episodes=[{"protected_id": "E03:b2", "terminal_timestamp_ns": 100,
                          "end_timestamp_ns": 34_000_100}],
        registered={"E03:b2": {"resume_started_ns": 34_000_100,
                                "first_token_ns": 200_000_000}},
        retry_timeline=[{"protected_id": "E03:b2", "timestamp_ns": 20_000_100}],
    )
    result = runner._retry_probe_coverage(controller)
    assert result["pre_resume_or_frontier_windows_ge_20ms"] == 1
    assert result["windows_with_exact_probe_sample"] == 1
    assert result["coverage_pass"] is True


def test_unobserved_34ms_pre_resume_window_fails_coverage():
    from types import SimpleNamespace
    controller = SimpleNamespace(
        poison_episodes=[{"protected_id": "E03:b2", "terminal_timestamp_ns": 100,
                          "end_timestamp_ns": 34_000_100}],
        registered={"E03:b2": {"resume_started_ns": 34_000_100,
                                "first_token_ns": 200_000_000}},
        retry_timeline=[],
    )
    result = runner._retry_probe_coverage(controller)
    assert result["coverage_pass"] is False
    assert result["uncovered_protected_ids"] == ["E03:b2"]


def test_retry_probe_coverage_threshold_matches_minimum_actionable_window():
    assert analysis.retry_probe_coverage_required(19.999) is False
    assert analysis.retry_probe_coverage_required(20) is True


def test_poison_episode_clock_starts_at_terminal_decision():
    assert analysis.poison_episode_start_ns(100, 150) == 100
    assert analysis.poison_episode_start_ns(150, 100) == 150


def test_poison_duration_uses_terminal_time_not_later_frontier_entry():
    episode = {
        "terminal_timestamp_ns": 100,
        "frontier_entered_ns": 150,
        "episode_start_timestamp_ns": 100,
    }
    result = analysis.finish_poison_episode(
        episode, end_timestamp_ns=200, end_reason="FIRST_TOKEN", first_token_ns=200,
    )
    assert result["poison_duration_ms"] == 0.0001
    assert result["frontier_entered_ns"] == 150


def test_shadow_monitor_attempts_first_probe_at_terminal_plus_20ms():
    from types import SimpleNamespace
    from experiments.phase4g_d.shadow import ProtectedHeadShadowController

    terminal_ns = time.monotonic_ns()
    samples = []
    controller = SimpleNamespace(
        _shadow_stop=threading.Event(),
        _shadow_wake_event=threading.Event(),
        _shadow_monitor_started_event=threading.Event(),
        _shadow_lock=threading.Lock(),
        lock=threading.Lock(),
        _active_poison=None,
        _terminal_timestamp_ns={"E03:b2": terminal_ns},
        registered={"E03:b2": {"decision": "PROGRESS_REJECTED", "resume_started_ns": None,
                                "first_token_ns": None}},
        retry_timeline=[], shadow_probe_errors=[], shadow_monitor_schedule_events=[],
        shadow_monitor_started=False, shadow_monitor_exited=False,
        shadow_monitor_loop_count=0, shadow_monitor_probe_attempt_count=0,
        shadow_monitor_resume_skip_count=0, shadow_monitor_fatal_error=None,
        shadow_monitor_cpu_ns=0, shadow_cpu_ns=0, shadow_client_cpu_ns=0,
        _sync_poison_episode=lambda _timestamp_ns: None,
        _record_protected_state=lambda _timestamp_ns: None,
    )

    def snapshot(_boundary_id):
        row = {"timestamp_ns": time.monotonic_ns(), "shadow_server_cpu_ns": 0,
               "shadow_client_cpu_ns": 0}
        samples.append(row)
        controller._shadow_stop.set()
        return row

    controller._snapshot_for_head = snapshot
    ProtectedHeadShadowController._shadow_monitor(controller)
    assert len(samples) == 1
    assert samples[0]["timestamp_ns"] >= analysis.first_retry_probe_deadline_ns(terminal_ns)
    assert controller.shadow_monitor_schedule_events[0]["outcome"] == "FIRST_PROBE_ATTEMPTED"
