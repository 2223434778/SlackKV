from experiments.phase4g_b.analysis import (
    FROZEN_RUN_ORDER,
    PROBE_INTERVAL_MS,
    background_is_valid,
    classify_result,
    feasibility_snapshot,
    longest_contiguous_true_ms,
    nearest_rank,
    partial_opportunity,
    progress_reserve_blocks,
    source_readiness_milestones,
    validate_matrix,
)


def test_progress_reserve_matches_psa_hypothetical_lease_rule():
    assert progress_reserve_blocks(1400, 0, candidate_matches_protected=True, candidate_blocks=1300) == 100
    assert progress_reserve_blocks(1400, 1200, candidate_matches_protected=False, candidate_blocks=1000) == 200


def test_local_prefix_reduces_external_physical_need():
    state = feasibility_snapshot(
        required_blocks=100, local_blocks=20, external_ready_blocks=80,
        free_blocks=90, progress_reserve=10, reserved_blocks=0,
        logical_enabled=False,
    )
    assert state["external_needed_blocks"] == 80
    assert state["source_ready"] and state["physical_fit"] and state["psa_safe_fit"]


def test_missing_source_and_safe_headroom_are_separate_constraints():
    state = feasibility_snapshot(
        required_blocks=100, local_blocks=0, external_ready_blocks=75,
        free_blocks=60, progress_reserve=20, reserved_blocks=0,
        logical_enabled=False,
    )
    assert state["missing_blocks"] == 25
    assert not state["source_ready"]
    assert not state["physical_fit"]
    assert not state["psa_safe_fit"]


def test_logical_reservation_is_an_independent_gate():
    state = feasibility_snapshot(
        required_blocks=1300, local_blocks=0, external_ready_blocks=1300,
        free_blocks=1500, progress_reserve=100, reserved_blocks=2500,
        logical_enabled=True, logical_budget=3510,
    )
    assert state["source_ready"] and state["psa_safe_fit"]
    assert not state["logical_fit"] and not state["full_feasible"]


def test_partial_opportunity_uses_progress_reserve_and_required_denominator():
    assert partial_opportunity(100, 80, 70, 20) == 0.5
    assert partial_opportunity(100, 80, 10, 20) == 0.0


def test_actionable_window_requires_consecutive_samples():
    samples = [
        {"timestamp_monotonic_ns": 0, "ok": True},
        {"timestamp_monotonic_ns": 10_000_000, "ok": True},
        {"timestamp_monotonic_ns": 30_000_000, "ok": True},
        {"timestamp_monotonic_ns": 80_000_000, "ok": True},
    ]
    assert longest_contiguous_true_ms(samples, lambda row: row["ok"]) == 30


def test_source_milestones_are_sampled_and_use_covered_blocks():
    samples = [
        {"timestamp_monotonic_ns": 1_000_000_000, "covered_blocks": 3},
        {"timestamp_monotonic_ns": 1_020_000_000, "covered_blocks": 6},
        {"timestamp_monotonic_ns": 1_040_000_000, "covered_blocks": 9},
        {"timestamp_monotonic_ns": 1_060_000_000, "covered_blocks": 10},
    ]
    result = source_readiness_milestones(samples, 10)
    assert result["ready_minus_4_ms"] == 20
    assert result["ready_minus_2_ms"] == 40
    assert result["ready_minus_1_ms"] == 40
    assert result["ready_minus_0_ms"] == 60


def test_frozen_result_A_thresholds():
    assert classify_result(
        m2_not_ready_count=8, m2_later_full_by_safe_start_count=5,
        m2_actionable_recoveries=3, m2_source_in_time_count=4,
        m2_source_in_time_without_safe_window=0, m2_source_in_time_with_partial=0,
        m2_source_in_time_physical_fit_count=4,
        m2_physical_fit_without_safe_window=0,
    ) == "RESULT_A_SOURCE_RECHECK_OPPORTUNITY"


def test_frozen_result_B_threshold():
    assert classify_result(
        m2_not_ready_count=8, m2_later_full_by_safe_start_count=1,
        m2_actionable_recoveries=0, m2_source_in_time_count=0,
        m2_source_in_time_without_safe_window=0, m2_source_in_time_with_partial=0,
        m2_source_in_time_physical_fit_count=0,
        m2_physical_fit_without_safe_window=0,
    ) == "RESULT_B_SOURCE_NOT_READY_IN_TIME"


def test_frozen_result_C_and_D_classification():
    common = dict(m2_not_ready_count=0, m2_later_full_by_safe_start_count=0,
                  m2_actionable_recoveries=0, m2_source_in_time_count=8,
                  m2_source_in_time_physical_fit_count=3)
    assert classify_result(**common, m2_source_in_time_without_safe_window=6,
                           m2_source_in_time_with_partial=7,
                           m2_physical_fit_without_safe_window=0) == "RESULT_C_PARTIAL_PROMOTION_OPPORTUNITY"
    assert classify_result(**common, m2_source_in_time_without_safe_window=6,
                           m2_source_in_time_with_partial=0,
                           m2_physical_fit_without_safe_window=5) == "RESULT_D_PSA_HEADROOM_BLOCKER"


def test_frozen_result_E_is_fallback():
    assert classify_result(
        m2_not_ready_count=0, m2_later_full_by_safe_start_count=0,
        m2_actionable_recoveries=0, m2_source_in_time_count=8,
        m2_source_in_time_without_safe_window=3, m2_source_in_time_with_partial=2,
        m2_source_in_time_physical_fit_count=5,
        m2_physical_fit_without_safe_window=2,
    ) == "RESULT_E_MIXED_OR_NO_CLEAR_BOTTLENECK"


def test_probe_cadence_is_frozen_at_twenty_ms():
    assert PROBE_INTERVAL_MS == 20.0


def test_background_gate_uses_chunk_tokens_gap_and_tail():
    assert background_is_valid(generated_tokens=8192, expected_tokens=8192,
                               max_idle_gap_ms=25, allowed_gap_ms=50,
                               tail_margin_ms=4000, minimum_tail_ms=3000)
    assert not background_is_valid(generated_tokens=8191, expected_tokens=8192,
                                   max_idle_gap_ms=25, allowed_gap_ms=50,
                                   tail_margin_ms=4000)
    assert not background_is_valid(generated_tokens=8192, expected_tokens=8192,
                                   max_idle_gap_ms=51, allowed_gap_ms=50,
                                   tail_margin_ms=4000)
    assert not background_is_valid(generated_tokens=8192, expected_tokens=8192,
                                   max_idle_gap_ms=25, allowed_gap_ms=50,
                                   tail_margin_ms=2999, minimum_tail_ms=3000)


def test_frozen_matrix_has_twelve_runs_and_expected_condition_balance():
    assert len(FROZEN_RUN_ORDER) == 12
    assert [row[0] for row in FROZEN_RUN_ORDER] == [
        "M0", "M1", "M1", "M0", "M2", "M3", "M3", "M2", "M2", "M3", "M3", "M2"
    ]
    assert {condition: sum(row[0] == condition for row in FROZEN_RUN_ORDER)
            for condition in ("M0", "M1", "M2", "M3")} == {"M0": 2, "M1": 2, "M2": 4, "M3": 4}


def test_no_background_and_bgv2_conditions_are_separated():
    assert all(not row[2] for row in FROZEN_RUN_ORDER if row[0] in {"M0", "M1"})
    assert all(row[2] for row in FROZEN_RUN_ORDER if row[0] in {"M2", "M3"})


def test_source_readiness_can_be_false_despite_capacity_fit():
    state = feasibility_snapshot(
        required_blocks=100, local_blocks=0, external_ready_blocks=99,
        free_blocks=500, progress_reserve=20, reserved_blocks=0,
        logical_enabled=False,
    )
    assert state["physical_fit"] and state["psa_safe_fit"]
    assert not state["source_ready"] and not state["full_feasible"]


def test_source_readiness_can_pass_while_physical_headroom_fails():
    state = feasibility_snapshot(
        required_blocks=100, local_blocks=0, external_ready_blocks=100,
        free_blocks=99, progress_reserve=0, reserved_blocks=0,
        logical_enabled=False,
    )
    assert state["source_ready"]
    assert not state["physical_fit"] and not state["psa_safe_fit"]


def test_external_ready_blocks_are_clipped_to_missing_prefix_blocks():
    state = feasibility_snapshot(
        required_blocks=100, local_blocks=30, external_ready_blocks=80,
        free_blocks=70, progress_reserve=0, reserved_blocks=0,
        logical_enabled=False,
    )
    assert state["external_needed_blocks"] == 70
    assert state["external_ready_needed_blocks"] == 70
    assert state["covered_blocks"] == 100


def test_exact_physical_fit_passes_but_exact_unsafe_headroom_does_not():
    state = feasibility_snapshot(
        required_blocks=100, local_blocks=0, external_ready_blocks=100,
        free_blocks=100, progress_reserve=1, reserved_blocks=0,
        logical_enabled=False,
    )
    assert state["physical_fit"] and not state["psa_safe_fit"]


def test_partial_opportunity_is_bounded_to_unit_interval():
    assert partial_opportunity(100, 500, 900, 0) == 1.0
    assert partial_opportunity(100, 80, 40, 70) == 0.0


def test_partial_opportunity_rejects_zero_required_prefix():
    import pytest
    with pytest.raises(ValueError):
        partial_opportunity(0, 0, 0, 0)


def test_contiguous_window_breaks_on_false_sample():
    samples = [
        {"timestamp_monotonic_ns": 0, "ok": True},
        {"timestamp_monotonic_ns": 20_000_000, "ok": False},
        {"timestamp_monotonic_ns": 40_000_000, "ok": True},
        {"timestamp_monotonic_ns": 60_000_000, "ok": True},
    ]
    assert longest_contiguous_true_ms(samples, lambda row: row["ok"]) == 20


def test_contiguous_window_breaks_on_sampling_gap_over_forty_ms():
    samples = [
        {"timestamp_monotonic_ns": 0, "ok": True},
        {"timestamp_monotonic_ns": 50_000_000, "ok": True},
    ]
    assert longest_contiguous_true_ms(samples, lambda row: row["ok"]) == 0


def test_source_milestones_report_missing_full_readiness():
    samples = [
        {"timestamp_monotonic_ns": 0, "covered_blocks": 1},
        {"timestamp_monotonic_ns": 20_000_000, "covered_blocks": 2},
    ]
    result = source_readiness_milestones(samples, 10)
    assert result["ready_minus_0_ms"] == "NA"


def test_nearest_rank_uses_project_definition():
    assert nearest_rank([1, 2, 3, 4], .95) == 4
    assert nearest_rank([1, 2, 3, 4], .50) == 2


def test_result_A_recovery_threshold_is_inclusive_at_half():
    assert classify_result(
        m2_not_ready_count=8, m2_later_full_by_safe_start_count=4,
        m2_actionable_recoveries=2, m2_source_in_time_count=0,
        m2_source_in_time_without_safe_window=0, m2_source_in_time_with_partial=0,
        m2_source_in_time_physical_fit_count=0, m2_physical_fit_without_safe_window=0,
    ) == "RESULT_A_SOURCE_RECHECK_OPPORTUNITY"


def test_result_B_requires_recovery_strictly_below_quarter():
    common = dict(m2_not_ready_count=8, m2_actionable_recoveries=0,
                  m2_source_in_time_count=0, m2_source_in_time_without_safe_window=0,
                  m2_source_in_time_with_partial=0, m2_source_in_time_physical_fit_count=0,
                  m2_physical_fit_without_safe_window=0)
    assert classify_result(m2_later_full_by_safe_start_count=1, **common) == "RESULT_B_SOURCE_NOT_READY_IN_TIME"
    assert classify_result(m2_later_full_by_safe_start_count=2, **common) != "RESULT_B_SOURCE_NOT_READY_IN_TIME"


def test_result_C_uses_partial_and_no_safe_window_majorities():
    args = dict(m2_not_ready_count=0, m2_later_full_by_safe_start_count=0,
                m2_actionable_recoveries=0, m2_source_in_time_count=10,
                m2_source_in_time_without_safe_window=6, m2_source_in_time_with_partial=6,
                m2_source_in_time_physical_fit_count=0, m2_physical_fit_without_safe_window=0)
    assert classify_result(**args) == "RESULT_C_PARTIAL_PROMOTION_OPPORTUNITY"


def test_result_D_denominator_is_source_ready_and_physically_fitting_jobs():
    args = dict(m2_not_ready_count=0, m2_later_full_by_safe_start_count=0,
                m2_actionable_recoveries=0, m2_source_in_time_count=20,
                m2_source_in_time_without_safe_window=0, m2_source_in_time_with_partial=0,
                m2_source_in_time_physical_fit_count=10, m2_physical_fit_without_safe_window=6)
    assert classify_result(**args) == "RESULT_D_PSA_HEADROOM_BLOCKER"
    args["m2_physical_fit_without_safe_window"] = 5
    assert classify_result(**args) != "RESULT_D_PSA_HEADROOM_BLOCKER"


def test_result_E_is_used_when_no_frozen_majority_gate_matches():
    args = dict(m2_not_ready_count=10, m2_later_full_by_safe_start_count=3,
                m2_actionable_recoveries=1, m2_source_in_time_count=10,
                m2_source_in_time_without_safe_window=4, m2_source_in_time_with_partial=5,
                m2_source_in_time_physical_fit_count=10, m2_physical_fit_without_safe_window=4)
    assert classify_result(**args) == "RESULT_E_MIXED_OR_NO_CLEAR_BOTTLENECK"


def test_complete_frozen_matrix_passes_validation():
    rows = [{"condition": condition, "valid": True, "run_id": f"r{i}"}
            for i, (condition, _, _) in enumerate(FROZEN_RUN_ORDER)]
    validate_matrix(rows)


def test_partial_matrix_cannot_be_finalized():
    import pytest
    rows = [{"condition": condition, "valid": True}
            for condition, _, _ in FROZEN_RUN_ORDER[:-1]]
    with pytest.raises(ValueError, match="partial or unbalanced"):
        validate_matrix(rows)


def test_invalid_run_cannot_be_finalized_even_when_count_balanced():
    import pytest
    rows = [{"condition": condition, "valid": True, "run_id": str(i)}
            for i, (condition, _, _) in enumerate(FROZEN_RUN_ORDER)]
    rows[-1]["valid"] = False
    with pytest.raises(ValueError, match="invalid Phase 4G-B runs"):
        validate_matrix(rows)
