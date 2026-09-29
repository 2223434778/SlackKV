from __future__ import annotations

import csv
import hashlib
import inspect
from pathlib import Path

import pytest

from experiments.phase4g_a.workload import ROOT, SOURCE
from experiments.phase4g_a_fix import background as bgv2
from experiments.phase4f_a.logic import progress_blocks
from experiments.phase4g_c.analysis import (
    GPU_BLOCKS,
    K2,
    P_SAFE_MS,
    RECHECK_INTERVAL_MS,
    RESERVATION_BUDGET_BLOCKS,
    RESUME_GUARD_MS,
    SOURCE_SHA,
    START_COMMIT,
    allocator_preflight,
    classify_attempt,
    classify_matrix,
    full_ready,
    missed_protected_self_promotion,
    normalize_incomplete_result,
    prompt_only_externalizable_tokens,
    protected_self_reserve,
    psa1_reserve,
    run_scoped_diagnostic_counts,
    selection_visibility,
    source_coverage_diagnostic,
    source_recheck_event,
    unique_source_trace_rows,
)


def test_frozen_start_and_workload_sha():
    assert START_COMMIT == "c1761e980b16eec5d5381e8cf0a6edd4d245bf09"
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA


def test_frozen_bg_and_controller_constants():
    assert bgv2.BG_TOTAL_OUTPUT_TOKENS > 0
    assert bgv2.BG_PROMPT_TOKENS == 9500
    assert P_SAFE_MS == 102
    assert RESUME_GUARD_MS == 110
    assert K2 == 2
    assert RESERVATION_BUDGET_BLOCKS == 3510
    assert GPU_BLOCKS == 4681
    assert RECHECK_INTERVAL_MS == 20


def test_psa1_formula_matches_phase4f_a_implementation():
    assert progress_blocks(20_432) == 1_278
    assert psa1_reserve(1_278, 0, False, 1_330) == 1_278
    assert psa1_reserve(1_278, 1271, True, 1271) == 7
    assert protected_self_reserve(1_278, 0, 1_271) == 7


def test_full_ready_requires_all_required_blocks():
    assert full_ready({"required_blocks": 100, "covered_blocks": 100})
    assert not full_ready({"required_blocks": 100, "covered_blocks": 99})


def test_prompt_only_offload_excludes_only_complete_decode_tail_blocks():
    assert prompt_only_externalizable_tokens(25_168, 24_968, 16) == 24_960
    diagnostic = source_coverage_diagnostic(
        full_prefix_tokens=25_168, previous_prompt_tokens=24_968,
        local_source_tokens=0, external_source_tokens=24_960,
    )
    assert diagnostic["prompt_only_decode_tail_tokens"] == 208
    assert diagnostic["full_prefix_source_shortfall_tokens"] == 208
    assert diagnostic["prompt_only_explained_shortfall_tokens"] == 208
    assert diagnostic["unexplained_source_shortfall_tokens"] == 0
    assert diagnostic["full_prefix_source_hit_exact"] is False


def test_explained_cache_source_shortfall_is_diagnostic_not_phase_gate():
    from experiments.phase4g_c import run_phase4g_c

    source = inspect.getsource(run_phase4g_c.run_one)
    assert 'if not result["source_conservation_pass"]' not in source
    assert "source_coverage_diagnostic(" in source


def test_duplicate_source_trace_for_one_request_counts_only_first_observation():
    rows = [
        {"request_id": "req-1", "session": "E05:turn:1", "timestamp_monotonic": 2.0,
         "local_cached_tokens": 0, "external_cached_tokens": 20656},
        {"request_id": "req-1", "session": "E05:turn:1", "timestamp_monotonic": 1.0,
         "local_cached_tokens": 0, "external_cached_tokens": 20384},
        {"request_id": "req-2", "session": "E04:turn:1", "timestamp_monotonic": 1.5,
         "local_cached_tokens": 16, "external_cached_tokens": 20320},
    ]
    unique, duplicate_count = unique_source_trace_rows(rows)
    assert duplicate_count == 1
    assert [row["request_id"] for row in unique] == ["req-1", "req-2"]
    assert sum(row["local_cached_tokens"] + row["external_cached_tokens"] for row in unique) == 40720


def test_per_run_allocator_and_counterfactual_counts_do_not_accumulate():
    guard_rows = [
        {"run_id": "c0-r1", "condition": "C0", "event": "CAPACITY_GUARD", "allocator_fit": False},
        {"run_id": "c0-r1", "condition": "C0", "event": "CAPACITY_GUARD", "allocator_fit": True},
        {"run_id": "c0-r2", "condition": "C0", "event": "CAPACITY_GUARD", "allocator_fit": False},
        {"run_id": "c1-r1", "condition": "C1", "event": "CAPACITY_GUARD", "allocator_fit": False},
    ]
    counterfactual_rows = [
        {"run_id": "c0-r1", "condition": "C0", "would_recover": True},
        {"run_id": "c0-r2", "condition": "C0", "would_recover": False},
        {"run_id": "c1-r1", "condition": "C1", "would_recover": True},
    ]
    assert run_scoped_diagnostic_counts("c0-r1", "C0", guard_rows, counterfactual_rows) == {
        "exact_allocator_rejects": 1,
        "missed_protected_self_promotion_count": 1,
    }
    assert run_scoped_diagnostic_counts("c0-r2", "C0", guard_rows, counterfactual_rows) == {
        "exact_allocator_rejects": 1,
        "missed_protected_self_promotion_count": 0,
    }
    assert run_scoped_diagnostic_counts("c1-r1", "C1", guard_rows, counterfactual_rows) == {
        "exact_allocator_rejects": 1,
        "missed_protected_self_promotion_count": 0,
    }


def test_phase4g_c_wait_uses_absolute_monotonic_deadline_and_spin_tail():
    from experiments.phase4g_c.run_phase4g_c import (
        PAUSE_WAIT_SPIN_WINDOW_S,
        phase4g_c_wait_until,
    )

    assert PAUSE_WAIT_SPIN_WINDOW_S == 0.050
    now = [0.0]
    sleeps = []

    def monotonic():
        value = now[0]
        now[0] += 0.00025
        return value

    def sleeper(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    target = 1.0
    phase4g_c_wait_until(target, monotonic=monotonic, sleeper=sleeper)
    assert len(sleeps) == 1
    assert sleeps[0] == pytest.approx(target - PAUSE_WAIT_SPIN_WINDOW_S)
    assert now[0] >= target


def test_client_thread_switch_interval_is_temporary_and_frozen():
    import sys

    from experiments.phase4g_c.run_phase4g_c import (
        PAUSE_THREAD_SWITCH_INTERVAL_S,
        precise_client_thread_switching,
    )

    original = sys.getswitchinterval()
    with precise_client_thread_switching():
        assert sys.getswitchinterval() == PAUSE_THREAD_SWITCH_INTERVAL_S == 0.0005
    assert sys.getswitchinterval() == original


def test_source_trace_request_identity_cannot_cross_sessions():
    with pytest.raises(ValueError, match="multiple sessions"):
        unique_source_trace_rows([
            {"request_id": "req-1", "session": "E04:turn:1", "timestamp_monotonic": 1.0},
            {"request_id": "req-1", "session": "E05:turn:1", "timestamp_monotonic": 2.0},
        ])


def test_recheck_transition_is_full_ready_and_initially_not_ready_only():
    sample = {
        "boundary_id": "E04:b0", "timestamp_monotonic_ns": 2_000_000_000,
        "required_blocks": 10, "covered_blocks": 10,
        "local_cached_blocks": 3, "external_ready_needed_blocks": 7,
    }
    event = source_recheck_event(sample, "SOURCE_NOT_READY", 1_000_000_000, 3_000_000_000)
    assert event is not None
    assert event["event"] == "SOURCE_READY_RECHECK"
    assert event["covered_blocks"] == event["required_blocks"] == 10
    assert event["time_since_initial_source_check_ms"] == 1000
    assert event["time_to_deadline_ms"] == 1000
    assert source_recheck_event({**sample, "covered_blocks": 9}, "SOURCE_NOT_READY", 0, 3_000_000_000) is None
    assert source_recheck_event(sample, "SOURCE_READY", 0, 3_000_000_000) is None


@pytest.mark.parametrize(
    ("candidate", "protected", "visible", "exact", "stale", "wrong"),
    [
        ("E01:b0", "E04:b0", False, True, True, True),
        ("E04:b0", "E04:b0", False, True, True, False),
        ("E01:b0", "E04:b0", False, False, False, False),
        ("E04:b0", "E04:b0", True, True, False, False),
    ],
)
def test_stale_visibility_and_candidate_identity(candidate, protected, visible, exact, stale, wrong):
    result = selection_visibility(candidate, protected, visible, exact)
    assert result["stale_protected_visibility"] is stale
    assert result["wrong_candidate_due_to_stale_source"] is wrong


def test_self_psa_does_not_conflate_full_footprint_and_progress_need():
    progress_need, full_blocks, current_lease = 1278, 1271, 0
    actual = psa1_reserve(progress_need, current_lease, False, 1330)
    self_candidate = protected_self_reserve(progress_need, current_lease, full_blocks)
    assert actual == 1278
    assert self_candidate == 7
    assert actual - self_candidate == 1271


def test_allocator_preflight_keeps_pool_free_and_schedulable_distinct():
    result = allocator_preflight(
        pool_free_blocks=1564, inflight_reserved_blocks=759,
        num_blocks_to_allocate=900,
    )
    assert result["pool_free_blocks"] == 1564
    assert result["allocator_available_blocks"] == 805
    assert result["allocator_required_blocks"] == 900
    assert result["allocator_fit"] is False
    assert result["watermark_blocks"] == 0


def test_full_sequence_gate_uses_pool_free_without_inflight_reservation():
    result = allocator_preflight(
        pool_free_blocks=100, inflight_reserved_blocks=80,
        num_blocks_to_allocate=90, full_sequence_must_fit=True,
    )
    assert result["allocator_available_blocks"] == 100
    assert result["allocator_fit"] is True


def test_legacy_guard_allocator_agreement_is_directional():
    false_admit = classify_attempt(
        legacy_required_blocks=1000, pool_free_blocks=1000,
        allocator_required_blocks=900, allocator_available_blocks=800,
    )
    assert false_admit["false_admit"] and not false_admit["false_reject"]
    false_reject = classify_attempt(
        legacy_required_blocks=1000, pool_free_blocks=900,
        allocator_required_blocks=800, allocator_available_blocks=900,
    )
    assert false_reject["false_reject"] and not false_reject["false_admit"]


def test_missed_protected_self_promotion_requires_all_existing_gates():
    fields = dict(candidate_id="E01:b0", protected_id="E04:b0",
                  protected_exact_ready=True, self_logical_fit=True,
                  self_psa_fit=True, self_legacy_physical_fit=True,
                  self_exact_allocator_fit=True, timing_permits=True)
    assert missed_protected_self_promotion(**fields)
    assert not missed_protected_self_promotion(**{**fields, "self_exact_allocator_fit": False})
    assert not missed_protected_self_promotion(**{**fields, "protected_exact_ready": False})


def test_result_a_thresholds_are_frozen_and_sufficient():
    c0 = {
        "selection_count": 100, "stale_protected_visibility_count": 30,
        "candidate_matches_protected_count": 30,
        "missed_protected_self_promotion_count": 1,
        "median_psa_reserve": 1000, "successful_promotions": 2,
    }
    c1 = {
        "selection_count": 100, "candidate_matches_protected_count": 60,
        "median_psa_reserve": 400, "successful_promotions": 3,
    }
    assert classify_matrix(c0, c1) == "RESULT_A_SOURCE_VISIBILITY_PSA_INTERACTION"


def test_result_e_is_fallback_when_no_causal_threshold_is_met():
    c0 = {"selection_count": 10, "stale_protected_visibility_count": 1,
          "candidate_matches_protected_count": 4, "median_psa_reserve": 500,
          "missed_protected_self_promotion_count": 0, "successful_promotions": 1}
    c1 = {"selection_count": 10, "candidate_matches_protected_count": 5,
          "median_psa_reserve": 450, "successful_promotions": 1,
          "attempt_count": 10, "false_admit_count": 0, "false_reject_count": 1}
    assert classify_matrix(c0, c1) == "RESULT_E_MIXED_VISIBILITY_PSA_ALLOCATOR"


def test_matrix_plan_is_four_pairs_and_no_performance_classifier():
    from experiments.phase4g_c import run_phase4g_c

    assert run_phase4g_c.RUN_ORDER == [
        (1, "C0"), (1, "C1"), (2, "C1"), (2, "C0"),
        (3, "C0"), (3, "C1"), (4, "C1"), (4, "C0"),
    ]
    source = inspect.getsource(run_phase4g_c)
    assert "H1" not in source and "H2" not in source and "H3" not in source


def test_smoke_gate_requires_c0_ignore_and_c1_consume_rechecks():
    from experiments.phase4g_c import run_phase4g_c

    source = inspect.getsource(run_phase4g_c.main)
    assert 'smoke_results[0]["recheck_behavior_updates"] != 0' in source
    assert 'smoke_results[1]["recheck_behavior_updates"] <= 0' in source


def test_vllm_allocator_trace_is_observational_not_a_new_decision():
    path = Path("/media/a6000/D/ZhaoBinHe/slackkv-phase2c-fix-sourceprobe/vllm/v1/core/kv_cache_manager.py")
    if not path.exists():
        pytest.skip("pinned A6000 vLLM source is only present on the experiment host")
    text = path.read_text(encoding="utf-8")
    assert "num_blocks_to_allocate > available_blocks" in text
    assert "_slackkv_allocator_truth(" in text
    assert "num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(" in text


def test_frozen_workload_shape_and_no_private_text_fields():
    with SOURCE.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 40
    assert {row["episode_id"] for row in rows} == {f"E{i:02d}" for i in range(8)}
    assert not {"prompt", "completion", "instance_id", "raw_tool_input"}.intersection(rows[0])


def test_source_probe_transition_never_accepts_partial_coverage():
    sample = {"boundary_id": "E00:b0", "timestamp_monotonic_ns": 5,
              "required_blocks": 1024, "covered_blocks": 1023}
    assert source_recheck_event(sample, "SOURCE_NOT_READY", 1, 10) is None
    sample["covered_blocks"] = 1024
    assert source_recheck_event(sample, "SOURCE_NOT_READY", 1, 10)["covered_blocks"] == 1024


def test_recheck_controller_keeps_base_candidate_selection_and_only_consumes_c1():
    from experiments.phase4g_a.controller import DynamicController
    from experiments.phase4g_c.controller import SourceRecheckController

    source = inspect.getsource(SourceRecheckController._candidate)
    observe = inspect.getsource(SourceRecheckController._observe_traces)
    consume = inspect.getsource(SourceRecheckController._consume_recheck)
    assert "super()._candidate()" in source
    assert "condition == \"C1\"" in consume
    assert "return selected" in source
    assert "def _candidate" in inspect.getsource(DynamicController)


@pytest.mark.parametrize(("condition", "visible", "updates"), [("C0", False, 0), ("C1", True, 1)])
def test_recheck_before_registration_preserves_exact_deadline_and_arm_semantics(
    tmp_path, condition, visible, updates,
):
    import json
    import time
    from experiments.phase4g_c.controller import SourceRecheckController

    now_ns = time.monotonic_ns()
    log = tmp_path / "server.log"
    log.write_text(
        "SLACKKV_OFFLOAD_TRACE " + json.dumps({
            "event": "SOURCE_NOT_READY", "timestamp_monotonic": (now_ns - 20_000_000) / 1e9,
            "boundary_id": "E00:b0",
        }) + "\n"
        + "SLACKKV_READY_PROBE " + json.dumps({
            "event": "SAMPLE", "timestamp_monotonic_ns": now_ns,
            "boundary_id": "E00:b0", "required_blocks": 1, "covered_blocks": 1,
            "local_cached_blocks": 0, "external_ready_needed_blocks": 1,
        }) + "\n",
        encoding="utf-8",
    )
    manifest = [{
        "boundary_id": "E00:b0", "lease_key": "E00:turn:1",
        "required_blocks": 1, "required_prefix_tokens": 16,
        "resume_isl": 32, "resume_osl": 2, "proxy_pause_ms": 100,
    }]
    controller = SourceRecheckController(
        manifest, log, now_ns, condition=condition, repetition=1,
    )
    controller._observe_traces()
    assert "E00:b0" not in controller.offload_events
    assert "E00:b0" in controller.pending_rechecks
    controller.register("E00:b0", [7] * 16, now_ns)
    assert ("E00:b0" in controller.offload_events) is visible
    assert controller.recheck_behavior_updates == updates
    assert controller.source_rechecks[0]["time_to_deadline_ms"] == pytest.approx(100)
    assert controller.source_rechecks[0]["covered_blocks"] == 1


def test_allocator_coordinator_preflight_api_is_read_only():
    path = Path("/media/a6000/D/ZhaoBinHe/slackkv-phase2c-fix-sourceprobe/vllm/v1/core/single_type_kv_cache_manager.py")
    if not path.exists():
        pytest.skip("pinned A6000 vLLM source is only present on the experiment host")
    source = path.read_text(encoding="utf-8")
    start = source.index("    def get_num_blocks_to_allocate(")
    end = source.index("    def allocate_new_computed_blocks(", start)
    body = source[start:end]
    assert "return num_new_blocks + num_evictable_blocks" in body
    assert "get_new_blocks(" not in body
    assert "touch(" not in body
    assert "ref_cnt" in body


def test_actual_allocate_slots_comparison_is_preserved_with_trace_around_it():
    path = Path("/media/a6000/D/ZhaoBinHe/slackkv-phase2c-fix-sourceprobe/vllm/v1/core/kv_cache_manager.py")
    if not path.exists():
        pytest.skip("pinned A6000 vLLM source is only present on the experiment host")
    source = path.read_text(encoding="utf-8")
    assert "if num_blocks_to_allocate > available_blocks:" in source
    assert source.count("_slackkv_allocator_truth(") >= 3


def test_source_readiness_counts_only_a_contiguous_prefix():
    import ast
    from collections.abc import Sequence

    path = Path("/media/a6000/D/ZhaoBinHe/slackkv-phase2c-fix-sourceprobe/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py")
    if not path.exists():
        pytest.skip("pinned A6000 vLLM source is only present on the experiment host")
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    helper = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_count_contiguous_ready_prefix"
    )
    module = ast.Module(body=[helper], type_ignores=[])
    namespace = {"Sequence": Sequence}
    exec(compile(module, str(path), "exec"), namespace)
    count = namespace["_count_contiguous_ready_prefix"]
    assert count([False, False, False], [True, False, True]) == (1, 1)
    assert count([True, False, False], [False, False, True]) == (1, 0)
    assert count([False, True, False], [True, False, True]) == (3, 2)


def test_failed_run_diagnostics_are_compact_and_never_overwritten(tmp_path, monkeypatch):
    import json
    from experiments.phase4g_c import run_phase4g_c

    monkeypatch.setattr(run_phase4g_c, "OUT", tmp_path / "results")
    monkeypatch.setattr(run_phase4g_c, "TMP_LOGS", tmp_path / "logs")
    first = run_phase4g_c._record_run_failure(
        "smoke_c0", {"error": "first"}, "C0", 0
    )
    second = run_phase4g_c._record_run_failure(
        "smoke_c0", {"error": "second"}, "C0", 0
    )
    assert first != second
    assert json.loads(first.read_text(encoding="utf-8"))["error"] == "first"
    assert json.loads(second.read_text(encoding="utf-8"))["error"] == "second"


def test_snapshot_shadow_preflight_does_not_allocate_or_touch_blocks():
    path = Path("/media/a6000/D/ZhaoBinHe/slackkv-phase2c-fix-sourceprobe/vllm/v1/core/sched/scheduler.py")
    if not path.exists():
        pytest.skip("pinned A6000 vLLM source is only present on the experiment host")
    source = path.read_text(encoding="utf-8")
    start = source.index("    def _slackkv_rca_allocator_preflight(")
    end = source.index("    def _slackkv_capacity_trace(", start)
    body = source[start:end]
    assert "get_num_blocks_to_allocate(" in body
    assert "get_new_blocks(" not in body
    assert ".touch(" not in body
    assert "allocator_available_blocks" in body
    assert "coordinator.scheduler_block_size" in body
    assert "kv_cache_manager.block_size" not in body


def test_snapshot_completion_preserves_payload_against_connector_prefetch_result():
    import ast
    from typing import Any

    path = Path("/media/a6000/D/ZhaoBinHe/slackkv-phase2c-fix-sourceprobe/vllm/v1/core/sched/scheduler.py")
    if not path.exists():
        pytest.skip("pinned A6000 vLLM source is only present on the experiment host")
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    helper = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_slackkv_capacity_finished_params"
    )
    namespace = {"Any": Any}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(path), "exec"), namespace)
    merge = namespace["_slackkv_capacity_finished_params"]
    snapshot = {"event": "SELECTION_SNAPSHOT", "free_blocks_before": 42}
    result = merge(
        {"slackkv_prefetch": {"status": "snapshot", "snapshot": snapshot}},
        {"slackkv_prefetch": {"status": "ok", "transfer_bytes": 0}, "other": 7},
        status="snapshot", reason="CONTROLLER_SELECTION", required_blocks=None,
        free_blocks=42, total_blocks=4681,
    )
    assert result["slackkv_prefetch"]["status"] == "snapshot"
    assert result["slackkv_prefetch"]["snapshot"] == snapshot
    assert result["slackkv_prefetch"]["transfer_bytes"] == 0
    assert result["other"] == 7


def test_actual_full_and_incremental_allocator_requirements_are_separate():
    result = allocator_preflight(pool_free_blocks=1200, inflight_reserved_blocks=300,
                                 num_blocks_to_allocate=800)
    assert result["pool_free_blocks"] == 1200
    assert result["allocator_available_blocks"] == 900
    assert result["allocator_required_blocks"] == 800
    assert result["full_sequence_must_fit"] is False


def test_stale_visibility_classification_is_not_called_performance_error():
    values = selection_visibility("E00:b0", "E04:b0", False, True)
    assert values == {
        "candidate_matches_protected": False,
        "stale_protected_visibility": True,
        "wrong_candidate_due_to_stale_source": True,
    }


def test_missed_self_promotion_requires_timing_and_allocation_gates():
    fields = dict(candidate_id="E00:b0", protected_id="E04:b0",
                  protected_exact_ready=True, self_logical_fit=True,
                  self_psa_fit=True, self_legacy_physical_fit=True,
                  self_exact_allocator_fit=True, timing_permits=True)
    assert missed_protected_self_promotion(**fields)
    for gate in ("self_logical_fit", "self_psa_fit", "self_legacy_physical_fit",
                 "self_exact_allocator_fit", "timing_permits"):
        assert not missed_protected_self_promotion(**{**fields, gate: False})


def test_allocator_capacity_race_is_reported_by_selection_and_attempt_snapshots():
    from experiments.phase4g_c.analysis import classify_capacity_reject
    assert classify_capacity_reject(True, False) == "ADMISSION_RACE"
    assert classify_capacity_reject(False, False) == "STRUCTURAL_CAPACITY_REJECT"
    assert classify_capacity_reject(None, False) == "SELECTION_SNAPSHOT_MISSING"
    assert classify_capacity_reject(True, True) == "NOT_REJECTED"


def test_fresh_matrix_has_four_counterbalanced_pairs():
    from experiments.phase4g_c.run_phase4g_c import RUN_ORDER
    assert len(RUN_ORDER) == 8
    assert [row[0] for row in RUN_ORDER].count(1) == 2
    assert [row[0] for row in RUN_ORDER].count(4) == 2
    assert all({condition for rep, condition in RUN_ORDER if rep == number} == {"C0", "C1"}
               for number in range(1, 5))
    assert len({f"fresh-server-r{rep}-{condition}" for rep, condition in RUN_ORDER}) == 8


@pytest.mark.parametrize(
    ("c1", "expected"),
    [
        ({"candidate_matches_protected_rate": .8, "source_ready_selection_opportunities": 20,
          "psa_blocked_source_ready_fraction": .7, "median_psa_reserve": 900,
          "median_allocator_need_now": 300}, "RESULT_B_PSA1_INTRINSIC_CONSERVATISM"),
        ({"candidate_matches_protected_rate": .8, "exact_allocator_reject_fraction": .7},
         "RESULT_C_ALLOCATOR_CAPACITY_DOMINANT"),
        ({"attempt_count": 10, "false_admit_count": 2, "false_reject_count": 0},
         "RESULT_D_GUARD_ALLOCATOR_MISMATCH"),
    ],
)
def test_result_b_c_d_classifier(c1, expected):
    from experiments.phase4g_c.analysis import classify_matrix_results
    c0 = {"selection_count": 10, "stale_protected_visibility_count": 0,
          "candidate_matches_protected_count": 8, "missed_protected_self_promotion_count": 0,
          "median_psa_reserve": 100, "successful_promotions": 1}
    base = {"selection_count": 10, "candidate_matches_protected_count": 8,
            "median_psa_reserve": 100, "successful_promotions": 1}
    assert expected in classify_matrix_results(c0, {**base, **c1})


def test_phase_has_no_partial_promotion_or_future_phase_invocation():
    from experiments.phase4g_c import controller as controller_module
    source = inspect.getsource(controller_module.SourceRecheckController)
    assert "partial_promotion" not in source
    assert "progressive_promotion" not in source


def test_capacity_fast_reject_and_exact_lease_helpers_remain_referenced():
    root = Path("/media/a6000/D/ZhaoBinHe/SlackKV/experiments")
    capacity = (root / "phase4d_b/capacity.py").read_text(encoding="utf-8")
    release = (root / "phase4d_a_fix/release_fix.py").read_text(encoding="utf-8")
    assert "INSUFFICIENT_SCHEDULABLE_BLOCKS" in capacity or "INSUFFICIENT_FREE_BLOCKS" in capacity
    assert "lease_key" in release and "LEASE_RELEASED" in release


def test_controller_does_not_change_reservation_or_timing_constants():
    from experiments.phase4g_c import controller as controller_module
    from experiments.phase4g_a.controller import DynamicController
    source = inspect.getsource(controller_module.SourceRecheckController)
    assert "latest_feasible_start(" in source
    assert "merge_blackouts(" in source
    assert "self.ledger.reserve(key" in source
    assert "self.ledger.release(key)" in inspect.getsource(DynamicController._harvest)
    assert "BUDGET_BLOCKS" not in source


def test_incomplete_result_normalization_does_not_duplicate_result_code():
    code, detail = normalize_incomplete_result(
        "EXPERIMENT_INVALID_MATRIX:EXPERIMENT_INVALID_MATRIX:measured_rep41_d"
    )
    assert code == "EXPERIMENT_INVALID_MATRIX"
