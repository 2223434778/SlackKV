from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.phase4g_a.controller import BUDGET_BLOCKS, P_SAFE_MS, RESUME_GUARD_MS
from experiments.phase4g_a.workload import SOURCE, SOURCE_SHA, build_manifest, load_source
from experiments.phase4g_b.analysis import FROZEN_RUN_ORDER, validate_matrix
from experiments.phase4g_b_fix import run
from experiments.phase4g_b_fix.audit import _delta_patch


def test_frozen_workload_and_runtime_parameters():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA
    assert len(load_source()) == 40
    assert len(build_manifest(load_source())) == 32
    assert (P_SAFE_MS, RESUME_GUARD_MS, BUDGET_BLOCKS) == (102.0, 110.0, 3510)


def test_frozen_matrix_is_reused_without_added_policy():
    assert run.original.RUN_ORDER is FROZEN_RUN_ORDER
    assert len(FROZEN_RUN_ORDER) == 12
    assert {policy for _, policy, _ in FROZEN_RUN_ORDER} == {"A", "D"}


def test_old_phase_results_are_not_the_new_destination():
    assert run.OUT.name == "phase4g_b_fix"
    assert run.OUT != run.ROOT / "results/phase4g_b"


def test_pre_fix_alloc_failure_is_parsed_and_reservation_subtracted(tmp_path: Path):
    path = tmp_path / "diagnostic.log"
    event = {"event": "ALLOCATE_NONE", "free_blocks": 1564,
             "reserved_blocks": 759, "local_tokens": 21264,
             "external_tokens": 16, "load_kv_async": True,
             "blocked_count": 1}
    path.write_text("prefix SLACKKV_PREFETCH_ALLOC_TRACE " + json.dumps(event) + "\n")
    evidence = run._pre_fix_evidence(path)
    assert evidence["rate_limited_events"] == 1
    assert evidence["schedulable_blocks_at_first"] == 805
    assert evidence["first"]["external_tokens"] == 16


def test_missing_alloc_failure_does_not_invent_root_cause(tmp_path: Path):
    path = tmp_path / "empty.log"
    path.write_text("no allocation trace\n")
    with pytest.raises(RuntimeError, match="did not capture"):
        run._pre_fix_evidence(path)


def test_partial_matrix_cannot_be_aggregated():
    with pytest.raises(ValueError, match="partial or unbalanced"):
        validate_matrix([{"condition": "M0", "valid": True}])


def test_fallback_does_not_alter_frozen_workload_or_old_result():
    assert run.START_COMMIT == "0a32b037641449462699cbe1dc7edfb199369a4d"
    assert (run.ROOT / "results/phase4g_b/summary.md").is_file()
    assert "INCOMPLETE_INVALID_RUN" in (run.ROOT / "results/phase4g_b/summary.md").read_text()


def test_native_alloc_failure_repair_is_guarded_and_before_waiting_break():
    scheduler = (run.original.VLLM_SOURCE / "vllm/v1/core/sched/scheduler.py").read_text()
    branch = scheduler.split("if new_blocks is None:", 1)[1].split(
        "if _is_slackkv_trace_resume(request):", 1
    )[0]
    assert "_slackkv_reject_blocked_native_prefetch(request)" in branch
    assert "INSUFFICIENT_SCHEDULABLE_BLOCKS" in branch
    assert "_slackkv_finish_capacity_guard_request" in branch
    assert "continue" in branch


def test_normal_and_unguarded_prefetch_are_excluded_from_repair():
    source = (run.original.VLLM_SOURCE / "vllm/v1/core/sched/scheduler.py").read_text()
    helper = source.split("def _slackkv_reject_blocked_native_prefetch", 1)[1].split("\n\n", 1)[0]
    assert "_is_slackkv_prefetch_only(request)" in helper
    assert "_is_slackkv_capacity_guarded(request)" in helper


def test_rejection_helper_retains_zero_transfer_and_zero_model_work():
    source = (run.original.VLLM_SOURCE / "vllm/v1/core/sched/scheduler.py").read_text()
    helper = source.split("def _slackkv_finish_capacity_guard_request", 1)[1].split(
        "def _slackkv_preflight_capacity_guards", 1
    )[0]
    for field in ("external_tokens_loaded", "transfer_bytes", "model_forward_count",
                  "generated_token_count", "mixed_prefetch_model_step_count"):
        assert f'"{field}": 0' in helper


def test_incremental_vllm_patch_is_limited_to_guarded_failure_and_test():
    patch = _delta_patch()
    assert "INSUFFICIENT_SCHEDULABLE_BLOCKS" in patch
    assert "_slackkv_reject_blocked_native_prefetch" in patch
    assert patch.count("--- a/") == 2
    assert "vllm/v1/core/sched/scheduler.py" in patch
    assert "test_slackkv_connector_first.py" in patch
