import pytest

from experiments.autonomous_round4.analyze import paired_effect, schedule, summarize
from experiments.autonomous_round4.resources import release_witness
from experiments.autonomous_round4.audit_disabled import equivalent


def test_frozen_independent_pair_order():
    rows = schedule()
    assert len(rows) == 12
    assert len({r["name"] for r in rows}) == 12
    assert [r["method"] for r in rows[:6]] == ["k2", "off", "off", "k2", "k2", "off"]
    assert [r["method"] for r in rows[6:]] == ["off", "k2", "k2", "off", "off", "k2"]


def fixture_pair():
    config = {k: 0 for k in ("kv_bytes", "source_sha", "execution_mode", "batch_invariant",
                            "allocator_trace", "scheduler_sha256", "allocator_sha256", "bg_tokens",
                            "backfill_limit", "continuation_credit", "working_set_epochs",
                            "gate", "source_directory", "runner_sha256", "replay_sha256", "git_head")}
    metrics = {k: 100 for k in ("session_e2e_median_ms", "resume_p95_ms", "resume_median_ms",
                              "BG_wall_ms", "BG_TPOT_ms", "mixed_generated_tokens_per_second")}
    return [{"config": dict(config, gate_enabled=enabled), "metrics": metrics.copy(),
             "inputs": {"a": "hash"}, "outputs": {"a": 128}} for enabled in (True, False)]


def test_compare_effect_and_fail_closed_on_work_change():
    a, b = fixture_pair()
    b["metrics"]["session_e2e_median_ms"] = 80
    assert paired_effect(a, b)["session_e2e_median_ms"] == pytest.approx(-20)
    b["outputs"]["a"] = 127
    with pytest.raises(ValueError, match="workload"):
        paired_effect(a, b)


def test_source_mismatch_is_not_a_paired_algorithm_result():
    a, b = fixture_pair()
    b["config"]["scheduler_sha256"] = "different"
    with pytest.raises(ValueError, match="configuration"):
        paired_effect(a, b)


def test_incomplete_conditions_are_never_success(tmp_path):
    result = summarize(tmp_path)
    assert result["gpu_hours_conservative"] == 0
    assert all(not c["complete"] and not c["promising"] for c in result["conditions"].values())


def test_shared_blocks_only_free_after_all_owners_complete():
    owners = {"a": [1, 2], "b": [2, 3]}
    assert release_witness(3, 0, owners, {1: 1, 2: 2, 3: 1}) == ["a", "b"]
    assert release_witness(1, 0, owners, {1: 1, 2: 2, 3: 1}) == ["a"]


def test_dma_reference_cannot_be_counted_as_free_after_completion():
    assert release_witness(2, 0, {"a": [1, 2]}, {1: 1, 2: 2}) is None
    assert release_witness(2, 2, {"a": [1]}, {1: 1}) == []
    with pytest.raises(ValueError, match="references"):
        release_witness(2, 0, {"a": [1], "b": [1]}, {1: 1})


def test_disabled_instrumentation_preserves_nested_native_branch():
    original = "def f(connector, ext):\n    if connector:\n        ext = get_ext()\n        if ext is None:\n            return None\n        x = ext\n    return 1\n"
    correct = original.replace("        if ext is None:\n", "        if ext is None:\n            if os.environ.get('SLACKKV_RESOURCE_TRACE') == '1':\n                snapshot()\n")
    assert equivalent(original, correct)
    broken = "def f(connector, ext):\n    if connector:\n        ext = get_ext()\n    if ext is None:\n        if os.environ.get('SLACKKV_RESOURCE_TRACE') == '1':\n            snapshot()\n            return None\n        x = ext\n    return 1\n"
    assert not equivalent(original, broken)
