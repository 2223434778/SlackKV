import pytest

from experiments.autonomous_fork.model import bytes_per_token, fork_footprint, streaming_cost
from experiments.autonomous_fork.analyze import analyze, compare, counter, remote_intervals, validate_inputs, critical_path
import hashlib
import json


def config(kv_heads=2):
    return dict(num_attention_heads=12, hidden_size=1536,
                num_hidden_layers=28, num_key_value_heads=kv_heads)


def test_model_agrees_with_pinned_allocator_layout():
    assert bytes_per_token(config()) == 28672
    assert bytes_per_token(config(4)) == 57344


def test_sharing_and_copy_opportunity_account_for_null_and_private_blocks():
    result = fork_footprint(config(), 24576, 256, 128, 8)
    assert result["duplicate_upper_bytes"] - result["unique_bytes"] == result["avoided_copy_upper_bytes"]
    assert result["unique_bytes"] == (1 + 1536 + 8 * 24) * 458752


def test_single_branch_has_no_sharing_opportunity():
    result = fork_footprint(config(), 24576, 256, 128, 1)
    assert result["avoided_copy_upper_bytes"] == 0


def test_wrong_model_or_dimensions_are_rejected():
    with pytest.raises(ValueError):
        bytes_per_token(dict(config(), use_sliding_window=True))
    with pytest.raises(ValueError):
        fork_footprint(config(), 0, 256, 128, 8)


def test_streaming_cannot_count_sharing_as_eight_transfers():
    result = streaming_cost(config(), 24576, 256, 26.279616)
    assert result["root_only_bytes"] == 180388626432
    assert result["dma_seconds_at_reference_bandwidth"] == pytest.approx(6.864204784)
    with pytest.raises(ValueError):
        streaming_cost(config(), 24576, 256, 0)


def test_absent_counter_is_unknown_and_zero_is_observed():
    assert counter("# HELP h help\nh{x=\"y\"} 0\n", "missing") is None
    assert counter('h{x="y"} 0\n', "h", 'x="y"') == 0
    assert counter('h{x="y"} 7\nh{x="z"} 11\n', "h", 'x="y"') == 7


def test_remote_waits_require_matching_identity_and_complete_intervals():
    start = dict(request_id="a", session="F0:0:0", timestamp_monotonic=1,
                 event="WAITING_FOR_REMOTE_KVS")
    ready = dict(start, timestamp_monotonic=1.1, event="REMOTE_KV_READY")
    assert remote_intervals([start, ready])[0]["elapsed_ms"] == pytest.approx(100)
    for trace in ([start], [ready], [start, start, ready], [start, dict(ready, timestamp_monotonic=.9)]):
        with pytest.raises(ValueError):
            remote_intervals(trace)


def test_comparison_rejects_changed_inputs_or_regimes_and_unknown_transfer():
    baseline = dict(name="a", inputs={"x": "hash"}, outputs={"x": "hash"},
                    kv_gib=4, gate=8, background_phase="decode", batch_invariant=False,
                    workflow_median_ms=100, branch_ttft_p95_ms=100, bg_median_ms=100,
                    h2d_bytes_actual=0)
    candidate = dict(baseline, name="b", workflow_median_ms=80)
    assert compare(baseline, candidate)["performance_screen_pass"]
    assert not compare(baseline, dict(candidate, h2d_bytes_actual=None))["performance_screen_pass"]
    assert not compare(baseline, dict(candidate, h2d_bytes_actual=10))["performance_screen_pass"]
    for field, value in (("inputs", {}), ("kv_gib", 8), ("background_phase", "cold")):
        with pytest.raises(ValueError):
            compare(baseline, dict(candidate, **{field: value}))


def test_input_hash_cannot_hide_different_actual_prompt_or_duplicate():
    row = dict(session="F0:0:0", prompt=[1, 2])
    recorded = dict(session=row["session"], prompt_sha256=hashlib.sha256(json.dumps(row["prompt"]).encode()).hexdigest())
    validate_inputs([row], [recorded])
    for rows in ([dict(row, prompt=[1, 3])], [row, row]):
        with pytest.raises(ValueError):
            validate_inputs(rows, [recorded])


def test_critical_branch_includes_tool_wait_and_issue_offset():
    rows = [dict(session="F0:0:0", issue_ns=0, first_ns=10, done_ns=30),
            dict(session="F0:0:1", issue_ns=40, first_ns=50, done_ns=60),
            dict(session="F0:1:0", issue_ns=5, first_ns=20, done_ns=50),
            dict(session="F0:1:1", issue_ns=70, first_ns=80, done_ns=100)]
    result = critical_path(rows, 0)
    assert result["last_branch"] == "F0:1:1"
    assert sum(result["components_ms"].values()) == pytest.approx(result["total_ms"])
    assert result["components_ms"]["tool_and_client_gap"] == 20 / 1e6


def test_live_budget_cannot_be_silently_treated_as_zero(tmp_path):
    (tmp_path / "execution.json").write_text(json.dumps(dict(status="RUNNING")))
    with pytest.raises(ValueError, match="finalized"):
        analyze(tmp_path)


def test_oracle_requires_explicit_separate_invariant_mode(monkeypatch):
    from experiments.autonomous_fork.run import main
    monkeypatch.setattr("sys.argv", ["run", "--name", "oracle", "--oracle"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
