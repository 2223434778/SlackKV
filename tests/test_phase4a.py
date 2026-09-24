"""Focused offline Phase 4A semantics and causal-order tests."""

import pytest

from experiments.phase4a.boundaries import consecutive_rounds, context_replayable, tool_signature, valid_timing
from experiments.phase4a.predictors import Boundary, online_predictions
from experiments.phase4a.run_phase4a import event_semantics, required_blocks


def test_required_blocks() -> None:
    assert required_blocks(0) == 0
    assert required_blocks(17) == 2
    with pytest.raises(ValueError):
        required_blocks(-1)


def test_consecutive_rounds() -> None:
    assert consecutive_rounds([8, 9, 10])
    assert not consecutive_rounds([8, 10, 11])


def test_history_waits_for_resume_not_earlier_decision() -> None:
    rows = online_predictions([Boundary(0, 200, "bash"), Boundary(50, 100, "read"), Boundary(120, 150, "bash")])
    assert rows[1]["global_prediction_ms"] is None
    assert rows[2]["global_prediction_ms"] == 50
    assert rows[2]["tool_prediction_ms"] == 50
    assert all(row["future_leakage_count"] == 0 for row in rows)


def test_cold_start_and_tool_fallback() -> None:
    rows = online_predictions([Boundary(0, 10, "bash"), Boundary(20, 30, "read")])
    assert rows[0]["global_prediction_ms"] is None
    assert rows[0]["tool_prediction_ms"] is None
    assert rows[1]["tool_prediction_ms"] == rows[1]["global_prediction_ms"] == 10


def test_simultaneous_decisions_do_not_order_by_future_resume() -> None:
    rows = online_predictions([Boundary(0, 200, "a"), Boundary(0, 100, "b")])
    assert [row["actual_slack_ms"] for row in rows] == [200, 100]
    assert all(row["global_prediction_ms"] is None for row in rows)


def test_tool_signature_is_order_independent() -> None:
    assert tool_signature(["Read", "bash", "bash"]) == tool_signature(["bash", "bash", "read"]) == "bashx2+read"


def test_invalid_timing_exclusion() -> None:
    assert not valid_timing(20, 10)
    assert not valid_timing(20, None)
    assert valid_timing(20, 20)


def test_context_shrink_exclusion() -> None:
    assert not context_replayable(4000, 3900)
    assert context_replayable(4000, 4100)


def test_no_tool_result_substitution() -> None:
    observed = [{"event_type": "tool_result", "source": "user.content.tool_result", "count": 1}]
    assert event_semantics(observed)["actual_resume_time"] == "UNOBSERVABLE"


def test_result_count_consistency() -> None:
    rows = online_predictions([Boundary(0, 10, "a"), Boundary(20, 30, "b"), Boundary(40, 50, "a")])
    assert len(rows) == 3
    assert sum(row["global_prediction_ms"] is None for row in rows) == 1
    assert sum(row["future_leakage_count"] for row in rows) == 0
