import math

import pytest

from experiments.phase4b.audit_helpers import (
    DATASET_REVISION,
    UPSTREAM_COMMIT,
    cached_input_tokens,
    consecutive_replayable_runs,
    reconstructed_slack_ms,
    required_blocks,
    reusable_tokens,
    seconds_to_ms,
    source_revision_is_pinned,
    timing_fields_available,
)


def test_session_boundaries_never_merge():
    rows = [
        {"session_id": "s1", "turn": 1, "isl": 10},
        {"session_id": "s2", "turn": 2, "isl": 20},
    ]
    grouped = consecutive_replayable_runs(rows, ("isl",))
    assert len(grouped["s1"]) == len(grouped["s2"]) == 1
    assert grouped["s1"][0][0]["session_id"] == "s1"


def test_turn_order_is_sorted_within_session():
    rows = [{"session_id": "s", "turn": 2, "x": 2}, {"session_id": "s", "turn": 1, "x": 1}]
    runs = consecutive_replayable_runs(rows, ("x",))["s"]
    assert [row["turn"] for row in runs[0]] == [1, 2]


def test_seconds_to_milliseconds():
    assert seconds_to_ms(0.25) == 250


def test_reconstructed_slack_formula():
    assert reconstructed_slack_ms(100.0, 200.0, 100.350) == pytest.approx(150)


def test_negative_slack_is_not_clipped():
    assert reconstructed_slack_ms(10.0, 101.0, 10.095) == pytest.approx(-6)


def test_isl_new_must_not_exceed_isl():
    assert cached_input_tokens(100, 30) == 70
    with pytest.raises(ValueError):
        cached_input_tokens(10, 11)


def test_reusable_tokens_are_conservative_minimum():
    assert reusable_tokens(100, 20, 140, 50) == 90


def test_required_blocks_rounds_up_by_sixteen():
    assert required_blocks(17) == 2
    assert required_blocks(0) == 0


def test_missing_e2e_means_no_timing_pair():
    assert not timing_fields_available({"request_time"})


def test_replayable_runs_split_on_missing_rows_and_turn_gaps():
    rows = [
        {"session_id": "s", "turn": 1, "x": 1},
        {"session_id": "s", "turn": 2, "x": None},
        {"session_id": "s", "turn": 3, "x": 3},
        {"session_id": "s", "turn": 5, "x": 5},
    ]
    runs = consecutive_replayable_runs(rows, ("x",))["s"]
    assert [[row["turn"] for row in run] for run in runs] == [[1], [3], [5]]


def test_deterministic_grouping():
    rows = [{"session_id": "s", "turn": n, "x": n} for n in (3, 1, 2)]
    assert consecutive_replayable_runs(rows, ("x",)) == consecutive_replayable_runs(rows, ("x",))


def test_revision_pinning_is_exact():
    assert source_revision_is_pinned(UPSTREAM_COMMIT)
    assert source_revision_is_pinned(DATASET_REVISION, DATASET_REVISION)
    assert not source_revision_is_pinned("main")
