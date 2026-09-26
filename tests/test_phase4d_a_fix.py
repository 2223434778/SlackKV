import json
import math
from pathlib import Path

import pytest

from experiments.phase4d_a.helpers import ReservationLedger, FROZEN_SHA256, first_boundary_jobs
from experiments.phase4d_a_fix.release_fix import (
    EXPECTED_GPU_BLOCKS,
    EXPECTED_WORKLOAD_SHA256,
    FROZEN_P_SAFE_MS,
    LeaseTraceReader,
    RuntimeReleaseQueue,
    issue_spread_ms,
    issue_spread_valid,
    parse_lease_trace,
    release_order_valid,
    service_envelope,
    smallest_prefix_groups,
    stalled_for_five_seconds,
    validate_stage_name,
    watchdog_classification,
)

ROOT = Path(__file__).resolve().parents[1]
WORKLOAD = ROOT / "results/phase4c/hybrid_workload.csv"


def lease_line(event, key, ts, blocks=1271):
    return "SLACKKV_LEASE_TRACE " + json.dumps({
        "event": event, "lease_key": key, "timestamp_monotonic": ts, "num_blocks": blocks,
    }, separators=(",", ":"))


def make_queue(budget=3510, episodes=("E04",)):
    ledger = ReservationLedger(budget)
    keys = {episode: f"{episode}:turn:1" for episode in episodes}
    blocks = {"E04": 1271, "E00": 1277, "E06": 1276, "E07": 1444, "E03": 1306}
    for episode in episodes:
        assert ledger.reserve(episode, blocks[episode])
    return RuntimeReleaseQueue(ledger, keys, {e: blocks[e] for e in episodes}, 1000.0), ledger


def test_frozen_workload_hash():
    from experiments.phase4d_a.helpers import sha256_file
    assert sha256_file(WORKLOAD) == EXPECTED_WORKLOAD_SHA256 == FROZEN_SHA256


def test_frozen_eight_job_and_totals():
    jobs = first_boundary_jobs(WORKLOAD)
    assert len(jobs) == 8
    assert sum(j["required_prefix_tokens"] for j in jobs) == 168016
    assert sum(j["required_blocks"] for j in jobs) == 10501


def test_runtime_gpu_capacity_is_pinned():
    assert EXPECTED_GPU_BLOCKS == 4681


def test_p_safe_is_pinned():
    assert FROZEN_P_SAFE_MS == 102


def test_exact_episode_to_physical_lease_mapping():
    queue, _ = make_queue()
    assert queue.episode_to_lease_key["E04"] == "E04:turn:1"


def test_regression_old_episode_key_does_not_match_turn_key():
    queue, ledger = make_queue()
    event = parse_lease_trace(lease_line("LEASE_RELEASED", "E04:turn:1", 1000.1))[0]
    # The old episode-ID equality search incorrectly found no physical release.
    assert event["lease_key"] != "E04"
    assert not any(e["event"] == "LEASE_RELEASED" and e["lease_key"] == "E04" for e in [event])
    queue.observe_lease_events([event])
    queue.enqueue_first_token("E04", 1_000_200_000_000)
    released = queue.drain(1_000_250_000_000)
    assert len(released) == 1
    assert released[0]["lease_key"] == "E04:turn:1"
    assert ledger.reserved_blocks == 0


def test_e04_never_matches_e040_turn_key():
    queue, ledger = make_queue()
    event = parse_lease_trace(lease_line("LEASE_RELEASED", "E040:turn:1", 1000.1))[0]
    queue.observe_lease_events([event])
    queue.enqueue_first_token("E04", 1_000_200_000_000)
    assert queue.drain(1_000_250_000_000) == []
    assert ledger.reserved_blocks == 1271


def test_no_physical_release_means_no_logical_release():
    queue, ledger = make_queue()
    queue.enqueue_first_token("E04", 1_000_200_000_000)
    assert queue.drain(1_000_250_000_000) == []
    assert ledger.reserved_blocks == 1271
    assert queue.logical_leaks == 1


def test_physical_after_logical_timestamp_is_rejected():
    assert not release_order_valid(200, 100)
    queue, ledger = make_queue()
    queue.observe_lease_events(parse_lease_trace(lease_line("LEASE_RELEASED", "E04:turn:1", 1000.3)))
    queue.enqueue_first_token("E04", 1_000_200_000_000)
    assert queue.drain(1_000_250_000_000) == []
    assert queue.unsafe_release_count == 1
    assert ledger.reserved_blocks == 1271


def test_valid_physical_before_logical_order():
    assert release_order_valid(100, 101)


def test_duplicate_drain_does_not_double_release():
    queue, ledger = make_queue()
    queue.observe_lease_events(parse_lease_trace(lease_line("LEASE_RELEASED", "E04:turn:1", 1000.1)))
    queue.enqueue_first_token("E04", 1_000_200_000_000)
    assert len(queue.drain(1_000_250_000_000)) == 1
    assert queue.drain(1_000_300_000_000) == []
    assert ledger.reserved_blocks == 0
    assert len(queue.events) == 1


def test_released_capacity_admits_later_job():
    queue, ledger = make_queue(episodes=("E04", "E00"))
    queue.observe_lease_events(parse_lease_trace(
        lease_line("LEASE_RELEASED", "E04:turn:1", 1000.1, 1271) + "\n" +
        lease_line("LEASE_RELEASED", "E00:turn:1", 1000.12, 1277)))
    queue.enqueue_first_token("E04", 1_000_200_000_000)
    queue.enqueue_first_token("E00", 1_000_210_000_000)
    released = queue.drain(1_000_250_000_000)
    assert {row["episode_id"] for row in released} == {"E04", "E00"}
    assert ledger.reserved_blocks == 0
    assert ledger.reserve("E06", 1276)
    assert ledger.reserved_blocks == 1276


def test_final_reservation_and_lease_counts_can_reach_zero():
    queue, ledger = make_queue()
    queue.observe_lease_events(parse_lease_trace(lease_line("LEASE_RELEASED", "E04:turn:1", 1000.1)))
    queue.enqueue_first_token("E04", 1_000_200_000_000)
    queue.drain(1_000_250_000_000)
    assert queue.logical_leaks == 0
    assert not ledger.active


def test_parser_extracts_structured_event_fields():
    event = parse_lease_trace(lease_line("LEASE_RELEASED", "E04:turn:1", 123.456, 1271))[0]
    assert event["event"] == "LEASE_RELEASED"
    assert event["lease_key"] == "E04:turn:1"
    assert event["timestamp_monotonic"] == pytest.approx(123.456)
    assert event["num_blocks"] == 1271


def test_parser_ignores_malformed_json():
    assert parse_lease_trace("SLACKKV_LEASE_TRACE {bad json") == []


def test_parser_ignores_non_lease_events():
    assert parse_lease_trace("ordinary text") == []


def test_smallest_prefix_selection_is_mechanical():
    jobs = first_boundary_jobs(WORKLOAD)
    selected = smallest_prefix_groups(jobs)
    assert selected["episode_ids"] == ["E04", "E05", "E06", "E00"]


def test_smallest_prefix_cumulative_N_counts():
    selected = smallest_prefix_groups(first_boundary_jobs(WORKLOAD))
    assert (selected["N1"], selected["N2"], selected["N3"], selected["N4"]) == (1271, 2545, 3821, 5098)


def test_n3_is_below_gpu_capacity():
    assert smallest_prefix_groups(first_boundary_jobs(WORKLOAD))["N3"] < EXPECTED_GPU_BLOCKS


def test_n4_crosses_gpu_capacity():
    assert EXPECTED_GPU_BLOCKS < smallest_prefix_groups(first_boundary_jobs(WORKLOAD))["N4"]


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"elapsed_s": 19.9, "pending": 1, "server_alive": True, "oom": False, "server_crash": False}, "IN_PROGRESS"),
        ({"elapsed_s": 20, "pending": 1, "server_alive": True, "oom": False, "server_crash": False}, "NON_PROGRESS"),
        ({"elapsed_s": 20, "pending": 0, "server_alive": True, "oom": False, "server_crash": False}, "COMPLETE"),
        ({"elapsed_s": 20, "pending": 1, "server_alive": False, "oom": False, "server_crash": True}, "INVALID_SERVER_FAILURE"),
        ({"elapsed_s": 20, "pending": 1, "server_alive": True, "oom": True, "server_crash": False}, "INVALID_SERVER_FAILURE"),
    ],
)
def test_watchdog_classification(kwargs, expected):
    assert watchdog_classification(**kwargs) == expected


def test_issue_spread_in_ms():
    assert issue_spread_ms([1_000_000, 11_000_000]) == 10
    assert issue_spread_valid(20)
    assert not issue_spread_valid(20.01)


def test_stalled_five_seconds_with_pending_work():
    assert stalled_for_five_seconds(1500, 6500, 1)
    assert not stalled_for_five_seconds(1500, 6499, 1)
    assert not stalled_for_five_seconds(1500, 6500, 0)


def test_service_budget_overrun_is_detected_without_retuning():
    audit = service_envelope([80, 102, 103])
    assert audit["budget_ms"] == 102
    assert audit["overrun_count"] == 1
    assert audit["status"] == "P_SAFE_102_INTEGRATION_ENVELOPE_FAIL"


def test_service_budget_boundary_passes():
    assert service_envelope([102])["status"] == "P_SAFE_102_INTEGRATION_ENVELOPE_PASS"


def test_phase_d_stage_b_is_forbidden():
    validate_stage_name("release-fix")
    validate_stage_name("liveness")
    with pytest.raises(ValueError):
        validate_stage_name("matrix")


@pytest.mark.parametrize("blocked", ["stage-b", "full-matrix", "policy-matrix"])
def test_no_stage_b_alias_is_invocable(blocked):
    with pytest.raises(ValueError):
        validate_stage_name(blocked)


def test_runtime_release_row_reports_exact_physical_then_logical_times():
    queue, _ = make_queue()
    queue.observe_lease_events(parse_lease_trace(lease_line("LEASE_RELEASED", "E04:turn:1", 1000.15)))
    queue.enqueue_first_token("E04", 1_000_200_000_000)
    row = queue.drain(1_000_250_000_000)[0]
    assert row["event"] == "RUNTIME_RELEASE_AFTER_PHYSICAL_LEASE"
    assert row["logical_minus_physical_ms"] == pytest.approx(100)
    assert row["reserved_blocks_before"] == 1271
    assert row["reserved_blocks_after"] == 0


def test_incremental_lease_reader_handles_partial_json_lines(tmp_path):
    path = tmp_path / "server.log"
    encoded = (lease_line("LEASE_RELEASED", "E04:turn:1", 1000.1) + "\n").encode()
    reader = LeaseTraceReader(path)
    path.write_bytes(encoded[:20])
    assert reader.read_events() == []
    with path.open("ab") as stream:
        stream.write(encoded[20:])
        stream.write((lease_line("LEASE_RELEASED", "E00:turn:1", 1000.2) + "\n").encode())
    events = reader.read_events()
    assert [event["lease_key"] for event in events] == ["E04:turn:1", "E00:turn:1"]
    assert reader.read_events() == []
