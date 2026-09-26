from __future__ import annotations

import hashlib
import json
from concurrent.futures import Future
from pathlib import Path

import pytest

from experiments.phase4d_a.helpers import (
    FROZEN_SHA256,
    ReservationLedger,
    gate_valid,
    source_conserved,
)
from experiments.phase4d_a_fix.release_fix import (
    EXPECTED_GPU_BLOCKS,
    FROZEN_P_SAFE_MS,
    RuntimeReleaseQueue,
    parse_lease_trace,
    smallest_prefix_groups,
)
from experiments.phase4d_b.capacity import (
    capacity_decision,
    capacity_invariant,
    rejected_request_lifecycle_valid,
    parse_capacity_trace,
    parse_prefetch_response,
    result_classification,
    unique_block_ids,
    validate_capacity_rejection,
    validate_mode,
    validate_required_blocks,
    watchdog_nonprogress,
)
from experiments.phase4d_b.capacity_client import build_payload
from experiments.phase4d_b.integration import FinalDrainReleaseQueue
from experiments.phase4d_b.controller import harvest_completed_promotions

ROOT = Path(__file__).resolve().parents[1]
JOBS_PATH = ROOT / "results/phase4d_a/boundary0_jobs.csv"
WORKLOAD_PATH = ROOT / "results/phase4c/hybrid_workload.csv"
EXPECTED_PROMOTIONS = {
    "status": "capacity_rejected",
    "reason": "INSUFFICIENT_FREE_BLOCKS",
    "required_blocks": 1277,
    "free_blocks": 860,
    "total_blocks": 4681,
    "external_tokens_loaded": 0,
    "transfer_bytes": 0,
    "model_forward_count": 0,
    "generated_token_count": 0,
    "mixed_prefetch_model_step_count": 0,
}


def load_jobs():
    import csv

    with JOBS_PATH.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_frozen_workload_hash():
    assert hashlib.sha256(WORKLOAD_PATH.read_bytes()).hexdigest() == FROZEN_SHA256


def test_eight_jobs_and_expected_totals():
    jobs = load_jobs()
    assert len(jobs) == 8
    assert sum(int(row["required_prefix_tokens"]) for row in jobs) == 168016
    assert sum(int(row["required_blocks"]) for row in jobs) == 10501


def test_expected_gpu_capacity_and_budget():
    assert EXPECTED_GPU_BLOCKS == 4681
    assert int(0.75 * EXPECTED_GPU_BLOCKS) == 3510


def test_frozen_service_budget():
    assert FROZEN_P_SAFE_MS == 102


def test_minimum_transfer_mib():
    assert sum(int(row["required_prefix_tokens"]) for row in load_jobs()) * 28672 / 2**20 == 4594.1875


@pytest.mark.parametrize(
    ("prompt_tokens", "required", "valid"),
    [(16 * n, n, True) for n in range(1, 13)]
    + [(16 * n + 1, n, False) for n in range(1, 13)]
    + [(16 * n, n + 1, False) for n in range(1, 13)],
)
def test_required_block_validation(prompt_tokens, required, valid):
    if valid:
        assert validate_required_blocks(prompt_tokens, required) == required
    else:
        with pytest.raises(ValueError, match="INVALID_CAPACITY_GUARD_REQUEST"):
            validate_required_blocks(prompt_tokens, required)


def test_required_blocks_reject_bool():
    with pytest.raises(ValueError):
        validate_required_blocks(16, True)


def test_guard_disabled_payload_preserves_historical_shape():
    assert build_payload([1, 2], capacity_guard=False) == {
        "prompt_token_ids": [1, 2]
    }


def test_guarded_payload_contains_only_requested_metadata():
    assert build_payload(
        [1] * 32, capacity_guard=True, required_blocks=2, lease_key="E06:turn:1"
    ) == {
        "prompt_token_ids": [1] * 32,
        "capacity_guard": True,
        "required_blocks": 2,
        "lease_key": "E06:turn:1",
    }


@pytest.mark.parametrize(("free", "required", "decision"), [(20, 20, "ADMIT"), (21, 20, "ADMIT"), (19, 20, "REJECT")])
def test_capacity_admit_iff_free_is_sufficient(free, required, decision):
    assert capacity_decision(free, required) == decision
    assert capacity_invariant(
        {"free_blocks_before": free, "required_blocks": required, "decision": decision}
    )


@pytest.mark.parametrize(
    "decision,free,expected",
    [("ADMIT", 1276, True), ("ADMIT", 1275, False), ("REJECT", 1275, True), ("REJECT", 1276, False)],
)
def test_capacity_trace_invariant(decision, free, expected):
    assert capacity_invariant(
        {"free_blocks_before": free, "required_blocks": 1276, "decision": decision}
    ) is expected


def test_reject_response_schema_is_zero_transfer_and_zero_compute():
    assert validate_capacity_rejection(EXPECTED_PROMOTIONS)
    assert parse_prefetch_response(EXPECTED_PROMOTIONS) == "capacity_rejected"


@pytest.mark.parametrize(
    "mutation",
    [
        {"external_tokens_loaded": 1},
        {"transfer_bytes": 1},
        {"model_forward_count": 1},
        {"generated_token_count": 1},
        {"mixed_prefetch_model_step_count": 1},
        {"free_blocks": 1277},
        {"status": "ok"},
    ],
)
def test_reject_schema_detects_invalid_lifecycle(mutation):
    invalid = {**EXPECTED_PROMOTIONS, **mutation}
    assert not validate_capacity_rejection(invalid)


def test_ok_prefetch_parser():
    assert parse_prefetch_response({"status": "ok"}) == "ok"


@pytest.mark.parametrize("status", ["failed", "timeout", "invalid_capacity_guard_request", None])
def test_prefetch_parser_fails_unknown_status(status):
    with pytest.raises(ValueError):
        parse_prefetch_response({"status": status})


def test_structured_capacity_trace_parser():
    body = "INFO SLACKKV_CAPACITY_TRACE " + json.dumps({
        "event": "CAPACITY_GUARD",
        "decision": "ADMIT",
        "required_blocks": 1276,
        "free_blocks_before": 2136,
    })
    events = parse_capacity_trace(body)
    assert len(events) == 1 and capacity_invariant(events[0])


def test_capacity_trace_ignores_bad_json_and_other_events():
    assert parse_capacity_trace(
        "SLACKKV_CAPACITY_TRACE {bad}\nSLACKKV_CAPACITY_TRACE " + json.dumps({"event": "OTHER"})
    ) == []


def test_block_ownership_deduplicates_ids():
    assert unique_block_ids([[1, 2, 2], [2, 3]]) == {1, 2, 3}


def test_source_conservation():
    assert source_conserved(1024, 166992, 168016)
    assert not source_conserved(1024, 166991, 168016)


def test_k2_limit():
    assert gate_valid(2, 2)
    assert not gate_valid(3, 2)


def test_five_second_prefetch_watchdog():
    assert watchdog_nonprogress(elapsed_seconds=5, pending=1, server_alive=True, oom=False, server_crash=False)
    assert not watchdog_nonprogress(elapsed_seconds=4.99, pending=1, server_alive=True, oom=False, server_crash=False)
    assert not watchdog_nonprogress(elapsed_seconds=10, pending=1, server_alive=False, oom=False, server_crash=False)
    assert not watchdog_nonprogress(elapsed_seconds=10, pending=1, server_alive=True, oom=True, server_crash=False)
    assert not watchdog_nonprogress(elapsed_seconds=10, pending=0, server_alive=True, oom=False, server_crash=False)


def test_mode_prohibits_performance_stage():
    for mode in ("g0", "g1", "g2-smoke", "g2-repeat"):
        validate_mode(mode)
    with pytest.raises(ValueError):
        validate_mode("stage-b")


def test_smallest_prefixes_cross_physical_capacity():
    groups = smallest_prefix_groups(load_jobs())
    assert groups["episode_ids"] == ["E04", "E05", "E06", "E00"]
    assert [groups[f"N{i}"] for i in range(1, 5)] == [1271, 2545, 3821, 5098]
    assert groups["N3"] < EXPECTED_GPU_BLOCKS < groups["N4"]


def test_exact_physical_lease_key_not_prefix_match():
    events = parse_lease_trace(
        'SLACKKV_LEASE_TRACE {"event":"LEASE_RELEASED","lease_key":"E04:turn:1","timestamp_monotonic":1.0,"num_blocks":1271}\n'
        'SLACKKV_LEASE_TRACE {"event":"LEASE_RELEASED","lease_key":"E040:turn:1","timestamp_monotonic":1.0,"num_blocks":1271}'
    )
    queue = RuntimeReleaseQueue(
        ReservationLedger(3510),
        {"E04": "E04:turn:1"},
        {"E04": 1271},
        0.0,
    )
    queue.ledger.reserve("E04", 1271)
    queue.enqueue_first_token("E04", 1_100_000_000)
    queue.observe_lease_events(events)
    assert queue.drain(1_200_000_000)[0]["lease_key"] == "E04:turn:1"


def test_capacity_reject_releases_logical_reservation_once():
    ledger = ReservationLedger(3510)
    assert ledger.reserve("E00", 1277)
    item = {
        "future": Future(),
        "lease_key": "E00:turn:1",
        "required_blocks": 1277,
    }
    item["future"].set_result(EXPECTED_PROMOTIONS)
    promotions = {"E00": item}
    states = {"E00": {"promoted": True, "leased": True}}
    events = []
    first = harvest_completed_promotions(
        promotions, states, ledger, events, now_monotonic=3.0
    )
    second = harvest_completed_promotions(
        promotions, states, ledger, events, now_monotonic=4.0
    )
    assert len(first) == 1 and second == []
    assert ledger.reserved_blocks == 0
    assert states["E00"]["promoted"] is False
    assert states["E00"]["leased"] is False
    assert len(events) == 1 and events[0]["event"] == "CAPACITY_REJECT_RELEASE"


def test_successful_promotion_keeps_logical_reservation():
    ledger = ReservationLedger(3510)
    assert ledger.reserve("E04", 1271)
    future = Future()
    future.set_result({"status": "ok"})
    item = {"future": future, "lease_key": "E04:turn:1", "required_blocks": 1271}
    events = []
    state = {"E04": {"leased": False}}
    harvest_completed_promotions(
        {"E04": item}, state, ledger, events, now_monotonic=5.0
    )
    assert ledger.reserved_blocks == 1271
    assert events[0]["event"] == "PROMOTION_PHYSICALLY_ADMITTED"
    assert state["E04"]["leased"] is True


def test_final_drain_sees_unqueued_live_reservations():
    ledger = ReservationLedger(3510)
    assert ledger.reserve("E03", 1306)
    queue = FinalDrainReleaseQueue(ledger, {"E03": "E03:turn:1"}, {"E03": 1306}, 0.0)
    assert queue.pending_release_count == 1
    queue.enqueue_first_token("E03", 1_000_000_000)
    queue.observe_lease_events(
        [{"event": "LEASE_RELEASED", "lease_key": "E03:turn:1", "timestamp_monotonic": 0.9}]
    )
    assert len(queue.drain(1_100_000_000)) == 1
    assert queue.pending_release_count == 0
    assert ledger.reserved_blocks == 0


    ledger = ReservationLedger(3510)
    future = Future()
    future.set_exception(TimeoutError("native endpoint timeout"))
    with pytest.raises(RuntimeError, match="native promotion failed"):
        harvest_completed_promotions(
            {"E04": {"future": future}},
            {"E04": {}},
            ledger,
            [],
            now_monotonic=5.0,
        )



def test_rejected_promotion_has_no_wait_lease_or_connector_side_effect():
    event = {"request_id": "prefetch-rejected-1", "lease_key": "E00:turn:1"}
    assert rejected_request_lifecycle_valid(
        event,
        EXPECTED_PROMOTIONS,
        [],
        [],
        [],
    )
    assert not rejected_request_lifecycle_valid(
        event, EXPECTED_PROMOTIONS,
        [{"request_id": "prefetch-rejected-1", "event": "WAITING_FOR_REMOTE_KVS"}], [], [],
    )

@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"g1_safe_pass": True, "g1_unsafe_pass": True, "smoke_pass": True, "repetitions_pass": True, "integration_reject_count": 1}, "RESULT_A_PHYSICAL_HEADROOM_RESTORES_LIVENESS"),
        ({"g1_safe_pass": True, "g1_unsafe_pass": True, "smoke_pass": True, "repetitions_pass": True, "integration_reject_count": 0}, "RESULT_B_GUARD_CORRECT_INTEGRATION_STABLE_NO_REAL_REJECTION"),
        ({"g1_safe_pass": True, "g1_unsafe_pass": True, "smoke_pass": False, "repetitions_pass": False, "integration_reject_count": 1}, "RESULT_C_GUARD_WORKS_BUT_LIVENESS_FAILURE_REMAINS"),
        ({"g1_safe_pass": False, "g1_unsafe_pass": True, "smoke_pass": False, "repetitions_pass": False, "integration_reject_count": 1}, "RESULT_D_PHYSICAL_GUARD_INVALID"),
    ],
)
def test_frozen_result_classification(kwargs, expected):
    assert result_classification(**kwargs) == expected


def test_phase3_native_client_source_unchanged():
    path = ROOT / "experiments/phase3f/native_client.py"
    assert path.exists()
    assert "capacity_rejected" not in path.read_text(encoding="utf-8")


def test_exact_lease_parser_requires_all_fields():
    good = parse_lease_trace(
        'SLACKKV_LEASE_TRACE {"event":"LEASE_RELEASED","lease_key":"E04:turn:1","timestamp_monotonic":1.25,"num_blocks":1271}'
    )
    bad = parse_lease_trace(
        'SLACKKV_LEASE_TRACE {"event":"LEASE_RELEASED","lease_key":"E04:turn:1"}'
    )
    assert len(good) == 1 and bad == []
def test_g0_guard_disabled_route_request_shape_and_status():
    from vllm.entrypoints.serve.dev.slackkv_prefetch.api_router import (
        SlackKVPrefetchRequest,
        validate_capacity_guard_request,
        validate_prefetch_status,
    )

    request = SlackKVPrefetchRequest(prompt_token_ids=[1, 2])
    validate_capacity_guard_request(
        request.prompt_token_ids, request.capacity_guard, request.required_blocks
    )
    validate_prefetch_status("ok", False)
    with pytest.raises(RuntimeError):
        validate_prefetch_status("capacity_rejected", False)


@pytest.mark.parametrize(
    ("tokens", "required", "valid"),
    [([1] * 16, 1, True), ([1] * 32, 2, True), ([1] * 15, 1, False), ([1] * 32, 1, False), ([1] * 16, None, False)],
)
def test_g0_endpoint_rejects_invalid_guard_metadata(tokens, required, valid):
    from fastapi import HTTPException
    from vllm.entrypoints.serve.dev.slackkv_prefetch.api_router import (
        validate_capacity_guard_request,
    )

    if valid:
        validate_capacity_guard_request(tokens, True, required)
    else:
        with pytest.raises(HTTPException) as exc:
            validate_capacity_guard_request(tokens, True, required)
        assert exc.value.status_code == 422
        assert exc.value.detail == "INVALID_CAPACITY_GUARD_REQUEST"


def test_g0_endpoint_status_parser_accepts_guarded_rejection_only():
    from vllm.entrypoints.serve.dev.slackkv_prefetch.api_router import (
        validate_prefetch_status,
    )

    validate_prefetch_status("ok", False)
    validate_prefetch_status("capacity_rejected", True)
    with pytest.raises(RuntimeError):
        validate_prefetch_status("capacity_rejected", False)
def test_g0_endpoint_keeps_legacy_status_contract():
    from vllm.entrypoints.serve.dev.slackkv_prefetch.api_router import (
        validate_prefetch_status,
    )

    validate_prefetch_status("ok", False)
    with pytest.raises(RuntimeError):
        validate_prefetch_status("capacity_rejected", False)
    validate_prefetch_status("capacity_rejected", True)
