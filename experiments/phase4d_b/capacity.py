from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

CAPACITY_TRACE_PREFIX = "SLACKKV_CAPACITY_TRACE "
ALLOWED_PREFETCH_STATUSES = {"ok", "capacity_rejected"}
ALLOWED_MODES = {"g0", "g1", "g2-smoke", "g2-repeat"}


def validate_required_blocks(prompt_tokens: int, required_blocks: object, block_size: int = 16) -> int:
    if (
        isinstance(required_blocks, bool)
        or not isinstance(required_blocks, int)
        or required_blocks <= 0
        or block_size <= 0
        or prompt_tokens <= 0
        or prompt_tokens % block_size
        or required_blocks != prompt_tokens // block_size
    ):
        raise ValueError("INVALID_CAPACITY_GUARD_REQUEST")
    return required_blocks


def capacity_decision(free_blocks: int, required_blocks: int) -> str:
    if free_blocks < 0 or required_blocks <= 0:
        raise ValueError("invalid block counts")
    return "ADMIT" if free_blocks >= required_blocks else "REJECT"


def capacity_invariant(event: Mapping[str, Any]) -> bool:
    try:
        free = int(event["free_blocks_before"])
        required = int(event["required_blocks"])
        decision = str(event["decision"])
    except (KeyError, TypeError, ValueError):
        return False
    return (decision == "ADMIT" and free >= required) or (
        decision == "REJECT" and free < required
    )


def parse_capacity_trace(text: str) -> list[dict[str, Any]]:
    events = []
    for line in text.splitlines():
        offset = line.find(CAPACITY_TRACE_PREFIX)
        if offset < 0:
            continue
        try:
            event = json.loads(line[offset + len(CAPACITY_TRACE_PREFIX) :])
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("event") == "CAPACITY_GUARD":
            events.append(event)
    return events


def unique_block_ids(groups: Iterable[Iterable[int]]) -> set[int]:
    return {int(block_id) for group in groups for block_id in group}


def validate_capacity_rejection(response: Mapping[str, Any]) -> bool:
    try:
        return (
            response.get("status") == "capacity_rejected"
            and response.get("reason") == "INSUFFICIENT_FREE_BLOCKS"
            and int(response["required_blocks"]) > int(response["free_blocks"])
            and int(response["total_blocks"]) > 0
            and int(response.get("external_tokens_loaded", -1)) == 0
            and int(response.get("transfer_bytes", -1)) == 0
            and int(response.get("model_forward_count", -1)) == 0
            and int(response.get("generated_token_count", -1)) == 0
            and int(response.get("mixed_prefetch_model_step_count", -1)) == 0
        )
    except (KeyError, TypeError, ValueError):
        return False


def parse_prefetch_response(response: Mapping[str, Any]) -> str:
    status = response.get("status")
    if status not in ALLOWED_PREFETCH_STATUSES:
        raise ValueError(f"unsupported capacity prefetch status: {status!r}")
    if status == "capacity_rejected" and not validate_capacity_rejection(response):
        raise ValueError("malformed capacity_rejected response")
    return str(status)


def watchdog_nonprogress(
    *, elapsed_seconds: float, pending: int, server_alive: bool, oom: bool, server_crash: bool
) -> bool:
    return elapsed_seconds >= 5 and pending > 0 and server_alive and not oom and not server_crash



def rejected_request_lifecycle_valid(
    event: Mapping[str, Any],
    response: Mapping[str, Any],
    scheduler_trace: Iterable[Mapping[str, Any]],
    lease_events: Iterable[Mapping[str, Any]],
    connector_only_lines: Iterable[str],
) -> bool:
    request_id = str(event.get("request_id") or "")
    if not request_id or not validate_capacity_rejection(response):
        return False
    return (
        not any(
            row.get("request_id") == request_id
            and row.get("event") == "WAITING_FOR_REMOTE_KVS"
            for row in scheduler_trace
        )
        and not any(
            row.get("request_id") == request_id
            and row.get("event") == "LEASE_ACQUIRED"
            for row in lease_events
        )
        and not any(
            "SLACKKV_CONNECTOR_ONLY_STEP" in line and request_id in line
            for line in connector_only_lines
        )
    )

def validate_mode(mode: str) -> None:
    if mode not in ALLOWED_MODES:
        raise ValueError("Phase 4D-B has no performance-matrix stage")


def result_classification(
    *, g1_safe_pass: bool, g1_unsafe_pass: bool, smoke_pass: bool,
    repetitions_pass: bool, integration_reject_count: int
) -> str:
    if not (g1_safe_pass and g1_unsafe_pass):
        return "RESULT_D_PHYSICAL_GUARD_INVALID"
    if not smoke_pass or not repetitions_pass:
        return "RESULT_C_GUARD_WORKS_BUT_LIVENESS_FAILURE_REMAINS"
    if integration_reject_count == 0:
        return "RESULT_B_GUARD_CORRECT_INTEGRATION_STABLE_NO_REAL_REJECTION"
    return "RESULT_A_PHYSICAL_HEADROOM_RESTORES_LIVENESS"
