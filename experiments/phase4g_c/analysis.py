"""Pure Phase 4G-C source-visibility and allocator-truth analysis helpers."""
from __future__ import annotations

from statistics import median
from typing import Any

SOURCE_SHA = "9b381336a70e73b0efbddbb79d2c88608c19663247e819c1e7a364efd2971105"
START_COMMIT = "c1761e980b16eec5d5381e8cf0a6edd4d245bf09"
GPU_BLOCKS = 4681
P_SAFE_MS = 102
RESUME_GUARD_MS = 110
K2 = 2
RESERVATION_BUDGET_BLOCKS = 3510
RECHECK_INTERVAL_MS = 20


def full_ready(sample: dict[str, Any]) -> bool:
    required = int(sample.get("required_blocks", 0))
    covered = int(sample.get("covered_blocks", 0))
    return required > 0 and covered >= required


def prompt_only_externalizable_tokens(
    full_prefix_tokens: int, previous_prompt_tokens: int, block_size: int = 16,
) -> int:
    if full_prefix_tokens < 0 or previous_prompt_tokens < 0 or block_size <= 0:
        raise ValueError("token lengths must be non-negative and block size positive")
    return (min(full_prefix_tokens, previous_prompt_tokens) // block_size) * block_size


def source_coverage_diagnostic(
    *, full_prefix_tokens: int, previous_prompt_tokens: int,
    local_source_tokens: int, external_source_tokens: int, block_size: int = 16,
) -> dict[str, int | bool]:
    if min(full_prefix_tokens, previous_prompt_tokens, local_source_tokens,
           external_source_tokens) < 0:
        raise ValueError("token counts must be non-negative")
    prompt_only_tokens = prompt_only_externalizable_tokens(
        full_prefix_tokens, previous_prompt_tokens, block_size,
    )
    actual = local_source_tokens + external_source_tokens
    shortfall = max(0, full_prefix_tokens - actual)
    decode_tail = full_prefix_tokens - prompt_only_tokens
    explained = min(shortfall, decode_tail)
    return {
        "full_prefix_tokens": full_prefix_tokens,
        "previous_prompt_tokens": previous_prompt_tokens,
        "prompt_only_externalizable_tokens": prompt_only_tokens,
        "prompt_only_decode_tail_tokens": decode_tail,
        "local_source_tokens": local_source_tokens,
        "external_source_tokens": external_source_tokens,
        "actual_source_hit_tokens": actual,
        "full_prefix_source_shortfall_tokens": shortfall,
        "prompt_only_explained_shortfall_tokens": explained,
        "unexplained_source_shortfall_tokens": shortfall - explained,
        "full_prefix_source_hit_exact": actual == full_prefix_tokens,
    }


def unique_source_trace_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Keep the first successful source-accounting event for each real request.

    The pinned scheduler can revisit a request while its computed-token count
    is still zero. Those trace rows are repeated observations, not additional
    source tokens, so raw rows remain available separately for auditing.
    """
    ordered = sorted(rows, key=lambda row: int(row.get("timestamp_monotonic_ns", 0))
                     if row.get("timestamp_monotonic_ns") is not None
                     else round(float(row.get("timestamp_monotonic", 0)) * 1e9))
    by_request: dict[str, dict[str, Any]] = {}
    for row in ordered:
        request_id = str(row.get("request_id", "")).strip()
        session = str(row.get("session", "")).strip()
        if not request_id or not session:
            raise ValueError("source trace rows require request_id and session")
        previous = by_request.get(request_id)
        if previous is not None and str(previous.get("session")) != session:
            raise ValueError("one source request_id cannot identify multiple sessions")
        by_request.setdefault(request_id, row)
    unique = list(by_request.values())
    return unique, len(rows) - len(unique)


def run_scoped_diagnostic_counts(
    run_id: str,
    condition: str,
    guard_rows: list[dict[str, Any]],
    counterfactual_rows: list[dict[str, Any]],
) -> dict[str, int]:
    """Count allocator and counterfactual events from exactly one fresh run."""
    return {
        "exact_allocator_rejects": sum(
            row.get("run_id") == run_id
            and row.get("condition") == condition
            and row.get("event") == "CAPACITY_GUARD"
            and row.get("allocator_fit") is False
            for row in guard_rows
        ),
        "missed_protected_self_promotion_count": sum(
            row.get("run_id") == run_id
            and row.get("condition") == condition == "C0"
            and bool(row.get("would_recover"))
            for row in counterfactual_rows
        ),
    }


def source_recheck_event(
    sample: dict[str, Any], initial_state: str, initial_check_ns: int,
    deadline_ns: int,
) -> dict[str, Any] | None:
    if initial_state != "SOURCE_NOT_READY" or not full_ready(sample):
        return None
    now_ns = int(sample["timestamp_monotonic_ns"])
    required = int(sample["required_blocks"])
    return {
        "event": "SOURCE_READY_RECHECK",
        "boundary_id": str(sample["boundary_id"]),
        "timestamp_ns": now_ns,
        "required_blocks": required,
        "local_cached_blocks": int(sample.get("local_cached_blocks", 0)),
        "external_ready_needed_blocks": int(sample.get("external_ready_needed_blocks", 0)),
        "covered_blocks": int(sample["covered_blocks"]),
        "initial_source_state": initial_state,
        "time_since_initial_source_check_ms": (now_ns - initial_check_ns) / 1e6,
        "time_to_deadline_ms": (deadline_ns - now_ns) / 1e6,
    }


def psa1_reserve(
    progress_blocks: int, existing_lease_blocks: int,
    candidate_matches_protected: bool, candidate_blocks: int,
) -> int:
    hypothetical_lease = max(
        existing_lease_blocks,
        candidate_blocks if candidate_matches_protected else 0,
    )
    return max(0, progress_blocks - hypothetical_lease)


def protected_self_reserve(
    progress_blocks: int, existing_lease_blocks: int, protected_full_blocks: int,
) -> int:
    return max(0, progress_blocks - max(existing_lease_blocks, protected_full_blocks))


def selection_visibility(
    candidate_id: str, protected_id: str,
    controller_visible: bool, exact_source_ready: bool,
) -> dict[str, bool]:
    stale = not controller_visible and exact_source_ready
    return {
        "candidate_matches_protected": candidate_id == protected_id,
        "stale_protected_visibility": stale,
        "wrong_candidate_due_to_stale_source": candidate_id != protected_id and stale,
    }


def missed_protected_self_promotion(
    *, candidate_id: str, protected_id: str, protected_exact_ready: bool,
    self_logical_fit: bool, self_psa_fit: bool, self_legacy_physical_fit: bool,
    self_exact_allocator_fit: bool, timing_permits: bool,
) -> bool:
    return bool(
        candidate_id != protected_id and protected_exact_ready and self_logical_fit
        and self_psa_fit and self_legacy_physical_fit and self_exact_allocator_fit
        and timing_permits
    )


def classify_attempt(
    *, legacy_required_blocks: int, pool_free_blocks: int,
    allocator_required_blocks: int, allocator_available_blocks: int,
) -> dict[str, Any]:
    legacy_fit = legacy_required_blocks <= pool_free_blocks
    allocator_fit = allocator_required_blocks <= allocator_available_blocks
    return {
        "legacy_guard_decision": "ADMIT" if legacy_fit else "REJECT",
        "allocator_fit": allocator_fit,
        "agreement": legacy_fit == allocator_fit,
        "false_admit": legacy_fit and not allocator_fit,
        "false_reject": not legacy_fit and allocator_fit,
    }


def classify_capacity_reject(selection_fit: bool | None, attempt_fit: bool) -> str:
    if attempt_fit:
        return "NOT_REJECTED"
    if selection_fit is True:
        return "ADMISSION_RACE"
    if selection_fit is False:
        return "STRUCTURAL_CAPACITY_REJECT"
    return "SELECTION_SNAPSHOT_MISSING"


def validate_fresh_matrix(
    expected_order: list[tuple[int, str]], runs: list[dict[str, Any]],
) -> bool:
    actual = [(int(row.get("repetition", -1)), str(row.get("condition", ""))) for row in runs]
    run_ids = [str(row.get("run_id", "")) for row in runs]
    return (
        len(expected_order) == 8 and actual == expected_order
        and len(set(run_ids)) == 8 and all(row.get("valid") is True for row in runs)
    )


def allocator_preflight(
    *, pool_free_blocks: int, inflight_reserved_blocks: int,
    num_blocks_to_allocate: int, full_sequence_must_fit: bool = False,
) -> dict[str, Any]:
    """Mirror the pinned vLLM 0.23.0 allocate_slots comparison inputs.

    This helper is for interpreting observed locals, not for predicting them.
    The runtime trace must source ``num_blocks_to_allocate`` from the actual
    coordinator call inside ``allocate_slots``.
    """
    available = pool_free_blocks if full_sequence_must_fit else (
        pool_free_blocks - inflight_reserved_blocks
    )
    return {
        "pool_free_blocks": pool_free_blocks,
        "inflight_reserved_blocks": 0 if full_sequence_must_fit else inflight_reserved_blocks,
        "allocator_available_blocks": available,
        "allocator_required_blocks": num_blocks_to_allocate,
        "watermark_blocks": 0,
        "full_sequence_must_fit": full_sequence_must_fit,
        "allocator_fit": num_blocks_to_allocate <= available,
    }


def classify_matrix(c0: dict[str, Any], c1: dict[str, Any]) -> str:
    return classify_matrix_results(c0, c1)[0]


def classify_matrix_results(c0: dict[str, Any], c1: dict[str, Any]) -> list[str]:
    stale_fraction = c0["stale_protected_visibility_count"] / max(1, c0["selection_count"])
    c0_rate = c0["candidate_matches_protected_count"] / max(1, c0["selection_count"])
    c1_rate = c1["candidate_matches_protected_count"] / max(1, c1["selection_count"])
    c0_h = float(c0.get("median_psa_reserve", 0))
    c1_h = float(c1.get("median_psa_reserve", 0))
    missed = int(c0.get("missed_protected_self_promotion_count", 0))
    results = []
    if (
        stale_fraction >= .25 and missed > 0 and c1_rate - c0_rate >= .25
        and c0_h > 0 and (c0_h - c1_h) / c0_h >= .50
        and c1.get("successful_promotions", 0) > c0.get("successful_promotions", 0)
    ):
        results.append("RESULT_A_SOURCE_VISIBILITY_PSA_INTERACTION")
    if (
        c1_rate >= .75
        and c1.get("source_ready_selection_opportunities", 0) > 0
        and c1.get("psa_blocked_source_ready_fraction", 0) > .50
        and c1.get("median_psa_reserve", 0) >= 2 * max(1, c1.get("median_allocator_need_now", 0))
    ):
        results.append("RESULT_B_PSA1_INTRINSIC_CONSERVATISM")
    if c1_rate >= .75 and c1.get("exact_allocator_reject_fraction", 0) > .50:
        results.append("RESULT_C_ALLOCATOR_CAPACITY_DOMINANT")
    attempts = max(1, int(c1.get("attempt_count", 0)))
    mismatch = (int(c1.get("false_admit_count", 0)) + int(c1.get("false_reject_count", 0))) / attempts
    if mismatch >= .20:
        results.append("RESULT_D_GUARD_ALLOCATOR_MISMATCH")
    return results or ["RESULT_E_MIXED_VISIBILITY_PSA_ALLOCATOR"]


def summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"median": None, "p90": None, "p95": None}
    ordered = sorted(values)
    def nr(q: float) -> float:
        import math
        return ordered[max(0, math.ceil(q * len(ordered)) - 1)]
    return {"median": float(median(ordered)), "p90": nr(.90), "p95": nr(.95)}


def normalize_incomplete_result(reason: str) -> tuple[str, str]:
    """Keep the result taxonomy separate from nested exception detail."""
    detail = str(reason).strip()
    prefix = "EXPERIMENT_INVALID_MATRIX:"
    while detail.startswith(prefix):
        detail = detail[len(prefix):]
    return "EXPERIMENT_INVALID_MATRIX", detail
