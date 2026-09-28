"""Small deterministic helpers for Phase 4G-B timeline analysis."""
from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

PROBE_INTERVAL_MS = 20.0
P_SAFE_MS = 102.0
RESUME_GUARD_MS = 110.0
RESERVATION_BUDGET_BLOCKS = 3510
FROZEN_RUN_ORDER = (
    ("M0", "A", False), ("M1", "D", False), ("M1", "D", False), ("M0", "A", False),
    ("M2", "A", True), ("M3", "D", True), ("M3", "D", True), ("M2", "A", True),
    ("M2", "A", True), ("M3", "D", True), ("M3", "D", True), ("M2", "A", True),
)


def nearest_rank(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered or not 0 < quantile <= 1:
        raise ValueError("nearest-rank requires values and a quantile in (0, 1]")
    return ordered[math.ceil(quantile * len(ordered)) - 1]


def validate_matrix(results: list[dict[str, Any]]) -> None:
    expected = {"M0": 2, "M1": 2, "M2": 4, "M3": 4}
    observed = {condition: sum(row.get("condition") == condition for row in results)
                for condition in expected}
    if len(results) != sum(expected.values()) or observed != expected:
        raise ValueError(f"partial or unbalanced Phase 4G-B matrix: {observed}")
    invalid = [row.get("run_id", "NA") for row in results if row.get("valid") is not True]
    if invalid:
        raise ValueError(f"invalid Phase 4G-B runs: {invalid}")


def background_is_valid(*, generated_tokens: int, expected_tokens: int,
                        max_idle_gap_ms: float, allowed_gap_ms: float,
                        tail_margin_ms: float, minimum_tail_ms: float = 0.0) -> bool:
    return (generated_tokens == expected_tokens
            and max_idle_gap_ms <= allowed_gap_ms
            and tail_margin_ms > 0
            and tail_margin_ms >= minimum_tail_ms)


def progress_reserve_blocks(
    protected_progress_blocks: int,
    protected_existing_lease_blocks: int,
    *,
    candidate_matches_protected: bool,
    candidate_blocks: int,
) -> int:
    hypothetical = max(
        protected_existing_lease_blocks,
        candidate_blocks if candidate_matches_protected else 0,
    )
    return max(0, protected_progress_blocks - hypothetical)


def feasibility_snapshot(
    *,
    required_blocks: int,
    local_blocks: int,
    external_ready_blocks: int,
    free_blocks: int,
    progress_reserve: int,
    reserved_blocks: int,
    logical_enabled: bool,
    logical_budget: int = RESERVATION_BUDGET_BLOCKS,
    source_ready: bool | None = None,
    timing_open: bool = True,
) -> dict[str, Any]:
    if min(required_blocks, local_blocks, external_ready_blocks, free_blocks, progress_reserve, reserved_blocks) < 0:
        raise ValueError("block counts cannot be negative")
    local = min(required_blocks, local_blocks)
    external_needed = required_blocks - local
    ready_external_needed = min(external_needed, external_ready_blocks)
    covered = local + ready_external_needed
    physical = free_blocks >= external_needed
    safe = physical and free_blocks - external_needed >= progress_reserve
    logical = not logical_enabled or reserved_blocks + required_blocks <= logical_budget
    source = covered == required_blocks if source_ready is None else bool(source_ready)
    return {
        "external_needed_blocks": external_needed,
        "external_ready_needed_blocks": ready_external_needed,
        "covered_blocks": covered,
        "missing_blocks": required_blocks - covered,
        "source_ready": source,
        "physical_fit": physical,
        "psa_safe_fit": safe,
        "logical_fit": logical,
        "timing_fit": bool(timing_open),
        "full_feasible": source and safe and logical and bool(timing_open),
    }


def partial_opportunity(
    required_blocks: int,
    external_ready_needed_blocks: int,
    free_blocks: int,
    progress_reserve_blocks: int,
) -> float:
    if required_blocks <= 0:
        raise ValueError("required_blocks must be positive")
    return min(
        required_blocks,
        max(0, external_ready_needed_blocks),
        max(0, free_blocks - progress_reserve_blocks),
    ) / required_blocks


def longest_contiguous_true_ms(
    samples: list[dict[str, Any]], predicate, *, max_gap_ms: float = 40.0
) -> float:
    """Duration supported by adjacent positive samples without a large gap."""
    ordered = sorted(samples, key=lambda row: int(row["timestamp_monotonic_ns"]))
    best = current = 0.0
    previous_ns: int | None = None
    previous_true = False
    for row in ordered:
        when = int(row["timestamp_monotonic_ns"])
        current_true = bool(predicate(row))
        if previous_ns is not None:
            delta_ms = (when - previous_ns) / 1e6
            if current_true and previous_true and 0 < delta_ms <= max_gap_ms:
                current += delta_ms
            else:
                current = 0.0
            best = max(best, current)
        previous_ns = when
        previous_true = current_true
    return best


def classify_result(
    *,
    m2_not_ready_count: int,
    m2_later_full_by_safe_start_count: int,
    m2_actionable_recoveries: int,
    m2_source_in_time_count: int,
    m2_source_in_time_without_safe_window: int,
    m2_source_in_time_with_partial: int,
    m2_source_in_time_physical_fit_count: int,
    m2_physical_fit_without_safe_window: int,
) -> str:
    """Apply the frozen A/B/C/D/E rule in priority order."""
    if m2_not_ready_count:
        recovery_fraction = m2_later_full_by_safe_start_count / m2_not_ready_count
        if recovery_fraction >= 0.50:
            actionable_fraction = (
                m2_actionable_recoveries / m2_later_full_by_safe_start_count
                if m2_later_full_by_safe_start_count else 0.0
            )
            if actionable_fraction >= 0.50:
                return "RESULT_A_SOURCE_RECHECK_OPPORTUNITY"
    if m2_not_ready_count and m2_later_full_by_safe_start_count / m2_not_ready_count < 0.25:
        return "RESULT_B_SOURCE_NOT_READY_IN_TIME"
    if m2_source_in_time_count:
        blocked_safe = m2_source_in_time_without_safe_window / m2_source_in_time_count
        partial = m2_source_in_time_with_partial / m2_source_in_time_count
        if blocked_safe > 0.50 and partial > 0.50:
            return "RESULT_C_PARTIAL_PROMOTION_OPPORTUNITY"
        if (m2_source_in_time_physical_fit_count
                and m2_physical_fit_without_safe_window / m2_source_in_time_physical_fit_count > 0.50):
            return "RESULT_D_PSA_HEADROOM_BLOCKER"
    return "RESULT_E_MIXED_OR_NO_CLEAR_BOTTLENECK"


def source_readiness_milestones(samples: list[dict[str, Any]], required_blocks: int) -> dict[str, float | str]:
    if required_blocks <= 0:
        raise ValueError("required_blocks must be positive")
    ordered = sorted(samples, key=lambda row: int(row["timestamp_monotonic_ns"]))
    origin = int(ordered[0]["timestamp_monotonic_ns"]) if ordered else 0
    result: dict[str, float | str] = {}
    for deficit in (4, 2, 1, 0):
        target = max(0, required_blocks - deficit)
        found = next(
            (row for row in ordered if int(row.get("covered_blocks", 0)) >= target),
            None,
        )
        result[f"ready_minus_{deficit}_ms"] = (
            (int(found["timestamp_monotonic_ns"]) - origin) / 1e6
            if found is not None else "NA"
        )
    return result
