"""Offline nullable allocator, protected-head and shadow-frontier analysis."""
from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

START_COMMIT = "7bada3599076fa4fbf8e0d73120c37dbe6285ca0"
WORKLOAD_SHA256 = "9b381336a70e73b0efbddbb79d2c88608c19663247e819c1e7a364efd2971105"
GPU_BLOCKS = 4681
P_SAFE_MS = 102
RESUME_GUARD_MS = 110
K2 = 2
RESERVATION_BUDGET_BLOCKS = 3510
RECHECK_INTERVAL_MS = 20
MIN_ACTIONABLE_WINDOW_MS = 20
TERMINAL_DECISIONS = {
    "CAPACITY_REJECTED", "PROGRESS_REJECTED", "RESERVATION_SKIP",
    "STALE_PROMOTION_CANCELLED",
}


def optional_int(value: Any) -> int | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        raise ValueError("boolean is not an allocator block count")
    parsed = int(value)
    if str(value).strip() not in {str(parsed), f"{parsed}.0"} and not isinstance(value, int):
        if float(value) != parsed:
            raise ValueError(f"allocator block count is not integral: {value!r}")
    return parsed


def classify_attempt_nullable(
    legacy_guard_decision: str | None,
    allocator_required_blocks: Any,
    allocator_available_blocks: Any,
) -> dict[str, bool | str]:
    """Classify only when both exact allocator operands were observed."""
    required = optional_int(allocator_required_blocks)
    available = optional_int(allocator_available_blocks)
    if required is None or available is None:
        return {
            "allocator_fit": "UNKNOWN", "agreement": "UNKNOWN",
            "false_admit": "UNKNOWN", "false_reject": "UNKNOWN",
        }
    legacy = str(legacy_guard_decision or "").upper()
    if legacy not in {"ADMIT", "REJECT"}:
        raise ValueError(f"unknown legacy guard decision: {legacy_guard_decision!r}")
    legacy_fit = legacy == "ADMIT"
    allocator_fit = required <= available
    return {
        "allocator_fit": allocator_fit,
        "agreement": legacy_fit == allocator_fit,
        "false_admit": legacy_fit and not allocator_fit,
        "false_reject": not legacy_fit and allocator_fit,
    }


def reanalyze_allocator_csv(path: str | Path) -> dict[str, Any]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    exact = []
    missing = 0
    for row in rows:
        required = optional_int(row.get("allocator_required_blocks"))
        available = optional_int(row.get("allocator_available_blocks"))
        if required is None or available is None:
            missing += 1
            continue
        exact.append(classify_attempt_nullable(
            row.get("legacy_guard_decision"), required, available,
        ))
    agreement = sum(row["agreement"] is True for row in exact)
    mismatch = sum(row["agreement"] is False for row in exact)
    false_admit = sum(row["false_admit"] is True for row in exact)
    false_reject = sum(row["false_reject"] is True for row in exact)
    result = {
        "source_artifact": str(path),
        "total_guard_observations": len(rows),
        "exact_known": len(exact),
        "exact_missing": missing,
        "known_agreement": agreement,
        "known_mismatch": mismatch,
        "known_false_admit": false_admit,
        "known_false_reject": false_reject,
        "unknown_excluded_from_mismatch_denominator": True,
        "previous_result_reassessment": (
            "RESULT_D_GUARD_ALLOCATOR_MISMATCH is not supported after missing-value correction"
        ),
    }
    if (len(rows), len(exact), missing, agreement, mismatch, false_admit, false_reject) != (
        123, 54, 69, 54, 0, 0, 0,
    ):
        raise ValueError(f"frozen allocator reanalysis counts changed: {result}")
    return result


def normalize_decision(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text.startswith("PROMOTION_"):
        text = text[len("PROMOTION_"):]
    return text


def is_nonresident_terminal(decision: Any) -> bool:
    return normalize_decision(decision) in TERMINAL_DECISIONS


def is_protected_poison_episode(
    *, protected_id: str | None, boundary_id: str, first_token_ns: Any,
    decision: Any,
) -> bool:
    """The frozen poison definition ends at first token, not resume arrival."""
    return (protected_id == boundary_id and first_token_ns is None
            and is_nonresident_terminal(decision))


def poison_episode_start_ns(terminal_timestamp_ns: int, frontier_entered_ns: int | None) -> int:
    """The frozen episode clock starts at the terminal decision.

    Frontier entry is retained separately for interpretation, but must not
    shorten the terminal-to-resume retry-observation interval.
    """
    del frontier_entered_ns
    return int(terminal_timestamp_ns)


def finish_poison_episode(
    episode: dict[str, Any], *, end_timestamp_ns: int, end_reason: str,
    first_token_ns: Any,
) -> dict[str, Any]:
    """Finalize a poison interval while retaining the actual first-token time."""
    result = dict(episode)
    start_ns = int(result.get("episode_start_timestamp_ns") or result["terminal_timestamp_ns"])
    end_ns = max(start_ns, int(end_timestamp_ns))
    result["end_timestamp_ns"] = end_ns
    result["episode_start_timestamp_ns"] = start_ns
    result["end_reason"] = end_reason
    result["first_token_ns"] = optional_int(first_token_ns)
    result["poison_duration_ms"] = (end_ns - start_ns) / 1e6
    return result


def psa1_need(
    progress_blocks: int,
    existing_lease_blocks: int,
    candidate_matches_frontier: bool,
    candidate_blocks: int,
) -> int:
    hypothetical_lease = max(
        existing_lease_blocks,
        candidate_blocks if candidate_matches_frontier else 0,
    )
    return max(0, progress_blocks - hypothetical_lease)


def psa_fit(free_blocks: int | None, candidate_blocks: int,
            need_blocks: int | None) -> bool | None:
    if free_blocks is None or need_blocks is None:
        return None
    return free_blocks - candidate_blocks >= need_blocks


def choose_frontiers(
    states: dict[str, dict[str, Any]],
    exact_source_ready: dict[str, bool | None],
    candidate_id: str | None,
) -> dict[str, str | None]:
    unfinished = [(key, row) for key, row in states.items() if row.get("first_token_ns") is None]
    key_order = lambda pair: (int(pair[1]["deadline_ns"]), pair[0])
    active = [(key, row) for key, row in unfinished
              if row.get("decision") == "REGISTERED" and row.get("resume_started_ns") is None]
    ready_active = [(key, row) for key, row in active
                    if exact_source_ready.get(key) is True]
    arrived = [(key, row) for key, row in unfinished
               if row.get("resume_started_ns") is not None]
    return {
        "F0": min(unfinished, key=key_order)[0] if unfinished else None,
        "F1": min(active, key=key_order)[0] if active else None,
        "F2": min(ready_active, key=key_order)[0] if ready_active else None,
        "F3": candidate_id,
        "F4": min(arrived, key=key_order)[0] if arrived else None,
    }


def retry_actionable(sample: dict[str, Any]) -> bool:
    """UNKNOWN is not actionable; every listed condition must be explicit."""
    return all(sample.get(key) is True for key in (
        "source_full_ready", "logical_reservation_fit", "timing_gate_open",
        "self_psa_fit", "allocator_fit",
    )) and sample.get("resume_started") is False and sample.get("first_token_seen") is False


def first_retry_probe_deadline_ns(terminal_timestamp_ns: int) -> int:
    return int(terminal_timestamp_ns) + RECHECK_INTERVAL_MS * 1_000_000


def retry_probe_coverage_required(observation_window_ms: float) -> bool:
    return float(observation_window_ms) >= MIN_ACTIONABLE_WINDOW_MS


def actionable_windows(
    samples: Iterable[dict[str, Any]], *, min_duration_ms: float = MIN_ACTIONABLE_WINDOW_MS,
    max_gap_ms: float = 40.0,
) -> list[dict[str, Any]]:
    ordered = sorted(samples, key=lambda row: int(row["timestamp_ns"]))
    windows: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for sample in ordered:
        if not bool(sample.get("retry_actionable")):
            if current:
                windows.append(current)
                current = []
            continue
        if current and (int(sample["timestamp_ns"]) - int(current[-1]["timestamp_ns"])) / 1e6 > max_gap_ms:
            windows.append(current)
            current = []
        current.append(sample)
    if current:
        windows.append(current)
    result = []
    for window in windows:
        duration = (int(window[-1]["timestamp_ns"]) - int(window[0]["timestamp_ns"])) / 1e6
        if duration >= min_duration_ms:
            result.append({
                "start_ns": int(window[0]["timestamp_ns"]),
                "end_ns": int(window[-1]["timestamp_ns"]),
                "duration_ms": duration,
                "sample_count": len(window),
            })
    return result


def classify_poison_type(
    *, retryable: bool, frontier_recoverable: bool,
    allocator_known_fit_fraction: float | None,
    f3_psa_fail_fraction: float | None,
) -> str:
    if retryable:
        return "Retryable head"
    if frontier_recoverable:
        return "Frontier-only"
    if allocator_known_fit_fraction is not None and allocator_known_fit_fraction < 0.5:
        return "Capacity-bound"
    if (allocator_known_fit_fraction is not None and allocator_known_fit_fraction >= 0.5
            and f3_psa_fail_fraction is not None and f3_psa_fail_fraction > 0.5):
        return "PSA-intrinsic"
    return "Mixed/unknown"


def classify_result(
    *, poison_episode_count: int, retry_recoverable_count: int,
    frontier_blocked_count: int, frontier_recoverable_count: int,
    relevant_allocator_known_count: int, relevant_allocator_reject_count: int,
    allocator_fit_f3_psa_fail_count: int,
) -> str:
    retry_fraction = retry_recoverable_count / max(1, poison_episode_count)
    frontier_fraction = frontier_recoverable_count / max(1, frontier_blocked_count)
    capacity_fraction = relevant_allocator_reject_count / max(1, relevant_allocator_known_count)
    psa_fraction = allocator_fit_f3_psa_fail_count / max(1, frontier_blocked_count)
    if retry_fraction >= 0.50 and retry_recoverable_count >= 8:
        return "RESULT_A_RETRYABLE_PROTECTED_HEAD_DOMINANT"
    if retry_fraction < 0.25 and frontier_fraction >= 0.50:
        return "RESULT_B_PROTECTED_FRONTIER_POISONING_DOMINANT"
    if (capacity_fraction > 0.50 and retry_recoverable_count == 0
            and frontier_recoverable_count == 0):
        return "RESULT_C_CAPACITY_DOMINANT"
    if psa_fraction > 0.50:
        return "RESULT_D_PSA_INTRINSIC"
    return "RESULT_E_MIXED_HEAD_CAPACITY_PSA"


def exact_truth_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    exact = [row for row in rows
             if optional_int(row.get("allocator_required_blocks")) is not None
             and optional_int(row.get("allocator_available_blocks")) is not None]
    return {
        "eligible_selections": len(rows),
        "exact_known": len(exact),
        "exact_unknown": len(rows) - len(exact),
        "coverage_fraction": len(exact) / len(rows) if rows else None,
    }


def _optional_number(value: Any) -> float | None:
    if value is None or (isinstance(value, str) and value.strip().upper() in
                         {"", "NA", "N/A", "NONE", "UNKNOWN"}):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _number_summary(values: list[float]) -> dict[str, float | int | None]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "median": None, "p90": None, "p95": None, "max": None}

    def quantile(q: float) -> float:
        return ordered[max(0, math.ceil(q * len(ordered)) - 1)]

    return {
        "count": len(ordered), "median": statistics.median(ordered),
        "p90": quantile(.90), "p95": quantile(.95), "max": ordered[-1],
    }


def summarize_protected_need(
    need_rows: list[dict[str, Any]], selection_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize PSA and allocator needs without treating missing truth as zero."""
    selection_match = {
        str(row.get("selection_id")): row.get("candidate_matches_protected")
        for row in selection_rows
    }
    groups: dict[str, list[dict[str, Any]]] = {
        "all": list(need_rows),
        "candidate_matches_protected": [],
        "candidate_differs_from_protected": [],
        "no_local_cached_blocks": [],
        "local_cached_blocks_positive": [],
    }
    for row in need_rows:
        match = selection_match.get(str(row.get("selection_id")))
        if match is True or (isinstance(match, str) and match.lower() == "true"):
            groups["candidate_matches_protected"].append(row)
        elif match is False or (isinstance(match, str) and match.lower() == "false"):
            groups["candidate_differs_from_protected"].append(row)
        local = _optional_number(row.get("local_cached_blocks"))
        if local == 0:
            groups["no_local_cached_blocks"].append(row)
        elif local is not None and local > 0:
            groups["local_cached_blocks_positive"].append(row)

    result: dict[str, Any] = {}
    ownership_fields = {
        "local_cached_blocks": "local_cached_blocks",
        "leased_blocks": "leased_blocks",
        "owned_blocks": "owned_blocks",
        "evictable_cached_blocks": "evictable_cached_blocks",
    }
    for group_name, rows in groups.items():
        exact_rows = [row for row in rows
                      if _optional_number(row.get("Need_allocator_now")) is not None
                      and _optional_number(row.get("allocator_available_now")) is not None]
        psa_all = [_optional_number(row.get("Need_PSA1")) for row in rows]
        psa_known = [_optional_number(row.get("Need_PSA1")) for row in exact_rows]
        allocator_need = [_optional_number(row.get("Need_allocator_now")) for row in exact_rows]
        paired = [(psa, alloc) for psa, alloc in zip(psa_known, allocator_need)
                  if psa is not None and alloc is not None]
        result[group_name] = {
            "selection_count": len(rows),
            "allocator_exact_known_count": len(exact_rows),
            "allocator_exact_unknown_count": len(rows) - len(exact_rows),
            "need_psa1_all": _number_summary([x for x in psa_all if x is not None]),
            "need_psa1_exact_known": _number_summary([x for x in psa_known if x is not None]),
            "need_allocator_now": _number_summary([x for x in allocator_need if x is not None]),
            "need_gap_psa1_minus_allocator": _number_summary([psa - alloc for psa, alloc in paired]),
            "need_ratio_psa1_over_max_1_allocator": _number_summary(
                [psa / max(1.0, alloc) for psa, alloc in paired]),
            "ownership": {
                key: _number_summary([number for row in rows
                                      if (number := _optional_number(row.get(field))) is not None])
                for key, field in ownership_fields.items()
            },
        }
    result["interpretation_guardrails"] = {
        "allocator_need_is_instantaneous_not_future_liveness_proof": True,
        "missing_allocator_values_excluded_not_zero_filled": True,
        "ownership_not_inferred_when_missing": True,
    }
    return result


def source_accounting(
    *, local_tokens: Any, external_tokens: Any, expected_tokens: Any,
    explained_shortfall_tokens: Any, unexplained_shortfall_tokens: Any,
    unique_resume_traces: Any, expected_resume_traces: int = 32,
) -> dict[str, Any]:
    """Validate the frozen prompt-only offload accounting without hiding gaps."""
    local = optional_int(local_tokens)
    external = optional_int(external_tokens)
    expected = optional_int(expected_tokens)
    explained = optional_int(explained_shortfall_tokens)
    unexplained = optional_int(unexplained_shortfall_tokens)
    traces = optional_int(unique_resume_traces)
    complete = all(value is not None for value in
                   (local, external, expected, explained, unexplained, traces))
    accounted = bool(
        complete
        and local + external + explained == expected
        and unexplained == 0
        and traces == expected_resume_traces
    )
    return {
        "valid": accounted,
        "local_source_tokens": local,
        "external_source_tokens": external,
        "expected_source_tokens": expected,
        "prompt_only_explained_shortfall_tokens": explained,
        "unexplained_source_shortfall_tokens": unexplained,
        "source_trace_unique_resume_count": traces,
        "expected_resume_traces": expected_resume_traces,
        "raw_local_plus_external_equals_expected": (
            local + external == expected if complete else None
        ),
        "accounting_equation": "local + external + explained_shortfall = expected",
    }


def timing_gate(abs_pause_errors_ms: list[float]) -> bool:
    if not abs_pause_errors_ms:
        return False
    ordered = sorted(float(value) for value in abs_pause_errors_ms)
    p95 = ordered[max(0, math.ceil(.95 * len(ordered)) - 1)]
    return p95 <= 10 and max(ordered) <= 30
