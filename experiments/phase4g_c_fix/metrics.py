"""Pure timing gates and frozen Phase 4G-C-FIX run-order checks."""
from __future__ import annotations

import math
from typing import Any, Iterable

START_COMMIT = "c10a6c7ac69e41b444d00497c5d258da6bfc71b0"
WORKLOAD_SHA256 = "9b381336a70e73b0efbddbb79d2c88608c19663247e819c1e7a364efd2971105"
VLLM_DIFF_SHA256 = "bfd789e6849715196435f18a57ed9bbefae7b4b9d3a78d69115614f27d1542e8"
BACKEND = "CLOCK_NANOSLEEP_ABS"
SWITCH_INTERVAL_S = 0.0005
PAUSE_P95_GATE_MS = 10.0
PAUSE_MAX_GATE_MS = 30.0
STAGE_T_ORDER = [
    (1, "C0"), (2, "C1"), (3, "C1"),
    (4, "C0"), (5, "C0"), (6, "C1"),
]
MEASURED_ORDER = [
    (1, "C0"), (1, "C1"), (2, "C1"), (2, "C0"),
    (3, "C0"), (3, "C1"), (4, "C1"), (4, "C0"),
]


def nearest_rank(values: Iterable[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("nearest_rank requires at least one value")
    if not 0 < q <= 1:
        raise ValueError("q must be in (0, 1]")
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def timing_gate(rows: list[dict[str, Any]]) -> dict[str, float | bool]:
    errors = [abs(float(row["pause_error_ms"])) for row in rows]
    if not errors:
        return {"p95_abs_ms": math.inf, "max_abs_ms": math.inf, "pass": False}
    p95 = nearest_rank(errors, 0.95)
    maximum = max(errors)
    return {
        "p95_abs_ms": p95,
        "max_abs_ms": maximum,
        "pass": p95 <= PAUSE_P95_GATE_MS and maximum <= PAUSE_MAX_GATE_MS,
    }


def summarize_timing(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    def signed(field: str, q: float) -> float | None:
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        return nearest_rank(values, q) if values else None

    def absolute(field: str, q: float) -> float | None:
        values = [abs(float(row[field])) for row in rows if row.get(field) is not None]
        return nearest_rank(values, q) if values else None

    return {
        "sleep_overshoot_P50_ms": signed("sleep_overshoot_ms", 0.50),
        "sleep_overshoot_P95_ms": signed("sleep_overshoot_ms", 0.95),
        "sleep_overshoot_P99_ms": signed("sleep_overshoot_ms", 0.99),
        "sleep_overshoot_max_ms": max(
            (float(row["sleep_overshoot_ms"]) for row in rows), default=None
        ),
        "sleep_overshoot_abs_P95_ms": absolute("sleep_overshoot_ms", 0.95),
        "dispatch_delay_P50_ms": signed("dispatch_delay_ms", 0.50),
        "dispatch_delay_P95_ms": signed("dispatch_delay_ms", 0.95),
        "dispatch_delay_P99_ms": signed("dispatch_delay_ms", 0.99),
        "dispatch_delay_max_ms": max(
            (float(row["dispatch_delay_ms"]) for row in rows), default=None
        ),
        "pause_error_P50_ms": signed("pause_error_ms", 0.50),
        "pause_error_P95_ms": signed("pause_error_ms", 0.95),
        "pause_error_P99_ms": signed("pause_error_ms", 0.99),
        "pause_error_max_ms": max(
            (float(row["pause_error_ms"]) for row in rows), default=None
        ),
        "pause_error_abs_P95_ms": absolute("pause_error_ms", 0.95),
        "pause_error_abs_P99_ms": absolute("pause_error_ms", 0.99),
        "pause_error_abs_max_ms": max(
            (abs(float(row["pause_error_ms"])) for row in rows), default=None
        ),
    }


def timing_root_cause(rows: list[dict[str, Any]], passed: bool) -> str:
    if passed:
        return "RESOLVED"
    sleep_values = [abs(float(row["sleep_overshoot_ms"])) for row in rows]
    dispatch_values = [abs(float(row["dispatch_delay_ms"])) for row in rows]
    sleep_p95 = nearest_rank(sleep_values, 0.95) if sleep_values else 0.0
    dispatch_p95 = nearest_rank(dispatch_values, 0.95) if dispatch_values else 0.0
    sleep_high = sleep_p95 > 10
    dispatch_high = dispatch_p95 > 10
    if sleep_high and dispatch_high:
        return "MIXED_TIMING_JITTER"
    if sleep_high:
        return "HOST_WAKEUP_JITTER"
    if dispatch_high:
        return "CLIENT_DISPATCH_JITTER"
    return "MIXED_TIMING_JITTER"


def validate_stage_t(run_rows: list[dict[str, Any]], timing_rows: list[dict[str, Any]]) -> bool:
    order = [(int(row["repetition"]), str(row["condition"])) for row in run_rows]
    return (
        order == STAGE_T_ORDER
        and len(run_rows) == 6
        and sum(row["condition"] == "C0" for row in run_rows) == 3
        and sum(row["condition"] == "C1" for row in run_rows) == 3
        and len(timing_rows) == 192
        and all(bool(row.get("complete")) for row in run_rows)
    )


def validate_measured_matrix(
    results: list[dict[str, Any]], expected_root: str,
) -> bool:
    if len(results) != 8 or any(
        "repetition" not in row or "condition" not in row for row in results
    ):
        return False
    order = [(int(row["repetition"]), str(row["condition"])) for row in results]
    return (
        order == MEASURED_ORDER
        and len(results) == 8
        and sum(row["condition"] == "C0" for row in results) == 4
        and sum(row["condition"] == "C1" for row in results) == 4
        and all(bool(row.get("valid")) for row in results)
        and all(str(row.get("artifact_root")) == expected_root for row in results)
        and sum(int(row.get("turns_complete", 0)) for row in results) == 320
        and sum(int(row.get("boundaries", 0)) for row in results) == 256
    )
