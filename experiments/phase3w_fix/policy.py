"""Frozen prediction-error model for Phase 3W-FIX.

Promotion code receives only ``scheduler_deadline_ms`` and
``reservation_end_ms``. Actual deadlines remain with resume dispatch and
post-run validation.
"""

from __future__ import annotations

from dataclasses import dataclass


ORACLE = "OracleDynamicBlock512-K2"
RAW = "RawPred-ClosedLoop-DynamicBlock512-K2"
SAFE = "SafePred200-ClosedLoop-DynamicBlock512-K2"
POLICIES = (ORACLE, RAW, SAFE)
PREDICTION_ERROR_MS = {
    **dict.fromkeys("ABCD", 200.0),
    **dict.fromkeys("EFGH", -50.0),
    **dict.fromkeys("IJKL", 120.0),
}
SAFETY_MARGIN_MS = 200.0
HANDOFF_GUARD_MS = 100.0
RESERVATION_BUDGET_BLOCKS = 512
SERVICE_BUDGET_MS = 40.0
K2 = 2


@dataclass(frozen=True)
class Prediction:
    session: str
    actual_deadline_ms: float
    prediction_error_ms: float
    predicted_deadline_ms: float
    scheduler_deadline_ms: float
    reservation_end_ms: float


def predict_deadline(actual_deadline_ms: float, error_ms: float) -> float:
    return actual_deadline_ms + error_ms


def scheduler_deadline(policy: str, actual_deadline_ms: float, predicted_deadline_ms: float) -> float:
    if policy == ORACLE:
        return actual_deadline_ms
    if policy == RAW:
        return predicted_deadline_ms
    if policy == SAFE:
        return predicted_deadline_ms - SAFETY_MARGIN_MS
    raise ValueError(f"unknown Phase 3W policy: {policy}")


def build_predictions(
    actual_deadlines_ms: dict[str, float],
    policy: str,
    errors_ms: dict[str, float] = PREDICTION_ERROR_MS,
) -> dict[str, Prediction]:
    """Build experiment bookkeeping; pass only scheduler fields to scheduling."""
    if policy not in POLICIES:
        raise ValueError(f"unknown Phase 3W-FIX policy: {policy}")
    out: dict[str, Prediction] = {}
    for session, actual in actual_deadlines_ms.items():
        error = errors_ms.get(session, 0.0)
        predicted = predict_deadline(actual, error)
        scheduler_d = scheduler_deadline(policy, actual, predicted)
        reservation_deadline = actual if policy == ORACLE else predicted
        out[session] = Prediction(
            session=session,
            actual_deadline_ms=actual,
            prediction_error_ms=error,
            predicted_deadline_ms=predicted,
            scheduler_deadline_ms=scheduler_d,
            reservation_end_ms=reservation_deadline + HANDOFF_GUARD_MS,
        )
    return out


def reservation_covers_actual(reservation_end_ms: float, actual_deadline_ms: float) -> bool:
    return reservation_end_ms >= actual_deadline_ms


def prediction_sign(error_ms: float) -> str:
    if error_ms > 0:
        return "LATE_PREDICTION"
    if error_ms < 0:
        return "EARLY_PREDICTION"
    return "ZERO_ERROR"
