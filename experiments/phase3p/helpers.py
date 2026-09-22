"""Pure controller and metric helpers for Phase 3P."""
from __future__ import annotations


DEADLINES_MS = {"A": 300.0, "B": 320.0, "C": 340.0, "D": 650.0, "E": 900.0, "F": 1300.0}
BLOCKING_SERVICE_MS = 70.0
ASYNC_SERVICE_MS = 40.0


def effective_latency_ms(first_token_ms: float, deadline_ms: float) -> float:
    return first_token_ms - deadline_ms


def internal_ready_miss(arrival_ms: float, t6_ms: float) -> bool:
    return arrival_ms < t6_ms


def gpu_ready_residency_ms(admission_ms: float, t6_ms: float) -> float:
    return max(0.0, admission_ms - t6_ms)


def async_resume_may_send(prefetch_future_done: bool) -> bool:
    """The async controller never uses HTTP future completion as a send gate."""
    return True


def configured_service_budget(policy: str) -> float | None:
    if policy == "Blocking70-K2":
        return BLOCKING_SERVICE_MS
    if policy == "Async40-K2":
        return ASYNC_SERVICE_MS
    if policy == "OnDemand-K2":
        return None
    raise ValueError(policy)
