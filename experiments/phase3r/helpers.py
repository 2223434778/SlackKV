"""Pure controller and metric helpers for Phase 3P."""
from __future__ import annotations


DEADLINES_MS = {"A":300.0,"B":320.0,"C":340.0,"D":360.0,"E":900.0,"F":920.0,"G":940.0,"H":960.0,"I":1500.0,"J":1520.0,"K":1540.0,"L":1560.0}
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
    if policy in ("Async40-K2", "CapacityAwareAsync40-K2"):
        return ASYNC_SERVICE_MS
    if policy == "OnDemand-K2":
        return None
    raise ValueError(policy)
GPU_KV_MIB=1024.0
SESSION_KV_MIB=112.0
CAPACITY_RESERVE_MIB=32.0
REQUIRED_FREE_MIB=144.0
def capacity_decision(usage_perc):
 free=GPU_KV_MIB*(1.0-usage_perc)
 return ('PROMOTE' if free>=REQUIRED_FREE_MIB else 'CAPACITY_SKIP',free)
def promotion_outcome(decision,t6_before_arrival,remote_wait):
 if decision=='CAPACITY_SKIP': return 'CAPACITY_SKIPPED_ONDEMAND'
 if not t6_before_arrival: return 'INTERNAL_READY_MISS'
 return 'READY_BUT_NOT_LOCAL' if remote_wait else 'PROMOTED_AND_LOCAL'
