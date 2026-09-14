"""Event-driven one-resource promotion simulator; times are seconds and bytes are decimal GB for costs."""
from dataclasses import dataclass
import math
from statistics import mean
from typing import Dict, List, Sequence

from .jobs import PromotionJob
from .policies import Policy, earliest_deadline, eligibility_time, queue_safe_idle_s


EPS = 1e-12


@dataclass(frozen=True)
class JobOutcome:
    job: PromotionJob
    start_time_s: float
    completion_time_s: float

    @property
    def stall_s(self) -> float:
        return max(0.0, self.completion_time_s - self.job.deadline_s)

    @property
    def prefetch_residency_s(self) -> float:
        return max(0.0, self.job.deadline_s - self.completion_time_s)

    @property
    def prefetch_vram_time_gbs(self) -> float:
        return self.job.kv_size_gb * self.prefetch_residency_s


@dataclass(frozen=True)
class SimulationResult:
    policy: Policy
    outcomes: Sequence[JobOutcome]
    busy_time_s: float


def simulate(jobs: Sequence[PromotionJob], policy: Policy) -> SimulationResult:
    if not jobs:
        return SimulationResult(policy, (), 0.0)
    future = sorted(((eligibility_time(job, policy), job) for job in jobs), key=lambda item: (item[0], item[1].job_id))
    pending: List[PromotionJob] = []
    outcomes: List[JobOutcome] = []
    index = 0
    now_s = future[0][0]
    busy_s = 0.0
    while index < len(future) or pending:
        while index < len(future) and future[index][0] <= now_s + EPS:
            pending.append(future[index][1]); index += 1
        if not pending:
            now_s = max(now_s, future[index][0])
            continue
        if policy is Policy.QUEUE_JIT:
            safe_idle_s = queue_safe_idle_s(now_s, pending)
            if safe_idle_s > EPS:
                idle_end = now_s + safe_idle_s
                next_release = future[index][0] if index < len(future) else math.inf
                now_s = min(idle_end, next_release)
                continue
        job = earliest_deadline(pending)
        pending.remove(job)
        start_s = now_s
        finish_s = start_s + job.service_time_s
        if finish_s < start_s:
            raise AssertionError("completion before start")
        outcomes.append(JobOutcome(job, start_s, finish_s))
        busy_s += job.service_time_s
        now_s = finish_s
    if len(outcomes) != len(jobs):
        raise AssertionError("not every job completed")
    return SimulationResult(policy, tuple(outcomes), busy_s)


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values); position = (len(ordered) - 1) * quantile; lo = int(position); hi = math.ceil(position)
    return ordered[lo] if lo == hi else ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def residency_capacity(outcomes: Sequence[JobOutcome]) -> Dict[str, float]:
    events = []
    for outcome in outcomes:
        if outcome.prefetch_residency_s > EPS:
            events.append((outcome.completion_time_s, 1, outcome.job.kv_size_gb))
            events.append((outcome.job.deadline_s, 0, -outcome.job.kv_size_gb))
    if not events:
        return {"peak_prefetched_gb": 0.0, "p95_prefetched_gb": 0.0, "mean_prefetched_gb": 0.0}
    events.sort(); level = 0.0; previous = events[0][0]; weighted = []; area = 0.0
    for time_s, _, change in events:
        duration = time_s - previous
        if duration > 0:
            weighted.append((level, duration)); area += level * duration
        level += change; previous = time_s
    total = sum(duration for _, duration in weighted)
    if total == 0:
        return {"peak_prefetched_gb": 0.0, "p95_prefetched_gb": 0.0, "mean_prefetched_gb": 0.0}
    cumulative = 0.0
    p95 = 0.0
    for level, duration in sorted(weighted):
        cumulative += duration
        if cumulative / total >= 0.95:
            p95 = level; break
    return {"peak_prefetched_gb": max(level for level, _ in weighted), "p95_prefetched_gb": p95, "mean_prefetched_gb": area / total}


def aggregate(result: SimulationResult) -> Dict[str, float]:
    outcomes = result.outcomes
    if not outcomes:
        return {"job_count": 0}
    stalls = [outcome.stall_s * 1000.0 for outcome in outcomes]
    residency = [outcome.prefetch_residency_s * 1000.0 for outcome in outcomes]
    first_release = min(outcome.job.release_time_s for outcome in outcomes)
    last_completion = max(outcome.completion_time_s for outcome in outcomes)
    span = last_completion - first_release
    prefetch = sum(outcome.prefetch_vram_time_gbs for outcome in outcomes)
    keep = sum(outcome.job.kv_size_gb * (outcome.job.deadline_s - outcome.job.release_time_s) for outcome in outcomes)
    values = {
        "job_count": len(outcomes), "deadline_miss_count": sum(value > EPS for value in stalls),
        "deadline_miss_rate": sum(value > EPS for value in stalls) / len(stalls), "stall_ms_mean": mean(stalls),
        "stall_ms_p50": percentile(stalls, .50), "stall_ms_p95": percentile(stalls, .95), "stall_ms_p99": percentile(stalls, .99),
        "max_stall_ms": max(stalls), "prefetch_residency_ms_mean": mean(residency), "prefetch_residency_ms_p50": percentile(residency, .50),
        "prefetch_residency_ms_p95": percentile(residency, .95), "prefetch_vram_time_gbs_total": prefetch,
        "prefetch_vram_time_gbs_per_job": prefetch / len(outcomes), "keep_reference_vram_time_gbs_total": keep,
        "prefetch_to_keep_ratio": prefetch / keep if keep else 0.0, "transfer_engine_utilization": result.busy_time_s / span if span else 0.0,
        "transfer_engine_idle_fraction": 1.0 - result.busy_time_s / span if span else 0.0,
        "total_transfer_bytes": sum(outcome.job.kv_size_bytes for outcome in outcomes),
        "completed_before_deadline_fraction": sum(value <= EPS for value in stalls) / len(stalls),
    }
    values.update(residency_capacity(outcomes)); values.update({"prefetch_budget_8gib_exceeded": values["peak_prefetched_gb"] > 8, "prefetch_budget_16gib_exceeded": values["peak_prefetched_gb"] > 16, "prefetch_budget_24gib_exceeded": values["peak_prefetched_gb"] > 24})
    return values
