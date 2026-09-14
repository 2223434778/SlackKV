from enum import Enum
from typing import Iterable, Optional

from .jobs import PromotionJob


class Policy(str, Enum):
    ON_DEMAND = "OnDemand"
    ASAP_EDF = "ASAP-EDF"
    NAIVE_JIT = "NaiveJIT"
    QUEUE_JIT = "SlackKV-QueueJIT"


def eligibility_time(job: PromotionJob, policy: Policy) -> float:
    if policy is Policy.ON_DEMAND:
        return job.deadline_s
    if policy is Policy.ASAP_EDF or policy is Policy.QUEUE_JIT:
        return job.release_time_s
    if policy is Policy.NAIVE_JIT:
        return max(job.release_time_s, job.deadline_s - job.service_time_s)
    raise ValueError(policy)


def earliest_deadline(jobs: Iterable[PromotionJob]) -> PromotionJob:
    return min(jobs, key=lambda job: (job.deadline_s, job.release_time_s, job.job_id))


def queue_safe_idle_s(now_s: float, pending: Iterable[PromotionJob]) -> float:
    """Minimum prefix slack for released jobs under non-preemptive EDF service."""
    total = 0.0
    safe_idle = float("inf")
    for job in sorted(pending, key=lambda item: (item.deadline_s, item.job_id)):
        total += job.service_time_s
        safe_idle = min(safe_idle, job.deadline_s - now_s - total)
    return safe_idle
