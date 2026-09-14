"""Deterministic Phase 1A synthetic workloads; all gaps are oracle resume gaps."""
import math
import random
from typing import List

from slackkv.simulator.jobs import PromotionJob
from slackkv.simulator.transfer_model import TransferModel


SIZES = (256 << 20, 1 << 30, 4 << 30)
WEIGHTS = (0.50, 0.35, 0.15)
GAPS_S = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)


def controlled_burst(job_count: int, seed: int, model: TransferModel, multiplier: float) -> List[PromotionJob]:
    rng = random.Random(seed); jobs = []
    for index in range(job_count):
        size = rng.choices(SIZES, weights=WEIGHTS, k=1)[0]
        release = rng.uniform(0.0, 0.020)
        gap = rng.choice(GAPS_S) * rng.uniform(0.90, 1.10)
        jobs.append(PromotionJob(f"burst-{seed}-{index}", release, release + gap, size, model.service_time_s(size, multiplier)))
    return jobs


def stochastic_stream(job_count: int, rho: float, seed: int, model: TransferModel, multiplier: float) -> List[PromotionJob]:
    if not 0 < rho < 1:
        raise ValueError("rho must be between zero and one")
    rng = random.Random(seed)
    expected_service = sum(weight * model.service_time_s(size, multiplier) for size, weight in zip(SIZES, WEIGHTS))
    arrival_rate = rho / expected_service
    release = 0.0; jobs = []
    for index in range(job_count):
        release += rng.expovariate(arrival_rate)
        size = rng.choices(SIZES, weights=WEIGHTS, k=1)[0]
        gap = math.exp(rng.uniform(math.log(0.1), math.log(10.0)))
        jobs.append(PromotionJob(f"stream-{seed}-{index}", release, release + gap, size, model.service_time_s(size, multiplier)))
    return jobs
