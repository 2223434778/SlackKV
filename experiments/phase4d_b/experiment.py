from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from experiments.phase4d_a.experiment import (
    ROOT,
    build_resume_prompts,
    first_boundary_jobs,
    gpu_capacity_blocks,
    materialize_and_evict,
    write_csv,
    write_json,
)
from experiments.phase4d_a.helpers import (
    FROZEN_SHA256,
    minimum_h2d_mib,
    reservation_budget,
    sha256_file,
)
from experiments.phase4d_a_fix.release_fix import EXPECTED_GPU_BLOCKS

OUT = ROOT / "results/phase4d_b"
JOBS_PATH = ROOT / "results/phase4d_a/boundary0_jobs.csv"
WORKLOAD_PATH = ROOT / "results/phase4c/hybrid_workload.csv"
RESERVATION_BUDGET = 3510
GPU_BYTES = 2_147_483_648
CPU_BYTES = 8_589_934_592
BLOCK_SIZE = 16
K2 = 2
P_SAFE_MS = 102
RESUME_GUARD_MS = 110


def frozen_jobs() -> list[dict[str, Any]]:
    actual_sha = sha256_file(WORKLOAD_PATH)
    if actual_sha != FROZEN_SHA256:
        raise RuntimeError(f"EXPERIMENT_INVALID_WORKLOAD:{actual_sha}")
    jobs = first_boundary_jobs(WORKLOAD_PATH)
    with JOBS_PATH.open(encoding="utf-8", newline="") as stream:
        frozen = {row["episode_id"]: row for row in csv.DictReader(stream)}
    if len(frozen) != 8:
        raise RuntimeError("EXPERIMENT_INVALID_WORKLOAD:frozen job artifact count")
    for job in jobs:
        stored = frozen.get(job["episode_id"])
        if stored is None:
            raise RuntimeError("EXPERIMENT_INVALID_WORKLOAD:job missing from frozen artifact")
        for field in ("config", "session_hash", "required_prefix_tokens", "required_blocks", "lease_key"):
            if str(job[field]) != str(stored[field]):
                raise RuntimeError(f"EXPERIMENT_INVALID_WORKLOAD:{job['episode_id']}:{field}")
        if not math.isclose(float(job["deadline_ms"]), float(stored["deadline_ms"]), abs_tol=1e-9):
            raise RuntimeError(f"EXPERIMENT_INVALID_WORKLOAD:{job['episode_id']}:deadline")
    if len(jobs) != 8:
        raise RuntimeError("EXPERIMENT_INVALID_WORKLOAD:expected exactly 8 jobs")
    if sum(int(job["required_prefix_tokens"]) for job in jobs) != 168016:
        raise RuntimeError("EXPERIMENT_INVALID_WORKLOAD:token total")
    if sum(int(job["required_blocks"]) for job in jobs) != 10501:
        raise RuntimeError("EXPERIMENT_INVALID_WORKLOAD:block total")
    if not math.isclose(minimum_h2d_mib(jobs), 4594.1875, abs_tol=1e-9):
        raise RuntimeError("EXPERIMENT_INVALID_WORKLOAD:minimum H2D")
    if reservation_budget(EXPECTED_GPU_BLOCKS, jobs) != RESERVATION_BUDGET:
        raise RuntimeError("EXPERIMENT_INVALID_WORKLOAD:reservation budget")
    return jobs


def reconstructed_data(jobs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    data = build_resume_prompts(jobs)
    if set(data) != {job["episode_id"] for job in jobs}:
        raise RuntimeError("EXPERIMENT_INVALID_WORKLOAD:Turn1 reconstruction")
    return data


def validate_runtime(log: Path, actual_block_size: int) -> dict[str, int]:
    if actual_block_size != BLOCK_SIZE:
        raise RuntimeError("EXPERIMENT_INVALID_ENVIRONMENT:block size")
    total_blocks = gpu_capacity_blocks(log, BLOCK_SIZE)
    if total_blocks != EXPECTED_GPU_BLOCKS:
        raise RuntimeError("EXPERIMENT_INVALID_ENVIRONMENT:GPU KV blocks")
    budget = math.floor(0.75 * total_blocks)
    if budget != RESERVATION_BUDGET:
        raise RuntimeError("EXPERIMENT_INVALID_ENVIRONMENT:reservation budget")
    return {"total_gpu_kv_blocks": total_blocks, "reservation_budget_blocks": budget}


def save_run_artifacts(rows_by_path: dict[str, list[dict[str, Any]]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for relative, rows in rows_by_path.items():
        path = OUT / relative
        if rows:
            write_csv(path, rows)
        elif not path.exists():
            write_csv(path, [])


def write_run_context(start_commit: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(
        OUT / "source_manifest.json",
        {
            "phase": "Phase 4D-B",
            "start_commit": start_commit,
            "phase4c_workload_sha256": hashlib.sha256(WORKLOAD_PATH.read_bytes()).hexdigest(),
            "frozen_jobs_path": str(JOBS_PATH.relative_to(ROOT)),
            "vllm_baseline_commit": "0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665",
        },
    )
