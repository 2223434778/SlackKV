#!/usr/bin/env python3
"""Run the fixed Phase 1A simulation matrix and preserve job-level audit rows."""
import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from generate_workloads import controlled_burst, stochastic_stream
from slackkv.simulator.policies import Policy
from slackkv.simulator.simulator import aggregate, simulate
from slackkv.simulator.transfer_model import SIZE_MIB, TransferModel


RAW_FIELDS = ["run_id", "policy", "seed", "job_id", "release_time_s", "deadline_s", "kv_size_bytes", "service_time_s", "start_time_s", "completion_time_s", "stall_ms", "prefetch_residency_ms", "prefetch_vram_time_gbs"]
AGG_FIELDS = ["experiment", "source_tier", "service_multiplier", "rho", "configured_job_count", "seed", "policy", "job_count", "deadline_miss_count", "deadline_miss_rate", "stall_ms_mean", "stall_ms_p50", "stall_ms_p95", "stall_ms_p99", "max_stall_ms", "prefetch_residency_ms_mean", "prefetch_residency_ms_p50", "prefetch_residency_ms_p95", "prefetch_vram_time_gbs_total", "prefetch_vram_time_gbs_per_job", "keep_reference_vram_time_gbs_total", "prefetch_to_keep_ratio", "transfer_engine_utilization", "transfer_engine_idle_fraction", "total_transfer_bytes", "completed_before_deadline_fraction", "peak_prefetched_gb", "p95_prefetched_gb", "mean_prefetched_gb", "prefetch_budget_8gib_exceeded", "prefetch_budget_16gib_exceeded", "prefetch_budget_24gib_exceeded"]
WORKLOAD_FIELDS = ["run_id", "experiment", "source_tier", "service_multiplier", "rho", "configured_job_count", "seed", "actual_job_count", "release_min_s", "release_max_s", "deadline_min_s", "deadline_max_s", "total_kv_bytes"]


def emit_configuration(experiment: str, source_tier: str, multiplier: float, rho: str, configured_count: int, seed: int, jobs, raw_rows: List[Dict], aggregate_rows: List[Dict], workload_rows: List[Dict]) -> None:
    run_id = f"{experiment}|{source_tier}|m{multiplier:g}|r{rho}|n{configured_count}|s{seed}"
    context = {"experiment": experiment, "source_tier": source_tier, "service_multiplier": multiplier, "rho": rho, "configured_job_count": configured_count, "seed": seed}
    workload_rows.append({**context, "actual_job_count": len(jobs), "release_min_s": min(job.release_time_s for job in jobs), "release_max_s": max(job.release_time_s for job in jobs), "deadline_min_s": min(job.deadline_s for job in jobs), "deadline_max_s": max(job.deadline_s for job in jobs), "total_kv_bytes": sum(job.kv_size_bytes for job in jobs)})
    for policy in Policy:
        result = simulate(jobs, policy)
        aggregate_rows.append({**context, "policy": policy.value, **aggregate(result)})
        for outcome in result.outcomes:
            raw_rows.append({"run_id": run_id, "policy": policy.value, "seed": seed, "job_id": outcome.job.job_id, "release_time_s": outcome.job.release_time_s, "deadline_s": outcome.job.deadline_s, "kv_size_bytes": outcome.job.kv_size_bytes, "service_time_s": outcome.job.service_time_s, "start_time_s": outcome.start_time_s, "completion_time_s": outcome.completion_time_s, "stall_ms": outcome.stall_s * 1000.0, "prefetch_residency_ms": outcome.prefetch_residency_s * 1000.0, "prefetch_vram_time_gbs": outcome.prefetch_vram_time_gbs})


def write_csv(path: Path, fields: Iterable[str], rows: Iterable[Dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format(value, ".9g") if isinstance(value, float) else value for key, value in row.items()})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/phase1a")
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--quick", action="store_true", help="small validation matrix; never used for final results")
    args = parser.parse_args()
    root = Path(args.repository_root).resolve(); output = (root / args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    raw_rows: List[Dict] = []; aggregate_rows: List[Dict] = []; workload_rows: List[Dict] = []
    burst_seeds = (11, 22, 33, 44, 55); stream_seeds = (101, 202, 303)
    stream_jobs = 100 if args.quick else 5000
    for tier in ("cpu",):
        model = TransferModel.from_phase0(root, tier)
        for count in (4, 8, 16, 32):
            for seed in burst_seeds:
                emit_configuration("controlled_burst", tier, 1.0, "", count, seed, controlled_burst(count, seed, model, 1.0), raw_rows, aggregate_rows, workload_rows)
        for rho in (.25, .50, .75, .90):
            for seed in stream_seeds:
                emit_configuration("stochastic_stream", tier, 1.0, rho, stream_jobs, seed, stochastic_stream(stream_jobs, rho, seed, model, 1.0), raw_rows, aggregate_rows, workload_rows)
        for multiplier in (1.0, 1.05, 1.15, 1.30):
            seed = 101
            emit_configuration("service_sensitivity", tier, multiplier, .75, stream_jobs, seed, stochastic_stream(stream_jobs, .75, seed, model, multiplier), raw_rows, aggregate_rows, workload_rows)
    peer_model = TransferModel.from_phase0(root, "peer")
    emit_configuration("peer_ablation", "peer", 1.0, .75, stream_jobs, 101, stochastic_stream(stream_jobs, .75, 101, peer_model, 1.0), raw_rows, aggregate_rows, workload_rows)
    write_csv(output / "metrics.csv", RAW_FIELDS, raw_rows)
    write_csv(output / "aggregate_metrics.csv", AGG_FIELDS, aggregate_rows)
    write_csv(output / "workload_summary.csv", WORKLOAD_FIELDS, workload_rows)
    cpu_model = TransferModel.from_phase0(root, "cpu")
    peer_model = TransferModel.from_phase0(root, "peer")
    config = {"phase": "Phase 1A — Queue-Aware JIT KV Promotion Simulator", "phase0_calibration_commit": cpu_model.phase0_commit, "calibration_p50_latency_ms": {"cpu_to_gpu0": {str(size): cpu_model.service_time_s(size) * 1000 for size in cpu_model.p50_latency_s_by_bytes}, "gpu1_to_gpu0": {str(size): peer_model.service_time_s(size) * 1000 for size in peer_model.p50_latency_s_by_bytes}}, "payload_sizes_mib": list(SIZE_MIB), "resource_model": "one serialized, non-preemptive inbound promotion resource to GPU0", "oracle_deadlines": True, "policies": [policy.value for policy in Policy], "controlled_burst": {"job_counts": [4, 8, 16, 32], "seeds": list(burst_seeds), "resume_gap_s": [0.1, .25, .5, 1, 2, 5, 10]}, "stochastic_stream": {"job_count_per_configuration": stream_jobs, "seeds": list(stream_seeds), "nominal_rho": [.25, .50, .75, .90], "arrival_process": "Poisson", "resume_gap_distribution": "log-uniform 0.1 to 10 s", "size_weights": {"256MiB": .50, "1GiB": .35, "4GiB": .15}}, "sensitivity": {"rho": .75, "seed": 101, "multipliers": [1.0, 1.05, 1.15, 1.30]}, "peer_ablation": {"rho": .75, "seed": 101}}
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "commands.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\nPYTHON=/home/a6000/anaconda3/bin/python\n$PYTHON -m pytest -q tests/test_phase1a_simulator.py\n$PYTHON experiments/phase1a/run_phase1a.py\n$PYTHON scripts/plot_phase1a.py\n")
    (output / "commands.sh").chmod(0o755)
    print(f"wrote {len(raw_rows)} job rows and {len(aggregate_rows)} aggregate rows to {output}")


if __name__ == "__main__":
    main()
