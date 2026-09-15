#!/usr/bin/env python3
"""Phase 2A: real single-GPU tensor offload/prefetch runtime validation."""
from __future__ import annotations

import csv
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "phase2a"
DEVICE = torch.device("cuda:0")
DTYPE = torch.float16
SIZES = (256 * 2**20, 1024 * 2**20, 4096 * 2**20)
WARMUP_RUNS = 3
MEASURED_RUNS = 10
JIT_PAUSES_S = (1.0, 2.0, 5.0)
JIT_MARGINS_S = (0.0, 0.250, 0.500)
JIT_REPETITIONS = 5
SAMPLE_COUNT = 1024


def numel_for_bytes(size_bytes: int, element_size_bytes: int) -> int:
    """Return an exactly representable contiguous tensor element count."""
    if size_bytes <= 0 or size_bytes % element_size_bytes:
        raise ValueError("requested tensor byte size must be positive and divisible")
    return size_bytes // element_size_bytes


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("cannot calculate percentile of no values")
    return float(torch.quantile(torch.tensor(values, dtype=torch.float64), q).item())


def sampled_equal(
    tensor: torch.Tensor, sample_indices: torch.Tensor, expected: torch.Tensor
) -> bool:
    observed = tensor.index_select(0, sample_indices.to(tensor.device)).cpu()
    return bool(torch.equal(observed, expected.cpu()))


def resume_metrics(completion_s: float, deadline_s: float) -> tuple[float, float]:
    return (
        max(0.0, completion_s - deadline_s),
        max(0.0, deadline_s - completion_s),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def phase0_p50_ms() -> dict[tuple[str, int], float]:
    values: dict[tuple[str, int], float] = {}
    with (ROOT / "results" / "phase0" / "summary_metrics.csv").open(
        newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            direction = row["direction"]
            size = int(row["size_bytes"])
            if direction in {"gpu0_to_pinned_cpu", "pinned_cpu_to_gpu0"}:
                values[(direction, size)] = float(row["latency_ms_p50"])
    return values


def sample_data(numel: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(20260915)
    indices = torch.randint(numel, (SAMPLE_COUNT,), generator=generator, dtype=torch.int64)
    if indices.unique().numel() != SAMPLE_COUNT:
        raise RuntimeError('fixed sample generator unexpectedly produced duplicate indices')
    expected = torch.arange(SAMPLE_COUNT, dtype=DTYPE)
    return indices, expected


def make_source(numel: int, indices: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    source = torch.zeros(numel, device=DEVICE, dtype=DTYPE)
    source.index_copy_(0, indices.to(DEVICE), expected.to(DEVICE))
    torch.cuda.synchronize(DEVICE)
    return source


def timed_copy(
    destination: torch.Tensor, source: torch.Tensor, stream: torch.cuda.Stream
) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        start.record(stream)
        destination.copy_(source, non_blocking=True)
        end.record(stream)
    end.synchronize()
    return float(start.elapsed_time(end))


def one_transfer_run(size_bytes: int, stream: torch.cuda.Stream) -> dict[str, Any]:
    numel = numel_for_bytes(size_bytes, torch.empty((), dtype=DTYPE).element_size())
    indices, expected = sample_data(numel)
    source = make_source(numel, indices, expected)
    cpu_buffer = torch.empty(numel, dtype=DTYPE, pin_memory=True)

    allocated_before = int(torch.cuda.memory_allocated(DEVICE))
    reserved_before = int(torch.cuda.memory_reserved(DEVICE))
    d2h_ms = timed_copy(cpu_buffer, source, stream)

    del source
    torch.cuda.synchronize(DEVICE)
    allocated_after_offload = int(torch.cuda.memory_allocated(DEVICE))
    reserved_after_offload = int(torch.cuda.memory_reserved(DEVICE))
    allocated_decreased = allocated_after_offload < allocated_before

    replacement = torch.empty(numel, device=DEVICE, dtype=DTYPE)
    replacement.fill_(1)
    torch.cuda.synchronize(DEVICE)
    replacement_allocated = int(torch.cuda.memory_allocated(DEVICE))
    replacement_ok = replacement.numel() == numel
    del replacement
    torch.cuda.synchronize(DEVICE)

    restored = torch.empty(numel, device=DEVICE, dtype=DTYPE)
    h2d_ms = timed_copy(restored, cpu_buffer, stream)
    correctness = sampled_equal(restored, indices, expected)

    row = {
        "size_bytes": size_bytes,
        "size_mib": size_bytes / 2**20,
        "d2h_ms": d2h_ms,
        "h2d_ms": h2d_ms,
        "roundtrip_ms": d2h_ms + h2d_ms,
        "allocated_before_bytes": allocated_before,
        "allocated_after_offload_bytes": allocated_after_offload,
        "reserved_before_bytes": reserved_before,
        "reserved_after_offload_bytes": reserved_after_offload,
        "replacement_allocated_bytes": replacement_allocated,
        "memory_allocated_decreased": allocated_decreased,
        "replacement_ok": replacement_ok,
        "correctness": correctness,
    }
    del restored, cpu_buffer
    torch.cuda.synchronize(DEVICE)
    return row


def transfer_experiment() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    phase0 = phase0_p50_ms()
    raw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    stream = torch.cuda.Stream(device=DEVICE)

    for size_bytes in SIZES:
        for _ in range(WARMUP_RUNS):
            warmup = one_transfer_run(size_bytes, stream)
            if not warmup["correctness"] or not warmup["replacement_ok"]:
                raise RuntimeError("warm-up correctness or replacement allocation failed")

        measured = [one_transfer_run(size_bytes, stream) for _ in range(MEASURED_RUNS)]
        raw_rows.extend({"phase": "transfer", "run": index, **row} for index, row in enumerate(measured))

        d2h = [float(row["d2h_ms"]) for row in measured]
        h2d = [float(row["h2d_ms"]) for row in measured]
        roundtrip = [float(row["roundtrip_ms"]) for row in measured]
        phase0_d2h = phase0[("gpu0_to_pinned_cpu", size_bytes)]
        phase0_h2d = phase0[("pinned_cpu_to_gpu0", size_bytes)]
        summary_rows.append(
            {
                "size_bytes": size_bytes,
                "size_mib": size_bytes / 2**20,
                "d2h_median_ms": statistics.median(d2h),
                "d2h_p95_ms": percentile(d2h, 0.95),
                "h2d_median_ms": statistics.median(h2d),
                "h2d_p95_ms": percentile(h2d, 0.95),
                "roundtrip_median_ms": statistics.median(roundtrip),
                "roundtrip_p95_ms": percentile(roundtrip, 0.95),
                "d2h_effective_gbps": (size_bytes / 1e9) / (statistics.median(d2h) / 1000),
                "h2d_effective_gbps": (size_bytes / 1e9) / (statistics.median(h2d) / 1000),
                "phase0_d2h_p50_ms": phase0_d2h,
                "phase0_h2d_p50_ms": phase0_h2d,
                "d2h_relative_error": abs(statistics.median(d2h) - phase0_d2h) / phase0_d2h,
                "h2d_relative_error": abs(statistics.median(h2d) - phase0_h2d) / phase0_h2d,
                "memory_reuse": "PASS"
                if all(row["memory_allocated_decreased"] and row["replacement_ok"] for row in measured)
                else "FAIL",
                "correctness": "PASS"
                if all(row["correctness"] for row in measured)
                else "FAIL",
            }
        )
    return raw_rows, summary_rows


def sleep_until(target_s: float) -> None:
    remaining = target_s - time.perf_counter()
    if remaining > 0:
        time.sleep(remaining)


def jit_experiment(estimated_h2d_ms: float) -> list[dict[str, Any]]:
    size_bytes = 1024 * 2**20
    numel = numel_for_bytes(size_bytes, torch.empty((), dtype=DTYPE).element_size())
    indices, expected = sample_data(numel)
    stream = torch.cuda.Stream(device=DEVICE)
    cpu_buffer = torch.empty(numel, dtype=DTYPE, pin_memory=True)
    raw_rows: list[dict[str, Any]] = []
    estimate_s = estimated_h2d_ms / 1000

    for pause_s in JIT_PAUSES_S:
        for margin_s in JIT_MARGINS_S:
            for repetition in range(JIT_REPETITIONS):
                source = make_source(numel, indices, expected)
                _ = timed_copy(cpu_buffer, source, stream)
                del source
                torch.cuda.synchronize(DEVICE)

                offload_complete_s = time.perf_counter()
                deadline_s = offload_complete_s + pause_s
                scheduled_start_s = deadline_s - estimate_s - margin_s
                sleep_until(scheduled_start_s)

                restored = torch.empty(numel, device=DEVICE, dtype=DTYPE)
                actual_start_s = time.perf_counter()
                h2d_ms = timed_copy(restored, cpu_buffer, stream)
                completion_s = time.perf_counter()
                stall_s, early_s = resume_metrics(completion_s, deadline_s)
                if not sampled_equal(restored, indices, expected):
                    raise RuntimeError("JIT round-trip sampled correctness failed")
                raw_rows.append(
                    {
                        "pause_s": pause_s,
                        "margin_ms": margin_s * 1000,
                        "repetition": repetition,
                        "estimated_h2d_ms": estimated_h2d_ms,
                        "actual_h2d_ms": h2d_ms,
                        "scheduled_prefetch_start_after_offload_ms": (scheduled_start_s - offload_complete_s) * 1000,
                        "actual_prefetch_start_after_offload_ms": (actual_start_s - offload_complete_s) * 1000,
                        "completion_after_offload_ms": (completion_s - offload_complete_s) * 1000,
                        "resume_stall_ms": stall_s * 1000,
                        "early_residency_ms": early_s * 1000,
                    }
                )
                del restored
                torch.cuda.synchronize(DEVICE)

    del cpu_buffer
    torch.cuda.synchronize(DEVICE)
    return raw_rows


def summarize_jit(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for pause_s in JIT_PAUSES_S:
        for margin_s in JIT_MARGINS_S:
            rows = [
                row
                for row in raw_rows
                if row["pause_s"] == pause_s and row["margin_ms"] == margin_s * 1000
            ]
            summaries.append(
                {
                    "pause_s": pause_s,
                    "margin_ms": margin_s * 1000,
                    "resume_stall_ms_median": statistics.median(
                        float(row["resume_stall_ms"]) for row in rows
                    ),
                    "early_residency_ms_median": statistics.median(
                        float(row["early_residency_ms"]) for row in rows
                    ),
                    "actual_h2d_ms_median": statistics.median(
                        float(row["actual_h2d_ms"]) for row in rows
                    ),
                }
            )
    return summaries


def gpu_environment() -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).splitlines()[0]
    except (OSError, subprocess.CalledProcessError):
        driver = "unavailable"
    return {
        "gpu_index": 0,
        "gpu_name": properties.name,
        "total_gpu_memory_bytes": properties.total_memory,
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "driver_version": driver,
        "python_version": sys.version,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 2A requires CUDA")
    torch.cuda.set_device(0)
    torch.manual_seed(20260915)
    RESULTS.mkdir(parents=True, exist_ok=True)

    raw_transfer, transfer_summary = transfer_experiment()
    write_csv(RESULTS / "transfer_metrics.csv", transfer_summary)
    jit_raw = jit_experiment(
        next(row["h2d_median_ms"] for row in transfer_summary if row["size_bytes"] == 1024 * 2**20)
    )
    write_csv(RESULTS / "jit_metrics.csv", summarize_jit(jit_raw))
    write_csv(RESULTS / "raw_runs.csv", raw_transfer)
    write_csv(RESULTS / "raw_jit_runs.csv", jit_raw)
    (RESULTS / "config.json").write_text(
        json.dumps(
            {
                "base_commit": "7f3c4966e7e1d6657cedd1a1a6bf73a3afe1a5e1",
                "gpu": 0,
                "dtype": str(DTYPE),
                "sizes_bytes": SIZES,
                "warmup_runs": WARMUP_RUNS,
                "measured_runs": MEASURED_RUNS,
                "jit_pauses_s": JIT_PAUSES_S,
                "jit_margins_ms": [margin * 1000 for margin in JIT_MARGINS_S],
                "jit_repetitions": JIT_REPETITIONS,
                "sample_count": SAMPLE_COUNT,
                "phase0_calibration_commit": "0337837b43ddf78821839d86be6a89ae2a7c164a",
                "no_empty_cache_during_primary_experiment": True,
            },
            indent=2,
        )
        + "\n"
    )
    (RESULTS / "environment.json").write_text(json.dumps(gpu_environment(), indent=2) + "\n")
    (RESULTS / "commands.sh").write_text(
        "python -m pytest -q tests/test_phase2a.py\n"
        "python experiments/phase2a/run_phase2a.py\n"
        "python scripts/plot_phase2a.py\n"
    )
    print(f"wrote {len(transfer_summary)} transfer summaries and {len(jit_raw)} JIT rows")


if __name__ == "__main__":
    main()
