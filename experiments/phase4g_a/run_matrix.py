"""Execute the frozen serial Phase 4G-A A/D paired matrix."""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
import subprocess
from pathlib import Path
from typing import Any

from experiments.phase4d_a import runtime
from experiments.phase4d_a.experiment import x_prompt
from experiments.phase4d_b.experiment import validate_runtime
from experiments.phase4g_a.run_phase4g_a import X_OUTPUT_TOKENS, run_lifecycle
from experiments.phase4g_a.workload import OUT, ROOT, SOURCE, SOURCE_SHA, build_manifest, load_source

SRC = runtime.SRC
START_COMMIT = "f4b19bd0ee0b7f3d4fe4b4732b8e8bcc4b2499f4"
RUN_ORDER = (("A", "D"), ("D", "A")) * 4


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def vllm_diff_sha() -> str:
    return hashlib.sha256(subprocess.check_output(["git", "diff"], cwd=SRC)).hexdigest()


def frozen_checks() -> str:
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA:
        raise RuntimeError("EXPERIMENT_INVALID_WORKLOAD:SHA")
    build_manifest(load_source())
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != START_COMMIT:
        raise RuntimeError(f"EXPERIMENT_INVALID_ENVIRONMENT:unexpected start commit {head}")
    return vllm_diff_sha()


def run_x_only(repetition: int, expected_vllm_sha: str) -> dict[str, Any]:
    if vllm_diff_sha() != expected_vllm_sha:
        raise RuntimeError("EXPERIMENT_INVALID_ENVIRONMENT:vLLM diff changed")
    label = f"x_only_{repetition}"
    run_dir = OUT / "runs" / label
    run_dir.mkdir(parents=True, exist_ok=True)
    proc, log = runtime.start_server(label, out=run_dir)
    try:
        _, block_size = runtime.wait_ready(proc)
        validate_runtime(log, block_size)
        warmup_prompt = runtime.tokenizer_ids("phase4g-a-warmup", label, 0, 20480)
        if len(runtime.send(warmup_prompt, 16, params={"max_offload_tokens": 0})["tokens"]) != 16:
            raise RuntimeError("warmup output mismatch")
        x = runtime.send(x_prompt(), X_OUTPUT_TOKENS, stream=True, params={"max_offload_tokens": 0})
        count = len(x["tokens"])
        duration = (x["done_ns"] - x["first_ns"]) / 1e6
        row = {"repetition": repetition, "X_generated_tokens": count,
               "X_decode_ms": duration, "X_TPOT_ms": duration / (count - 1) if count > 1 else "NA",
               "valid": count == X_OUTPUT_TOKENS and proc.poll() is None}
        write_json(run_dir / "run.json", row)
        if not row["valid"]:
            raise RuntimeError(f"EXPERIMENT_INVALID_MATRIX:{label}")
        return row
    finally:
        runtime.stop_server(proc, log, run_dir, label)
        if log.exists():
            lines = [line[:1200] for line in log.read_text(encoding="utf-8", errors="replace").splitlines()
                     if any(marker in line for marker in ("ERROR", "OOM", "GPU KV cache size"))]
            (OUT / "server_logs").mkdir(parents=True, exist_ok=True)
            (OUT / "server_logs" / f"{label}_compact.log").write_text("\n".join(lines[-200:]) + "\n", encoding="utf-8")


def csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_matrix() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fingerprint = frozen_checks()
    smoke = {policy: json.loads((OUT / f"lifecycle_smoke_{policy.lower()}.json").read_text())
             for policy in ("A", "D")}
    if not all(row["valid"] and row["X_tail_margin_ms"] >= 3000 for row in smoke.values()):
        raise RuntimeError("EXPERIMENT_INVALID_LIFECYCLE:Stage L gate")
    write_json(OUT / "run_order.json", {"repetitions": RUN_ORDER})
    write_json(OUT / "config.json", {"phase": "4G-A", "policies": ["OnDemand-K2", "FullSlackKV+PSA1-K2"],
        "K": 2, "P_SAFE_ms": 102, "resume_guard_ms": 110, "reservation_budget_blocks": 3510,
        "GPU_KV_bytes": 2147483648, "CPU_external_KV_bytes": 8589934592, "X_prompt_tokens": 9500,
        "X_output_tokens": X_OUTPUT_TOKENS, "workload_sha256": SOURCE_SHA,
        "pre_materialized_boundary_count": 0})
    write_json(OUT / "source_manifest.json", {"start_commit": START_COMMIT,
        "workload_sha256": SOURCE_SHA, "vllm_upstream_commit": "0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665",
        "vllm_working_diff_sha256": fingerprint})
    write_json(OUT / "environment.json", {"vllm": "0.23.0", "model": str(runtime.MODEL),
        "dtype": "float16", "TP": 1, "GPU": "GPU0", "GPU_KV_blocks": 4681,
        "block_size": 16, "python": subprocess.check_output([str(runtime.ENV_ROOT / "bin/python"), "--version"], text=True).strip(),
        "nvidia_smi": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader", "-i", "0"], text=True).strip()})
    (OUT / "commands.sh").write_text(
        "PYTHONPATH=/media/a6000/D/ZhaoBinHe/slackkv-phase2c-fix-sourceprobe:$PWD "
        "/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python -m experiments.phase4g_a.run_matrix\n",
        encoding="utf-8")
    x_rows: list[dict[str, Any]] = []
    measured: list[dict[str, Any]] = []
    try:
        for repetition in range(1, 5):
            x_rows.append(run_x_only(repetition, fingerprint))
            csv_rows(OUT / "x_only_runs.csv", x_rows)
        for repetition, pair in enumerate(RUN_ORDER, 1):
            for policy in pair:
                if vllm_diff_sha() != fingerprint:
                    raise RuntimeError("EXPERIMENT_INVALID_ENVIRONMENT:vLLM diff changed")
                result = run_lifecycle(policy, "measured", repetition)
                if not result["valid"] or result["X_tail_margin_ms"] <= 0:
                    raise RuntimeError(f"EXPERIMENT_INVALID_MATRIX:rep{repetition}:{policy}")
                measured.append({"repetition": repetition, "policy": result["policy"],
                                 "valid": result["valid"], "turns_complete": result["turns_complete"],
                                 "boundaries": result["boundaries"],
                                 "X_tail_margin_ms": result["X_tail_margin_ms"]})
                csv_rows(OUT / "measured_progress.csv", measured)
        for repetition in range(5, 9):
            x_rows.append(run_x_only(repetition, fingerprint))
            csv_rows(OUT / "x_only_runs.csv", x_rows)
        if len(x_rows) != 8 or len(measured) != 16:
            raise RuntimeError("EXPERIMENT_INVALID_MATRIX:incomplete matrix")
        from experiments.phase4g_a.finalize_phase4g_a import finalize
        finalize()
    except BaseException as exc:
        write_json(OUT / "failure_report.json", {
            "phase": "4G-A", "result": "EXPERIMENT_INVALID_MATRIX", "error_type": type(exc).__name__,
            "error": str(exc), "X_only_completed": len(x_rows), "measured_completed": len(measured),
            "partial_matrix_not_aggregated": True})
        raise


if __name__ == "__main__":
    run_matrix()
