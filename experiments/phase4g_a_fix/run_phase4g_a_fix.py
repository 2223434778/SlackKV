"""Run Phase 4G-A-FIX with a frozen, sustained four-request background."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from experiments.phase4d_a import runtime
from experiments.phase4d_a.experiment import x_prompt
from experiments.phase4d_b.experiment import validate_runtime
from experiments.phase4g_a import run_phase4g_a as legacy
from experiments.phase4g_a.workload import ROOT, SOURCE, SOURCE_SHA, build_manifest, load_source
from experiments.phase4g_a_fix.background import (
    BG_CHUNK_OUTPUT_TOKENS,
    BG_CHUNKS,
    BG_POST_FIRST_TOKENS,
    BG_PROMPT_TOKENS,
    BG_TOTAL_OUTPUT_TOKENS,
    CALIBRATION_TAIL_MS,
    MAX_IDLE_GAP_MS,
    adjust_local_hit_attribution,
    background_tails,
    background_valid,
    run_chunks,
    summarize_chunks,
)

OUT = ROOT / "results/phase4g_a_fix"
TEMP_LOGS = Path("/tmp/slackkv-phase4g-a-fix-server-logs")
START_COMMIT = "1d9fa6ade07698071a7b74cae7241781b943e10b"
VLLM_EXPECTED_DIFF_SHA = "f0f8e327cc7ff4eb1797852a24e56e0e7f4a77b62e0f9439e88405fd5b470ae4"
BG_CAL_ORDER = (("A", 1), ("D", 2), ("D", 3), ("A", 4))
RUN_ORDER = (("A", "D"), ("D", "A")) * 4
POLICY_NAMES = {"A": "OnDemand-K2", "D": "FullSlackKV+PSA1-K2"}
ACTIVE_BACKGROUND: dict[str, Any] = {}
ORIGINAL_START_SERVER = runtime.start_server


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    existed = path.exists()
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        if not existed:
            writer.writeheader()
        writer.writerows(rows)


def vllm_diff_sha() -> str:
    diff = subprocess.check_output(["git", "diff", "--binary"], cwd=runtime.SRC)
    return hashlib.sha256(diff).hexdigest()


def frozen_checks(*, require_start_commit: bool = True) -> str:
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA:
        raise RuntimeError("EXPERIMENT_INVALID_WORKLOAD:Phase4C SHA mismatch")
    build_manifest(load_source())
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if require_start_commit and head != START_COMMIT:
        raise RuntimeError(f"EXPERIMENT_INVALID_ENVIRONMENT:start commit {head}")
    source_sha = vllm_diff_sha()
    if source_sha != VLLM_EXPECTED_DIFF_SHA:
        raise RuntimeError(f"EXPERIMENT_INVALID_ENVIRONMENT:vLLM fingerprint {source_sha}")
    return source_sha


def start_sustained_background():
    prompt = x_prompt()
    if len(prompt) != BG_PROMPT_TOKENS:
        raise RuntimeError("background prompt token count mismatch")
    ready = threading.Event()
    first: dict[str, int] = {}

    def on_first(timestamp_ns: int) -> None:
        first["ns"] = timestamp_ns
        ready.set()

    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(run_chunks, runtime, prompt, on_first)
    ACTIVE_BACKGROUND.clear()
    ACTIVE_BACKGROUND.update({"pool": pool, "future": future})
    if not ready.wait(180):
        pool.shutdown(wait=False, cancel_futures=True)
        raise TimeoutError("BGv2 X0 first token timeout")
    return pool, future, first["ns"]


def _temporary_server_start(label: str, connector: bool = True,
                            gpu_bytes: int = 2147483648, out: Path | None = None):
    TEMP_LOGS.mkdir(parents=True, exist_ok=True)
    return ORIGINAL_START_SERVER(label, connector=connector, gpu_bytes=gpu_bytes, out=TEMP_LOGS)


def _persist_background_rows(
    chunks: list[dict[str, Any]], stage: str, repetition: int, policy: str
) -> None:
    append_csv(OUT / "background_chunks.csv", [
        {"stage": stage, "repetition": repetition, "policy": policy, **chunk}
        for chunk in chunks
    ])


def _decorate_agent_run(result: dict[str, Any], label: str) -> dict[str, Any]:
    bg = ACTIVE_BACKGROUND["future"].result(timeout=360)
    ACTIVE_BACKGROUND["pool"].shutdown(wait=True)
    chunks = adjust_local_hit_attribution(bg["chunks"], result.get("source_trace", []))
    result["background_chunks"] = chunks
    result["BG_chunks_complete"] = len(chunks)
    result["BG_total_generated_tokens"] = len(bg["tokens"])
    result["BG_total_wall_ms"] = (int(bg["done_ns"]) - int(bg["first_ns"])) / 1e6
    result["BG_weighted_TPOT_ms"] = summarize_chunks(chunks)["weighted_TPOT_ms"]
    result["BG_max_inter_chunk_gap_ms"] = summarize_chunks(chunks)["max_inter_chunk_idle_gap_ms"]
    last_first = max(int(row["first_ns"]) for row in result["turn_rows"])
    last_finish = max(int(row["finish_ns"]) for row in result["turn_rows"])
    first_tail, finish_tail = background_tails(int(bg["done_ns"]), last_first, last_finish)
    result["BG_tail_first_token_ms"] = first_tail
    result["BG_tail_request_finish_ms"] = finish_tail
    result["X_tokens"] = len(bg["tokens"])
    result["X_first_ns"] = int(bg["first_ns"])
    result["X_finish_ns"] = int(bg["done_ns"])
    result["X_decode_ms"] = result["BG_total_wall_ms"]
    result["X_TPOT_ms"] = result["BG_weighted_TPOT_ms"]
    result["X_tail_margin_ms"] = first_tail
    # The request-finish endpoint is the Phase 4G-A-FIX validity endpoint.
    result["valid"] = bool(
        result.get("valid")
        and len(bg["tokens"]) == BG_TOTAL_OUTPUT_TOKENS
        and result["BG_max_inter_chunk_gap_ms"] <= MAX_IDLE_GAP_MS
        and finish_tail > 0
    )
    result_path = OUT / "runs" / label / "run.json"
    write_json(result_path, result)
    _persist_background_rows(chunks, result["stage"], int(result["repetition"]), result["policy"])
    return result


def run_agent(policy: str, stage: str, repetition: int) -> dict[str, Any]:
    if policy not in {"A", "D"}:
        raise ValueError("Phase 4G-A-FIX contains only policies A and D")
    if vllm_diff_sha() != VLLM_EXPECTED_DIFF_SHA:
        raise RuntimeError("EXPERIMENT_INVALID_ENVIRONMENT:vLLM mechanism fingerprint changed")
    configure_legacy_lifecycle()
    original_start = runtime.start_server
    runtime.start_server = _temporary_server_start
    label = f"{stage}_rep{repetition}_{policy.lower()}"
    ACTIVE_BACKGROUND.clear()
    try:
        result = legacy.run_lifecycle(policy, stage, repetition)
        return _decorate_agent_run(result, label)
    except BaseException:
        run_path = OUT / "runs" / label / "run.json"
        future = ACTIVE_BACKGROUND.get("future")
        if future is not None and future.done() and run_path.exists():
            try:
                saved = json.loads(run_path.read_text(encoding="utf-8"))
                bg = future.result()
                if "turn_rows" in saved:
                    chunks = adjust_local_hit_attribution(bg["chunks"], saved.get("source_trace", []))
                    saved["background_chunks"] = chunks
                    saved["BG_chunks_complete"] = len(chunks)
                    saved["BG_total_generated_tokens"] = len(bg["tokens"])
                    last_first = max(int(row["first_ns"]) for row in saved["turn_rows"])
                    last_finish = max(int(row["finish_ns"]) for row in saved["turn_rows"])
                    saved["BG_tail_first_token_ms"], saved["BG_tail_request_finish_ms"] = background_tails(
                        int(bg["done_ns"]), last_first, last_finish
                    )
                    saved["valid"] = False
                    write_json(run_path, saved)
                    _persist_background_rows(chunks, stage, repetition, saved.get("policy", policy))
                ACTIVE_BACKGROUND["pool"].shutdown(wait=True)
            except BaseException:
                pass
        raise
    finally:
        runtime.start_server = original_start


def configure_legacy_lifecycle() -> None:
    legacy.OUT = OUT
    legacy.start_background = start_sustained_background
    # The legacy lifecycle owns resume orchestration, but its X-length check
    # must follow this phase's four fixed chunks instead of the old single X.
    legacy.X_OUTPUT_TOKENS = BG_TOTAL_OUTPUT_TOKENS


def warmup(proc: Any, log: Path, label: str) -> None:
    _, block_size = runtime.wait_ready(proc)
    validate_runtime(log, block_size)
    prompt = runtime.tokenizer_ids("phase4g-a-warmup", label, 0, 20480)
    response = runtime.send(prompt, 16, params={"max_offload_tokens": 0})
    if len(response["tokens"]) != 16:
        raise RuntimeError("warmup output mismatch")


def run_bg_only(repetition: int, side: str) -> dict[str, Any]:
    if side not in {"pre", "post"}:
        raise ValueError("background-only side must be pre or post")
    if vllm_diff_sha() != VLLM_EXPECTED_DIFF_SHA:
        raise RuntimeError("EXPERIMENT_INVALID_ENVIRONMENT:vLLM mechanism fingerprint changed")
    label = f"bg_only_{side}_{repetition}"
    run_dir = OUT / "runs" / label
    run_dir.mkdir(parents=True, exist_ok=True)
    proc, log = _temporary_server_start(label, out=run_dir)
    try:
        warmup(proc, log, label)
        result = run_chunks(runtime, x_prompt())
        chunks = result["chunks"]
        summary = summarize_chunks(chunks)
        if summary["total_generated_tokens"] != BG_TOTAL_OUTPUT_TOKENS:
            raise RuntimeError("RUN_INVALID_BACKGROUND:BG-only output count")
        row = {
            "side": side,
            "repetition": repetition,
            "chunks_complete": summary["chunks_complete"],
            "total_generated_tokens": summary["total_generated_tokens"],
            "BGWall_ms": summary["wall_ms"],
            "weighted_decode_TPOT_ms": summary["weighted_TPOT_ms"],
            "max_inter_chunk_idle_gap_ms": summary["max_inter_chunk_idle_gap_ms"],
            "valid": (proc.poll() is None
                      and summary["max_inter_chunk_idle_gap_ms"] <= MAX_IDLE_GAP_MS),
        }
        write_json(run_dir / "run.json", {**row, "chunks": chunks})
        _persist_background_rows(chunks, f"bg_only_{side}", repetition, "BG-only")
        existing = list(csv.DictReader((OUT / "bg_only_runs.csv").open(encoding="utf-8"))) if (OUT / "bg_only_runs.csv").exists() else []
        write_csv(OUT / "bg_only_runs.csv", existing + [row])
        if not row["valid"]:
            raise RuntimeError("RUN_INVALID_BACKGROUND:bg-only server or continuity gate")
        return row
    finally:
        runtime.stop_server(proc, log, run_dir, label)


def save_initial_artifacts() -> dict[str, Any]:
    unexpected = ([path.name for path in OUT.iterdir() if path.name != "failures"]
                  if OUT.exists() else [])
    if unexpected:
        raise RuntimeError("Phase 4G-A-FIX output directory already contains data; refusing to mix attempts")
    OUT.mkdir(parents=True, exist_ok=True)
    vllm_sha = frozen_checks()
    env = {
        "python": subprocess.check_output([str(runtime.ENV_ROOT / "bin/python"), "--version"], text=True).strip(),
        "pytorch": subprocess.check_output([str(runtime.ENV_ROOT / "bin/python"), "-c", "import torch; print(torch.__version__); print(torch.version.cuda)"], text=True).splitlines(),
        "vllm": "0.23.0",
        "vllm_upstream_commit": "0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665",
        "vllm_working_diff_sha256": vllm_sha,
        "model": str(runtime.MODEL),
        "dtype": "float16",
        "tensor_parallel": 1,
        "cuda_visible_devices": "0",
        "gpu_and_driver": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader", "-i", "0"], text=True).strip(),
        "max_model_len": 32768,
        "GPU_KV_bytes": 2147483648,
        "CPU_external_KV_bytes": 8589934592,
        "block_size": 16,
    }
    source = {
        "start_commit": START_COMMIT,
        "workload_sha256": SOURCE_SHA,
        "vllm_upstream_commit": "0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665",
        "vllm_working_diff_sha256": vllm_sha,
        "prior_phase_algorithm_fingerprint_sha256": VLLM_EXPECTED_DIFF_SHA,
        "algorithm_fingerprint_unchanged": vllm_sha == VLLM_EXPECTED_DIFF_SHA,
    }
    config = {
        "phase": "4G-A-FIX",
        "policies": [POLICY_NAMES["A"], POLICY_NAMES["D"]],
        "BG_prompt_tokens": BG_PROMPT_TOKENS,
        "BG_chunk_output_tokens": BG_CHUNK_OUTPUT_TOKENS,
        "BG_chunks": BG_CHUNKS,
        "BG_total_output_tokens": BG_TOTAL_OUTPUT_TOKENS,
        "max_inter_chunk_idle_gap_ms": MAX_IDLE_GAP_MS,
        "calibration_required_tail_ms": CALIBRATION_TAIL_MS,
        "K": 2,
        "P_SAFE_ms": 102,
        "resume_guard_ms": 110,
        "reservation_budget_blocks": 3510,
        "workload_sha256": SOURCE_SHA,
        "measured_repetitions": 8,
        "measured_runs": 16,
    }
    write_json(OUT / "config.json", config)
    write_json(OUT / "environment.json", env)
    write_json(OUT / "source_manifest.json", source)
    write_json(OUT / "background_protocol.json", {
        "protocol": "BGv2_fixed_work_chunked",
        "prompt_tokens": BG_PROMPT_TOKENS,
        "chunk_output_tokens": BG_CHUNK_OUTPUT_TOKENS,
        "chunks": BG_CHUNKS,
        "total_output_tokens": BG_TOTAL_OUTPUT_TOKENS,
        "background_post_first_tokens": BG_POST_FIRST_TOKENS,
        "same_prompt_each_chunk": True,
        "max_inter_chunk_idle_gap_ms": MAX_IDLE_GAP_MS,
        "coverage_endpoint": "last_agent_request_finished",
        "calibration_required_margin_ms": CALIBRATION_TAIL_MS,
        "frozen_before_calibration": True,
    })
    write_json(OUT / "run_order.json", {
        "BG_CAL": BG_CAL_ORDER,
        "measured": RUN_ORDER,
        "BG_only_pre": [1, 2, 3, 4],
        "BG_only_post": [5, 6, 7, 8],
    })
    (OUT / "commands.sh").write_text(
        "set -euo pipefail\n"
        "cd /media/a6000/D/ZhaoBinHe/SlackKV\n"
        "/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python -m pytest -q tests/test_phase4g_a_fix.py\n"
        "/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python -m experiments.phase4g_a_fix.run_phase4g_a_fix all\n",
        encoding="utf-8",
    )
    write_json(OUT / "progress.json", {"phase": "4G-A-FIX", "stage": "initialized", "runs": []})
    return {"config": config, "environment": env, "source": source}


def _record_progress(stage: str, row: dict[str, Any]) -> None:
    path = OUT / "progress.json"
    progress = json.loads(path.read_text(encoding="utf-8"))
    progress["stage"] = stage
    progress.setdefault("runs", []).append(row)
    write_json(path, progress)


def finalize_stopped(result: str, reason: str, stage: str) -> None:
    write_json(OUT / "validation_report.json", {
        "phase": "4G-A-FIX",
        "start_commit": START_COMMIT,
        "workload_sha256": SOURCE_SHA,
        "final_result": result,
        "stopped_stage": stage,
        "failure_reason": reason,
        "measured_matrix_not_aggregated": True,
    })
    (OUT / "summary.md").write_text(
        f"# Phase 4G-A-FIX\n\nFinal result: **{result}**.\n\n"
        f"The experiment stopped during {stage}: {reason}. No incomplete measured matrix was aggregated.\n",
        encoding="utf-8",
    )


def run_all() -> None:
    save_initial_artifacts()
    try:
        calibration_rows = []
        for policy, repetition in BG_CAL_ORDER:
            _record_progress("BG-CAL", {"policy": POLICY_NAMES[policy], "repetition": repetition, "status": "started"})
            result = run_agent(policy, "smoke", repetition)
            last_finish = max(int(row["finish_ns"]) for row in result["turn_rows"])
            ok, tail = background_valid(result["background_chunks"], last_finish, calibration=True)
            row = {
                "policy": result["policy"],
                "repetition": repetition,
                "last_first_ms": (max(int(item["first_ns"]) for item in result["turn_rows"]) - int(result["X_first_ns"])) / 1e6,
                "last_finish_ms": (last_finish - int(result["X_first_ns"])) / 1e6,
                "BG_final_done_ms": (int(result["X_finish_ns"]) - int(result["X_first_ns"])) / 1e6,
                "tail_first_ms": result["BG_tail_first_token_ms"],
                "tail_finish_ms": tail,
                "max_inter_chunk_gap_ms": result["BG_max_inter_chunk_gap_ms"],
                "chunks_complete": result["BG_chunks_complete"],
                "valid": bool(result["valid"] and ok),
            }
            calibration_rows.append(row)
            write_csv(OUT / "background_calibration.csv", calibration_rows)
            _record_progress("BG-CAL", {"policy": result["policy"], "repetition": repetition, "valid": row["valid"], "tail_finish_ms": tail})
            if not row["valid"]:
                raise RuntimeError(f"EXPERIMENT_INVALID_BACKGROUND_PROTOCOL:BG-CAL {policy}{repetition}")
        min_tail = min(float(row["tail_finish_ms"]) for row in calibration_rows)
        if min_tail < CALIBRATION_TAIL_MS:
            finalize_stopped("EXPERIMENT_INVALID_BACKGROUND_PROTOCOL", f"minimum BG-CAL tail {min_tail:.3f} ms < {CALIBRATION_TAIL_MS:.0f} ms", "BG-CAL")
            return
        write_json(OUT / "background_protocol_frozen.json", {
            "protocol": "BGv2_fixed_work_chunked",
            "BG_CAL_valid": len(calibration_rows),
            "BG_CAL_tail_finish_ms": [float(row["tail_finish_ms"]) for row in calibration_rows],
            "minimum_BG_CAL_tail_finish_ms": min_tail,
            "required_tail_finish_ms": CALIBRATION_TAIL_MS,
            "max_inter_chunk_idle_gap_ms": max(float(row["max_inter_chunk_gap_ms"]) for row in calibration_rows),
            "frozen_for_matrix": True,
        })

        bg_only_rows = []
        for repetition in range(1, 5):
            row = run_bg_only(repetition, "pre")
            if not row["valid"]:
                raise RuntimeError(f"RUN_INVALID_BACKGROUND:bg-only pre {repetition}")
            bg_only_rows.append(row)
            _record_progress("BG-only-pre", row)

        measured_rows = []
        for repetition, pair in enumerate(RUN_ORDER, 1):
            for policy in pair:
                result = run_agent(policy, "measured", repetition)
                if not result["valid"]:
                    raise RuntimeError(f"EXPERIMENT_INVALID_MATRIX:measured rep{repetition} {policy}")
                row = {
                    "repetition": repetition,
                    "policy": result["policy"],
                    "valid": result["valid"],
                    "turns_complete": result["turns_complete"],
                    "boundaries": result["boundaries"],
                    "BG_chunks_complete": result["BG_chunks_complete"],
                    "BG_total_generated_tokens": result["BG_total_generated_tokens"],
                    "BG_max_inter_chunk_gap_ms": result["BG_max_inter_chunk_gap_ms"],
                    "BG_tail_first_token_ms": result["BG_tail_first_token_ms"],
                    "BG_tail_request_finish_ms": result["BG_tail_request_finish_ms"],
                }
                measured_rows.append(row)
                write_csv(OUT / "measured_progress.csv", measured_rows)
                _record_progress("measured", row)

        bg_post_rows = []
        for repetition in range(5, 9):
            row = run_bg_only(repetition, "post")
            if not row["valid"]:
                raise RuntimeError(f"RUN_INVALID_BACKGROUND:bg-only post {repetition}")
            bg_post_rows.append(row)
            _record_progress("BG-only-post", row)
        if len(measured_rows) != 16 or len(bg_only_rows) != 4 or len(bg_post_rows) != 4:
            raise RuntimeError("EXPERIMENT_INVALID_MATRIX:incomplete run count")
        from experiments.phase4g_a_fix.finalize_phase4g_a_fix import finalize
        finalize()
    except BaseException as exc:
        stage = str(json.loads((OUT / "progress.json").read_text()).get("stage", "unknown"))
        failure = {"phase": "4G-A-FIX", "result": "EXPERIMENT_INVALID_MATRIX", "stage": stage,
                   "error_type": type(exc).__name__, "error": str(exc),
                   "partial_matrix_not_aggregated": True}
        write_json(OUT / "failure_report.json", failure)
        if stage == "BG-CAL" and "EXPERIMENT_INVALID_BACKGROUND_PROTOCOL" in str(exc):
            finalize_stopped("EXPERIMENT_INVALID_BACKGROUND_PROTOCOL", str(exc), stage)
        elif stage.startswith("BG-only"):
            finalize_stopped("EXPERIMENT_INVALID_BACKGROUND_PROTOCOL", str(exc), stage)
        else:
            finalize_stopped("EXPERIMENT_INVALID_MATRIX", str(exc), stage)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("all",))
    args = parser.parse_args()
    if args.stage == "all":
        run_all()


if __name__ == "__main__":
    main()
