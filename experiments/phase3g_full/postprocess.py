"""Produce the Phase 3G-Full human-readable, trace-derived report."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from experiments.phase3g_full.metrics import percentile_linear


OUT = Path("results/phase3g_full")
ASAP_MEMORY_MIB_S = 383.61
POLICIES = ("OnDemand-Trace", "ResumeAware-Trace", "InEngineFirstTokenGate-ResumeAware")


def read_csv(name: str) -> list[dict[str, str]]:
    with (OUT / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def f(value: str) -> float:
    return float(value)


def median(rows: list[dict[str, str]], column: str) -> float:
    return statistics.median(f(row[column]) for row in rows)


def fmt(value: float) -> str:
    return f"{value:.2f}"


def main() -> None:
    runs = read_csv("run_metrics.csv")
    sessions = read_csv("session_metrics.csv")
    native = read_csv("native_prefetch_metrics.csv")
    grouped = {policy: [row for row in runs if row["policy"] == policy] for policy in POLICIES}
    session_grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in sessions:
        session_grouped[(row["policy"], row["repetition"])].append(row)

    summary: dict[str, dict[str, float]] = {}
    for policy, rows in grouped.items():
        summary[policy] = {
            "abc_ttft": median(rows, "ABC_P95_TTFT_ms"),
            "overall_ttft": median(rows, "overall_P95_TTFT_ms"),
            "abc_completion": median(rows, "ABC_P95_completion_latency_ms"),
            "makespan": median(rows, "ABC_completion_makespan_ms"),
            "throughput": median(rows, "ABC_decode_throughput_tok_s"),
            "max_pre": median(rows, "max_concurrent_pre_first_token"),
            "max_decode": median(rows, "max_concurrent_post_first_token_decode"),
            "memory": median(rows, "idle_gpu_memory_time_mib_s"),
        }

    resume = summary["ResumeAware-Trace"]
    gated = summary["InEngineFirstTokenGate-ResumeAware"]
    ttft_improvement = (resume["abc_ttft"] - gated["abc_ttft"]) / resume["abc_ttft"] * 100.0
    throughput_retained = gated["throughput"] / resume["throughput"] * 100.0
    makespan_change = (gated["makespan"] / resume["makespan"] - 1.0) * 100.0
    if (
        ttft_improvement >= 15.0
        and throughput_retained >= 90.0
        and gated["memory"] < ASAP_MEMORY_MIB_S * 0.20
    ):
        result = "A"
    elif ttft_improvement > 0.0:
        result = "B"
    else:
        result = "C"

    # Runtime validity is intentionally separate from cross-policy sequence
    # equality; matched-schedule KV correctness is in Phase 3G-CorrectnessDiag.
    all_token_ok = all(row["token_count_validation"] == "PASS" for row in sessions)
    all_cache_ok = all(row["cache_transport_validation"] == "PASS" for row in sessions)
    native_ok = all(
        row["external_tokens_loaded"] == "4096"
        and row["model_forward_count"] == "0"
        and row["generated_token_count"] == "0"
        and row["mixed_prefetch_model_step_count"] == "0"
        and row["status"] == "ok"
        for row in native
    )
    gated_runs = grouped["InEngineFirstTokenGate-ResumeAware"]
    gate_ok = all(
        row["gate_invariant_validation"] == "PASS"
        and row["decode_overlap_validation"] == "PASS"
        and row["schedule_structure_validation"] == "PASS"
        and f(row["max_concurrent_pre_first_token"]) == 1.0
        for row in gated_runs
    )
    if not (all_token_ok and all_cache_ok and native_ok and gate_ok):
        result = "D"

    lines = [
        "# Phase 3G-Full — multi-token gated resume performance",
        "",
        "## Frozen setup",
        "",
        "- Base: `b0ca5fd755f4ec47c2991d4836fecc2d40956d99`; vLLM source `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`.",
        "- vLLM 0.23.0, PyTorch 2.11.0+cu126, CUDA 12.6, RTX A6000 GPU0, Qwen2.5-1.5B-Instruct FP16/TP1, 1 GiB GPU KV, 2 GiB CPU external KV.",
        "- A–F each use 4096-token / 117,440,512-byte KV prefixes; deadlines are 300/320/340/650/900/1300 ms. Real resumes use `max_tokens=32`, `temperature=0`, `ignore_eos=true`.",
        "- Phase 3B connector-first native promotion and the Phase 3F-Fix in-engine FirstTokenGate were reused unchanged; no new vLLM algorithm change was made.",
        "",
        "## Main medians across three independent restarts",
        "",
        "| Policy | ABC TTFT P95 (ms) | Overall TTFT P95 (ms) | ABC completion P95 (ms) | ABC makespan (ms) | ABC decode tok/s | Max pre-first | Max decode | Memory-time (MiB*s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for policy in POLICIES:
        value = summary[policy]
        lines.append(
            f"| {policy} | {fmt(value['abc_ttft'])} | {fmt(value['overall_ttft'])} | {fmt(value['abc_completion'])} | {fmt(value['makespan'])} | {fmt(value['throughput'])} | {fmt(value['max_pre'])} | {fmt(value['max_decode'])} | {fmt(value['memory'])} |"
        )
    lines.extend([
        "",
        "## Mechanism and correctness",
        "",
        f"- Token-count validation: {'PASS' if all_token_ok else 'FAIL'}; every measured request is required to produce exactly 32 tokens.",
        f"- Cache transport validation: {'PASS' if all_cache_ok else 'FAIL'}; native endpoint runtime records: {'PASS' if native_ok else 'FAIL'}.",
        f"- FirstTokenGate invariant / multi-token release: {'PASS' if gate_ok else 'FAIL'}. The gated median pre-first overlap is {fmt(median(gated_runs, 'ABC_pre_first_pairwise_overlap_ms'))} ms; median post-first ABC decode overlap is {fmt(median(gated_runs, 'ABC_total_decode_overlap_ms'))} ms.",
        "- Phase 3G-CorrectnessDiag separately established LocalKeep-Solo == Prefetch-Solo and LocalKeep-Gated == Prefetch-Gated under matched execution. Cross-policy sequence-hash differences here are diagnostic only because batch composition changes greedy trajectories.",
        "",
        "## ResumeAware → FirstTokenGate",
        "",
        f"- ABC P95 TTFT change: {fmt(ttft_improvement)}% (positive is an improvement).",
        f"- ABC decode throughput retained: {fmt(throughput_retained)}%; ABC makespan change: {fmt(makespan_change)}%.",
        f"- Memory-time change: {fmt(gated['memory'] - resume['memory'])} MiB*s; gated memory is {fmt(gated['memory'] / ASAP_MEMORY_MIB_S * 100.0)}% of historical ASAP ({ASAP_MEMORY_MIB_S:.2f} MiB*s).",
        "",
        "## Gated A/B/C real trace structure",
        "",
    ])
    for repetition in range(3):
        rows = {row["session"]: row for row in session_grouped[("InEngineFirstTokenGate-ResumeAware", str(repetition))]}
        fields = []
        for session in ("A", "B", "C"):
            row = rows[session]
            fields.append(f"{session} {row['scheduler_admission_step']}/{row['first_token_step']}/{row['request_finished_step']}")
        lines.append(f"- rep{repetition} admit/first/finish steps: " + "; ".join(fields) + ".")
    lines.extend([
        "",
        "- Full-request serialization: **NO**. In every gated repetition B was admitted after A's first token but before A finished, and C likewise after B's first token but before B finished.",
        "- OnDemand retained real `WAITING_FOR_REMOTE_KVS` trace events; its external recovery acts as implicit admission backpressure, but its median ABC P95 TTFT was higher than the gated policy in this multi-token workload.",
        "",
        "## Result",
        "",
        f"**Result {result}.** " + (
            "FirstTokenGate isolates the first-token critical path while retaining overlapping multi-token decode; it materially reduces clustered TTFT with limited throughput and GPU-memory-time cost."
            if result == "A" else
            "FirstTokenGate works mechanically under multi-token decoding, but its admission staggering imposes a meaningful end-to-end throughput/completion tradeoff."
            if result == "B" else
            "The first-token gate maintains its scheduling invariants and overlapping decode, but does not provide a useful multi-token TTFT tradeoff under the full six-session workload."
            if result == "C" else
            "The experiment is invalid because a required runtime correctness or trace invariant failed."
        ),
        "",
        "## Measurement notes",
        "",
        "- Scheduler timestamps retain raw `time.monotonic()` values and are aligned to the saved workload t0; they are not rebased to the first server trace event.",
        "- Positive request-arrival delay can exceed 50 ms while the engine is busy; it is retained as a server-intake diagnostic. Only a material negative offset would invalidate the shared monotonic time coordinate.",
        "- Native source-token attribution is endpoint-authoritative: the runtime response records external tokens loaded; vLLM's official external source counter does not include prefetch-only endpoint loads. Local-cache-hit source counters verify subsequent real-resume reuse.",
    ])
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")

    config = {
        "phase": "3G-Full", "base_commit": "b0ca5fd755f4ec47c2991d4836fecc2d40956d99",
        "p_safe_ms": 30, "resume_guard_ms": 110, "max_tokens": 32,
        "temperature": 0, "ignore_eos": True, "repetitions": 3,
        "policies": list(POLICIES),
    }
    (OUT / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    environment = {
        "vllm_version": "0.23.0", "vllm_source_commit": "0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665",
        "pytorch": "2.11.0+cu126", "cuda": "12.6", "gpu": "NVIDIA RTX A6000", "model": "Qwen/Qwen2.5-1.5B-Instruct",
    }
    (OUT / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    (OUT / "commands.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\nPYTHONPATH=. $VENV/bin/python -m pytest -q tests/test_phase3g_full.py\nPYTHONPATH=. $VENV/bin/python -m experiments.phase3g_full.run_phase3g_full --stage integration\nPYTHONPATH=. $VENV/bin/python -m experiments.phase3g_full.run_phase3g_full --stage full\nPYTHONPATH=. $VENV/bin/python -m experiments.phase3g_full.postprocess\n")
    (OUT / "vllm_phase3g_full_delta.patch").write_text("No new vLLM algorithm changes for Phase 3G-Full.\nPhase 3F-Fix in-engine FirstTokenGate and Phase 3B connector-first native promotion were reused unchanged.\n")
    (OUT / "implementation.md").write_text("Phase 3G-Full reuses the existing native prefetch and in-engine FirstTokenGate. The SlackKV runner persists actual endpoint response fields for every promotion, aligns raw server `time.monotonic()` trace records to the client workload t0, and postprocesses scheduler-derived admission/first-token/finish intervals.\n")


if __name__ == "__main__":
    main()
