from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase3o"
POLICIES = ("OnDemand-K2", "Blocking70-K2", "Async40-K2")


def read_rows(name: str) -> list[dict[str, str]]:
    with (OUT / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def median(rows: list[dict[str, str]], key: str) -> float:
    return statistics.median(float(row[key]) for row in rows)


def change_pct(new: float, old: float) -> float:
    return (new - old) / old * 100.0


def main() -> None:
    run = read_rows("run_metrics.csv")
    session = read_rows("session_metrics.csv")
    grouped = {policy: [row for row in run if row["policy"] == policy] for policy in POLICIES}
    table = {
        policy: {
            "p95": median(rows, "ABC_P95_effective_resume_latency_ms"),
            "completion": median(rows, "ABC_P95_effective_completion_latency_ms"),
            "decode": median(rows, "ABC_decode_throughput_tok_s"),
            "x_slowdown": median(rows, "X_slowdown"),
            "combined": median(rows, "combined_decode_throughput_tok_s"),
            "makespan": median(rows, "ABC_completion_makespan_ms"),
            "memory": median(rows, "promotion_memory_time_mib_s"),
        }
        for policy, rows in grouped.items()
    }
    async_rows = grouped["Async40-K2"]
    async_sessions = [row for row in session if row["policy"] == "Async40-K2"]
    async_gate = all(
        row[key] == "PASS"
        for row in async_rows
        for key in ("local_hit_validation", "token_validation", "background_validation", "gate_validation", "native_validation")
    ) and all(row["internal_ready_miss_count"] == "0" for row in async_rows)
    blocking, asynchronous, ondemand = table["Blocking70-K2"], table["Async40-K2"], table["OnDemand-K2"]
    if not async_gate:
        result, direction = "B", "ASYNC_BUDGET_CALIBRATION"
    elif asynchronous["p95"] <= 1.10 * blocking["p95"] and (
        asynchronous["memory"] <= 0.8 * blocking["memory"] or asynchronous["p95"] <= 0.9 * blocking["p95"]
    ) and asynchronous["combined"] >= 0.95 * blocking["combined"]:
        result, direction = "A", "FULL_ABCDEF_E2E"
    else:
        result, direction = "C", "ASYNC_DISPATCHER_MODEL"
    comparison = {
        "effective_resume_latency_pct": change_pct(asynchronous["p95"], blocking["p95"]),
        "memory_time_pct": change_pct(asynchronous["memory"], blocking["memory"]),
        "x_slowdown_pct": change_pct(asynchronous["x_slowdown"], blocking["x_slowdown"]),
        "combined_throughput_pct": change_pct(asynchronous["combined"], blocking["combined"]),
        "vs_ondemand_resume_pct": change_pct(asynchronous["p95"], ondemand["p95"]),
        "vs_ondemand_combined_pct": change_pct(asynchronous["combined"], ondemand["combined"]),
    }
    (OUT / "config.json").write_text(json.dumps({"blocking_service_budget_ms": 70, "async_internal_ready_service_budget_ms": 40, "resume_guard_ms": 110, "gate_limit": 2, "result": result, "next_direction": direction}, indent=2) + "\n", encoding="utf-8")
    internal = sum(row["arrival_minus_T6_ms"] != "NA" and float(row["arrival_minus_T6_ms"]) >= 0 for row in async_sessions)
    local = sum(row["local_cache_hit_tokens"] == "4096.0" for row in async_sessions)
    waits = sum(row["WAITING_FOR_REMOTE_KVS_seen"] == "True" for row in async_sessions)
    lines = [
        "# Phase 3O — asynchronous native promotion end to end", "",
        f"- Async internal-ready: {internal}/9; local cache success: {local}/9; remote waits: {waits}.",
        f"- Async max outstanding HTTP prefetch calls: {max(int(row['max_outstanding_prefetch_http_calls']) for row in async_rows)}; maximum overlapping T0→T6 intervals: {max(int(row['max_overlapping_internal_prefetch_intervals']) for row in async_rows)}.",
        "", "| Policy | ABC P95 effective resume | ABC P95 completion | ABC decode tok/s | X slowdown | Combined tok/s | Makespan | Promotion memory-time |", "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines += [f"| {policy} | {values['p95']:.2f} | {values['completion']:.2f} | {values['decode']:.2f} | {values['x_slowdown']:.3f} | {values['combined']:.2f} | {values['makespan']:.2f} | {values['memory']:.2f} |" for policy, values in table.items()]
    lines += ["", f"- Async vs Blocking: effective resume {comparison['effective_resume_latency_pct']:.2f}%; memory-time {comparison['memory_time_pct']:.2f}%; X slowdown {comparison['x_slowdown_pct']:.2f}%; combined throughput {comparison['combined_throughput_pct']:.2f}%.", f"- Result {result}; next direction `{direction}`.", "", "T6 rather than HTTP client return defines GPU-ready residency. Per-session local cache success is validated by the run-global exact 12288 local-token source-counter delta plus each real scheduler trace showing no remote-KV wait; source counters are not request-labeled under concurrent K2 serving."]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "implementation.md").write_text("# Implementation\n\nPhase 3O reuses Phase 3M/3N T0–T7 instrumentation and Phase 3H K=2 unchanged. Blocking70 waits on each native endpoint future; Async40 launches endpoint calls from the same EDF/blackout QueueJIT targets and sends resumes at original deadlines without awaiting endpoint return.\n", encoding="utf-8")
    (OUT / "vllm_phase3o_delta.patch").write_text("No new vLLM source changes for Phase 3O. Phase 3M T0-T7 instrumentation and Phase 3H K=2 gate reused unchanged.\n", encoding="utf-8")
    print(json.dumps({"result": result, "next_direction": direction, "comparison": comparison}, indent=2))


if __name__ == "__main__":
    main()
