"""Preserve an incomplete Phase 4E-A matrix without performance aggregation."""

from __future__ import annotations

import csv
import json
import shutil
import statistics
from collections import Counter
from pathlib import Path

from experiments.phase4d_a.experiment import write_csv, write_json
from experiments.phase4e_a.metrics import POLICIES

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/phase4e_a"
FAILURE_NAME = "measured_rep1_pos3_GuardedJITplusLease-Oracle-K2"


def rows(name: str) -> list[dict[str, str]]:
    path = OUT / name
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def finalize() -> dict:
    x = rows("x_only_runs.csv")
    smoke = rows("semantic_smoke.csv")
    measured = rows("run_metrics.csv")
    sessions = rows("session_metrics.csv")
    failing = next(row for row in measured if row["policy"] == "GuardedJIT+Lease-Oracle-K2" and row["repetition"] == "1")
    last_first = max(float(row["first_token_ms"]) for row in sessions if row["stage"] == "measured" and row["repetition"] == "1" and row["policy"] == failing["policy"])
    x_end = float(failing["X_decode_ms"])
    if not last_first > x_end or failing["X_active_through_last_first"] != "False":
        raise RuntimeError("failure evidence does not satisfy frozen background-invalid gate")
    failure = OUT / "failures" / FAILURE_NAME
    failure.mkdir(parents=True, exist_ok=True)
    write_json(failure / "run_config.json", {"policy": failing["policy"], "repetition": 1, "run_order_position": 3, "P_SAFE_ms": 102, "K": 2, "background_output_tokens": 256})
    write_json(failure / "failure.json", {"classification": "RUN_INVALID_BACKGROUND", "last_agent_FIRST_TOKEN_ms": last_first, "X_finished_ms": x_end, "agent_first_after_X_ms": last_first - x_end, "run_metrics": failing})
    for name in ("session_metrics.csv", "prefetch_metrics.csv", "capacity_trace.csv", "scheduler_trace.csv", "lease_events.csv", "resource_metrics.csv"):
        selected = [row for row in rows(name) if row.get("stage") == "measured" and row.get("repetition") == "1" and row.get("policy") == failing["policy"]]
        if selected:
            write_csv(failure / name, selected)
    source_log = OUT / "server_logs" / f"{FAILURE_NAME}_compact.log"
    if source_log.exists():
        shutil.copy2(source_log, failure / "server_compact.log")
    valid_smoke = sum(row["valid"] == "True" for row in smoke)
    valid_measured = sum(row["valid"] == "True" for row in measured)
    token_correct = sum(int(row["generated_tokens"]) == int(row["OSL"]) for row in sessions if row["stage"] == "measured")
    identity_mismatch = Counter(row["episode_id"] for row in sessions if row["identity_match"] == "False")
    report = {"phase": "4E-A", "result": "EXPERIMENT_INVALID_MATRIX", "reason": "RUN_INVALID_BACKGROUND",
              "X_only_valid": sum(row["valid"] == "True" for row in x), "X_only_median_decode_ms": statistics.median(float(row["X_decode_ms"]) for row in x),
              "X_only_median_TPOT_ms": statistics.median(float(row["X_TPOT_ms"]) for row in x),
              "stage_a_valid": valid_smoke, "stage_b_runs_started": len(measured), "stage_b_runs_valid": valid_measured,
              "stage_b_expected": 16, "measured_resumes_observed": sum(int(row["resume_count"]) for row in measured),
              "measured_token_count_correct": token_correct, "failing_policy": failing["policy"],
              "failing_repetition": 1, "failing_run_order_position": 3,
              "last_agent_FIRST_TOKEN_ms": last_first, "X_finished_ms": x_end,
              "agent_first_after_X_ms": last_first - x_end,
              "failing_run_source_conservation": int(failing["local_source_tokens"]) + int(failing["external_source_tokens"]) == 168016,
              "failing_run_K2_max_pre_first": int(failing["K2_max_pre_first"]),
              "failing_run_logical_leaks": int(failing["logical_leaks"]),
              "failing_run_lease_leaks": int(failing["lease_leaks"]),
              "output_identity_mismatch_counts": dict(identity_mismatch),
              "H1": "NOT_EVALUATED", "H2": "NOT_EVALUATED", "H3": "NOT_EVALUATED"}
    write_json(OUT / "validation_report.json", report)
    # The required schema exists, but no incomplete policy aggregate is reported.
    write_csv(OUT / "policy_summary.csv", [{"policy": policy.name, "valid_repetitions": sum(row["policy"] == policy.name and row["valid"] == "True" for row in measured), "deadline_P95_median": "NA", "total_H2D_median_MiB": "NA", "reason": "INCOMPLETE_COUNTERBALANCED_MATRIX"} for policy in POLICIES])
    (OUT / "paired_deltas.csv").write_text("repetition,comparison,P95_absolute_delta_ms,P95_relative_delta,total_H2D_delta_MiB,duplicate_H2D_delta_MiB,X_slowdown_delta,combined_tok_s_delta\n", encoding="utf-8")
    summary = f"""# Phase 4E-A — frozen hybrid performance matrix

**EXPERIMENT_INVALID_MATRIX**. The predeclared background-X validity gate failed in Stage B repetition 1, position 3. The driver stopped immediately; no incomplete matrix is interpreted as a performance result.

## Objective and frozen setup

This was the first controlled four-policy ablation on the frozen hybrid trace-driven workload: real coding-agent token/KV structure plus empirical TraceToolProxy timing, not real request-timestamp-faithful Agent execution. The eight first-boundary jobs contain 168,016 reusable tokens and 10,501 blocks (minimum H2D 4,594.1875 MiB). Runtime remained Qwen2.5-1.5B-Instruct, vLLM 0.23.0, FP16, GPU0, 32K, 2 GiB GPU KV (4,681 blocks), 8 GiB external KV, K2, P_SAFE=102 ms, 110-ms resume guard, and 3,510-block logical budget. No vLLM source mechanism was changed.

## Execution and failure

- X-only baseline: 4/4 valid, median decode {report['X_only_median_decode_ms']:.2f} ms and median TPOT {report['X_only_median_TPOT_ms']:.3f} ms.
- Stage A semantic smoke: {valid_smoke}/4 valid in frozen A→B→C→D order. All smoke runs completed 8/8 resumes, source conservation and K2, with X active through the last Agent first token.
- Stage B: only {len(measured)}/16 scheduled runs started; {valid_measured} valid. Repetition 1 positions 1 (OnDemand) and 2 (GuardedJIT) passed. Position 3 (GuardedJIT+Lease) completed 8/8 requests with exact output counts, source conservation, K2 maximum 2, zero lease leaks, zero OOM and no server crash, but failed background validity.
- In that failing run, X finished at t0+{x_end:.2f} ms. The last Agent FIRST_TOKEN (E03) occurred at t0+{last_first:.2f} ms, **{last_first-x_end:.2f} ms after X ended**. This is a direct comparison of the saved monotonic-t0 scheduler/client events, not a parsing artifact.
- The driver did not start position 4 or repetitions 2–4. It did not retry just one policy, change X length, alter workload/deadlines, or recombine partial data.

## Diagnostic observations, not performance claims

The complete failure record is `failures/{FAILURE_NAME}/`. The invalid C run had local/external source tokens {failing['local_source_tokens']}/{failing['external_source_tokens']}, three capacity admits and five rejects, and 0 logical/physical lease leaks. A single request (E04) had an output SHA different from its Phase 4C solo reference across multiple policies, while output lengths were exact; cross-policy batch-sensitive sequence identity is diagnostic, not a token-count failure. Because Stage B is incomplete, there is **no** policy median table, paired delta, H1/H2/H3 verdict, P95 superiority, H2D ablation conclusion, or statistical claim. `policy_summary.csv` marks medians NA rather than fabricating values.

## Next-step status

Phase 4E-A is not a valid completed performance matrix. No Phase 4E-B or all-boundary replay was started.
"""
    (OUT / "summary.md").write_text(summary, encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(finalize(), sort_keys=True))
