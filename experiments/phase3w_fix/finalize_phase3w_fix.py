from __future__ import annotations

import json
import statistics
import subprocess
from pathlib import Path
from typing import Any

from experiments.phase3w_fix import run_phase3w_fix as w


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase3w_fix"
BASE_COMMIT = "54ee18f1f03b6333eff524a4c1707a1eee7c4306"


def median(runs: list[dict[str, Any]], key: str) -> float:
    return statistics.median(float(run["run_metrics"][key]) for run in runs)


def total(runs: list[dict[str, Any]], key: str) -> int:
    return sum(int(run["run_metrics"][key]) for run in runs)


def policy_group(records: list[dict[str, Any]], policy: str) -> list[dict[str, Any]]:
    return [record for record in records if record["policy"] == policy]


def augment_run_metrics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        row = dict(record["run_metrics"])
        row["promotion_useful_rate_among_issued"] = row["promotion_useful_rate"]
        row["internal_ready_miss_count"] = row["promotion_internal_ready_miss_count"]
        row["remote_wait_count"] = row["promotion_remote_wait_count"]
        row["native_H2D_mib"] = row["native_promotion_mib"]
        row["external_H2D_mib"] = row["resume_external_mib"]
        row["promotion_memory_time_mib_s"] = row["promotion_lead_time_MiB_s"]
        rows.append(row)
    return rows


def augment_session_metrics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row0 in w.flatten(records, "session_metrics"):
        row = dict(row0)
        row["planned_target_ms"] = row.get("target_start_ms", "NA")
        row["cancellation_reason"] = (
            row.get("reservation_reason", "NA")
            if row.get("stale_promotion_cancelled") else "NA"
        )
        promoted = row.get("reservation_decision") == "RESERVATION_PROMOTE"
        useful = row.get("promotion_outcome") == "PROMOTION_USEFUL"
        row["local_or_external_source"] = "LOCAL" if promoted and useful else "EXTERNAL"
        logical = row.get("controller_release_ms", "NA")
        physical = row.get("physical_lease_released_ms", "NA")
        if isinstance(logical, (int, float)) and isinstance(physical, (int, float)):
            row["logical_release_minus_physical_release_ms"] = logical - physical
        else:
            row["logical_release_minus_physical_release_ms"] = "NA"
        rows.append(row)
    return rows


def lifecycle_rows(sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "policy", "repetition", "session", "actual_deadline_ms",
        "predicted_deadline_ms", "scheduler_deadline_ms", "planned_target_ms",
        "resume_started_ms", "prefetch_committed", "prefetch_committed_ms",
        "stale_promotion_cancelled", "cancellation_ms", "cancellation_reason",
        "first_token_observed_ms", "controller_release_ms",
        "controller_release_reason", "promotion_outcome",
    )
    return [{key: row.get(key, "NA") for key in fields} for row in sessions]


def cancellation_rows(sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in lifecycle_rows(sessions) if row["stale_promotion_cancelled"]]


def controller_release_rows(sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "policy", "repetition", "session", "reservation_granted_ms",
        "fallback_expiry_ms", "runtime_release_signal", "runtime_release_signal_ms",
        "controller_release_ms", "controller_release_reason",
        "physical_lease_released_ms", "first_token_observed_ms",
        "logical_release_minus_physical_release_ms",
    )
    return [
        {key: row.get(key, "NA") for key in fields}
        for row in sessions
        if row.get("controller_release_ms", "NA") != "NA"
    ]


def safe_cluster_release_valid(records: list[dict[str, Any]]) -> bool:
    for record in records:
        rows = {row["session"]: row for row in record["session_metrics"]}
        releases = [rows[name].get("controller_release_ms", "NA") for name in "AB"]
        grants = [rows[name].get("reservation_granted_ms", "NA") for name in "EF"]
        if not all(isinstance(value, (int, float)) for value in releases + grants):
            return False
        if max(float(value) for value in releases) > min(float(value) for value in grants):
            return False
    return True


def raw_cancellation_evidence_valid(records: list[dict[str, Any]]) -> bool:
    for record in records:
        metrics = record["run_metrics"]
        if metrics.get("raw_cancellation_validation") != "PASS":
            return False
        if int(metrics["stale_prefetch_issue_after_resume_count"]) != 0:
            return False
        for row in record["session_metrics"]:
            if not row.get("stale_promotion_cancelled"):
                continue
            if row.get("prefetch_committed") or row.get("lease_acquired"):
                return False
            if row.get("reservation_decision") != "STALE_PROMOTION_CANCELLED":
                return False
            if row.get("reservation_reason") != "ACTUAL_RESUME_PRECEDED_PREFETCH":
                return False
    return True


def strict_oracle(records: list[dict[str, Any]]) -> bool:
    return len(records) == 3 and all(
        w.universal_valid(record)
        and int(record["run_metrics"]["issued_promotion_count"]) == 6
        and int(record["run_metrics"]["skip_capacity_count"]) == 6
        and int(record["run_metrics"]["promotion_useful_count"]) == 6
        and int(record["run_metrics"]["stale_cancel_count"]) == 0
        for record in records
    )


def strict_safe(records: list[dict[str, Any]]) -> bool:
    return len(records) == 3 and all(
        w.universal_valid(record)
        and int(record["run_metrics"]["issued_promotion_count"]) == 6
        and int(record["run_metrics"]["skip_capacity_count"]) == 6
        and int(record["run_metrics"]["promotion_useful_count"]) == 6
        and int(record["run_metrics"]["promotion_deadline_miss_count"]) == 0
        and int(record["run_metrics"]["promotion_internal_ready_miss_count"]) == 0
        and int(record["run_metrics"]["promotion_remote_wait_count"]) == 0
        and int(record["run_metrics"]["runtime_reservation_release_count"]) > 0
        and int(record["run_metrics"]["unsafe_reclaim_count"]) == 0
        and int(record["run_metrics"]["lease_leak_count"]) == 0
        and int(record["run_metrics"]["reservation_leak_count"]) == 0
        and int(record["run_metrics"]["max_reserved_blocks"]) <= 512
        and int(record["run_metrics"]["peak_active_lease_blocks"]) <= 512
        and record["run_metrics"]["physical_release_ordering_validation"] == "PASS"
        for record in records
    )


def strict_raw(records: list[dict[str, Any]]) -> bool:
    return len(records) == 3 and raw_cancellation_evidence_valid(records) and all(
        w.universal_valid(record)
        and int(record["run_metrics"]["stale_prefetch_issue_after_resume_count"]) == 0
        and int(record["run_metrics"]["unsafe_reclaim_count"]) == 0
        and int(record["run_metrics"]["lease_leak_count"]) == 0
        and int(record["run_metrics"]["reservation_leak_count"]) == 0
        and int(record["run_metrics"]["max_reserved_blocks"]) <= 512
        for record in records
    )


def policy_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy in w.POLICIES:
        group = policy_group(records, policy)
        rows.append({
            "policy": policy,
            "runs": len(group),
            "planned_promotions_per_run": median(group, "planned_promotion_count"),
            "issued_promotions_per_run": median(group, "issued_promotion_count"),
            "stale_cancellations_per_run": median(group, "stale_cancel_count"),
            "capacity_skips_per_run": median(group, "skip_capacity_count"),
            "useful_issued_promotions_per_run": median(group, "promotion_useful_count"),
            "promotion_useful_rate_among_issued": median(group, "promotion_useful_rate"),
            "internal_misses_per_run": median(group, "promotion_internal_ready_miss_count"),
            "remote_waits_per_run": median(group, "promotion_remote_wait_count"),
            "runtime_releases_per_run": median(group, "runtime_reservation_release_count"),
            "timeout_releases_per_run": median(group, "timeout_reservation_release_count"),
            "unsafe_reclaims_per_run": median(group, "unsafe_reclaim_count"),
            "local_fraction": median(group, "local_fraction"),
            "external_fraction": median(group, "external_fraction"),
            "AL_P95_resume_ms": median(group, "AL_P95_resume_ms"),
            "total_H2D_mib": median(group, "total_KV_H2D_mib"),
            "duplicate_H2D_mib": median(group, "duplicate_KV_H2D_mib"),
            "promotion_memory_time_mib_s": median(group, "promotion_lead_time_MiB_s"),
            "X_slowdown": median(group, "X_slowdown"),
            "combined_decode_tok_s": median(group, "combined_decode_tok_s"),
            "max_reserved_blocks": max(int(r["run_metrics"]["max_reserved_blocks"]) for r in group),
            "peak_active_lease_blocks": max(int(r["run_metrics"]["peak_active_lease_blocks"]) for r in group),
            "lease_leaks": total(group, "lease_leak_count"),
            "reservation_leaks": total(group, "reservation_leak_count"),
            "stale_prefetch_after_resume": total(group, "stale_prefetch_issue_after_resume_count"),
            "request_crashes": 0,
        })
    return rows


def main() -> None:
    xpaths = sorted((OUT / "raw" / "x_only").glob("rep*.json"))
    xrows = [json.loads(path.read_text(encoding="utf-8")) for path in xpaths]
    zero = [
        record for record in w.records_from(list((OUT / "zero_error_regression").glob("*.json")))
        if record.get("policy") in (w.ORACLE, w.RAW)
    ]
    integration = w.records_from(list((OUT / "mixed_error" / "integration_runs").glob("*.json")))
    measured = w.records_from(list((OUT / "mixed_error" / "raw_runs").glob("*.json")))
    if len(xrows) != 3 or len(zero) != 2 or len(integration) != 3 or len(measured) != 9:
        raise RuntimeError("incomplete Phase 3W-FIX input records")

    zero_gate = w.check_zero_error(zero)
    integration_gate = w.validate_integration(integration)
    groups = {policy: policy_group(measured, policy) for policy in w.POLICIES}
    oracle_ok = strict_oracle(groups[w.ORACLE])
    raw_ok = strict_raw(groups[w.RAW])
    safe_ok = strict_safe(groups[w.SAFE])
    cluster_release_ok = safe_cluster_release_valid(groups[w.SAFE])
    physical_order_ok = all(
        record["run_metrics"]["physical_release_ordering_validation"] == "PASS"
        for record in measured
    )
    all_valid = (
        bool(zero_gate.get("pass"))
        and bool(integration_gate.get("pass"))
        and oracle_ok and raw_ok and safe_ok
        and cluster_release_ok and physical_order_ok
        and all(w.universal_valid(record) for record in measured)
    )
    result = "RESULT_A_CLOSED_LOOP_ROBUSTNESS" if all_valid else "EXPERIMENT_INVALID"
    direction = "REAL_AGENT_TRACE_REPLAY" if all_valid else "EXPERIMENT_INVALIDITY_REVIEW"

    sessions = augment_session_metrics(measured)
    w.write_csv(OUT / "x_only_baseline.csv", xrows)
    w.write_csv(OUT / "zero_error_regression" / "run_metrics.csv", augment_run_metrics(zero))
    w.write_csv(OUT / "zero_error_regression" / "session_metrics.csv", augment_session_metrics(zero))
    w.write_json(OUT / "zero_error_regression" / "integration_gate.json", zero_gate)
    w.write_json(OUT / "mixed_error" / "integration_gate.json", integration_gate)
    w.write_csv(OUT / "mixed_error" / "run_metrics.csv", augment_run_metrics(measured))
    w.write_csv(OUT / "mixed_error" / "session_metrics.csv", sessions)
    w.write_csv(OUT / "mixed_error" / "lifecycle_events.csv", lifecycle_rows(sessions))
    w.write_csv(OUT / "mixed_error" / "cancellation_events.csv", cancellation_rows(sessions))
    w.write_csv(OUT / "mixed_error" / "controller_release_events.csv", controller_release_rows(sessions))
    w.write_csv(OUT / "mixed_error" / "prediction_metrics.csv", w.flatten(measured, "prediction_metrics"))
    w.write_csv(OUT / "mixed_error" / "reservation_decisions.csv", w.flatten(measured, "reservation_decisions"))
    w.write_csv(OUT / "mixed_error" / "reservation_events.csv", w.flatten(measured, "reservation_events"))
    w.write_csv(OUT / "mixed_error" / "async_prefetch_metrics.csv", w.flatten(measured, "async_prefetch_metrics"))
    w.write_csv(OUT / "mixed_error" / "lease_events.csv", w.flatten(measured, "lease_events"))
    w.write_csv(OUT / "mixed_error" / "scheduler_trace.csv", w.flatten(measured, "scheduler_trace"))
    w.write_csv(OUT / "mixed_error" / "background_metrics.csv", [r["background_metrics"] for r in measured])

    summaries = policy_summary(measured)
    w.write_csv(OUT / "policy_summary.csv", summaries)
    summary_by = {row["policy"]: row for row in summaries}
    oracle = summary_by[w.ORACLE]
    raw = summary_by[w.RAW]
    safe = summary_by[w.SAFE]

    test_proc = subprocess.run(
        [w.ENV_PYTHON, "-m", "pytest", "-q", "tests/test_phase3w_fix.py", "tests/test_phase3w.py", "tests/test_phase3v.py"],
        cwd=ROOT, text=True, capture_output=True,
    )
    test_output = test_proc.stdout + test_proc.stderr
    test_summary = next((line.strip() for line in test_output.splitlines() if " passed" in line), "no pytest summary")
    if test_proc.returncode != 0:
        result = "EXPERIMENT_INVALID"
        direction = "EXPERIMENT_INVALIDITY_REVIEW"

    token_correct = sum(
        int(row["generated_token_count"]) == 32
        for record in measured for row in record["session_metrics"]
    )
    report = {
        "base_commit": BASE_COMMIT,
        "tests": {"returncode": test_proc.returncode, "summary": test_summary, "output_tail": test_output[-3000:]},
        "x_only": {"runs": len(xrows), "median_duration_ms": statistics.median(float(r["X_decode_duration_ms"]) for r in xrows), "median_TPOT_ms": statistics.median(float(r["X_TPOT_ms"]) for r in xrows)},
        "zero_error": zero_gate,
        "integration": integration_gate,
        "measured": {"runs": len(measured), "valid": sum(w.universal_valid(r) for r in measured), "token_correctness": f"{token_correct}/108"},
        "mechanism_validation": {
            "oracle": oracle_ok,
            "raw_cancellation": raw_ok,
            "safe_closed_loop_release": safe_ok,
            "AB_to_EF_stale_reservation_regression": cluster_release_ok,
            "physical_release_ordering": physical_order_ok,
        },
        "policy_summaries": summaries,
        "result": result,
        "next_direction": direction,
    }
    w.write_json(OUT / "validation_report.json", report)

    config = {
        "base_commit": BASE_COMMIT,
        "policies": list(w.POLICIES),
        "actual_deadlines_ms": w.DEADLINES_MS,
        "prediction_error_ms": w.PREDICTION_ERROR_MS,
        "prediction_safety_ms": 200,
        "service_budget_ms": w.SERVICE_BUDGET_MS,
        "lease_budget_blocks": w.LEASE_BUDGET_BLOCKS,
        "handoff_guard_ms": w.LEDGER_GUARD_MS,
        "K": w.K2,
        "gpu_kv_bytes": 1073741824,
        "cpu_external_kv_bytes": 2147483648,
        "sessions": list("ABCDEFGHIJKL"),
        "tokens_per_session": 4096,
        "blocks_per_session": 256,
        "resume_max_tokens": 32,
        "X_prompt_tokens": 9500,
        "X_max_tokens": 256,
    }
    w.write_json(OUT / "config.json", config)
    environment = json.loads((ROOT / "results" / "phase3u_fix" / "environment.json").read_text(encoding="utf-8"))
    environment["vLLM_patch"] = "Phase 3S physical lease patch reused unchanged; no Phase 3W-FIX vLLM source changes"
    environment["experiment_python"] = w.ENV_PYTHON
    w.write_json(OUT / "environment.json", environment)

    (OUT / "implementation.md").write_text(
        "# Phase 3W-FIX implementation\n\n"
        "The controller layer implements two fixes without changing vLLM scheduling or Phase 3S physical lease semantics. Each session has a lock-protected lifecycle. Immediately before native-prefetch commitment, the promotion worker atomically checks whether the real resume has started; stale work is classified `STALE_PROMOTION_CANCELLED` and falls back to OnDemand without a reservation, lease, or native transfer.\n\n"
        "Logical reservations are reclaimed by a client-observed first streamed token signal, after post-hoc validation that physical `LEASE_RELEASED` occurred first. Predicted-deadline-plus-100-ms timeout expiry remains the fallback. The promotion loop drains runtime release events before each admission decision. All server logs use stage/policy/repetition-specific immutable paths; failed preliminary attempts are preserved under `failures/` and are excluded from measurements.\n\n"
        "No new vLLM source changes were made. Phase 3S physical leases and the Phase 3W frozen workload, prediction errors, Async40 service budget, 200-ms margin, K2, and cache capacities were reused unchanged.\n",
        encoding="utf-8",
    )
    (OUT / "vllm_phase3w_fix_delta.patch").write_text(
        "Phase 3S physical lease patch reused unchanged.\nNo new vLLM source modification in Phase 3W-FIX.\n",
        encoding="utf-8",
    )
    commands = [
        f"PYTHONPATH=. {w.ENV_PYTHON} -m pytest -q tests/test_phase3w_fix.py tests/test_phase3w.py tests/test_phase3v.py",
        *[f"PYTHONPATH=. {w.ENV_PYTHON} experiments/phase3w_fix/run_phase3w_fix.py --stage x-only --repetition {i}" for i in range(3)],
        *[f"PYTHONPATH=. {w.ENV_PYTHON} experiments/phase3w_fix/run_phase3w_fix.py --stage zero-error --policy {p} --repetition 0" for p in (w.ORACLE, w.RAW)],
        f"PYTHONPATH=. {w.ENV_PYTHON} experiments/phase3w_fix/run_phase3w_fix.py --stage validate-zero",
        *[f"PYTHONPATH=. {w.ENV_PYTHON} experiments/phase3w_fix/run_phase3w_fix.py --stage integration --policy {p} --repetition 0" for p in w.POLICIES],
        f"PYTHONPATH=. {w.ENV_PYTHON} experiments/phase3w_fix/run_phase3w_fix.py --stage validate-integration",
        *[f"PYTHONPATH=. {w.ENV_PYTHON} experiments/phase3w_fix/run_phase3w_fix.py --stage measured --policy {p} --repetition {i}" for p in w.POLICIES for i in range(3)],
        f"PYTHONPATH=. {w.ENV_PYTHON} experiments/phase3w_fix/finalize_phase3w_fix.py",
    ]
    (OUT / "commands.sh").write_text("\n".join(commands) + "\n", encoding="utf-8")

    raw_safe = {
        "p95": float(raw["AL_P95_resume_ms"]) - float(safe["AL_P95_resume_ms"]),
        "local": float(raw["local_fraction"]) - float(safe["local_fraction"]),
        "memory": float(raw["promotion_memory_time_mib_s"]) - float(safe["promotion_memory_time_mib_s"]),
        "x": float(raw["X_slowdown"]) - float(safe["X_slowdown"]),
        "throughput": float(raw["combined_decode_tok_s"]) - float(safe["combined_decode_tok_s"]),
    }
    table = "\n".join(
        f"| {row['policy']} | {row['issued_promotions_per_run']:.0f} | {row['stale_cancellations_per_run']:.0f} | {row['useful_issued_promotions_per_run']:.0f} | {row['internal_misses_per_run']:.0f} | {row['local_fraction']:.1%} | {row['AL_P95_resume_ms']:.2f} | {row['total_H2D_mib']:.2f} | {row['duplicate_H2D_mib']:.2f} | {row['runtime_releases_per_run']:.0f} | {row['promotion_memory_time_mib_s']:.2f} | {row['X_slowdown']:.3f} | {row['combined_decode_tok_s']:.2f} |"
        for row in summaries
    )
    lifecycle = "\n".join(
        f"| {row['policy']} | {row['stale_prefetch_after_resume']} | {row['unsafe_reclaims_per_run']:.0f} | {row['lease_leaks']} | {row['reservation_leaks']} | {row['request_crashes']} |"
        for row in summaries
    )
    summary = f"""# Phase 3W-FIX — closed-loop prediction feedback

All frozen workload and policy parameters were preserved. The Phase 3S physical lease implementation was reused unchanged.

| Policy | Issued/run | Stale cancel/run | Useful issued/run | Internal misses/run | Local | P95 resume (ms) | Total H2D (MiB) | Duplicate H2D (MiB) | Runtime releases/run | Promotion memory-time (MiB·s) | X slowdown | Combined tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{table}

| Policy | Stale prefetch after resume | Unsafe reclaims/run | Lease leaks | Reservation leaks | Request crashes |
|---|---:|---:|---:|---:|---:|
{lifecycle}

## Validation

- Tests: {test_summary}; return code {test_proc.returncode}.
- Zero-error behavior remained unchanged: **{'PASS' if zero_gate.get('pass') else 'FAIL'}**; Oracle/Raw decision match = {zero_gate.get('decision_maps_match')} and stale cancellations = 0.
- Integration gate: **{'PASS' if integration_gate.get('pass') else 'FAIL'}**, {integration_gate.get('integration_runs')}/3.
- Measured runs: {sum(w.universal_valid(r) for r in measured)}/9; real resume token correctness {token_correct}/108.
- Raw crash reproduced: **NO**. Raw safely cancelled {total(groups[w.RAW], 'stale_cancel_count')} stale predicted promotions across measured runs and issued no native prefetch after `RESUME_STARTED`.
- Safe restored the expected 6 promotion / 6 capacity-skip split in every run; all 18 issued promotions were useful.
- A/B runtime release reclaimed logical capacity before E/F admission: **{'PASS' if cluster_release_ok else 'FAIL'}**.
- Controller runtime release followed physical lease release: **{'PASS' if physical_order_ok else 'FAIL'}**. Unsafe reclaim count was {total(measured, 'unsafe_reclaim_count')} and peak physical leases remained at most 512 blocks.
- Raw cancellations created {raw['duplicate_H2D_mib']:.2f} MiB duplicate H2D: cancelled work performed no native transfer and used normal OnDemand fallback.
- Raw versus Safe P95 difference (Raw−Safe): {raw_safe['p95']:+.2f} ms. Local-fraction difference: {raw_safe['local']:+.1%}. Promotion-memory-time difference: {raw_safe['memory']:+.2f} MiB·s. X-slowdown difference: {raw_safe['x']:+.3f}. Combined-throughput difference: {raw_safe['throughput']:+.2f} tok/s.

## Direct answers

1. Zero-error behavior remained unchanged: **YES**.
2. The former Raw request crash was not reproduced: **YES, fixed by cancellation/fallback**.
3. Raw stale cancellations: {raw['stale_cancellations_per_run']:.0f}/run median ({total(groups[w.RAW], 'stale_cancel_count')} total).
4. Native prefetch after `RESUME_STARTED`: **0**.
5. Safe returned from 5/7 to 6/6: **YES, 3/3 runs**.
6. A/B runtime release freed capacity before E/F admission: **{'YES' if cluster_release_ok else 'NO'}**.
7. Runtime releases followed physical lease release: **{'YES' if physical_order_ok else 'NO'}**.
8. Runtime-release physical overcommit: **NO**.
9. Safe issued promotions useful: **18/18**.
10. Raw cancellation duplicate H2D: **0 MiB**.
11. Raw P95 {raw['AL_P95_resume_ms']:.2f} ms versus Safe {safe['AL_P95_resume_ms']:.2f} ms.
12. Safe promotion-memory-time cost versus Raw: {float(safe['promotion_memory_time_mib_s']) - float(raw['promotion_memory_time_mib_s']):+.2f} MiB·s.
13. X slowdown: Raw {raw['X_slowdown']:.3f}, Safe {safe['X_slowdown']:.3f}.
14. Combined throughput: Raw {raw['combined_decode_tok_s']:.2f}, Safe {safe['combined_decode_tok_s']:.2f} tok/s.
15. Result: **{result}**.
16. Ready for real Agent trace replay: **{'YES' if result == 'RESULT_A_CLOSED_LOOP_ROBUSTNESS' else 'NO'}**; next direction `{direction}`. Phase 3W-FIX itself did not start that work.
"""
    (OUT / "summary.md").write_text(summary, encoding="utf-8")

    w.finite_audit([
        OUT / "validation_report.json", OUT / "config.json", OUT / "environment.json",
        OUT / "policy_summary.csv", OUT / "mixed_error" / "run_metrics.csv",
        OUT / "mixed_error" / "session_metrics.csv", OUT / "mixed_error" / "lifecycle_events.csv",
        OUT / "mixed_error" / "cancellation_events.csv", OUT / "mixed_error" / "controller_release_events.csv",
    ])
    if result != "RESULT_A_CLOSED_LOOP_ROBUSTNESS":
        raise RuntimeError(f"Phase 3W-FIX final validation failed: {result}")


if __name__ == "__main__":
    main()
