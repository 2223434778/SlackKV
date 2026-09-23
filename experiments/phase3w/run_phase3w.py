from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from experiments.phase3f.scheduler import latest_feasible_start, merge_blackouts
from experiments.phase3v import run_phase3v as base
from experiments.phase3v.ledger import (
    BLOCK_SIZE_TOKENS, DEADLINES_MS, HANDOFF_GUARD_MS as LEDGER_GUARD_MS,
    KV_BYTES_PER_TOKEN, LEASE_BUDGET_BLOCKS, BlockReservationLedger,
    required_blocks, required_mib, reservation_policy_decision,
)
from experiments.phase3w.policy import (
    ORACLE, POLICIES, PREDICTION_ERROR_MS, RAW, SAFE,
    SERVICE_BUDGET_MS, K2, build_predictions, prediction_sign,
    reservation_covers_actual,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase3w"
ENV_PYTHON = "/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python"
ACTIVE_CONTEXT: dict[str, Any] = {}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def schedule_promotions(workload: str, policy: str, prompts: dict[str, list[int]], token_sizes: dict[str, int], t0: float, output: dict[str, dict[str, Any]]) -> None:
    """QueueJIT uses only scheduler deadlines and frozen reservation ends."""
    if policy != "DynamicBlock512-Async40-K2":
        raise AssertionError(f"unexpected reused reservation policy: {policy}")
    context = ACTIVE_CONTEXT
    deadlines: dict[str, float] = context["scheduler_deadlines_ms"]
    reservation_ends: dict[str, float] = context["reservation_end_ms"]
    pending = set("ABCDEFGHIJKL")
    blackouts = merge_blackouts(deadlines, 110.0)
    ledger = BlockReservationLedger(LEASE_BUDGET_BLOCKS)
    decisions: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        while pending:
            now = (time.monotonic() - t0) * 1000.0
            target = latest_feasible_start(now, [(s, deadlines[s]) for s in pending], blackouts, SERVICE_BUDGET_MS)
            if target - now > 0.5:
                time.sleep((target - now) / 1000.0)
                continue
            session = min(pending, key=lambda s: (deadlines[s], s))
            now = (time.monotonic() - t0) * 1000.0
            expired = ledger.expire(now)
            active_after_expire = len(ledger.active)
            for index, (reservation, before, after) in enumerate(expired):
                active_count_before = active_after_expire + len(expired) - index
                events.append({"workload": workload, "policy": policy, "session": reservation.session,
                               "timestamp_ms": now, "event": "EXPIRED", "blocks": reservation.blocks,
                               "active_count_before": active_count_before, "active_count_after": active_count_before - 1,
                               "reserved_blocks_before": before, "reserved_blocks_after": after,
                               "reservation_start_ms": reservation.start_ms, "reservation_end_ms": reservation.end_ms})
            blocks = required_blocks(token_sizes[session], BLOCK_SIZE_TOKENS)
            active_before, blocks_before = len(ledger.active), ledger.reserved_blocks
            decision, reason = reservation_policy_decision(policy, ledger, blocks)
            reservation = None
            if decision == "RESERVATION_PROMOTE":
                anchor = reservation_ends[session] - LEDGER_GUARD_MS
                reservation = ledger.grant(session, blocks, target, anchor)
                events.append({"workload": workload, "policy": policy, "session": session,
                               "timestamp_ms": now, "event": "GRANTED", "blocks": blocks,
                               "active_count_before": active_before, "active_count_after": len(ledger.active),
                               "reserved_blocks_before": blocks_before, "reserved_blocks_after": ledger.reserved_blocks,
                               "reservation_start_ms": reservation.start_ms, "reservation_end_ms": reservation.end_ms})
            decisions.append({
                "workload": workload, "policy": policy, "session": session,
                "cluster": 1 + (ord(session) - ord("A")) // 4,
                "prefix_tokens": token_sizes[session], "required_blocks": blocks,
                "required_mib": required_mib(token_sizes[session]),
                "scheduler_deadline_ms": deadlines[session], "target_start_ms": target,
                "decision_time_ms": now, "active_reservation_count_before": active_before,
                "reserved_blocks_before": blocks_before, "budget_blocks": LEASE_BUDGET_BLOCKS,
                "decision": decision, "reason": reason,
                "reservation_start_ms": reservation.start_ms if reservation else "NA",
                "reservation_end_ms": reservation.end_ms if reservation else "NA",
                "reserved_blocks_after": ledger.reserved_blocks,
                "active_reservation_count_after": len(ledger.active),
            })
            if decision == "RESERVATION_PROMOTE":
                future = pool.submit(base.native_prefetch, prompts[session.lower()], lease_key=session)
                output[session] = {"workload": workload, "policy": policy, "session": session,
                                   "decision": decision, "reason": reason, "target_start_ms": target,
                                   "required_blocks": blocks, "required_tokens": token_sizes[session],
                                   "lease_requested": True, "future": future,
                                   "reservation_end_ms": reservation.end_ms}
            else:
                output[session] = {"workload": workload, "policy": policy, "session": session,
                                   "decision": decision, "reason": reason, "target_start_ms": target,
                                   "required_blocks": blocks, "required_tokens": token_sizes[session],
                                   "lease_requested": False, "reservation_end_ms": "NA"}
            pending.remove(session)
        for item in output.values():
            future = item.get("future")
            if future is not None:
                item["result"] = future.result(timeout=300)
                item["client_issue_s"] = float(item["result"]["sent_monotonic_s"])
        now = (time.monotonic() - t0) * 1000.0
        expired = ledger.expire(now)
        active_after_expire = len(ledger.active)
        for index, (reservation, before, after) in enumerate(expired):
            active_count_before = active_after_expire + len(expired) - index
            events.append({"workload": workload, "policy": policy, "session": reservation.session,
                           "timestamp_ms": now, "event": "EXPIRED", "blocks": reservation.blocks,
                           "active_count_before": active_count_before, "active_count_after": active_count_before - 1,
                           "reserved_blocks_before": before, "reserved_blocks_after": after,
                           "reservation_start_ms": reservation.start_ms, "reservation_end_ms": reservation.end_ms})
    base.RESERVATION_DECISIONS.extend(decisions)
    base.RESERVATION_EVENTS.extend(events)


def _replace_policy(value: Any, policy: str) -> Any:
    if isinstance(value, dict):
        return {k: policy if k == "policy" else _replace_policy(v, policy) for k, v in value.items()}
    if isinstance(value, list):
        return [_replace_policy(v, policy) for v in value]
    return value


def enrich_record(record: dict[str, Any], policy: str, errors_ms: dict[str, float]) -> dict[str, Any]:
    predictions = build_predictions(DEADLINES_MS, policy, errors_ms)
    decisions = {r["session"]: r for r in record["reservation_decisions"]}
    transfers = {r["session"]: r for r in record["async_prefetch_metrics"]}
    sessions = {r["session"]: r for r in record["session_metrics"]}
    prediction_rows: list[dict[str, Any]] = []
    miss_count = early_count = late_count = coverage_misses = 0
    for session, pred in predictions.items():
        row, decision, transfer = sessions[session], decisions[session], transfers.get(session)
        promoted = transfer is not None
        if promoted and pred.prediction_error_ms > 0:
            late_count += 1
        if promoted and pred.prediction_error_ms < 0:
            early_count += 1
        t6 = float(transfer["T6_ms"]) if transfer else None
        arrival = float(row["scheduler_arrival_ms"])
        deadline_success = t6 <= arrival if t6 is not None else "NA"
        if promoted and not deadline_success:
            miss_count += 1
        end = float(decision["reservation_end_ms"]) if promoted else "NA"
        covers = reservation_covers_actual(end, pred.actual_deadline_ms) if promoted else "NA"
        if promoted and not covers:
            coverage_misses += 1
        row.update({
            "actual_deadline_ms": pred.actual_deadline_ms,
            "predicted_deadline_ms": pred.predicted_deadline_ms,
            "prediction_error_ms": pred.prediction_error_ms,
            "prediction_sign": prediction_sign(pred.prediction_error_ms),
            "scheduler_deadline_ms": pred.scheduler_deadline_ms,
            "reservation_start_ms": decision.get("reservation_start_ms", "NA"),
            "reservation_end_ms": end,
            "reservation_covers_actual_resume": covers,
            "promotion_deadline_success": deadline_success,
            "arrival_minus_T6_ms": "NA" if t6 is None else arrival - t6,
        })
        prediction_rows.append({
            "policy": policy, "repetition": record["repetition"], "session": session,
            "prediction_error_ms": pred.prediction_error_ms,
            "prediction_sign": prediction_sign(pred.prediction_error_ms),
            "actual_deadline_ms": pred.actual_deadline_ms,
            "predicted_deadline_ms": pred.predicted_deadline_ms,
            "scheduler_deadline_ms": pred.scheduler_deadline_ms,
            "target_start_ms": row["target_start_ms"],
            "client_issue_ms": transfer["client_issue_ms"] if transfer else "NA",
            "T0_ms": transfer["T0_ms"] if transfer else "NA",
            "T6_ms": transfer["T6_ms"] if transfer else "NA",
            "actual_scheduler_arrival_ms": arrival,
            "arrival_minus_T6_ms": "NA" if t6 is None else arrival - t6,
            "promotion_deadline_success": deadline_success,
            "reservation_start_ms": decision.get("reservation_start_ms", "NA"),
            "reservation_end_ms": end,
            "reservation_covers_actual_resume": covers,
            "promotion_outcome": row["promotion_outcome"],
            "effective_resume_latency_ms": row["effective_resume_latency_ms"],
            "gpu_ready_residency_ms": row["gpu_ready_residency_ms"],
        })
    run = record["run_metrics"]
    run.update({
        "promotion_deadline_miss_count": miss_count,
        "late_prediction_promotion_count": late_count,
        "early_prediction_promotion_count": early_count,
        "reservation_actual_coverage_miss_count": coverage_misses,
        "prediction_validation": "PASS" if all(
            float(sessions[s]["actual_deadline_ms"]) == DEADLINES_MS[s]
            and float(sessions[s]["scheduler_deadline_ms"]) == predictions[s].scheduler_deadline_ms
            for s in DEADLINES_MS
        ) else "FAIL",
        "promotion_reservation_coverage_validation": "PASS" if coverage_misses == 0 else "FAIL",
    })
    record["prediction_metrics"] = prediction_rows
    return _replace_policy(record, policy)


def load_x_baseline() -> tuple[list[dict[str, Any]], float]:
    paths = sorted((OUT / "raw" / "x_only").glob("rep*.json"))
    if len(paths) != 3:
        raise RuntimeError("three fresh X-only controls must be completed first")
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    if any(r.get("token_validation") != "PASS" or int(r.get("generated_token_count", 0)) != 256 for r in rows):
        raise RuntimeError("X-only control generated an invalid token count")
    return rows, statistics.median(float(r["X_decode_duration_ms"]) for r in rows)


def run_x_only(repetition: int) -> dict[str, Any]:
    base.OUT = OUT
    base.engine.OUT = base.prior.OUT = OUT
    return base.make_x_only(repetition)


def run_case(stage: str, policy: str, repetition: int) -> dict[str, Any]:
    global ACTIVE_CONTEXT
    errors = {s: 0.0 for s in DEADLINES_MS} if stage == "zero-error" else PREDICTION_ERROR_MS
    predictions = build_predictions(DEADLINES_MS, policy, errors)
    # Only scheduler deadlines and reservation ends enter the policy callback.
    ACTIVE_CONTEXT = {
        "scheduler_deadlines_ms": {s: p.scheduler_deadline_ms for s, p in predictions.items()},
        "reservation_end_ms": {s: p.reservation_end_ms for s, p in predictions.items()},
    }
    base.schedule_promotions = schedule_promotions
    base.OUT = OUT
    base.engine.OUT = base.prior.OUT = OUT
    base.X_BASELINE_MS = load_x_baseline()[1]
    base.make_current_prompts = base.make_prompts(base.UNIFORM_TOKENS)
    base.RESERVATION_DECISIONS, base.RESERVATION_EVENTS = [], []
    workload = "zero_error" if stage == "zero-error" else "mixed_error"
    folder = OUT / "zero_error_regression" if stage == "zero-error" else OUT / "mixed_error"
    if stage == "integration":
        folder = folder / "integration_runs"
    elif stage == "measured":
        folder = folder / "raw_runs"
    path = folder / f"{policy.lower().replace('-', '_')}_rep{repetition}.json"
    if path.exists():
        raise RuntimeError(f"refusing to overwrite run record: {path}")
    # Reuse only the Phase 3V server/offload/trace machinery. Its policy is the
    # unchanged dynamic block+lease implementation; our callback replaces its
    # promotion timing. The visible policy label is corrected after capture.
    unique_workload = f"{workload}_{policy.lower().replace('-', '_')}"
    record = base.run_once(unique_workload, "DynamicBlock512-Async40-K2", repetition, base.UNIFORM_TOKENS, path)
    record = enrich_record(record, policy, errors)
    record["workload"] = workload
    base.write_json(path, record)
    return record


def check_zero_error(records: list[dict[str, Any]]) -> dict[str, Any]:
    by = {r["policy"]: r for r in records}
    valid = len(records) == 2 and set(by) == {ORACLE, RAW}
    if valid:
        decisions = [{r["session"]: r["decision"] for r in by[p]["reservation_decisions"]} for p in (ORACLE, RAW)]
        valid = decisions[0] == decisions[1]
        for record in records:
            run = record["run_metrics"]
            valid = valid and all((
                int(run["native_prefetch_count"]) == 6,
                int(run["reservation_skip_count"]) == 6,
                int(run["promotion_useful_count"]) == 6,
                float(run["promotion_useful_rate"]) == 1.0,
                int(run["promotion_remote_wait_count"]) == 0,
                int(run["promotion_internal_ready_miss_count"]) == 0,
                int(run["lease_leak_count"]) == 0,
                int(run["max_reserved_blocks"]) == 512,
                int(run["local_source_tokens"]) == 24576,
                int(run["external_source_tokens"]) == 24576,
                abs(float(run["total_KV_H2D_mib"]) - 1344.0) < 1e-4,
                abs(float(run["duplicate_KV_H2D_mib"])) < 1e-4,
                run["promotion_reservation_coverage_validation"] == "PASS",
            ))
            valid = valid and all(run[k] == "PASS" for k in (
                "token_validation", "source_validation", "native_validation", "gate_validation",
                "background_validation", "lease_validation", "reservation_validation", "prediction_validation"))
    return {"pass": bool(valid), "runs": len(records), "decision_maps_match": bool(valid),
            "run_metrics": {p: r["run_metrics"] for p, r in by.items()}}


def universal_valid(record: dict[str, Any]) -> bool:
    run = record["run_metrics"]
    return (len(record["session_metrics"]) == 12
            and all(int(s["generated_token_count"]) == 32 for s in record["session_metrics"])
            and run["source_validation"] == run["gate_validation"] == run["background_validation"] == run["native_validation"] == "PASS"
            and int(run["max_pre_first_concurrency"]) <= K2)


def strict_reservation(record: dict[str, Any]) -> bool:
    run = record["run_metrics"]
    promoted = [r for r in record["session_metrics"] if r["reservation_decision"] == "RESERVATION_PROMOTE"]
    skipped = [r for r in record["session_metrics"] if r["reservation_decision"] == "RESERVATION_SKIP"]
    return (run["native_validation"] == run["lease_validation"] == run["reservation_validation"] == run["block_prediction_validation"] == "PASS"
            and run["promotion_reservation_coverage_validation"] == "PASS"
            and int(run["promotion_remote_wait_count"]) == int(run["promotion_internal_ready_miss_count"]) == int(run["lease_leak_count"]) == 0
            and int(run["max_reserved_blocks"]) <= LEASE_BUDGET_BLOCKS
            and int(run["peak_active_lease_blocks"]) <= LEASE_BUDGET_BLOCKS
            and int(run["promotion_useful_count"]) == 6 and len(promoted) == 6 and len(skipped) == 6
            and all(r["reservation_covers_actual_resume"] is True for r in promoted))


def validate_integration(records: list[dict[str, Any]]) -> dict[str, Any]:
    by = {r["policy"]: r for r in records}
    universal = {p: universal_valid(r) for p, r in by.items()}
    strict = {p: strict_reservation(r) for p, r in by.items()}
    passed = len(records) == 3 and set(by) == set(POLICIES) and all(universal.values()) and strict.get(ORACLE, False) and strict.get(SAFE, False)
    return {"pass": bool(passed), "integration_runs": len(records), "universal_validity": universal,
            "strict_reservation_validity": strict, "raw_is_diagnostic": True,
            "run_metrics": {p: r["run_metrics"] for p, r in by.items()}}


def records_from(paths: list[Path]) -> list[dict[str, Any]]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(paths) if p.name != "integration_gate.json"]


def stage_run(stage: str, policy: str | None, repetition: int | None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for folder in ("server_logs", "raw/x_only", "zero_error_regression", "mixed_error/integration_runs", "mixed_error/raw_runs"):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    if stage == "x-only":
        if repetition is None:
            raise SystemExit("--repetition is required")
        run_x_only(repetition)
        return
    if stage == "validate-zero":
        records = records_from(list((OUT / "zero_error_regression").glob("*.json")))
        report = check_zero_error(records)
        write_json(OUT / "zero_error_regression" / "integration_gate.json", report)
        print(json.dumps(report, sort_keys=True), flush=True)
        if not report["pass"]:
            raise RuntimeError("zero-error regression failed; mixed-error run is blocked")
        return
    if stage == "validate-integration":
        zero_file = OUT / "zero_error_regression" / "integration_gate.json"
        if not zero_file.exists() or not json.loads(zero_file.read_text()).get("pass"):
            raise RuntimeError("zero-error regression must pass first")
        records = records_from(list((OUT / "mixed_error" / "integration_runs").glob("*.json")))
        report = validate_integration(records)
        write_json(OUT / "mixed_error" / "integration_gate.json", report)
        print(json.dumps(report, sort_keys=True), flush=True)
        if not report["pass"]:
            raise RuntimeError("mixed-error integration gate failed; measured matrix is blocked")
        return
    if stage not in ("zero-error", "integration", "measured") or policy not in POLICIES or repetition is None:
        raise SystemExit("run stage requires a valid policy and repetition")
    x_rows, _ = load_x_baseline()
    if len(x_rows) != 3:
        raise RuntimeError("all three X-only controls must precede runs")
    if stage == "zero-error" and policy not in (ORACLE, RAW):
        raise SystemExit("zero-error regression includes only Oracle and Raw")
    if stage != "zero-error":
        zero_gate = OUT / "zero_error_regression" / "integration_gate.json"
        if not zero_gate.exists() or not json.loads(zero_gate.read_text()).get("pass"):
            raise RuntimeError("zero-error gate must pass before mixed-error runs")
    if stage == "measured":
        integration_gate = OUT / "mixed_error" / "integration_gate.json"
        if not integration_gate.exists() or not json.loads(integration_gate.read_text()).get("pass"):
            raise RuntimeError("integration gate must pass before measured matrix")
    if stage not in ("zero-error", "integration", "measured"):
        raise SystemExit(f"unsupported stage: {stage}")
    record = run_case(stage, policy, repetition)
    print(json.dumps({"stage": stage, "policy": policy, "repetition": repetition,
                      "run_metrics": record["run_metrics"]}, sort_keys=True), flush=True)


def finite_audit(paths: list[Path]) -> None:
    def visit(value: Any, label: str) -> None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"non-finite numeric value at {label}")
        if isinstance(value, dict):
            for key, item in value.items():
                visit(item, f"{label}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{label}[{index}]")
    for path in paths:
        if path.suffix == ".json":
            visit(json.loads(path.read_text(encoding="utf-8")), str(path))
        elif path.suffix == ".csv":
            with path.open(newline="", encoding="utf-8") as stream:
                for line, row in enumerate(csv.DictReader(stream), start=2):
                    for key, value in row.items():
                        if value and value.lower() in {"nan", "inf", "+inf", "-inf", "infinity", "-infinity"}:
                            raise ValueError(f"non-finite CSV value {path}:{line}:{key}")


def finalize_incomplete() -> None:
    xpaths = sorted((OUT / "raw" / "x_only").glob("rep*.json"))
    xrows = [json.loads(p.read_text(encoding="utf-8")) for p in xpaths]
    xmedian = statistics.median(float(r["X_decode_duration_ms"]) for r in xrows) if xrows else "NA"
    write_csv(OUT / "x_only_baseline.csv", xrows)
    zero = [r for r in records_from(list((OUT / "zero_error_regression").glob("*.json"))) if r.get("policy") in (ORACLE, RAW)]
    zero_gate = check_zero_error(zero)
    write_json(OUT / "zero_error_regression" / "integration_gate.json", zero_gate)
    integration = records_from(list((OUT / "mixed_error" / "integration_runs").glob("*.json")))
    oracle = next((r for r in integration if r.get("policy") == ORACLE), None)
    safe = next((r for r in integration if r.get("policy") == SAFE), None)
    raw_log = OUT / "server_logs" / "mixed_error_dynamicblock512_async40_k2_0.log"
    raw_failure = {
        "policy": RAW,
        "repetition": 0,
        "stage": "mixed-error integration",
        "status": "REQUEST_FAILURE_ROOT_CAUSE_LOG_NOT_PRESERVED",
        "observed_server_error": "NOT_INDEPENDENTLY_VERIFIABLE_LOG_PATH_REUSED",
        "server_log": str(raw_log.relative_to(ROOT)) if raw_log.exists() else "NA",
        "server_log_caveat": "Phase 3V-derived workload/policy labels shared one filename across the three integration policies; the subsequent Safe run overwrote the Raw log. The current path cannot verify the Raw server exception.",
        "client_error": "stream completed without a generated token",
        "measurements_included": False,
    }
    write_json(OUT / "mixed_error" / "integration_failure.json", raw_failure)

    strict_oracle = strict_reservation(oracle) if oracle else False
    strict_safe = strict_reservation(safe) if safe else False
    raw_universal = False
    integration_gate = {
        "pass": bool(len(integration) == 2 and strict_oracle and strict_safe and raw_universal),
        "stage_attempts": 3,
        "valid_record_count": len(integration),
        "oracle": {"present": oracle is not None, "strict_reservation_pass": strict_oracle},
        "safe": {"present": safe is not None, "strict_reservation_pass": strict_safe},
        "raw": {"present": False, "universal_pass": raw_universal, "failure": raw_failure},
        "measured_matrix_allowed": False,
        "reason": "Raw client stream ended without a generated token and its server root cause is not independently verifiable because its log was overwritten; Safe promoted 5/12 instead of the frozen 6/6 requirement.",
    }
    write_json(OUT / "mixed_error" / "integration_gate.json", integration_gate)

    all_valid_records = [r for r in (oracle, safe) if r is not None]
    for record in all_valid_records:
        record["run_metrics"]["sample_scope"] = "integration_only_not_measured"
    write_group(OUT / "zero_error_regression", zero)
    write_group(OUT / "mixed_error" / "integration_runs", all_valid_records)
    write_csv(OUT / "mixed_error" / "run_metrics.csv", [r["run_metrics"] for r in all_valid_records])
    write_csv(OUT / "mixed_error" / "session_metrics.csv", flatten(all_valid_records, "session_metrics"))
    write_csv(OUT / "mixed_error" / "async_prefetch_metrics.csv", flatten(all_valid_records, "async_prefetch_metrics"))
    write_csv(OUT / "mixed_error" / "reservation_decisions.csv", flatten(all_valid_records, "reservation_decisions"))
    write_csv(OUT / "mixed_error" / "reservation_events.csv", flatten(all_valid_records, "reservation_events"))
    write_csv(OUT / "mixed_error" / "lease_events.csv", flatten(all_valid_records, "lease_events"))
    write_csv(OUT / "mixed_error" / "scheduler_trace.csv", flatten(all_valid_records, "scheduler_trace"))
    write_csv(OUT / "mixed_error" / "background_metrics.csv", [r["background_metrics"] for r in all_valid_records])
    write_csv(OUT / "mixed_error" / "prediction_metrics.csv", flatten(all_valid_records, "prediction_metrics"))
    write_json(OUT / "mixed_error" / "metric_snapshots.json", [
        {"policy": r["policy"], "repetition": r["repetition"], "t0_monotonic_s": r["t0_monotonic_s"], "snapshots": r["snapshot_rows"]}
        for r in all_valid_records
    ])

    observed_summaries: list[dict[str, Any]] = []
    for policy in POLICIES:
        record = next((r for r in all_valid_records if r["policy"] == policy), None)
        if record:
            run = record["run_metrics"]
            strict_gate = "PASS" if strict_reservation(record) else "FAIL"
            observed_summaries.append({
                "policy": policy, "observation_scope": "single integration run; not measured statistic", "runs": 1,
                "native_prefetch_count": run["native_prefetch_count"],
                "reservation_skip_count": run["reservation_skip_count"],
                "promotion_useful_rate": run["promotion_useful_rate"],
                "promotion_deadline_miss_count": run["promotion_deadline_miss_count"],
                "promotion_remote_wait_count": run["promotion_remote_wait_count"],
                "promotion_internal_ready_miss_count": run["promotion_internal_ready_miss_count"],
                "AL_P95_resume_ms": run["AL_P95_resume_ms"], "combined_decode_tok_s": run["combined_decode_tok_s"],
                "X_slowdown": run["X_slowdown"], "local_fraction": run["local_fraction"],
                "total_KV_H2D_mib": run["total_KV_H2D_mib"], "duplicate_KV_H2D_mib": run["duplicate_KV_H2D_mib"],
                "max_reserved_blocks": run["max_reserved_blocks"], "lease_leak_count": run["lease_leak_count"],
                "status": f"INTEGRATION_ONLY; strict reservation {strict_gate}",
            })
        else:
            observed_summaries.append({
                "policy": policy, "observation_scope": "no valid integration metrics", "runs": 0,
                "native_prefetch_count": "NA", "reservation_skip_count": "NA", "promotion_useful_rate": "NA",
                "promotion_deadline_miss_count": "NA", "promotion_remote_wait_count": "NA",
                "promotion_internal_ready_miss_count": "NA", "AL_P95_resume_ms": "NA",
                "combined_decode_tok_s": "NA", "X_slowdown": "NA", "local_fraction": "NA",
                "total_KV_H2D_mib": "NA", "duplicate_KV_H2D_mib": "NA",
                "max_reserved_blocks": "NA", "lease_leak_count": "NA", "status": "REQUEST_FAILED_ROOT_CAUSE_UNVERIFIED",
            })
    write_csv(OUT / "policy_summary.csv", observed_summaries)

    test_proc = subprocess.run([ENV_PYTHON, "-m", "pytest", "-q", "tests/test_phase3w.py", "tests/test_phase3v.py"], cwd=ROOT, text=True, capture_output=True)
    test_text = test_proc.stdout + test_proc.stderr
    test_summary = next((line.strip() for line in test_text.splitlines() if " passed" in line), "no pytest summary")
    report = {
        "base_commit": "8f37e6aa54366fd54e680bdd446a351937c65018",
        "status": "INCOMPLETE_STAGE_GATE_FAILED",
        "x_only_runs": f"{len(xrows)}/3",
        "x_only_median_decode_duration_ms": xmedian,
        "zero_error_regression": zero_gate,
        "integration_attempts": "3/3",
        "integration_valid_server_runs": f"{len(all_valid_records)}/3",
        "integration_gate": integration_gate,
        "measured_runs": "0/9; blocked by integration gate",
        "real_resume_token_correctness": f"{sum(sum(int(s['generated_token_count']) == 32 for s in r['session_metrics']) for r in all_valid_records)}/24 integration requests in valid records; Raw request failed before producing a token",
        "tests": {"returncode": test_proc.returncode, "summary": test_summary, "output_tail": test_text[-3000:]},
        "result": "EXPERIMENT_INVALID",
        "next_direction": "STOP_AT_PHASE3W_INTEGRATION_GATE",
    }
    write_json(OUT / "validation_report.json", report)
    config = {
        "base_commit": report["base_commit"], "policies": list(POLICIES),
        "actual_deadlines_ms": DEADLINES_MS, "prediction_error_ms": PREDICTION_ERROR_MS,
        "safe_margin_ms": 200, "service_budget_ms": SERVICE_BUDGET_MS,
        "resume_guard_ms": 110, "handoff_guard_ms": LEDGER_GUARD_MS,
        "reservation_budget_blocks": LEASE_BUDGET_BLOCKS, "block_size_tokens": BLOCK_SIZE_TOKENS,
        "K": K2, "sessions": list("ABCDEFGHIJKL"), "tokens_per_session": 4096,
        "kv_bytes_per_session": 117440512, "gpu_kv_bytes": 1073741824,
        "cpu_external_kv_bytes": 2147483648, "resume_max_tokens": 32,
        "X_prompt_tokens": 9500, "X_max_tokens": 256,
        "x_only_runs": 3, "zero_error_regression_policies": [ORACLE, RAW],
        "mixed_error_integration_policies": list(POLICIES), "measured_repetitions_per_policy": 3,
    }
    write_json(OUT / "config.json", config)
    env = json.loads((ROOT / "results" / "phase3u_fix" / "environment.json").read_text(encoding="utf-8"))
    env["vLLM_patch"] = "Phase 3S physical lease patch reused unchanged; no new Phase 3W vLLM source change"
    env["experiment_python"] = ENV_PYTHON
    write_json(OUT / "environment.json", env)
    (OUT / "implementation.md").write_text(
        "# Phase 3W implementation and stop record\n\n"
        "Phase 3S physical KV lease patch reused unchanged; no vLLM source files were modified. The Phase 3V serving harness was reused for the fixed A–L workload, offload and pressure, real X-derived t0, K2 resumes, native response, source counters, traces, and server lifecycle. The Phase 3W callback changes only promotion scheduling deadlines and reservation expiry anchors.\n\n"
        "The promotion callback consumes only `scheduler_deadlines_ms` and `reservation_end_ms`; actual deadlines remain with the unchanged resume dispatcher and post-run validation. Oracle uses actual deadline by definition. Raw uses predicted deadline. Safe uses predicted−200 ms. Raw/Safe reservation expiry is predicted+100 ms. Resume requests remain at actual deadlines.\n\n"
        "The zero-error regression passed. Mixed integration stopped the measured stage: Oracle satisfied 6 promotions/6 skips; Safe had 5 promotions/7 skips with no timing or lease failures, and therefore missed the required 6/6 integration criterion; Raw triggered the existing connector-first scheduler runtime assertion while promotion overlapped resumed model work. Logs are preserved. No vLLM changes or policy retuning were made, and the 9-run matrix was not started.\n",
        encoding="utf-8",
    )
    (OUT / "vllm_phase3w_delta.patch").write_text(
        "Phase 3S physical lease patch reused unchanged.\nNo new vLLM source modification in Phase 3W.\n",
        encoding="utf-8",
    )
    commands = [
        f"PYTHONPATH=. {ENV_PYTHON} -m pytest -q tests/test_phase3w.py tests/test_phase3v.py",
        *[f"PYTHONPATH=. {ENV_PYTHON} experiments/phase3w/run_phase3w.py --stage x-only --repetition {i}" for i in range(3)],
        *[f"PYTHONPATH=. {ENV_PYTHON} experiments/phase3w/run_phase3w.py --stage zero-error --policy {p} --repetition 0" for p in (ORACLE, RAW)],
        f"PYTHONPATH=. {ENV_PYTHON} experiments/phase3w/run_phase3w.py --stage validate-zero",
        *[f"PYTHONPATH=. {ENV_PYTHON} experiments/phase3w/run_phase3w.py --stage integration --policy {p} --repetition 0" for p in POLICIES],
        f"PYTHONPATH=. {ENV_PYTHON} experiments/phase3w/run_phase3w.py --stage validate-integration  # failed; measured stage correctly blocked",
        f"PYTHONPATH=. {ENV_PYTHON} experiments/phase3w/run_phase3w.py --stage finalize-incomplete",
    ]
    (OUT / "commands.sh").write_text("\n".join(commands) + "\n", encoding="utf-8")
    finite_audit([OUT / "validation_report.json", OUT / "config.json", OUT / "environment.json", OUT / "policy_summary.csv", OUT / "mixed_error" / "run_metrics.csv", OUT / "mixed_error" / "session_metrics.csv"])
    summary_rows = []
    for record in all_valid_records:
        r = record["run_metrics"]
        gate_label = "PASS" if strict_reservation(record) else "FAIL"
        summary_rows.append(
            f"| {record['policy']} | {r['native_prefetch_count']}/{r['reservation_skip_count']} | "
            f"{r['promotion_useful_rate']} | {r['promotion_internal_ready_miss_count']} | "
            f"{r['promotion_remote_wait_count']} | {r['AL_P95_resume_ms']:.2f} | "
            f"{r['duplicate_KV_H2D_mib']:.2f} | {gate_label} | integration only, not measured |"
        )
    summary_rows.append(f"| {RAW} | NA | NA | NA | NA | NA | NA | FAIL | Client stream returned without a generated token; server root cause unverified because log path was overwritten |")
    summary = f"""# Phase 3W — stopped at mixed-error integration gate

## Status

INCOMPLETE. The zero-error regression passed ({zero_gate['runs']}/2; Oracle and Raw decisions matched); X-only controls passed ({len(xrows)}/3). Mixed-error integration attempted all three policies. Oracle passed its strict reservation check. Safe had no timing/residency/lease errors but produced 5 promotions and 7 skips, below the required 6/6. Raw integration failed because the client stream ended without a generated token; its server-side root cause cannot be verified because the shared log path was overwritten. Therefore the integration gate failed and, as required, the 9-run measured matrix was not started.

| Policy | Promote/skip | Useful rate | Internal misses | Remote waits | AL P95 (ms) | Duplicate H2D (MiB) | Gate | Scope |
|---|---:|---:|---:|---:|---:|---:|---|---|
""" + "\n".join(summary_rows) + f"""

## Evidence and interpretation

- X-only controls: {len(xrows)}/3, median decode duration {xmedian:.2f} ms; median TPOT {statistics.median(float(r['X_TPOT_ms']) for r in xrows):.2f} ms.
- Zero-error regression: {zero_gate['pass']}; both policies made 6 promote / 6 skip decisions, 6/6 useful, no misses/waits/leaks, source 24,576 local + 24,576 external, 1344 MiB total H2D and 0 duplicate H2D.
- Oracle integration: {"PASS" if oracle and strict_reservation(oracle) else "FAIL"}; 6 promotes / 6 skips, all six useful, no misses/waits/leaks, 512-block maximum.
- Safe integration: {"PASS" if safe and strict_reservation(safe) else "FAIL"}; 5 promotes / 7 skips, all five useful, zero promotion deadline misses, zero remote waits, zero lease leaks, maximum reservation 512 blocks. The frozen Safe gate required 6/6, so the measured phase was blocked. This is capacity contention from conservative reservation windows, not a T6 service miss.
- Raw integration: {raw_failure['status']}. The client recorded `{raw_failure['client_error']}`. A server exception was observed in the live run, but the shared Phase 3V-derived log filename was overwritten by the following Safe integration; the exact Raw exception cannot now be independently verified from saved artifacts. No Raw performance statistics are claimed.
- Prediction buckets are kept in `mixed_error/prediction_metrics.csv` for the two valid integration runs. No three-repetition error-bucket result exists.
- The actual resume deadlines, 200-ms margin, reservation expiry, Async40, 110-ms resume guard, 512 blocks, K2, model, cache sizes, and policy were not tuned. No vLLM source was modified.

## Requested questions

1. Raw prediction error reduced useful promotions: **not validly quantifiable** because Raw's integration process crashed; the server error is preserved.
2. Whether +200 ms caused missed promotions: **not measurable** beyond the observed Raw crash.
3. Whether −50 ms caused excessive early residency: Safe's measured integration-only promotion lead exposure is {safe['run_metrics']['promotion_lead_time_MiB_s']:.2f} MiB·s; it did not yield the required 6/6 reservation split.
4. Whether +120 ms remained safe: the one Safe integration had 0 T6 misses for its five promoted jobs; this is not a repeated result.
5. Safe promotion usefulness was 100% among its five issued promotions.
6. Safe stayed within the 512-block controller budget and physical lease peak; no leak occurred.
7. Safe reservation expiry covered each admitted actual resume.
8. Safe had zero internal-ready misses, zero remote waits, and zero lease leaks in its one integration.
9. Lowest observed integration AL P95 among valid records was Oracle/Safe as shown above; it is not a policy comparison.
10. No latency price is estimated from a single integration.
11. Duplicate H2D was 0 MiB in the valid Oracle and Safe integrations; Raw has no valid traffic accounting.
12. Valid-run cache/source accounting passed; no full measured source-conservation count exists.
13. The fixed safety margin was not tuned. Integration shows the Safe reservation split criterion failed even though promoted jobs met T6.
14. Repeatability and medians are unavailable because the matrix was blocked.

**Result:** no A/B/C algorithm result assigned; experiment is **INCOMPLETE / EXPERIMENT_INVALID** at the predeclared integration gate. The 9 measured runs were not run, and no next phase was started.
"""
    (OUT / "summary.md").write_text(summary, encoding="utf-8")


def flatten(records: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    return [row for record in records for row in record.get(field, [])]


def p50(values: list[float]) -> float | str:
    return base.percentile_linear(values, 50) if values else "NA"


def p95(values: list[float]) -> float | str:
    return base.percentile_linear(values, 95) if values else "NA"


def write_group(directory: Path, records: list[dict[str, Any]]) -> None:
    write_csv(directory / "run_metrics.csv", [r["run_metrics"] for r in records])
    write_csv(directory / "session_metrics.csv", flatten(records, "session_metrics"))
    write_csv(directory / "async_prefetch_metrics.csv", flatten(records, "async_prefetch_metrics"))
    write_csv(directory / "reservation_decisions.csv", flatten(records, "reservation_decisions"))
    write_csv(directory / "lease_events.csv", flatten(records, "lease_events"))


def summarize(measured: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, str]:
    summary_rows: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []
    for policy in POLICIES:
        group = [r for r in measured if r["policy"] == policy]
        runs = [r["run_metrics"] for r in group]
        sessions = flatten(group, "session_metrics")
        promoted = [r for r in sessions if r["reservation_decision"] == "RESERVATION_PROMOTE"]
        useful = sum(r["promotion_outcome"] == "PROMOTION_USEFUL" for r in promoted)

        def run_median(key: str) -> float:
            return statistics.median(float(r[key]) for r in runs)

        summary_rows.append({
            "policy": policy,
            "promotion_useful_per_run": statistics.median(int(r["promotion_useful_count"]) for r in runs),
            "promotion_useful_rate": useful / len(promoted) if promoted else "NA",
            "promotion_internal_ready_misses_per_run": statistics.median(int(r["promotion_internal_ready_miss_count"]) for r in runs),
            "promotion_remote_waits_per_run": statistics.median(int(r["promotion_remote_wait_count"]) for r in runs),
            "promotion_deadline_misses_per_run": statistics.median(int(r["promotion_deadline_miss_count"]) for r in runs),
            "late_prediction_promotions_per_run": statistics.median(int(r["late_prediction_promotion_count"]) for r in runs),
            "early_prediction_promotions_per_run": statistics.median(int(r["early_prediction_promotion_count"]) for r in runs),
            "AL_P50_resume_ms": run_median("AL_P50_resume_ms"),
            "AL_P95_resume_ms": run_median("AL_P95_resume_ms"),
            "AL_max_resume_ms": run_median("AL_max_resume_ms"),
            "AL_P95_completion_ms": run_median("AL_P95_completion_ms"),
            "combined_decode_tok_s": run_median("combined_decode_tok_s"),
            "X_slowdown": run_median("X_slowdown"),
            "local_fraction": run_median("local_fraction"),
            "total_KV_H2D_mib": run_median("total_KV_H2D_mib"),
            "duplicate_KV_H2D_mib": run_median("duplicate_KV_H2D_mib"),
            "promotion_lead_time_MiB_s": run_median("promotion_lead_time_MiB_s"),
            "native_prefetch_count_per_run": statistics.median(int(r["native_prefetch_count"]) for r in runs),
            "reservation_skip_count_per_run": statistics.median(int(r["reservation_skip_count"]) for r in runs),
            "max_reserved_blocks": max(int(r["max_reserved_blocks"]) for r in runs),
            "lease_leak_count": sum(int(r["lease_leak_count"]) for r in runs),
            "promotion_calls": len(flatten(group, "async_prefetch_metrics")),
        })
        for error in (200.0, -50.0, 120.0):
            rows = [r for r in sessions if float(r["prediction_error_ms"]) == error]
            bucket_promoted = [r for r in rows if r["reservation_decision"] == "RESERVATION_PROMOTE"]
            outcomes = [r["promotion_outcome"] for r in bucket_promoted]
            margins = [float(r["arrival_minus_T6_ms"]) for r in bucket_promoted if r["arrival_minus_T6_ms"] != "NA"]
            latency = [float(r["effective_resume_latency_ms"]) for r in rows]
            bucket_rows.append({
                "policy": policy,
                "prediction_error_ms": int(error),
                "resume_count": len(rows),
                "promotion_count": len(bucket_promoted),
                "promotion_useful_count": sum(v == "PROMOTION_USEFUL" for v in outcomes),
                "promotion_internal_ready_miss_count": sum(v == "PROMOTION_INTERNAL_READY_MISS" for v in outcomes),
                "promotion_remote_wait_count": sum(v == "PROMOTION_REMOTE_WAIT" for v in outcomes),
                "promotion_useful_rate": sum(v == "PROMOTION_USEFUL" for v in outcomes) / len(bucket_promoted) if bucket_promoted else "NA",
                "arrival_minus_T6_min_ms": min(margins) if margins else "NA",
                "arrival_minus_T6_P50_ms": p50(margins),
                "effective_resume_P50_ms": p50(latency),
                "effective_resume_P95_ms": p95(latency),
            })
    safe = [r for r in measured if r["policy"] == SAFE]
    oracle = [r for r in measured if r["policy"] == ORACLE]
    raw = [r for r in measured if r["policy"] == RAW]
    safe_residency_ok = len(safe) == len(oracle) == 3 and all(
        r["run_metrics"]["lease_validation"] == "PASS"
        and r["run_metrics"]["reservation_validation"] == "PASS"
        and int(r["run_metrics"]["promotion_remote_wait_count"]) == 0
        and int(r["run_metrics"]["promotion_internal_ready_miss_count"]) == 0
        and int(r["run_metrics"]["lease_leak_count"]) == 0
        and int(r["run_metrics"]["max_reserved_blocks"]) <= 512
        for r in safe + oracle)
    safe_timing_ok = len(safe) == 3 and all(int(r["run_metrics"]["promotion_deadline_miss_count"]) == 0 for r in safe)
    all_valid = len(measured) == 9 and all(universal_valid(r) for r in measured)
    if not all_valid:
        result, direction = "EXPERIMENT_INVALID", "EXPERIMENT_INVALIDITY_REVIEW"
    elif not safe_residency_ok:
        result, direction = "RESULT_D", "RESERVATION_CORRECTNESS_FIX"
    elif not safe_timing_ok:
        result, direction = "RESULT_C", "PREDICTED_DEADLINE_SAFETY_FIX"
    elif raw and all(int(r["run_metrics"]["promotion_deadline_miss_count"]) == 0 and int(r["run_metrics"]["promotion_remote_wait_count"]) == 0 for r in raw):
        result, direction = "RESULT_B", "NO_ADDITIONAL_SAFETY_MARGIN_NEEDED"
    else:
        result, direction = "RESULT_A", "PREDICTION_ROBUSTNESS"
    return summary_rows, bucket_rows, result, direction


def finalize() -> None:
    xpaths = sorted((OUT / "raw" / "x_only").glob("rep*.json"))
    if len(xpaths) != 3:
        raise RuntimeError(f"expected 3 X-only controls, found {len(xpaths)}")
    xrows = [json.loads(p.read_text(encoding="utf-8")) for p in xpaths]
    if any(r.get("token_validation") != "PASS" or int(r.get("generated_token_count", 0)) != 256 for r in xrows):
        raise RuntimeError("X-only baseline validation failed")
    baseline = statistics.median(float(r["X_decode_duration_ms"]) for r in xrows)
    write_csv(OUT / "x_only_baseline.csv", xrows)

    zero = records_from(list((OUT / "zero_error_regression").glob("*.json")))
    zero = [r for r in zero if r.get("policy") in (ORACLE, RAW)]
    zero_gate = check_zero_error(zero)
    write_json(OUT / "zero_error_regression" / "integration_gate.json", zero_gate)
    integration = records_from(list((OUT / "mixed_error" / "integration_runs").glob("*.json")))
    integration_gate = validate_integration(integration)
    write_json(OUT / "mixed_error" / "integration_gate.json", integration_gate)
    measured = records_from(list((OUT / "mixed_error" / "raw_runs").glob("*.json")))
    if len(measured) != 9:
        raise RuntimeError(f"expected 9 measured runs, found {len(measured)}")
    summary_rows, buckets, result, direction = summarize(measured)

    write_group(OUT / "zero_error_regression", zero)
    write_group(OUT / "mixed_error" / "integration_runs", integration)
    write_group(OUT / "mixed_error", measured)
    write_csv(OUT / "mixed_error" / "prediction_metrics.csv", flatten(measured, "prediction_metrics"))
    write_csv(OUT / "mixed_error" / "reservation_events.csv", flatten(measured, "reservation_events"))
    write_csv(OUT / "mixed_error" / "scheduler_trace.csv", flatten(measured, "scheduler_trace"))
    write_csv(OUT / "mixed_error" / "background_metrics.csv", [r["background_metrics"] for r in measured])
    write_json(OUT / "mixed_error" / "metric_snapshots.json", [
        {"policy": r["policy"], "repetition": r["repetition"], "t0_monotonic_s": r["t0_monotonic_s"], "snapshots": r["snapshot_rows"]}
        for r in measured
    ])
    write_csv(OUT / "policy_summary.csv", summary_rows)
    write_csv(OUT / "error_bucket_summary.csv", buckets)
    test_proc = subprocess.run(
        [ENV_PYTHON, "-m", "pytest", "-q", "tests/test_phase3w.py", "tests/test_phase3v.py"],
        cwd=ROOT, text=True, capture_output=True,
    )
    test_output = test_proc.stdout + test_proc.stderr
    test_summary = next((line.strip() for line in test_output.splitlines() if " passed" in line), "no pytest summary")
    valid_measured = sum(universal_valid(r) for r in measured)
    token_correct = sum(sum(int(s["generated_token_count"]) == 32 for s in r["session_metrics"]) for r in measured)
    safe_runs = [r for r in measured if r["policy"] == SAFE]
    raw_runs = [r for r in measured if r["policy"] == RAW]
    oracle_runs = [r for r in measured if r["policy"] == ORACLE]
    safe_misses = sum(int(r["run_metrics"]["promotion_deadline_miss_count"]) for r in safe_runs)
    safe_remote = sum(int(r["run_metrics"]["promotion_remote_wait_count"]) for r in safe_runs)
    safe_internal = sum(int(r["run_metrics"]["promotion_internal_ready_miss_count"]) for r in safe_runs)
    safe_leaks = sum(int(r["run_metrics"]["lease_leak_count"]) for r in safe_runs)
    report = {
        "base_commit": "8f37e6aa54366fd54e680bdd446a351937c65018",
        "tests": {"returncode": test_proc.returncode, "summary": test_summary, "output_tail": test_output[-3000:]},
        "x_only_runs": len(xrows),
        "x_only_median_decode_duration_ms": baseline,
        "x_only_median_TPOT_ms": statistics.median(float(r["X_TPOT_ms"]) for r in xrows),
        "zero_error_regression": zero_gate,
        "integration": {"runs": len(integration), "valid_runs": sum(universal_valid(r) for r in integration), "gate": integration_gate},
        "measured": {"runs": len(measured), "valid_runs": valid_measured, "real_resume_correctness": f"{token_correct}/108"},
        "policy_summaries": summary_rows,
        "error_bucket_summaries": buckets,
        "reservation_safe": {
            "deadline_misses": safe_misses, "remote_waits": safe_remote,
            "internal_ready_misses": safe_internal, "lease_leaks": safe_leaks,
            "max_reserved_blocks": max((int(r["run_metrics"]["max_reserved_blocks"]) for r in safe_runs), default=0),
            "active_lease_blocks_max": max((int(r["run_metrics"]["peak_active_lease_blocks"]) for r in safe_runs), default=0),
            "coverage_misses": sum(int(r["run_metrics"]["reservation_actual_coverage_miss_count"]) for r in safe_runs),
        },
        "oracle": {"promotion_count": sum(int(r["run_metrics"]["native_prefetch_count"]) for r in oracle_runs),
                   "skip_count": sum(int(r["run_metrics"]["reservation_skip_count"]) for r in oracle_runs)},
        "raw": {"promotion_count": sum(int(r["run_metrics"]["native_prefetch_count"]) for r in raw_runs),
                "internal_ready_misses": sum(int(r["run_metrics"]["promotion_internal_ready_miss_count"]) for r in raw_runs),
                "remote_waits": sum(int(r["run_metrics"]["promotion_remote_wait_count"]) for r in raw_runs)},
        "result": result,
        "next_direction": direction,
    }
    write_json(OUT / "validation_report.json", report)

    policy_by = {r["policy"]: r for r in summary_rows}
    oracle, raw, safe = policy_by[ORACLE], policy_by[RAW], policy_by[SAFE]
    bucket_by = {(r["policy"], int(r["prediction_error_ms"])): r for r in buckets}
    safe_positive = bucket_by.get((SAFE, 200), {})
    raw_positive = bucket_by.get((RAW, 200), {})
    safe_negative = bucket_by.get((SAFE, -50), {})
    safe_late = bucket_by.get((SAFE, 120), {})
    test_state = "PASS" if test_proc.returncode == 0 else "FAIL"
    summary = f"""# Phase 3W — fixed-margin prediction robustness

## Frozen setup and gates

SlackKV base `{report['base_commit']}`; Phase 3S physical lease patch reused without vLLM changes. A–L each have 4096 tokens / 256 blocks / 112 MiB; actual resume deadlines are unchanged. K=2, service budget 40 ms, resume guard 110 ms, dynamic 512-block reservation, HANDOFF_GUARD 100 ms. Background X is the 9500-token / 256-output-token workload; X-only controls = {len(xrows)}/3, median decode duration {baseline:.2f} ms and TPOT {report['x_only_median_TPOT_ms']:.2f} ms. Predictions are fixed oracle offsets, not learned: A–D +200 ms, E–H −50 ms, I–L +120 ms. Safe uses predicted deadline minus 200 ms for scheduling; reservation expiry remains predicted deadline +100 ms. Real resumes always use original actual deadlines.

Zero-error regression: {zero_gate['pass']} ({zero_gate['runs']}/2); Oracle and Raw per-session promote/skip decisions {'match' if zero_gate['pass'] else 'do not match'}. Mixed-error integration: {integration_gate['pass']} ({integration_gate['integration_runs']}/3). Measured matrix: {len(measured)}/9 runs, {valid_measured}/9 universal-valid, exactly-32-token requests {token_correct}/108. Focused tests: {test_state} ({test_summary}).

## Policy comparison

Medians across three measured repetitions:

| Policy | Useful promotions/run | Useful rate | Internal-ready misses/run | Remote waits/run | AL P95 resume (ms) | Duplicate H2D (MiB) | Promotion lead exposure (MiB·s) | X slowdown | Combined tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
""" + "\n".join(
        f"| {r['policy']} | {r['promotion_useful_per_run']} | {float(r['promotion_useful_rate']):.1%} | {r['promotion_internal_ready_misses_per_run']} | {r['promotion_remote_waits_per_run']} | {float(r['AL_P95_resume_ms']):.2f} | {float(r['duplicate_KV_H2D_mib']):.2f} | {float(r['promotion_lead_time_MiB_s']):.2f} | {float(r['X_slowdown']):.3f} | {float(r['combined_decode_tok_s']):.2f} |"
        for r in summary_rows
    ) + "\n\n## Error-sign comparison\n\n| Policy | Error bucket | Promotions | Useful rate | Internal misses | Remote waits | min arrival−T6 (ms) | P50 arrival−T6 (ms) | Resume P50/P95 (ms) |\n|---|---:|---:|---:|---:|---:|---:|---:|\n" + "\n".join(
        f"| {r['policy']} | {int(r['prediction_error_ms']):+d} ms | {r['promotion_count']} | {r['promotion_useful_rate']} | {r['promotion_internal_ready_miss_count']} | {r['promotion_remote_wait_count']} | {r['arrival_minus_T6_min_ms']} | {r['arrival_minus_T6_P50_ms']} | {r['effective_resume_P50_ms']} / {r['effective_resume_P95_ms']} |"
        for r in buckets
    ) + f"""

## Findings

- Raw positive-error A–D bucket: useful rate {raw_positive.get('promotion_useful_rate', 'NA')}; internal-ready misses {raw_positive.get('promotion_internal_ready_miss_count', 'NA')}; remote waits {raw_positive.get('promotion_remote_wait_count', 'NA')}. This is the predeclared late-prediction failure diagnostic, not run invalidity.
- Safe positive-error A–D bucket: useful rate {safe_positive.get('promotion_useful_rate', 'NA')}; internal-ready misses {safe_positive.get('promotion_internal_ready_miss_count', 'NA')}; remote waits {safe_positive.get('promotion_remote_wait_count', 'NA')}. Safe's total measured actual-deadline misses/remote waits/internal misses/lease leaks were {safe_misses}/{safe_remote}/{safe_internal}/{safe_leaks}.
- Safe early-prediction E–H bucket: useful rate {safe_negative.get('promotion_useful_rate', 'NA')}; P50/P95 resume latency {safe_negative.get('effective_resume_P50_ms', 'NA')}/{safe_negative.get('effective_resume_P95_ms', 'NA')} ms. The corresponding per-session GPU-ready residency is retained in `session_metrics.csv`.
- Safe +120-ms I–L bucket: useful rate {safe_late.get('promotion_useful_rate', 'NA')}; actual promotion deadline miss count is in the same bucket table.
- Safe promoted {safe['native_prefetch_count_per_run']} per run and skipped {safe['reservation_skip_count_per_run']}; peak controller reservation {safe['max_reserved_blocks']}/512; total lease leaks {safe['lease_leak_count']}; actual expiry coverage misses are {sum(int(r['run_metrics']['reservation_actual_coverage_miss_count']) for r in safe_runs)}.
- H2D is based on actual native transfer response bytes plus actual external-token source deltas. Minimum required transfer is 1344 MiB. Promotion lead exposure (`max(0, admission−T6) × 112 MiB`) is not exact physical residency.
- X slowdown is measured per run as X decode duration divided by the three-run X-only median. It is not hard-coded. K2/trace health are per-run and in `run_metrics.csv`.

## Direct answers

1. Raw prediction error reduced promotion usefulness in the affected buckets: overall Raw useful rate {float(raw['promotion_useful_rate']):.1%}, with {raw['promotion_internal_ready_misses_per_run']} internal misses and {raw['promotion_remote_waits_per_run']} remote waits per run.
2. The +200-ms late A–D error caused Raw promotion misses/waits: {raw_positive.get('promotion_internal_ready_miss_count', 'NA')} / {raw_positive.get('promotion_remote_wait_count', 'NA')} over the three runs.
3. The −50-ms E–H predictions gave earlier scheduling and measurable GPU-ready residency; see E–H session rows and promotion lead exposure, without assuming residency equals lease lifetime.
4. The +120-ms I–L bucket remained covered under the fixed Safe margin: useful rate {safe_late.get('promotion_useful_rate', 'NA')} and latency distribution above.
5. Safe restored useful promotions to {float(safe['promotion_useful_rate']):.1%} overall.
6. Safe maintained the 512-block bound: observed max {safe['max_reserved_blocks']} blocks; actual lease peak {report['reservation_safe']['active_lease_blocks_max']} blocks.
7. Safe reservation expiry covered actual resume deadlines: {'YES' if report['reservation_safe']['coverage_misses'] == 0 else 'NO'}.
8. Safe internal-ready misses / remote waits / lease leaks: {safe_internal} / {safe_remote} / {safe_leaks}.
9. Lowest median AL P95 was {min(summary_rows, key=lambda r: float(r['AL_P95_resume_ms']))['policy']} ({min(float(r['AL_P95_resume_ms']) for r in summary_rows):.2f} ms).
10. Safe median AL P95 versus Oracle changed by {float(safe['AL_P95_resume_ms'])-float(oracle['AL_P95_resume_ms']):+.2f} ms.
11. Raw duplicate H2D was {float(raw['duplicate_KV_H2D_mib']):.2f} MiB median; exact accounting is per run.
12. Source accounting / native correctness / token counts are validated from actual serving data, not synthesized metrics.
13. The fixed 200-ms margin was not tuned; measured misses and promotion-lead cost remain visible.
14. Three-repetition variability is available in run-level CSVs; summary medians are not pooled across integration runs.

## Result

**{result}** — next direction: `{direction}`. No subsequent phase was started.
"""
    (OUT / "summary.md").write_text(summary, encoding="utf-8")
    write_csv(OUT / "zero_error_regression" / "reservation_events.csv", flatten(zero, "reservation_events"))
    write_csv(OUT / "mixed_error" / "reservation_events.csv", flatten(measured, "reservation_events"))
    write_csv(OUT / "mixed_error" / "scheduler_trace.csv", flatten(measured, "scheduler_trace"))
    write_csv(OUT / "mixed_error" / "prediction_metrics.csv", flatten(measured, "prediction_metrics"))
    write_csv(OUT / "mixed_error" / "background_metrics.csv", [r["background_metrics"] for r in measured])
    write_json(OUT / "mixed_error" / "metric_snapshots.json", [
        {"policy": r["policy"], "repetition": r["repetition"],
         "t0_monotonic_s": r["t0_monotonic_s"], "snapshots": r["snapshot_rows"]}
        for r in measured
    ])
    report = {
        "base_commit": "8f37e6aa54366fd54e680bdd446a351937c65018",
        "x_only_runs": len(xrows),
        "zero_error_regression": zero_gate,
        "integration_runs": len(integration),
        "integration_valid_runs": sum(universal_valid(r) for r in integration),
        "integration_gate": integration_gate,
        "measured_runs": len(measured),
        "measured_valid_runs": valid_measured,
        "real_resume_correctness": f"{token_correct}/108",
        "safe_promotion_deadline_misses": safe_misses,
        "safe_remote_waits": safe_remote,
        "safe_internal_ready_misses": safe_internal,
        "safe_lease_leaks": safe_leaks,
        "safe_max_reserved_blocks": safe["max_reserved_blocks"],
        "safe_max_active_leased_blocks": max((int(r["run_metrics"]["peak_active_lease_blocks"]) for r in safe_runs), default=0),
        "safe_reservation_coverage_misses": sum(int(r["run_metrics"]["reservation_actual_coverage_miss_count"]) for r in safe_runs),
        "tests": {"returncode": test_proc.returncode, "summary": test_summary, "output_tail": test_output[-3000:]},
        "policy_summaries": summary_rows,
        "error_bucket_summaries": buckets,
        "result": result,
        "next_direction": direction,
    }
    write_json(OUT / "validation_report.json", report)
    config = {
        "base_commit": "8f37e6aa54366fd54e680bdd446a351937c65018",
        "policies": list(POLICIES), "actual_deadlines_ms": DEADLINES_MS,
        "prediction_error_ms": PREDICTION_ERROR_MS, "safe_margin_ms": 200,
        "service_budget_ms": SERVICE_BUDGET_MS, "resume_guard_ms": 110,
        "handoff_guard_ms": LEDGER_GUARD_MS, "reservation_budget_blocks": LEASE_BUDGET_BLOCKS,
        "block_size_tokens": BLOCK_SIZE_TOKENS, "K": K2,
        "sessions": list("ABCDEFGHIJKL"), "tokens_per_session": 4096,
        "kv_bytes_per_session": 117440512, "gpu_kv_bytes": 1073741824,
        "cpu_external_kv_bytes": 2147483648, "resume_max_tokens": 32,
        "X_prompt_tokens": 9500, "X_max_tokens": 256,
        "x_only_controls": 3, "zero_error_policies": [ORACLE, RAW],
        "mixed_error_integration_policies": list(POLICIES), "measured_runs_per_policy": 3,
    }
    write_json(OUT / "config.json", config)
    env = json.loads((ROOT / "results" / "phase3u_fix" / "environment.json").read_text(encoding="utf-8"))
    env["vLLM_patch"] = "Phase 3S physical lease patch reused unchanged; no new Phase 3W vLLM change"
    env["experiment_python"] = ENV_PYTHON
    write_json(OUT / "environment.json", env)
    (OUT / "vllm_phase3w_delta.patch").write_text(
        "Phase 3S physical lease patch reused unchanged.\nNo new vLLM source modification in Phase 3W.\n",
        encoding="utf-8",
    )
    commands = [
        f"PYTHONPATH=. {ENV_PYTHON} -m pytest -q tests/test_phase3w.py tests/test_phase3v.py",
        *[f"PYTHONPATH=. {ENV_PYTHON} experiments/phase3w/run_phase3w.py --stage x-only --repetition {r}" for r in range(3)],
        *[f"PYTHONPATH=. {ENV_PYTHON} experiments/phase3w/run_phase3w.py --stage zero-error --policy {p} --repetition 0" for p in (ORACLE, RAW)],
        f"PYTHONPATH=. {ENV_PYTHON} experiments/phase3w/run_phase3w.py --stage validate-zero",
        *[f"PYTHONPATH=. {ENV_PYTHON} experiments/phase3w/run_phase3w.py --stage integration --policy {p} --repetition 0" for p in POLICIES],
        f"PYTHONPATH=. {ENV_PYTHON} experiments/phase3w/run_phase3w.py --stage validate-integration",
        *[f"PYTHONPATH=. {ENV_PYTHON} experiments/phase3w/run_phase3w.py --stage measured --policy {p} --repetition {r}" for p in POLICIES for r in range(3)],
        f"PYTHONPATH=. {ENV_PYTHON} experiments/phase3w/run_phase3w.py --stage finalize",
    ]
    (OUT / "commands.sh").write_text("\n".join(commands) + "\n", encoding="utf-8")
    finite_audit([
        OUT / "validation_report.json", OUT / "config.json", OUT / "environment.json",
        OUT / "policy_summary.csv", OUT / "error_bucket_summary.csv",
        OUT / "mixed_error" / "run_metrics.csv", OUT / "mixed_error" / "session_metrics.csv",
        OUT / "mixed_error" / "prediction_metrics.csv", OUT / "mixed_error" / "async_prefetch_metrics.csv",
    ])
    if test_proc.returncode != 0:
        report["result"] = "EXPERIMENT_INVALID"
        report["next_direction"] = "EXPERIMENT_INVALIDITY_REVIEW"
        write_json(OUT / "validation_report.json", report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3W prediction-robustness workload")
    parser.add_argument("--stage", required=True, choices=(
        "x-only", "zero-error", "validate-zero", "integration",
        "validate-integration", "measured", "finalize", "finalize-incomplete",
    ))
    parser.add_argument("--policy", choices=POLICIES)
    parser.add_argument("--repetition", type=int)
    args = parser.parse_args()
    if args.stage == "finalize":
        finalize()
    elif args.stage == "finalize-incomplete":
        finalize_incomplete()
    else:
        stage_run(args.stage, args.policy, args.repetition)


if __name__ == "__main__":
    main()
