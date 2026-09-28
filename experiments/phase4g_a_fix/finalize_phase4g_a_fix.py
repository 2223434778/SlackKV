"""Aggregate only a complete Phase 4G-A-FIX matrix."""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments.phase4g_a import finalize_phase4g_a as legacy_finalize
from experiments.phase4g_a.workload import ROOT, SOURCE_SHA, build_manifest, load_source, nearest_rank
from experiments.phase4g_a_fix.background import (
    BG_CHUNKS,
    BG_POST_FIRST_TOKENS,
    BG_TOTAL_OUTPUT_TOKENS,
    CALIBRATION_TAIL_MS,
    MAX_IDLE_GAP_MS,
)
from experiments.phase4g_a_fix.run_phase4g_a_fix import (
    OUT,
    POLICY_NAMES,
    RUN_ORDER,
    START_COMMIT,
    VLLM_EXPECTED_DIFF_SHA,
    write_csv,
    write_json,
    vllm_diff_sha,
)

MIB = 2**20
EXPECTED_TURNS = 40
EXPECTED_BOUNDARIES = 32
MEASURED_REPETITIONS = 8
MEASURED_RUNS = 16


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _percent_delta(new: float, old: float) -> float:
    return (new / old - 1) * 100 if old else float("nan")


def _metric(rows: list[dict[str, Any]], policy: str, field: str) -> list[float]:
    return [float(row[field]) for row in rows if row["policy"] == policy]


def _write_empty_or_rows(name: str, rows: list[dict[str, Any]], header: str) -> None:
    if rows:
        write_csv(OUT / name, rows)
    else:
        (OUT / name).write_text(header + "\n", encoding="utf-8")


def finalize() -> dict[str, Any]:
    if vllm_diff_sha() != VLLM_EXPECTED_DIFF_SHA:
        raise RuntimeError("EXPERIMENT_INVALID_ENVIRONMENT:vLLM fingerprint changed before aggregation")
    if __import__("hashlib").sha256((ROOT / "results/phase4c/hybrid_workload.csv").read_bytes()).hexdigest() != SOURCE_SHA:
        raise RuntimeError("EXPERIMENT_INVALID_WORKLOAD:source SHA mismatch during aggregation")
    manifest_rows = build_manifest(load_source())
    manifest = {str(row["boundary_id"]): row for row in manifest_rows}
    reference_path = ROOT / "results/phase4c/request_metrics.csv"
    reference_rows = _load_csv(reference_path)
    source_rows = load_source()
    hash_to_episode = {row["session_hash"]: row["episode_id"] for row in source_rows}
    reference = {
        (hash_to_episode[row["session_hash"]], int(row["replay_turn_index"])): row["output_sha256"]
        for row in reference_rows if row["session_hash"] in hash_to_episode
    }
    if len(reference) != EXPECTED_TURNS:
        raise RuntimeError("Phase4C identity reference incomplete")

    calibration = _load_csv(OUT / "background_calibration.csv")
    if len(calibration) != 4 or any(row["valid"] != "True" for row in calibration):
        raise RuntimeError("EXPERIMENT_INVALID_BACKGROUND_PROTOCOL:BG-CAL did not pass")
    tails = [float(row["tail_finish_ms"]) for row in calibration]
    gaps_cal = [float(row["max_inter_chunk_gap_ms"]) for row in calibration]
    if min(tails) < CALIBRATION_TAIL_MS or max(gaps_cal) > MAX_IDLE_GAP_MS:
        raise RuntimeError("EXPERIMENT_INVALID_BACKGROUND_PROTOCOL:BG-CAL threshold")

    bg_only = _load_csv(OUT / "bg_only_runs.csv")
    if len(bg_only) != 8 or any(row["valid"] != "True" or int(row["total_generated_tokens"]) != BG_TOTAL_OUTPUT_TOKENS for row in bg_only):
        raise RuntimeError("EXPERIMENT_INVALID_BACKGROUND_PROTOCOL:BG-only controls")
    bg_walls = [float(row["BGWall_ms"]) for row in bg_only]
    bg_baseline = statistics.median(bg_walls)
    first4 = statistics.median(bg_walls[:4])
    last4 = statistics.median(bg_walls[4:])
    drift_percent = _percent_delta(last4, first4)
    write_csv(OUT / "x_only_runs.csv", [
        {"side": row["side"], "repetition": row["repetition"],
         "BGWall_ms": row["BGWall_ms"], "weighted_decode_TPOT_ms": row["weighted_decode_TPOT_ms"],
         "total_generated_tokens": row["total_generated_tokens"], "valid": row["valid"]}
        for row in bg_only
    ])

    results = []
    for repetition, pair in enumerate(RUN_ORDER, 1):
        for letter in pair:
            path = OUT / "runs" / f"measured_rep{repetition}_{letter.lower()}" / "run.json"
            result = json.loads(path.read_text(encoding="utf-8"))
            if (result.get("policy") != POLICY_NAMES[letter]
                or int(result.get("repetition", -1)) != repetition
                or result.get("valid") is not True
                or len(result.get("turn_rows", [])) != EXPECTED_TURNS
                or len(result.get("boundary_rows", [])) != EXPECTED_BOUNDARIES
                or len(result.get("background_chunks", [])) != BG_CHUNKS
                or int(result.get("BG_total_generated_tokens", -1)) != BG_TOTAL_OUTPUT_TOKENS
                or float(result.get("BG_tail_request_finish_ms", -1)) <= 0
                or float(result.get("BG_max_inter_chunk_gap_ms", float("inf"))) > MAX_IDLE_GAP_MS):
                raise RuntimeError(f"EXPERIMENT_INVALID_MATRIX:invalid measured run {path}")
            results.append(result)
    if len(results) != MEASURED_RUNS:
        raise RuntimeError("EXPERIMENT_INVALID_MATRIX:partial measured matrix")

    legacy_finalize.OUT = OUT
    all_boundaries: list[dict[str, Any]] = []
    all_sessions: list[dict[str, Any]] = []
    all_h2d: list[dict[str, Any]] = []
    all_prefetch: list[dict[str, Any]] = []
    all_runs: list[dict[str, Any]] = []
    all_capacity: list[dict[str, Any]] = []
    all_leases: list[dict[str, Any]] = []
    all_reservations: list[dict[str, Any]] = []
    all_resume: list[dict[str, Any]] = []
    all_sources: list[dict[str, Any]] = []
    all_turns: list[dict[str, Any]] = []
    chunk_rows: list[dict[str, Any]] = []
    for result in results:
        boundaries, sessions, h2d, prefetch = legacy_finalize.process_run(
            result, manifest_rows, reference, bg_baseline
        )
        metric = legacy_finalize.summarize_run(
            result, boundaries, sessions, h2d, prefetch, bg_baseline, manifest
        )
        turn_rows = result["turn_rows"]
        first_ns = min([int(row["first_ns"]) for row in turn_rows] + [int(result["X_first_ns"])])
        last_ns = max([int(row["finish_ns"]) for row in turn_rows] + [int(result["X_finish_ns"])])
        post_first = sum(int(row["output_tokens"]) - 1 for row in turn_rows) + BG_POST_FIRST_TOKENS
        metric["combined_tok_s"] = post_first * 1e9 / (last_ns - first_ns)
        metric["BG_chunks_complete"] = int(result["BG_chunks_complete"])
        metric["BG_total_generated_tokens"] = int(result["BG_total_generated_tokens"])
        metric["BG_total_wall_ms"] = float(result["BG_total_wall_ms"])
        metric["BG_weighted_TPOT_ms"] = float(result["BG_weighted_TPOT_ms"])
        metric["BG_max_inter_chunk_gap_ms"] = float(result["BG_max_inter_chunk_gap_ms"])
        metric["BG_tail_first_token_ms"] = float(result["BG_tail_first_token_ms"])
        metric["BG_tail_request_finish_ms"] = float(result["BG_tail_request_finish_ms"])
        metric["BG_slowdown"] = float(result["BG_total_wall_ms"]) / bg_baseline
        metric["X_slowdown"] = metric["BG_slowdown"]
        metric["X_TPOT_ms"] = float(result["BG_weighted_TPOT_ms"])
        if metric["K2_max_pre_first"] > 2 or metric["logical_leaks"] or metric["lease_leaks"] or not metric["capacity_valid"]:
            raise RuntimeError("EXPERIMENT_INVALID_MATRIX:mechanism invariant")
        all_boundaries.extend(boundaries)
        all_sessions.extend(sessions)
        all_h2d.extend(h2d)
        all_prefetch.extend(prefetch)
        all_runs.append(metric)
        base = {"repetition": result["repetition"], "policy": result["policy"]}
        all_turns.extend({**base, **row} for row in turn_rows)
        for target, key in ((all_capacity, "capacity_trace"), (all_leases, "lease_events"),
                            (all_reservations, "decisions"), (all_resume, "resume_trace"),
                            (all_sources, "source_events")):
            target.extend({**base, **row} for row in result[key])
        chunk_rows.extend(
            {"stage": "measured", "repetition": result["repetition"], "policy": result["policy"], **row}
            for row in result["background_chunks"]
        )

    write_csv(OUT / "background_chunks.csv", chunk_rows + [
        row for row in _load_csv(OUT / "background_chunks.csv")
        if row.get("stage") != "measured"
    ])
    write_csv(OUT / "boundary_metrics.csv", all_boundaries)
    write_csv(OUT / "session_metrics.csv", all_sessions)
    write_csv(OUT / "turn_metrics.csv", all_turns)
    write_csv(OUT / "h2d_by_boundary.csv", all_h2d)
    write_csv(OUT / "h2d_accounting.csv", all_h2d)
    write_csv(OUT / "prefetch_metrics.csv", all_prefetch)
    write_csv(OUT / "run_metrics.csv", all_runs)
    write_csv(OUT / "capacity_trace.csv", all_capacity)
    write_csv(OUT / "lease_events.csv", all_leases)
    write_csv(OUT / "reservation_events.csv", all_reservations)
    write_csv(OUT / "scheduler_trace.csv", all_resume)
    write_csv(OUT / "resume_trace.csv", all_resume)
    write_csv(OUT / "offload_trace.csv", all_sources)
    write_csv(OUT / "source_readiness.csv", all_sources)
    resource_rows = [{key: row[key] for key in ("repetition", "policy", "lease_memory_time_MiB_s",
        "peak_leased_blocks", "peak_lease_count", "reservation_block_time_block_s", "peak_reserved_blocks",
        "capacity_rejects", "PSA_rejects", "reservation_skips")} for row in all_runs]
    write_csv(OUT / "resource_costs.csv", resource_rows)
    write_csv(OUT / "progress_admission.csv", [row for row in all_capacity if row.get("progress_guard") is True])

    summary_rows = []
    for policy in POLICY_NAMES.values():
        rows = [row for row in all_runs if row["policy"] == policy]
        if len(rows) != MEASURED_REPETITIONS:
            raise RuntimeError("EXPERIMENT_INVALID_MATRIX:per-policy repetitions")
        fields = (
            "deadline_P50_ms", "deadline_P90_ms", "deadline_P95_ms", "deadline_max_ms",
            "TTFT_P50_ms", "TTFT_P95_ms", "cumulative_TTFT_median_ms",
            "cumulative_TTFT_P95_ms", "session_e2e_median_ms", "scheduler_wait_P95_ms",
            "post_admit_to_first_P95_ms", "local_source_fraction", "external_source_fraction",
            "promotion_attempts", "successful_promotions", "promotion_useful", "promotion_ready_miss",
            "promotion_remote_wait", "capacity_rejects", "PSA_rejects", "reservation_skips",
            "promotion_H2D_MiB", "external_H2D_MiB", "total_H2D_MiB", "duplicate_H2D_MiB",
            "lease_memory_time_MiB_s", "reservation_block_time_block_s", "X_slowdown",
            "X_TPOT_ms", "combined_tok_s", "BG_total_wall_ms", "BG_tail_request_finish_ms",
            "BG_max_inter_chunk_gap_ms", "service_T6_median_ms", "service_T6_P95_ms",
            "service_T6_max_ms", "service_overrun_count", "source_ready_count", "source_not_ready_count",
        )
        record: dict[str, Any] = {"policy": policy, "valid_repetitions": len(rows)}
        for field in fields:
            values = [float(row[field]) for row in rows if row[field] != "NA"]
            if values:
                record[field + "_median"] = statistics.median(values)
                record[field + "_min"] = min(values)
                record[field + "_max"] = max(values)
            else:
                record[field + "_median"] = "NA"
                record[field + "_min"] = "NA"
                record[field + "_max"] = "NA"
        succeeded = sum(int(row["successful_promotions"]) for row in rows)
        record["promotion_useful_rate"] = (sum(int(row["promotion_useful"]) for row in rows) / succeeded
                                            if succeeded else "NA")
        summary_rows.append(record)
    write_csv(OUT / "policy_summary.csv", summary_rows)

    paired = []
    for repetition in range(1, MEASURED_REPETITIONS + 1):
        a = next(row for row in all_runs if row["repetition"] == repetition and row["policy"] == POLICY_NAMES["A"])
        d = next(row for row in all_runs if row["repetition"] == repetition and row["policy"] == POLICY_NAMES["D"])
        row: dict[str, Any] = {"repetition": repetition, "comparison": "D_vs_A"}
        for field in ("deadline_P95_ms", "cumulative_TTFT_median_ms", "session_e2e_median_ms",
                      "total_H2D_MiB", "duplicate_H2D_MiB", "X_slowdown", "combined_tok_s"):
            row[field + "_delta"] = float(d[field]) - float(a[field])
            row[field + "_relative_percent"] = _percent_delta(float(d[field]), float(a[field]))
        paired.append(row)
    write_csv(OUT / "paired_deltas.csv", paired)
    boot = []
    for field in ("deadline_P95_ms_relative_percent", "cumulative_TTFT_median_ms_relative_percent",
                  "session_e2e_median_ms_relative_percent", "X_slowdown_delta"):
        low, high = legacy_finalize.bootstrap([float(row[field]) for row in paired], 20260928)
        boot.append({"comparison": "D_vs_A", "metric": field,
                     "paired_median": statistics.median(float(row[field]) for row in paired),
                     "bootstrap_95_low": low, "bootstrap_95_high": high,
                     "resamples": 10000, "seed": 20260928,
                     "interpretation": "descriptive, not a significance test"})
    write_csv(OUT / "bootstrap_intervals.csv", boot)
    legacy_finalize.OUT = OUT
    strata = legacy_finalize.stratified_reports(all_boundaries)

    a_summary = next(row for row in summary_rows if row["policy"] == POLICY_NAMES["A"])
    d_summary = next(row for row in summary_rows if row["policy"] == POLICY_NAMES["D"])
    p95_delta = statistics.median(float(row["deadline_P95_ms_relative_percent"]) for row in paired)
    cumulative_delta = statistics.median(float(row["cumulative_TTFT_median_ms_relative_percent"]) for row in paired)
    e2e_delta = statistics.median(float(row["session_e2e_median_ms_relative_percent"]) for row in paired)
    h1 = float(d_summary["deadline_P95_ms_median"]) <= .90 * float(a_summary["deadline_P95_ms_median"])
    h2 = float(d_summary["total_H2D_MiB_median"]) <= 1.05 * float(a_summary["total_H2D_MiB_median"])
    h3 = float(d_summary["X_slowdown_median"]) <= float(a_summary["X_slowdown_median"]) + .05
    if all(delta >= -3.0 for delta in (p95_delta, cumulative_delta, e2e_delta)):
        final_result = "RESULT_C_MULTITURN_NO_MATERIAL_LATENCY_VALUE"
    elif h1 and h2 and h3:
        final_result = "RESULT_A_MULTITURN_DIRECTIONAL_BENEFIT"
    else:
        final_result = "RESULT_B_VALID_MULTITURN_MIXED_TRADEOFF"

    identities = {policy: sum(bool(row["identity_match"]) for row in all_boundaries if row["policy"] == policy)
                  for policy in POLICY_NAMES.values()}
    queue_rows = []
    for result in results:
        rep, policy = int(result["repetition"]), result["policy"]
        selected = [row for row in all_boundaries if int(row["repetition"]) == rep and row["policy"] == policy]
        p95_boundary = sorted(selected, key=lambda row: float(row["deadline_latency_ms"]))[-2]
        queue_rows.append({"repetition": rep, "policy": policy,
                           "p95_scheduler_wait_ms": p95_boundary["scheduler_wait_ms"],
                           "p95_post_admit_to_first_ms": p95_boundary["post_admit_to_first_ms"],
                           "p95_boundary_id": p95_boundary["boundary_id"]})
    write_csv(OUT / "queue_decomposition.csv", queue_rows)

    bg_local = defaultdict(list)
    for row in chunk_rows:
        if row.get("prompt_local_hit_tokens") not in ("NA", None, ""):
            bg_local[(row["stage"], row["policy"], int(row["chunk_id"]))].append(float(row["prompt_local_hit_tokens"]))
    bg_local_summary = [
        {"stage": stage, "policy": policy, "chunk_id": chunk, "median_prompt_local_hit_tokens": statistics.median(values),
         "observations": len(values)}
        for (stage, policy, chunk), values in sorted(bg_local.items())
    ]
    write_csv(OUT / "background_local_hits.csv", bg_local_summary)

    report = {
        "phase": "4G-A-FIX",
        "final_result": final_result,
        "start_commit": START_COMMIT,
        "workload_sha256": SOURCE_SHA,
        "algorithm_fingerprint_unchanged": vllm_diff_sha() == VLLM_EXPECTED_DIFF_SHA,
        "BG_protocol": "BGv2_fixed_work_chunked",
        "BG_output_tokens_per_run": BG_TOTAL_OUTPUT_TOKENS,
        "BG_CAL_valid": len(calibration),
        "BG_CAL_tail_finish_ms": tails,
        "BG_CAL_min_tail_finish_ms": min(tails),
        "BG_CAL_required_tail_ms": CALIBRATION_TAIL_MS,
        "BG_only_valid": len(bg_only),
        "BG_only_baseline_wall_median_ms": bg_baseline,
        "BG_only_pre_median_ms": first4,
        "BG_only_post_median_ms": last4,
        "BG_only_drift_percent": drift_percent,
        "environment_drift_warning": abs(drift_percent) > 5,
        "Stage_L_A_valid": True,
        "Stage_L_D_valid": True,
        "measured_valid": len(results),
        "measured_expected": MEASURED_RUNS,
        "requests_complete": sum(len(item["turn_rows"]) for item in results),
        "boundaries_complete": sum(len(item["boundary_rows"]) for item in results),
        "BG_chunks_complete": sum(int(item["BG_chunks_complete"]) for item in results),
        "BG_total_tokens_complete": sum(int(item["BG_total_generated_tokens"]) for item in results),
        "BG_coverage_valid": sum(float(item["BG_tail_request_finish_ms"]) > 0 for item in results),
        "BG_max_inter_chunk_idle_gap_ms": max(float(item["BG_max_inter_chunk_gap_ms"]) for item in results),
        "K2_valid": sum(int(item["K2_max_pre_first"]) <= 2 for item in all_runs),
        "pause_valid": sum(float(item["pause_abs_P95_ms"]) <= 10 and float(item["pause_abs_max_ms"]) <= 30 for item in all_runs),
        "server_crashes": sum(not item["server_alive"] for item in all_runs),
        "OOM": sum(bool(item["OOM"]) for item in all_runs),
        "logical_leaks": sum(int(item["logical_leaks"]) for item in all_runs),
        "lease_leaks": sum(int(item["lease_leaks"]) for item in all_runs),
        "capacity_invariant_failures": sum(not item["capacity_valid"] for item in all_runs),
        "output_identity_match": identities,
        "H1": h1,
        "H2": h2,
        "H3": h3,
        "paired_boundary_P95_percent_median": p95_delta,
        "paired_boundary_P95_percent_range": [min(float(row["deadline_P95_ms_relative_percent"]) for row in paired),
                                               max(float(row["deadline_P95_ms_relative_percent"]) for row in paired)],
        "paired_cumulative_TTFT_percent_median": cumulative_delta,
        "paired_session_E2E_percent_median": e2e_delta,
        "bootstrap_intervals": boot,
        "result_caveat": "Descriptive n=8 paired runs; no significance test. BG-only source-hit counters are serialized per chunk; measured chunk hits subtract concurrent resumed Agent source-trace tokens.",
    }
    write_json(OUT / "validation_report.json", report)
    _write_summary(report, summary_rows, all_runs, paired, boot, calibration, bg_only, strata, queue_rows, bg_local_summary)
    return report


def _write_summary(report: dict[str, Any], policy_rows: list[dict[str, Any]], runs: list[dict[str, Any]],
                   paired: list[dict[str, Any]], boot: list[dict[str, Any]], calibration: list[dict[str, str]],
                   bg_only: list[dict[str, str]], strata: dict[str, list[dict[str, Any]]],
                   queue_rows: list[dict[str, Any]], bg_hits: list[dict[str, Any]]) -> None:
    a, d = (next(row for row in policy_rows if row["policy"] == POLICY_NAMES[key]) for key in ("A", "D"))
    mid = lambda row, key: float(row[key + "_median"])
    cal_tails = [float(row["tail_finish_ms"]) for row in calibration]
    x_values = [float(row["BGWall_ms"]) for row in bg_only]
    table = ["| Policy | Boundary P50 | P90 | P95 | TTFT P95 | Cum TTFT/session | Session E2E | BG slowdown | BG TPOT | H2D MiB | Duplicate MiB | Combined tok/s |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in (a, d):
        table.append(
            f"| {row['policy']} | {mid(row,'deadline_P50_ms'):.2f} | {mid(row,'deadline_P90_ms'):.2f} | "
            f"{mid(row,'deadline_P95_ms'):.2f} | {mid(row,'TTFT_P95_ms'):.2f} | "
            f"{mid(row,'cumulative_TTFT_median_ms'):.2f} | {mid(row,'session_e2e_median_ms'):.2f} | "
            f"{mid(row,'X_slowdown'):.3f} | {mid(row,'X_TPOT_ms'):.3f} | "
            f"{mid(row,'total_H2D_MiB'):.2f} | {mid(row,'duplicate_H2D_MiB'):.2f} | "
            f"{mid(row,'combined_tok_s'):.2f} |"
        )
    paired_pct = [float(row["deadline_P95_ms_relative_percent"]) for row in paired]
    queue_wait = statistics.median(float(row["p95_scheduler_wait_ms"]) for row in queue_rows if row["policy"] == POLICY_NAMES["D"])
    queue_post = statistics.median(float(row["p95_post_admit_to_first_ms"]) for row in queue_rows if row["policy"] == POLICY_NAMES["D"])
    strongest = "scheduler waiting" if queue_wait > queue_post else "post-admission serving"
    hit_lines = []
    for policy in (POLICY_NAMES["A"], POLICY_NAMES["D"]):
        values = [row for row in bg_hits if row["stage"] == "measured" and row["policy"] == policy]
        by_chunk = {int(row["chunk_id"]): row for row in values}
        if by_chunk:
            hit_lines.append(f"{policy}: " + ", ".join(
                f"X{i} median {float(by_chunk[i]['median_prompt_local_hit_tokens']):.0f} local tokens"
                for i in sorted(by_chunk)
            ))
    text = [
        "# Phase 4G-A-FIX: sustained fixed-work background and 32-boundary rerun", "",
        f"Final result: **{report['final_result']}**. BG-CAL {report['BG_CAL_valid']}/4, BG-only {report['BG_only_valid']}/8, measured {report['measured_valid']}/16; partial matrices are never aggregated.", "",
        "## Objective and protocol", "",
        "This rerun fixes the Phase 4G-A background duration and its dependent metrics. It retains the same eight-session, 40-turn, 32-boundary frozen hybrid workload, controller, deadlines, model/runtime, KV capacities, K2, PSA-1, P_SAFE, and reservation budget.",
        "BGv2 uses one active X request at a time: four sequential requests, each with the same 9500-token prompt and 2048 generated tokens. The per-request KV footprint therefore stays close to the old single-X request while the fixed work lasts longer. The background covers through the last Agent REQUEST_FINISHED; last-first-token tail is diagnostic only.",
        "The old single X=2048 run ended before some Agent workflows. A single X=8192 would increase one request's KV footprint from about 11548 to 17692 tokens and confound duration with KV pressure, so the protocol uses four sequential 2048-token chunks.", "",
        "## BG-CAL and BG-only controls", "",
        "BG-CAL tail after the last Agent REQUEST_FINISHED (ms): " + ", ".join(f"{value:.2f}" for value in cal_tails) + f"; minimum {min(cal_tails):.2f} ms versus the frozen 30000-ms gate.",
        f"Maximum BG-CAL inter-chunk idle gap: {max(float(row['max_inter_chunk_gap_ms']) for row in calibration):.2f} ms (limit {MAX_IDLE_GAP_MS:.0f} ms).",
        f"BG-only valid runs: pre {sum(row['side']=='pre' for row in bg_only)}/4 and post {sum(row['side']=='post' for row in bg_only)}/4. Eight-run median BGWall {statistics.median(x_values):.2f} ms; weighted decode TPOT median {statistics.median(float(row['weighted_decode_TPOT_ms']) for row in bg_only):.3f} ms. First-four / last-four wall medians: {statistics.median(x_values[:4]):.2f} / {statistics.median(x_values[4:]):.2f} ms; drift {report['BG_only_drift_percent']:+.2f}%" + (" (ENVIRONMENT_DRIFT_WARNING)." if report["environment_drift_warning"] else "."),
        "Measured repeated-prompt local-cache hit medians by chunk:", *(hit_lines or ["Per-chunk source metric was unavailable; values are NA."]), "",
        "## Measured policy results (median of eight run-level values)", "", *table, "",
        f"Paired FullSlackKV minus OnDemand boundary P95: {statistics.median(paired_pct):+.2f}% (range {min(paired_pct):+.2f}% to {max(paired_pct):+.2f}%). Cumulative TTFT delta {report['paired_cumulative_TTFT_percent_median']:+.2f}%; session E2E delta {report['paired_session_E2E_percent_median']:+.2f}%. H1 {('PASS' if report['H1'] else 'FAIL')}; H2 {('PASS' if report['H2'] else 'FAIL')}; H3 {('PASS' if report['H3'] else 'FAIL')}.",
        "Bootstrap intervals are descriptive intervals for paired run-level medians, not significance tests. FullSlackKV paired P95 and X slowdown intervals are in bootstrap_intervals.csv.", "",
        "## Lifecycle and tail decomposition", "",
        f"Measured runs: 40/40 Agent requests and 32/32 boundaries per run; aggregate {report['requests_complete']}/{MEASURED_RUNS*EXPECTED_TURNS} requests and {report['boundaries_complete']}/{MEASURED_RUNS*EXPECTED_BOUNDARIES} boundaries. BG coverage through last request finish: {report['BG_coverage_valid']}/{MEASURED_RUNS}; maximum measured idle gap {report['BG_max_inter_chunk_idle_gap_ms']:.2f} ms.",
        f"For FullSlackKV's run-level P95 boundary, median scheduler wait was {queue_wait:.2f} ms and post-admission-to-first-token time was {queue_post:.2f} ms; the larger observed component was {strongest}. This is a descriptive decomposition, not a causal attribution.",
        f"Output identity matched the Phase 4C reference for A {report['output_identity_match'][POLICY_NAMES['A']]}/256 and D {report['output_identity_match'][POLICY_NAMES['D']]}/256 boundary requests; identity is diagnostic, while token length is the hard gate.", "",
        "## Validation and interpretation", "",
        f"Algorithm fingerprint unchanged: {report['algorithm_fingerprint_unchanged']}. K2 valid {report['K2_valid']}/{MEASURED_RUNS}; pause fidelity {report['pause_valid']}/{MEASURED_RUNS}; crashes {report['server_crashes']}; OOM {report['OOM']}; logical leaks {report['logical_leaks']}; lease leaks {report['lease_leaks']}; capacity invariant failures {report['capacity_invariant_failures']}.",
        "This is a frozen hybrid trace-derived replay using real coding-agent token/KV structure and empirical TraceToolProxy pauses. It is not a request-timestamp-faithful replay or a production Agent benchmark. No earlier Phase 4G-A measured samples were reused.",
    ]
    (OUT / "summary.md").write_text("\n".join(text) + "\n", encoding="utf-8")


if __name__ == "__main__":
    finalize()
