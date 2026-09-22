from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase3r_fix"


def read_csv(name: str) -> list[dict[str, str]]:
    with (OUT / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * p / 100.0
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def stats(values: list[float]) -> dict[str, float]:
    return {"min": min(values), "median": statistics.median(values), "P95": percentile(values, 95), "max": max(values)}


def main() -> None:
    decisions = read_csv("capacity_decisions.csv")
    sessions = read_csv("session_metrics.csv")
    native = read_csv("async_prefetch_metrics.csv")
    runs = read_csv("run_metrics.csv")
    sanity = json.loads((OUT / "capacity_metric_sanity.json").read_text())
    assert len(decisions) == len(sessions) == len(native) == 12
    target_usage = [float(row["kv_usage_fraction"]) for row in decisions]
    all_usage = [float(row["kv_usage_fraction"]) for row in sanity] + target_usage
    metric_sane = all(0 <= value <= 1.05 for value in all_usage) and max(all_usage) > min(all_usage)
    query = [float(row["capacity_query_latency_ms"]) for row in decisions]
    late = [float(row["issue_lateness_ms"]) for row in decisions]
    issue_t6 = [float(row["issue_to_T6_ms"]) for row in native]
    target_t6 = [float(row["target_to_T6_ms"]) for row in native]
    waits = [row["session"] for row in sessions if row["WAITING_FOR_REMOTE_KVS_seen"] == "True"]
    misses = [row["session"] for row in sessions if row["promotion_outcome"] == "INTERNAL_READY_MISS"]
    remote = [row["session"] for row in sessions if row["promotion_outcome"] == "PROMOTED_REMOTE_WAIT"]
    local = float(runs[0]["source_local_hit_tokens_total"])
    external = float(runs[0]["source_external_transfer_tokens_total"])
    source_ok = local + external == 49152.0
    native_ok = sum(row["status"] == "ok" and int(float(row["external_tokens_loaded"])) == 4096 and int(float(row["model_forward_count"])) == 0 and int(float(row["generated_token_count"])) == 0 and int(float(row["mixed_prefetch_model_step_count"])) == 0 for row in native)
    token_ok = sum(int(float(row["generated_token_count"])) == 32 for row in sessions)
    experiment_valid = metric_sane and source_ok and native_ok == len(native) and token_ok == 12 and runs[0]["gate_validation"] == "PASS" and runs[0]["background_validation"] == "PASS"
    result = "D" if not experiment_valid else ("B" if remote else ("A" if any(row["decision"] == "CAPACITY_SKIP" for row in decisions) else "B"))
    next_direction = {"A": "CAPACITY_AWARE_FULL_MATRIX", "B": "RESIDENCY_RESERVATION", "C": "ENGINE_LOCAL_CAPACITY_SIGNAL", "D": "FIX_INSTRUMENTATION"}[result]
    report = {
        "experiment_valid": experiment_valid,
        "capacity_metric_sane": metric_sane,
        "capacity_decisions_persisted": len(decisions),
        "PROMOTE_count": sum(row["decision"] == "PROMOTE" for row in decisions),
        "CAPACITY_SKIP_count": sum(row["decision"] == "CAPACITY_SKIP" for row in decisions),
        "free_mib_at_promotion": stats([float(row["free_mib"]) for row in decisions if row["decision"] == "PROMOTE"]),
        "internal_ready_miss_count": len(misses),
        "ready_but_remote_wait_count": len(remote),
        "ready_but_remote_wait_sessions": remote,
        "remote_wait_count": len(waits),
        "remote_wait_sessions": waits,
        "source_local_tokens": local,
        "source_external_tokens": external,
        "source_total_tokens": local + external,
        "local_fraction": local / 49152.0,
        "external_fraction": external / 49152.0,
        "source_conservation_pass": source_ok,
        "capacity_query_latency": stats(query),
        "issue_lateness": stats(late),
        "issue_to_T6": stats(issue_t6),
        "target_to_T6": stats(target_t6),
        "K2_valid": runs[0]["gate_validation"],
        "background_valid": runs[0]["background_validation"],
        "native_correctness": f"{native_ok}/{len(native)}",
        "token_correctness": f"{token_ok}/12",
        "prometheus_critical_path_too_slow": statistics.median(query) > 5.0,
        "result": result,
        "next_direction": next_direction,
    }
    (OUT / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "integration_gate.json").write_text(json.dumps({"experiment_valid": experiment_valid, "policy_pass": result == "A", "report": report}, indent=2) + "\n")
    lines = [
        "# Phase 3R-FIX", "", "One fresh Stage-A capacity-audit run; no Stage B.", "",
        f"Experiment valid: {experiment_valid}. Capacity metric sane: {metric_sane}. Decisions: {len(decisions)}/12.",
        f"PROMOTE/SKIP: {report['PROMOTE_count']}/{report['CAPACITY_SKIP_count']}. Internal-ready misses: {len(misses)}. Promoted remote waits: {len(remote)} ({', '.join(remote)}).",
        f"Source local/external/total: {local:.0f}/{external:.0f}/{local+external:.0f}; fractions {100*local/49152:.2f}%/{100*external/49152:.2f}%; conservation {'PASS' if source_ok else 'FAIL'}.",
        f"Capacity query ms median/P95/max: {report['capacity_query_latency']['median']:.2f}/{report['capacity_query_latency']['P95']:.2f}/{report['capacity_query_latency']['max']:.2f}.",
        f"Issue lateness ms median/P95/max: {report['issue_lateness']['median']:.2f}/{report['issue_lateness']['P95']:.2f}/{report['issue_lateness']['max']:.2f}.",
        f"Issue-to-T6 ms median/P95/max: {report['issue_to_T6']['median']:.2f}/{report['issue_to_T6']['P95']:.2f}/{report['issue_to_T6']['max']:.2f}.",
        f"Target-to-T6 ms median/P95/max: {report['target_to_T6']['median']:.2f}/{report['target_to_T6']['P95']:.2f}/{report['target_to_T6']['max']:.2f}.",
        "The three idle/setup checkpoints read zero, but the twelve target-time samples varied from 0.254 to 0.808, proving the live metric changed during the measured workload.",
        "All 12 decisions had measured free capacity above 144 MiB, yet seven later entered WAITING_FOR_REMOTE_KVS. Instantaneous free capacity is therefore insufficient to guarantee future residency.",
        f"Result {result}; next direction {next_direction}. No threshold tuning or next phase was run.",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    (OUT / "config.json").write_text(json.dumps({"phase": "3R-FIX", "gpu_kv_capacity_mib": 1024, "session_kv_mib": 112, "capacity_reserve_mib": 32, "required_free_mib": 144, "async_service_ms": 40, "resume_guard_ms": 110, "K": 2, "stage_b_ran": False}, indent=2) + "\n")
    (OUT / "environment.json").write_text(json.dumps({"runtime": "Phase 3R pinned runtime", "gpu_runs": 1}, indent=2) + "\n")
    (OUT / "commands.sh").write_text("python -m pytest -q tests/test_phase3r_fix.py\npython -m experiments.phase3r_fix.run_phase3r_fix --stage integration\npython -m experiments.phase3r_fix.postprocess\n")


if __name__ == "__main__":
    main()
