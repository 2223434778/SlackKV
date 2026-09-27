from __future__ import annotations

import csv
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.phase4d_a.experiment import write_csv, write_json
from experiments.phase4e_a.metrics import POLICIES, classify, hypothesis_flags, median_range

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/phase4e_a"


def read_csv(name: str) -> list[dict[str, str]]:
    path = OUT / name
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def numeric(rows: list[dict[str, str]], field: str) -> list[float]:
    return [float(row[field]) for row in rows if row.get(field) not in (None, "", "NA")]


def pct(new: float, old: float) -> float:
    return 100 * (new / old - 1) if old else float("nan")


def paired_deltas(runs: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_rep = {(int(row["repetition"]), row["policy"]): row for row in runs}
    comparisons = {
        "D_vs_A": (POLICIES[3].name, POLICIES[0].name),
        "C_vs_B": (POLICIES[2].name, POLICIES[1].name),
        "D_vs_C": (POLICIES[3].name, POLICIES[2].name),
    }
    rows = []
    for repetition in range(1, 5):
        for label, (left_name, right_name) in comparisons.items():
            left = by_rep.get((repetition, left_name))
            right = by_rep.get((repetition, right_name))
            if not left or not right:
                continue
            left_p95 = float(left["deadline_latency_P95"])
            right_p95 = float(right["deadline_latency_P95"])
            rows.append({"repetition": repetition, "comparison": label,
                "P95_absolute_delta_ms": left_p95 - right_p95,
                "P95_relative_delta": (left_p95 / right_p95 - 1) if right_p95 else "NA",
                "total_H2D_delta_MiB": float(left["total_H2D_MiB"]) - float(right["total_H2D_MiB"]),
                "duplicate_H2D_delta_MiB": float(left["duplicate_H2D_MiB"]) - float(right["duplicate_H2D_MiB"]),
                "X_slowdown_delta": float(left["X_slowdown"]) - float(right["X_slowdown"]),
                "combined_tok_s_delta": float(left["combined_tok_s"]) - float(right["combined_tok_s"])})
    return rows


def finalize() -> dict[str, Any]:
    x = read_csv("x_only_runs.csv")
    smoke = read_csv("semantic_smoke.csv")
    runs = read_csv("run_metrics.csv")
    sessions = read_csv("session_metrics.csv")
    native = read_csv("prefetch_metrics.csv")
    stage_a_valid = sum(row.get("valid") == "True" for row in smoke)
    stage_b_valid = sum(row.get("valid") == "True" for row in runs)
    if len(x) != 4 or stage_a_valid != 4 or stage_b_valid != 16:
        raise RuntimeError(f"cannot aggregate invalid/incomplete matrix: X={len(x)}, StageA={stage_a_valid}, StageB={stage_b_valid}")
    measured_sessions = [row for row in sessions if row["stage"] == "measured"]
    measured_native = [row for row in native if row["stage"] == "measured"]
    if len(measured_sessions) != 128:
        raise RuntimeError(f"expected 128 measured resumes, got {len(measured_sessions)}")
    policy_rows = []
    extended = {}
    for policy in POLICIES:
        subset = [row for row in runs if row["policy"] == policy.name]
        if len(subset) != 4:
            raise RuntimeError(f"policy {policy.name} missing measured repetition")
        stats = {field: median_range(numeric(subset, field)) for field in (
            "deadline_latency_P50", "deadline_latency_P95", "deadline_latency_max", "TTFT_P50", "TTFT_P95",
            "local_source_fraction", "total_H2D_MiB", "duplicate_H2D_MiB", "promotion_attempts",
            "capacity_rejects", "reservation_skips", "X_slowdown", "combined_tok_s",
        )}
        lease_values = numeric(subset, "lease_memory_time_MiB_s")
        stats["lease_memory_time_MiB_s"] = median_range(lease_values) if lease_values else None
        successful = sum(int(float(row["capacity_admits"])) for row in subset)
        useful = sum(int(float(row["promotion_useful"])) for row in subset)
        rate = useful / successful if successful else "NA"
        policy_rows.append({"policy": policy.name, "valid_repetitions": 4,
            "deadline_P50_median": stats["deadline_latency_P50"]["median"],
            "deadline_P95_median": stats["deadline_latency_P95"]["median"],
            "deadline_max_median": stats["deadline_latency_max"]["median"],
            "TTFT_P50_median": stats["TTFT_P50"]["median"],
            "TTFT_P95_median": stats["TTFT_P95"]["median"],
            "local_fraction_median": stats["local_source_fraction"]["median"],
            "total_H2D_median_MiB": stats["total_H2D_MiB"]["median"],
            "duplicate_H2D_median_MiB": stats["duplicate_H2D_MiB"]["median"],
            "promotion_attempts_median": stats["promotion_attempts"]["median"],
            "capacity_rejects_median": stats["capacity_rejects"]["median"],
            "reservation_skips_median": stats["reservation_skips"]["median"],
            "useful_promotion_rate": rate,
            "lease_memory_time_median": stats["lease_memory_time_MiB_s"]["median"] if lease_values else "NA",
            "X_slowdown_median": stats["X_slowdown"]["median"],
            "combined_tok_s_median": stats["combined_tok_s"]["median"]})
        extended[policy.name] = {"stats": stats, "successful_promotions": successful, "useful_promotions": useful,
            "remote_wait_promotions": sum(int(float(row["promotion_remote_wait"])) for row in subset),
            "ready_misses": sum(int(float(row["promotion_ready_miss"])) for row in subset),
            "capacity_rejects": sum(int(float(row["capacity_rejects"])) for row in subset),
            "reservation_skips": sum(int(float(row["reservation_skips"])) for row in subset)}
    write_csv(OUT / "policy_summary.csv", policy_rows)
    paired = paired_deltas(runs)
    if len(paired) != 12:
        raise RuntimeError("paired delta matrix incomplete")
    write_csv(OUT / "paired_deltas.csv", paired)
    by_name = {row["policy"]: row for row in policy_rows}
    a, b, c, d = [by_name[policy.name] for policy in POLICIES]
    flags = hypothesis_flags(
        {"deadline_latency_P95": float(a["deadline_P95_median"]), "total_H2D_MiB": float(a["total_H2D_median_MiB"]), "X_slowdown": float(a["X_slowdown_median"])},
        {"deadline_latency_P95": float(d["deadline_P95_median"]), "total_H2D_MiB": float(d["total_H2D_median_MiB"]), "X_slowdown": float(d["X_slowdown_median"])},
    )
    result = classify(stage_a_valid, stage_b_valid, flags)
    token_correct = sum(int(row["generated_tokens"]) == int(row["OSL"]) for row in measured_sessions)
    identity = Counter(row["policy"] for row in measured_sessions if row["identity_match"] == "False")
    report = {"phase": "4E-A", "result": result, "X_only_valid": sum(row["valid"] == "True" for row in x),
        "stage_a_valid": stage_a_valid, "stage_b_valid": stage_b_valid, "measured_resumes": len(measured_sessions),
        "token_correct": token_correct, "H1": flags["H1"], "H2": flags["H2"], "H3": flags["H3"],
        "output_identity_mismatch_by_policy": dict(identity), "policy_details": extended,
        "server_crashes": sum(row["server_crash"] == "True" for row in runs),
        "OOM": sum(row["OOM"] == "True" for row in runs),
        "logical_leaks": sum(int(row["logical_leaks"]) for row in runs),
        "lease_leaks": sum(int(row["lease_leaks"]) for row in runs),
        "unsafe_releases": sum(int(row["unsafe_releases"]) for row in runs),
        "service_overruns": sum(int(row["service_overruns"]) for row in runs),
    }
    write_json(OUT / "validation_report.json", report)
    main_table = ["| Policy | P95 Resume ms | Server TTFT P95 ms | Local % | Total H2D MiB | Duplicate H2D MiB | Useful Promotion | Lease Memory-Time MiB·s | X Slowdown | Combined tok/s |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in policy_rows:
        rate = row["useful_promotion_rate"]
        rate_text = "NA" if rate == "NA" else f"{float(rate)*100:.1f}%"
        lease_text = "NA" if row["lease_memory_time_median"] == "NA" else f"{float(row['lease_memory_time_median']):.1f}"
        main_table.append(f"| {row['policy']} | {float(row['deadline_P95_median']):.1f} | {float(row['TTFT_P95_median']):.1f} | {float(row['local_fraction_median'])*100:.1f}% | {float(row['total_H2D_median_MiB']):.1f} | {float(row['duplicate_H2D_median_MiB']):.1f} | {rate_text} | {lease_text} | {float(row['X_slowdown_median']):.3f} | {float(row['combined_tok_s_median']):.1f} |")
    questions = {"B vs A": "guarded oracle JIT", "C vs B": "physical lease", "D vs C": "logical reservation"}
    pairs = [("B vs A", b, a), ("C vs B", c, b), ("D vs C", d, c)]
    ablation = ["| Comparison | Question | P95 Δ ms | H2D Δ MiB | Duplicate H2D Δ MiB | X slowdown Δ | Interpretation |", "|---|---|---:|---:|---:|---:|---|"]
    for label, left, right in pairs:
        p95_delta = float(left["deadline_P95_median"]) - float(right["deadline_P95_median"])
        h2d_delta = float(left["total_H2D_median_MiB"]) - float(right["total_H2D_median_MiB"])
        dup_delta = float(left["duplicate_H2D_median_MiB"]) - float(right["duplicate_H2D_median_MiB"])
        x_delta = float(left["X_slowdown_median"]) - float(right["X_slowdown_median"])
        interpretation = ("lower P95" if p95_delta < 0 else "higher P95" if p95_delta > 0 else "same P95") + "; " + ("less duplicate traffic" if dup_delta < 0 else "more duplicate traffic" if dup_delta > 0 else "same duplicate traffic")
        ablation.append(f"| {label} | {questions[label]} | {p95_delta:+.1f} | {h2d_delta:+.1f} | {dup_delta:+.1f} | {x_delta:+.3f} | {interpretation} |")
    p95_change = pct(float(d["deadline_P95_median"]), float(a["deadline_P95_median"]))
    text = ["# Phase 4E-A — Frozen hybrid performance matrix", "", f"**{result}**. This is a controlled directional comparison, not a statistical-significance or SOTA claim.", "",
        "## Objective and frozen setup", "", "First-boundary frozen hybrid trace-driven workload: real coding-agent token/KV structure plus empirical TraceToolProxy timing, not real request-timestamp-faithful Agent execution. Eight paused Agent sessions; 168,016 reusable tokens; 10,501 blocks; minimum KV H2D 4,594.1875 MiB. Qwen2.5-1.5B-Instruct, vLLM 0.23.0, FP16, GPU0, 32K context, 2 GiB GPU KV (4,681 blocks), 8 GiB CPU external KV, K2, P_SAFE=102 ms, guard=110 ms, logical budget=3,510 blocks. No vLLM algorithm change in this phase.", "",
        "## Results (median of four fresh, counterbalanced runs per policy)", "", *main_table, "",
        "All shown medians are repetition-level medians; four runs are not treated as independent request-level significance samples. Per-policy min/max and per-repetition metrics are in `validation_report.json` and `run_metrics.csv`.", "",
        "## Component ablation", "", *ablation, "",
        f"Full SlackKV vs OnDemand median P95 change: {p95_change:+.1f}%. Frozen H1 (≥10% reduction): {'PASS' if flags['H1'] else 'FAIL'}; H2 (≤5% extra H2D): {'PASS' if flags['H2'] else 'FAIL'}; H3 (X slowdown ≤OnDemand+0.05): {'PASS' if flags['H3'] else 'FAIL'}.", "",
        f"GuardedJIT without lease: {extended[b['policy']]['remote_wait_promotions']} promoted remote waits across measured runs; duplicate H2D median {float(b['duplicate_H2D_median_MiB']):.1f} MiB. This is the post-T6 eviction diagnostic. C vs B tests the incremental physical lease; D vs C tests future logical reservation and selectivity. A similar C/D result would mean no measurable incremental reservation benefit on this frozen workload.", "",
        "Lease memory-time is exact T6→LEASE_RELEASED for C/D. A/B show NA; their promotion-lead exposure is not physical residency.", "",
        "## Correctness and limitations", "",
        f"X-only 4/4; semantic smoke {stage_a_valid}/4; measured {stage_b_valid}/16; measured resumes {len(measured_sessions)}/128; token counts {token_correct}/128; crashes {report['server_crashes']}; OOM {report['OOM']}; logical leaks {report['logical_leaks']}; lease leaks {report['lease_leaks']}; unsafe releases {report['unsafe_releases']}. Output SHA differences are diagnostic; counts by policy: {dict(identity)}. Service-envelope >102 ms: {report['service_overruns']}.", "",
        "The policy source tree and frozen workload are identical across all runs. Setup, materialization, warmup and pressure are excluded from source-counter/H2D measurement. No synchronous Prometheus query occurs between deadline and first token. The four-repetition matrix supports directional observations only; no predictor, all-boundary replay, sensitivity sweep or external SOTA comparison was performed.", "",
        "## Next-step status", "", "Phase 4E-A complete; no later phase was started.", ""]
    (OUT / "summary.md").write_text("\n".join(text), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(finalize(), sort_keys=True))
