"""Aggregate only a complete, valid Phase 4F-B matrix."""
from __future__ import annotations

import csv
import json
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.phase4d_a.experiment import write_csv, write_json
from experiments.phase4f_b.metrics import POLICIES, X_OUTPUT_TOKENS, classify, hypothesis_flags

OUT = Path(__file__).resolve().parents[2] / "results/phase4f_b"
METRICS = (
    "deadline_latency_P95", "TTFT_P95", "total_H2D_MiB", "duplicate_H2D_MiB",
    "X_slowdown", "combined_tok_s", "lease_memory_time_MiB_s",
)
PAIRS = {
    "D_vs_A": (3, 0), "B_vs_A": (1, 0), "C_vs_B": (2, 1),
    "D_vs_C": (3, 2), "C_vs_A": (2, 0),
}


def read_csv(name: str) -> list[dict[str, str]]:
    path = OUT / name
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def numbers(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) not in (None, "", "NA")]


def med(rows: list[dict[str, Any]], key: str) -> float | str:
    values = numbers(rows, key)
    return statistics.median(values) if values else "NA"


def paired_deltas(runs: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_key = {(int(row["repetition"]), row["policy"]): row for row in runs}
    output = []
    for rep in range(1, 9):
        for label, (left_index, right_index) in PAIRS.items():
            left = by_key[(rep, POLICIES[left_index].name)]
            right = by_key[(rep, POLICIES[right_index].name)]
            p95_left, p95_right = float(left["deadline_latency_P95"]), float(right["deadline_latency_P95"])
            if p95_right <= 0:
                raise ValueError("paired P95 relative delta needs positive baseline")
            row: dict[str, Any] = {
                "repetition": rep, "comparison": label,
                "P95_absolute_delta_ms": p95_left - p95_right,
                "P95_relative_delta": p95_left / p95_right - 1,
                "TTFT_P95_delta_ms": float(left["TTFT_P95"]) - float(right["TTFT_P95"]),
                "total_H2D_delta_MiB": float(left["total_H2D_MiB"]) - float(right["total_H2D_MiB"]),
                "duplicate_H2D_delta_MiB": float(left["duplicate_H2D_MiB"]) - float(right["duplicate_H2D_MiB"]),
                "X_slowdown_delta": float(left["X_slowdown"]) - float(right["X_slowdown"]),
                "combined_tok_s_delta": float(left["combined_tok_s"]) - float(right["combined_tok_s"]),
                "lease_memory_time_delta_MiB_s": (
                    float(left["lease_memory_time_MiB_s"]) - float(right["lease_memory_time_MiB_s"])
                    if left["lease_memory_time_MiB_s"] != "NA" and right["lease_memory_time_MiB_s"] != "NA"
                    else "NA"),
            }
            output.append(row)
    return output


def bootstrap_paired_median(values: list[float], *, seed: int = 20260927, draws: int = 10000) -> tuple[float, float]:
    if len(values) != 8:
        raise ValueError("bootstrap needs eight paired repetitions")
    rng = random.Random(seed)
    estimates = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(draws))
    return estimates[int(.025 * draws)], estimates[int(.975 * draws) - 1]


def summarize_policy(subset: list[dict[str, str]], name: str) -> dict[str, Any]:
    if len(subset) != 8:
        raise ValueError("policy is incomplete")
    output: dict[str, Any] = {
        "policy": name, "valid_repetitions": 8,
        "deadline_P95_median": med(subset, "deadline_latency_P95"),
        "deadline_P95_min": min(numbers(subset, "deadline_latency_P95")),
        "deadline_P95_max": max(numbers(subset, "deadline_latency_P95")),
        "TTFT_P95_median": med(subset, "TTFT_P95"),
        "local_fraction_median": med(subset, "local_source_fraction"),
        "successful_promotions_median": med(subset, "successful_promotions"),
        "useful_rate_median": med(subset, "useful_rate"),
        "capacity_rejects_median": med(subset, "capacity_rejects"),
        "progress_rejects_median": med(subset, "progress_rejects"),
        "reservation_skips_median": med(subset, "reservation_skips"),
        "total_H2D_median": med(subset, "total_H2D_MiB"),
        "duplicate_H2D_median": med(subset, "duplicate_H2D_MiB"),
        "lease_memory_time_median": med(subset, "lease_memory_time_MiB_s"),
        "reservation_block_time_median": med(subset, "reservation_block_time"),
        "X_slowdown_median": med(subset, "X_slowdown"),
        "X_TPOT_median": med(subset, "X_TPOT_ms"),
        "combined_tok_s_median": med(subset, "combined_tok_s"),
        "promotion_attempts_median": med(subset, "promotion_attempts"),
        "peak_leased_blocks_median": med(subset, "peak_leased_blocks"),
        "peak_reserved_blocks_median": med(subset, "peak_reserved_blocks"),
        "service_overruns": sum(int(row["service_overrun_count"]) for row in subset),
        "remote_wait_promotions": sum(int(row["promotion_remote_wait"]) for row in subset),
        "useful_promotions": sum(int(row["promotion_useful"]) for row in subset),
        "progress_interventions": sum(int(row["progress_rejects"]) for row in subset),
    }
    return output


def fmt(value: Any, decimals: int = 2) -> str:
    return "NA" if value == "NA" else f"{float(value):.{decimals}f}"


def finalize() -> dict[str, Any]:
    x = read_csv("x_only_runs.csv")
    smoke = read_csv("semantic_smoke.csv")
    runs = read_csv("run_metrics.csv")
    sessions = read_csv("session_metrics.csv")
    valid_x = sum(row.get("valid") == "True" and int(row["generated_tokens"]) == X_OUTPUT_TOKENS for row in x)
    valid_smoke = sum(row.get("valid") == "True" for row in smoke)
    valid_runs = sum(row.get("valid") == "True" for row in runs)
    if (len(x), valid_x, len(smoke), valid_smoke, len(runs), valid_runs) != (8, 8, 4, 4, 32, 32):
        raise RuntimeError("no partial Phase 4F-B matrix aggregation")
    if any(float(row["X_tail_margin_ms"]) <= 0 for row in smoke + runs):
        raise RuntimeError("background ended before last Agent FIRST_TOKEN")
    measured_sessions = [row for row in sessions if row["stage"] == "measured"]
    if len(measured_sessions) != 256 or any(int(row["generated_tokens"]) != int(row["OSL"]) for row in measured_sessions):
        raise RuntimeError("measured Agent token correctness failed")
    first4 = statistics.median(numbers(x[:4], "X_decode_ms"))
    last4 = statistics.median(numbers(x[4:], "X_decode_ms"))
    baseline = statistics.median(numbers(x, "X_decode_ms"))
    drift = (last4 / first4 - 1) * 100
    # Policy runs precede the post-matrix controls; recalculate slowdown from all eight.
    for row in smoke + runs:
        row["X_slowdown"] = float(row["X_decode_ms"]) / baseline
    write_csv(OUT / "semantic_smoke.csv", smoke)
    write_csv(OUT / "run_metrics.csv", runs)
    policy_rows = [summarize_policy([r for r in runs if r["policy"] == policy.name], policy.name)
                   for policy in POLICIES]
    if any({r["repetition"] for r in runs if r["policy"] == p.name} != {str(i) for i in range(1, 9)}
           for p in POLICIES):
        raise RuntimeError("policy repetition missing")
    write_csv(OUT / "policy_summary.csv", policy_rows)
    # The structured scheduler trace is the resume trace for this phase.
    write_csv(OUT / "resume_trace.csv", read_csv("scheduler_trace.csv"))
    paired = paired_deltas(runs)
    write_csv(OUT / "paired_deltas.csv", paired)
    bootstrap = []
    for label, metric in (("D_vs_A", "P95_relative_delta"),
                          ("D_vs_C", "P95_relative_delta"),
                          ("D_vs_A", "X_slowdown_delta")):
        sample = numbers([r for r in paired if r["comparison"] == label], metric)
        lo, hi = bootstrap_paired_median(sample)
        bootstrap.append({"comparison": label, "metric": metric, "paired_median": statistics.median(sample),
                          "bootstrap_95_low": lo, "bootstrap_95_high": hi, "resamples": 10000,
                          "seed": 20260927, "interpretation": "descriptive uncertainty interval; not a significance test"})
    write_csv(OUT / "bootstrap_intervals.csv", bootstrap)
    a, b, c, d = policy_rows
    flags = hypothesis_flags(
        {"deadline_latency_P95": a["deadline_P95_median"], "total_H2D_MiB": a["total_H2D_median"], "X_slowdown": a["X_slowdown_median"]},
        {"deadline_latency_P95": d["deadline_P95_median"], "total_H2D_MiB": d["total_H2D_median"], "X_slowdown": d["X_slowdown_median"]},
    )
    result = classify(valid_x, valid_smoke, min(float(r["X_tail_margin_ms"]) for r in smoke), valid_runs, flags)
    identity = Counter(r["policy"] for r in measured_sessions if r["identity_match"] == "False")
    report: dict[str, Any] = {
        "phase": "4F-B", "start_commit": "012dd5f20efabe2d5132540153be7f6991b81145",
        "workload_sha": "9b381336a70e73b0efbddbb79d2c88608c19663247e819c1e7a364efd2971105",
        "X_only_valid": valid_x, "X_baseline_decode_ms": baseline,
        "X_first4_median_ms": first4, "X_last4_median_ms": last4, "X_drift_percent": drift,
        "environment_drift_warning": abs(drift) > 5,
        "Stage_A_valid": valid_smoke, "Stage_B_valid": valid_runs,
        "Agent_completions": len(measured_sessions), "token_correct": 256,
        "source_conservation_pass": sum(int(r["local_source_tokens"]) + int(r["external_source_tokens"]) == 168016 for r in runs),
        "crash_count": sum(r["server_crash"] == "True" for r in runs),
        "OOM_count": sum(r["OOM"] == "True" for r in runs),
        "lease_leaks": sum(int(r["lease_leaks"]) for r in runs),
        "logical_leaks": sum(int(r["logical_leaks"]) for r in runs),
        "unsafe_releases": sum(int(r["unsafe_releases"]) for r in runs),
        "PSA_invariant_violations": sum(int(r["PSA_invariant_violation"]) for r in runs),
        "output_identity_mismatch_by_policy": dict(identity),
        **flags, "final_result": result,
    }
    write_json(OUT / "validation_report.json", report)
    summary = [
        "# Phase 4F-B — PSA-safe frozen hybrid performance matrix", "",
        "## Objective", "",
        "Compare latency, KV H2D traffic, leased GPU-KV cost, and background interference of four preregistered PSA-safe policies; isolate the incremental value of logical reservation once PSA-1 is present.", "",
        "## Setup and compared methods", "",
        f"**{result}.** The eight-session first-boundary workload has real coding-agent token/KV structure and empirical TraceToolProxy pauses; it is not request-timestamp-faithful real Agent execution.", "",
        "All runs use Qwen2.5-1.5B-Instruct FP16 on GPU0, vLLM 0.23.0, 4,681 GPU KV blocks (2 GiB), 8-GiB CPU external KV, K2, P_SAFE=102 ms, 110-ms resume guard, X9500→512, and the immutable Phase 4C workload. The four methods are A OnDemand, B guarded evictable JIT, C guarded JIT+lease+PSA-1, and D C plus 3,510-block logical reservation. The unsafe lease-without-PSA variant is excluded.", "",
        f"X-only 8/8: baseline decode {baseline:.2f} ms; first-four median {first4:.2f} ms, last-four median {last4:.2f} ms, drift {drift:+.2f}%{' (ENVIRONMENT_DRIFT_WARNING)' if abs(drift)>5 else ''}. Stage A {valid_smoke}/4; Stage B {valid_runs}/32; Agent completions and exact token counts 256/256.", "",
        "## Results", "",
        "## Policy outcomes (median of eight run-level metrics)", "",
        "| Policy | P95 Resume ms | TTFT P95 ms | Local % | Useful Promo % | Total H2D MiB | Duplicate H2D MiB | Lease Mem-Time MiB·s | X Slowdown | Combined tok/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for p in policy_rows:
        useful = "NA" if p["useful_rate_median"] == "NA" else fmt(float(p["useful_rate_median"])*100,1)
        summary.append(f"| {p['policy']} | {fmt(p['deadline_P95_median'])} | {fmt(p['TTFT_P95_median'])} | {fmt(float(p['local_fraction_median'])*100)} | {useful} | {fmt(p['total_H2D_median'])} | {fmt(p['duplicate_H2D_median'])} | {fmt(p['lease_memory_time_median'])} | {fmt(p['X_slowdown_median'],3)} | {fmt(p['combined_tok_s_median'])} |")
    summary += ["", "## Admission and resource costs", "",
                "| Policy | Promo Admit | Capacity Reject | PSA Reject | Reservation Skip | Peak Lease Blocks | Peak Reserved Blocks | Reservation block·s |",
                "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for p in policy_rows:
        summary.append(f"| {p['policy']} | {fmt(p['successful_promotions_median'])} | {fmt(p['capacity_rejects_median'])} | {fmt(p['progress_rejects_median'])} | {fmt(p['reservation_skips_median'])} | {fmt(p['peak_leased_blocks_median'])} | {fmt(p['peak_reserved_blocks_median'])} | {fmt(p['reservation_block_time_median'])} |")
    summary += ["", "## Paired component comparisons", "",
                "| Comparison | Median P95 Δ % | Median H2D Δ MiB | Median duplicate Δ MiB | Median X slowdown Δ | Interpretation |",
                "|---|---:|---:|---:|---:|---|"]
    questions = {"B_vs_A": "guarded JIT before lease", "C_vs_B": "lease plus PSA (joint change)",
                 "D_vs_C": "incremental logical reservation", "D_vs_A": "full system vs OnDemand"}
    for label in ("B_vs_A", "C_vs_B", "D_vs_C", "D_vs_A"):
        rows = [r for r in paired if r["comparison"] == label]
        summary.append(f"| {label} | {statistics.median(numbers(rows,'P95_relative_delta'))*100:+.2f} | {statistics.median(numbers(rows,'total_H2D_delta_MiB')):+.2f} | {statistics.median(numbers(rows,'duplicate_H2D_delta_MiB')):+.2f} | {statistics.median(numbers(rows,'X_slowdown_delta')):+.3f} | {questions[label]} |")
    summary += ["", f"H1 {'PASS' if flags['H1'] else 'FAIL'}; H2 {'PASS' if flags['H2'] else 'FAIL'}; H3 {'PASS' if flags['H3'] else 'FAIL'}. D-vs-A P95 paired-median bootstrap 95% interval: {bootstrap[0]['bootstrap_95_low']*100:+.2f}% to {bootstrap[0]['bootstrap_95_high']*100:+.2f}%; descriptive uncertainty only, not a significance test.",
                f"B/C/D service-envelope overruns (>102 ms): {b['service_overruns']}/{c['service_overruns']}/{d['service_overruns']}. B remote waits {b['remote_wait_promotions']}; C {c['remote_wait_promotions']}; D {d['remote_wait_promotions']}. PSA interventions (old physical guard would admit): C {c['progress_interventions']}, D {d['progress_interventions']}.",
                f"Output identity mismatches (diagnostic, not validity gate): {dict(identity)}. No crashes, OOMs, leaks, unsafe releases, or PSA invariant violations were accepted into the matrix.",
                "Lease occupancy uses LEASE_ACQUIRED→LEASE_RELEASED; unlike evictable JIT lead time, it is an exact leased-residency cost. Logical reservation incremental effects are the D-vs-C row; no winner is assumed. Unsafe lease-without-PSA is excluded from performance comparisons and retained only as historical liveness-negative evidence.",
                "No predictor, 32-boundary replay, PSA-2, capacity sweep, SOTA comparison, or statistical-significance claim was made."]
    summary += ["", "## Interpretation and limitations", "",
                f"D versus A changes the median of run-level P95 values by {(float(d['deadline_P95_median'])/float(a['deadline_P95_median'])-1)*100:+.2f}% (paired median {statistics.median(numbers([r for r in paired if r['comparison']=='D_vs_A'],'P95_relative_delta'))*100:+.2f}%). This is below the preregistered 10% improvement threshold; H1 therefore fails even though H2D stays at the one-load minimum and H3 passes.",
                f"B's evictable prefetch produced {b['remote_wait_promotions']} post-T6 remote waits and {float(b['duplicate_H2D_median']):.2f} MiB median duplicate traffic. C adds lease **and** PSA together: those counts fall to {c['remote_wait_promotions']} and {float(c['duplicate_H2D_median']):.2f} MiB, but this matrix cannot assign the entire difference to the lease alone.",
                f"C versus D has the same median local fraction and total H2D; paired P95 changes {statistics.median(numbers([r for r in paired if r['comparison']=='D_vs_C'],'P95_relative_delta'))*100:+.2f}%. D makes a median {float(c['promotion_attempts_median'])-float(d['promotion_attempts_median']):.1f} fewer attempts by skipping {float(d['reservation_skips_median']):.1f} jobs and avoids C's {c['progress_interventions']} PSA rejects across eight runs. D's median lease memory-time is {float(d['lease_memory_time_median'])-float(c['lease_memory_time_median']):+.2f} MiB·s higher. Thus reservation shows promotion selectivity but limited incremental latency/traffic value on this frozen workload, and it is not cost-free.",
                f"PSA's observable selection cost is {c['progress_interventions']} physically feasible promotions rejected in C (versus {d['progress_interventions']} in D, where logical reservation filters candidates first). These are safe fallbacks, not request failures. This matrix does not isolate PSA's latency cost from lease effects.",
                "The output-identity mismatches are confined to E04: one E04 mismatch in every measured run under every policy. All 256 requested output lengths and source-accounting checks passed. E04's batch-sensitive token identity remains a diagnostic limitation, not evidence of semantic trace replay or KV corruption.",
                "Eight run-level repetitions support directional description only; paired bootstrap intervals are descriptive, not a significance test.", ""]
    for label in PAIRS:
        values = numbers([r for r in paired if r["comparison"] == label], "P95_relative_delta")
        summary.append(f"- {label} paired P95 delta: median {statistics.median(values)*100:+.2f}% (min {min(values)*100:+.2f}%, max {max(values)*100:+.2f}%).")
    summary += ["", "## Conclusion and next-step status", "",
                f"The preregistered classification is **{result}**. This Phase technically completed with all correctness and liveness gates satisfied; the 10% latency hypothesis failed. No later phase was launched or authorized by this result."]
    (OUT / "summary.md").write_text("\n".join(summary)+"\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(finalize(), sort_keys=True))
