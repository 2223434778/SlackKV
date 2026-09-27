from __future__ import annotations

import csv
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.phase4d_a.experiment import write_csv, write_json
from experiments.phase4e_a_fix.metrics import (
    POLICIES, X_OUTPUT_TOKENS, classify, hypothesis_flags, median_range,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/phase4e_a_fix"


def read_csv(name: str) -> list[dict[str, str]]:
    path = OUT / name
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def values(rows: list[dict[str, str]], field: str) -> list[float]:
    return [float(row[field]) for row in rows if row.get(field) not in (None, "", "NA")]


def median_or_na(rows: list[dict[str, str]], field: str) -> float | str:
    selected = values(rows, field)
    return statistics.median(selected) if selected else "NA"


def fmt(value: float | str, decimals: int = 2) -> str:
    return "NA" if value == "NA" else f"{float(value):.{decimals}f}"


def paired_deltas(runs: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_rep = {(int(row["repetition"]), row["policy"]): row for row in runs}
    pairs = {
        "D_vs_A": (POLICIES[3].name, POLICIES[0].name),
        "C_vs_B": (POLICIES[2].name, POLICIES[1].name),
        "D_vs_C": (POLICIES[3].name, POLICIES[2].name),
    }
    output: list[dict[str, Any]] = []
    for repetition in range(1, 5):
        for comparison, (left_name, right_name) in pairs.items():
            left = by_rep[(repetition, left_name)]
            right = by_rep[(repetition, right_name)]
            left_p95 = float(left["deadline_latency_P95"])
            right_p95 = float(right["deadline_latency_P95"])
            output.append({
                "repetition": repetition, "comparison": comparison,
                "P95_absolute_delta_ms": left_p95 - right_p95,
                "P95_relative_delta": left_p95 / right_p95 - 1 if right_p95 else "NA",
                "total_H2D_delta_MiB": float(left["total_H2D_MiB"]) - float(right["total_H2D_MiB"]),
                "duplicate_H2D_delta_MiB": float(left["duplicate_H2D_MiB"]) - float(right["duplicate_H2D_MiB"]),
                "X_slowdown_delta": float(left["X_slowdown"]) - float(right["X_slowdown"]),
                "combined_tok_s_delta": float(left["combined_tok_s"]) - float(right["combined_tok_s"]),
            })
    return output


def summarize_policy(subset: list[dict[str, str]], name: str) -> dict[str, Any]:
    def med(field: str) -> float | str:
        return median_or_na(subset, field)

    admits = sum(int(float(row["capacity_admits"])) for row in subset)
    useful = sum(int(float(row["promotion_useful"])) for row in subset)
    p95 = values(subset, "deadline_latency_P95")
    margin = values(subset, "X_tail_margin_ms")
    return {
        "policy": name, "valid_repetitions": len(subset),
        "deadline_P50_median": med("deadline_latency_P50"),
        "deadline_P95_median": statistics.median(p95),
        "deadline_P95_min": min(p95), "deadline_P95_max": max(p95),
        "TTFT_P95_median": med("TTFT_P95"),
        "local_fraction_median": med("local_source_fraction"),
        "promotion_attempts_median": med("promotion_attempts"),
        "capacity_admits_median": med("capacity_admits"),
        "capacity_rejects_median": med("capacity_rejects"),
        "reservation_skips_median": med("reservation_skips"),
        "useful_rate": useful / admits if admits else "NA",
        "useful_promotions": useful,
        "remote_wait_promotions": sum(int(float(row["promotion_remote_wait"])) for row in subset),
        "ready_miss_promotions": sum(int(float(row["promotion_ready_miss"])) for row in subset),
        "service_overrun_count": sum(int(float(row["service_overrun_count"])) for row in subset),
        "service_overrun_rate_median": med("service_overrun_rate"),
        "promotion_T6_P95_median": med("promotion_T6_P95_ms"),
        "total_H2D_median": med("total_H2D_MiB"),
        "duplicate_H2D_median": med("duplicate_H2D_MiB"),
        "lease_memory_time_median": med("lease_memory_time_MiB_s"),
        "promotion_lead_exposure_median": med("promotion_lead_exposure_MiB_s"),
        "X_decode_median": med("X_decode_ms"),
        "X_TPOT_median": med("X_TPOT_ms"),
        "X_tail_margin_min": min(margin),
        "X_tail_margin_median": statistics.median(margin),
        "X_slowdown_median": med("X_slowdown"),
        "combined_tok_s_median": med("combined_tok_s"),
    }


def finalize() -> dict[str, Any]:
    x = read_csv("x_only_runs.csv")
    smoke = read_csv("semantic_smoke.csv")
    runs = read_csv("run_metrics.csv")
    sessions = read_csv("session_metrics.csv")
    protocol = json.loads((OUT / "background_protocol.json").read_text(encoding="utf-8"))
    x_valid = sum(row.get("valid") == "True" and int(row.get("generated_tokens", -1)) == X_OUTPUT_TOKENS for row in x)
    a_valid = sum(row.get("valid") == "True" for row in smoke)
    b_valid = sum(row.get("valid") == "True" for row in runs)
    margin = float(protocol["minimum_smoke_tail_margin_ms"])
    if (len(x), x_valid, len(smoke), a_valid, len(runs), b_valid) != (4, 4, 4, 4, 16, 16):
        raise RuntimeError("cannot aggregate incomplete or invalid Phase4E-A-FIX matrix")
    if margin < 1000 or any(float(row["X_tail_margin_ms"]) <= 0 for row in runs):
        raise RuntimeError("background protocol validation failed")
    measured_sessions = [row for row in sessions if row["stage"] == "measured"]
    if len(measured_sessions) != 128:
        raise RuntimeError("expected exactly 128 measured Agent resumes")
    token_correct = sum(int(row["generated_tokens"]) == int(row["OSL"]) for row in measured_sessions)
    if token_correct != 128:
        raise RuntimeError("measured Agent output-token count mismatch")
    policy_rows = []
    for policy in POLICIES:
        subset = [row for row in runs if row["policy"] == policy.name]
        if len(subset) != 4 or {row["repetition"] for row in subset} != {"1", "2", "3", "4"}:
            raise RuntimeError(f"incomplete policy {policy.name}")
        policy_rows.append(summarize_policy(subset, policy.name))
    metric_fields = (
        "deadline_latency_P50", "deadline_latency_P95", "TTFT_P95",
        "local_source_fraction", "total_H2D_MiB", "duplicate_H2D_MiB",
        "promotion_attempts", "capacity_rejects", "reservation_skips",
        "service_overrun_rate", "lease_memory_time_MiB_s",
        "X_decode_ms", "X_TPOT_ms", "X_tail_margin_ms", "X_slowdown",
        "combined_tok_s",
    )
    ranges = {
        policy.name: {
            field: median_range(values([row for row in runs if row["policy"] == policy.name], field))
            for field in metric_fields
            if values([row for row in runs if row["policy"] == policy.name], field)
        }
        for policy in POLICIES
    }
    write_json(OUT / "metric_ranges.json", ranges)
    write_csv(OUT / "policy_summary.csv", policy_rows)
    paired = paired_deltas(runs)
    write_csv(OUT / "paired_deltas.csv", paired)
    a, b, c, d = policy_rows
    flags = hypothesis_flags(
        {"deadline_latency_P95": a["deadline_P95_median"], "total_H2D_MiB": a["total_H2D_median"], "X_slowdown": a["X_slowdown_median"]},
        {"deadline_latency_P95": d["deadline_P95_median"], "total_H2D_MiB": d["total_H2D_median"], "X_slowdown": d["X_slowdown_median"]},
    )
    result = classify(x_valid, a_valid, margin, b_valid, flags)
    identity = Counter(row["policy"] for row in measured_sessions if row["identity_match"] == "False")
    report = {
        "phase": "4E-A-FIX", "start_commit": "6c870afb2a47620a5b84be8aab0b3bf734ef5fc3",
        "workload_sha": "9b381336a70e73b0efbddbb79d2c88608c19663247e819c1e7a364efd2971105",
        "X_output_tokens": X_OUTPUT_TOKENS, "X_only_valid": x_valid,
        "Stage_A_valid": a_valid, "Stage_A_min_X_tail_margin_ms": margin,
        "Stage_B_expected": 16, "Stage_B_valid": b_valid,
        "resume_count": len(measured_sessions), "token_correct_count": token_correct,
        "source_conservation_pass": sum(int(row["local_source_tokens"]) + int(row["external_source_tokens"]) == 168016 for row in runs),
        "crash_count": sum(row["server_crash"] == "True" for row in runs),
        "OOM_count": sum(row["OOM"] == "True" for row in runs),
        "logical_leaks": sum(int(row["logical_leaks"]) for row in runs),
        "lease_leaks": sum(int(row["lease_leaks"]) for row in runs),
        "unsafe_releases": sum(int(row["unsafe_releases"]) for row in runs),
        "service_overrun_count": sum(int(row["service_overrun_count"]) for row in runs),
        "output_identity_mismatch_by_policy": dict(identity),
        "H1": flags["H1"], "H2": flags["H2"], "H3": flags["H3"],
        "final_result": result,
    }
    write_json(OUT / "validation_report.json", report)
    summary_lines = [
        "# Phase 4E-A-FIX — background-safe frozen performance matrix",
        "",
        f"**{result}.** Phase4E-A remains **EXPERIMENT_INVALID_MATRIX**: its 256-token X ended 143.44 ms before the final Agent FIRST_TOKEN in measured repetition 1, position 3. This new phase changes only the fixed X duration and X-dependent harness formulas; no old performance samples are reused.",
        "",
        "The workload is the frozen hybrid trace-driven first boundary: real coding-agent token/KV structure and empirical TraceToolProxy timing, not request-timestamp-faithful Agent execution. All policies use the same Phase4D-B vLLM mechanism, 2-GiB GPU KV, 8-GiB external KV, K2, P_SAFE=102 ms, and 3510-block logical budget when applicable.",
        "",
        f"Fresh X-only: {x_valid}/4, median decode {statistics.median(values(x, 'X_decode_ms')):.2f} ms, median TPOT {statistics.median(values(x, 'X_TPOT_ms')):.3f} ms; all X requests generated {X_OUTPUT_TOKENS} tokens. Stage A: {a_valid}/4, minimum X tail margin {margin:.2f} ms (frozen gate ≥1000 ms). Stage B: {b_valid}/16; Agent resumes {len(measured_sessions)}/128, exact output lengths {token_correct}/128.",
        "",
        "## Policy outcomes",
        "",
        "| Policy | P95 Resume ms | TTFT P95 ms | Local % | Total H2D MiB | Duplicate H2D MiB | Useful Promo | Service Overrun | Lease Mem-Time MiB·s | X Slowdown | Combined tok/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in policy_rows:
        useful = "NA" if row["useful_rate"] == "NA" else f"{row['useful_rate'] * 100:.1f}%"
        overrun = "NA" if row["service_overrun_rate_median"] == "NA" else f"{row['service_overrun_rate_median'] * 100:.1f}%"
        summary_lines.append(f"| {row['policy']} | {fmt(row['deadline_P95_median'])} | {fmt(row['TTFT_P95_median'])} | {fmt(row['local_fraction_median'] * 100)} | {fmt(row['total_H2D_median'])} | {fmt(row['duplicate_H2D_median'])} | {useful} | {overrun} | {fmt(row['lease_memory_time_median'])} | {fmt(row['X_slowdown_median'],3)} | {fmt(row['combined_tok_s_median'])} |")
    summary_lines += [
        "",
        "## Background protocol",
        "",
        "| Policy | Stage-A tail margin ms | Stage-B min tail margin ms | Stage-B median tail margin ms | X decode ms | X TPOT ms | X slowdown |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in policy_rows:
        stage_a = protocol["smoke_tail_margin_by_policy"][row["policy"]]
        summary_lines.append(f"| {row['policy']} | {stage_a:.2f} | {row['X_tail_margin_min']:.2f} | {row['X_tail_margin_median']:.2f} | {fmt(row['X_decode_median'])} | {fmt(row['X_TPOT_median'],3)} | {fmt(row['X_slowdown_median'],3)} |")
    summary_lines += [
        "",
        "## Component ablation",
        "",
        "| Comparison | Main question | P95 Δ ms | H2D Δ MiB | Overrun Δ count | Lease memory Δ MiB·s | X slowdown Δ |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for label, left, right, question in (("B vs A", b, a, "guarded oracle JIT"), ("C vs B", c, b, "physical lease"), ("D vs C", d, c, "logical reservation")):
        lease_delta = "NA" if left["lease_memory_time_median"] == "NA" or right["lease_memory_time_median"] == "NA" else fmt(left["lease_memory_time_median"] - right["lease_memory_time_median"])
        summary_lines.append(f"| {label} | {question} | {left['deadline_P95_median'] - right['deadline_P95_median']:+.2f} | {left['total_H2D_median'] - right['total_H2D_median']:+.2f} | {left['service_overrun_count'] - right['service_overrun_count']:+d} | {lease_delta} | {left['X_slowdown_median'] - right['X_slowdown_median']:+.3f} |")
    d_vs_a = (d["deadline_P95_median"] / a["deadline_P95_median"] - 1) * 100 if a["deadline_P95_median"] else float("nan")
    summary_lines += [
        "",
        f"Full SlackKV vs OnDemand median P95: {d_vs_a:+.2f}%. H1 {'PASS' if flags['H1'] else 'FAIL'}; H2 {'PASS' if flags['H2'] else 'FAIL'}; H3 {'PASS' if flags['H3'] else 'FAIL'}. Paired repetition-level deltas and ranges are in `paired_deltas.csv`; no statistical-significance claim is made.",
        f"B/C/D successful-promotion service overruns (>102 ms), measured across four repetitions: {b['service_overrun_count']}/{c['service_overrun_count']}/{d['service_overrun_count']}. Differences are consistent with promotion contention, not proof of PCIe causality.",
        f"Without lease, B had {b['remote_wait_promotions']} post-T6 promoted remote waits and median duplicate H2D {b['duplicate_H2D_median']:.2f} MiB. C had {c['remote_wait_promotions']} such waits and {c['duplicate_H2D_median']:.2f} MiB duplicate H2D. Lease memory-time is measured T6→LEASE_RELEASED for C/D; B's lead exposure is not exact physical residency.",
        f"Without logical reservation, C attempted a median {c['promotion_attempts_median']:.1f} promotions, with {c['capacity_rejects_median']:.1f} capacity rejects. Full D attempted {d['promotion_attempts_median']:.1f}, skipped {d['reservation_skips_median']:.1f}, and had {d['capacity_rejects_median']:.1f} capacity rejects. The P95 and memory-time columns show whether the extra selection mechanism helped or cost resources.",
        f"Measured correctness: source conservation {report['source_conservation_pass']}/16, crashes {report['crash_count']}, OOM {report['OOM_count']}, logical leaks {report['logical_leaks']}, physical lease leaks {report['lease_leaks']}, unsafe releases {report['unsafe_releases']}. Output identity mismatches are diagnostic: {dict(identity)}.",
        "",
        "All 4 policy medians are computed from four fresh repetition-level values, not 32 independent Agent samples. No predictor, full-32-boundary replay, SOTA comparison, or later phase was run.",
    ]
    summary_lines += ["", "## Paired repetition-level changes", ""]
    for comparison in ("D_vs_A", "C_vs_B", "D_vs_C"):
        subset = [row for row in paired if row["comparison"] == comparison]
        relative = values(subset, "P95_relative_delta")
        absolute = values(subset, "P95_absolute_delta_ms")
        summary_lines.append(
            f"- {comparison}: P95 relative median {statistics.median(relative) * 100:+.2f}% "
            f"(range {min(relative) * 100:+.2f}% to {max(relative) * 100:+.2f}%); "
            f"absolute median {statistics.median(absolute):+.2f} ms "
            f"(range {min(absolute):+.2f} to {max(absolute):+.2f} ms)."
        )
    summary_lines += [
        "",
        "Per-policy median/min/max for all principal metrics are in `metric_ranges.json`; no p-values are inferred from four repetitions.",
    ]
    (OUT / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(finalize(), sort_keys=True))
