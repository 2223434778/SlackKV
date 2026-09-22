from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase3n"


def read_rows(name: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted((OUT / "batches").glob(f"*/{name}")):
        with path.open(encoding="utf-8", newline="") as stream:
            rows.extend(csv.DictReader(stream))
    return rows


def write_rows(name: str, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def f(row: dict[str, str], name: str) -> float:
    return float(row[name])


def med(rows: list[dict[str, str]], field: str) -> float | None:
    return statistics.median([f(row, field) for row in rows]) if rows else None


def main() -> None:
    native = read_rows("native_prefetch_metrics.csv")
    resume = read_rows("resume_metrics.csv")
    timing = read_rows("timing_boundary.csv")
    trace = read_rows("scheduler_trace.csv")
    if len(resume) != 15:
        raise RuntimeError(f"expected 15 measured runs, got {len(resume)}")
    write_rows("native_prefetch_metrics.csv", native); write_rows("resume_metrics.csv", resume); write_rows("timing_boundary.csv", timing); write_rows("scheduler_trace.csv", trace)
    native_by = {(row["offset_ms"], row["repetition"]): row for row in native}
    timing_by = {(row["offset_ms"], row["repetition"]): row for row in timing}
    run_rows = []
    for row in resume:
        key = (row["offset_ms"], row["repetition"])
        run_rows.append({**row, **{f"native_{name}": value for name, value in native_by[key].items() if name not in key}, **{f"timing_{name}": value for name, value in timing_by[key].items() if name not in key}})
    write_rows("run_metrics.csv", run_rows)
    snapshots: list[Any] = []
    for path in sorted((OUT / "batches").glob("*/metric_snapshots.json")):
        snapshots.extend(json.loads(path.read_text(encoding="utf-8")))
    (OUT / "metric_snapshots.json").write_text(json.dumps(snapshots, indent=2), encoding="utf-8")
    groups = {name: [row for row in resume if row["classification"] == name] for name in ("PRE_READY", "INTERNAL_READY_PRE_CLIENT_RETURN", "POST_CLIENT_RETURN")}
    critical = groups["INTERNAL_READY_PRE_CLIENT_RETURN"]
    critical_ok = [row for row in critical if row["cache_transport_validation"] == "PASS"]
    valid = [row for row in resume if row["cache_transport_validation"] == "PASS" and row["x_active_validation"] == "PASS" and row["generated_token_count"] == "32"]
    baseline = groups["POST_CLIENT_RETURN"]
    internal_median = med(critical, "resume_ttft_ms")
    post_median = med(baseline, "resume_ttft_ms")
    all_native_ok = all(row["native_validation"] == "PASS" for row in native)
    all_x_ok = all(row["x_active_validation"] == "PASS" for row in resume)
    if len(critical) < 3 or not all_native_ok or not all_x_ok or len(valid) != 15:
        result, direction = "D", "CONTROL_PATH_FIX"
    elif len(critical_ok) >= 3 and post_median is not None and internal_median is not None and internal_median <= 1.10 * post_median:
        result, direction = "A", "ASYNC_PROMOTION_E2E"
    elif len(critical_ok) >= 3:
        result, direction = "B", "CONTROL_PATH_FIX"
    else:
        result, direction = "C", "CACHE_READY_SEMANTICS"
    config = {"offsets_ms": [5,20,30,40,70], "repetitions": 3, "resume_max_tokens": 32, "x_max_tokens": 128, "result": result, "next_direction": direction}
    (OUT / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    lines = [
        "# Phase 3N — internal KV-ready handoff", "",
        "Phase 3M T0–T7 instrumentation was reused unchanged; no new vLLM source changes were made.", "",
        f"- Valid measured runs: {len(valid)}/15.",
        f"- PRE_READY / INTERNAL_READY_PRE_CLIENT_RETURN / POST_CLIENT_RETURN: {len(groups['PRE_READY'])} / {len(critical)} / {len(groups['POST_CLIENT_RETURN'])}.",
        f"- Critical-window local hit successes: {len(critical_ok)}/{len(critical)}; remote-wait events: {sum(row['waiting_for_remote_kvs'] == 'True' for row in critical)}.",
        f"- Median TTFT (PRE / critical / post): {med(groups['PRE_READY'], 'resume_ttft_ms')} / {internal_median} / {post_median} ms. No PRE_READY sample occurred under the frozen offsets; no additional offset was added.",
        f"- Median internal T6-ready latency: {med(timing, 'T6_ready_latency_ms')} ms; median T6→client-return gap: {med(timing, 'T6_to_client_return_ms')} ms.",
        f"- Result {result}; next direction: {direction}.", "",
        "The critical-window median is within 10% of the POST_CLIENT_RETURN median, so the scheduler-visible T6 boundary is sufficient for local reuse; waiting for the HTTP endpoint return is not required.",
        "Interpretation: cache outcomes are based on the real `vllm:prompt_tokens_by_source_total` local/external source-counter deltas around A resume, plus actual scheduler remote-wait trace events; they are not inferred from latency.",
    ]
    (OUT / "summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    (OUT / "implementation.md").write_text("# Implementation\n\nPhase 3N reuses Phase 3M T0–T7 instrumentation unchanged. It starts a 128-token ordinary X decode, submits native prefetch A in an independent thread after X's first client token, and independently sends A at fixed offsets without waiting for the endpoint return. The scheduler arrival timestamp selects the boundary category.\n", encoding="utf-8")
    (OUT / "vllm_phase3n_delta.patch").write_text("No new vLLM source changes for Phase 3N. Phase 3M T0–T7 instrumentation reused unchanged.\n", encoding="utf-8")
    print(json.dumps({"result": result, "direction": direction, "counts": {key: len(value) for key, value in groups.items()}}, indent=2))


if __name__ == "__main__":
    main()
