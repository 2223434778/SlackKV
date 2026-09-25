"""Extract anonymized TraceToolProxy durations and assign frozen pauses."""
from __future__ import annotations

import csv
import json
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from replay_helpers import quantile, trace_tool_proxy_quantiles

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase4c"
DB = ROOT / "external" / "tracelab_v002" / "syfi_coding_trace.duckdb"
EXPECTED_SHA = "a7bab286bc640844560850965ccf47975cf66407154132abaab90f27ec9be744"
SEED = 20260925


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def qsummary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p50": None, "p90": None, "p95": None, "max": None}
    return {"count": len(values), "min": min(values), "p50": quantile(values, .5),
            "p90": quantile(values, .9), "p95": quantile(values, .95), "max": max(values)}


def main():
    import hashlib
    actual_sha = hashlib.sha256(DB.read_bytes()).hexdigest()
    if actual_sha != EXPECTED_SHA:
        raise RuntimeError(f"EXPERIMENT_INVALID_SOURCE:TraceLab SHA mismatch {actual_sha}")
    con = duckdb.connect(str(DB), read_only=True)
    rows = con.execute("""
        SELECT round_pk,
               MIN(emitted_at) FILTER (WHERE emitted_at IS NOT NULL AND result_at IS NOT NULL) AS first_emit,
               MAX(result_at) FILTER (WHERE emitted_at IS NOT NULL AND result_at IS NOT NULL) AS last_result,
               COUNT(*) FILTER (WHERE emitted_at IS NOT NULL AND result_at IS NOT NULL) AS complete_calls
        FROM tool_calls
        GROUP BY round_pk
        HAVING COUNT(*) FILTER (WHERE emitted_at IS NOT NULL AND result_at IS NOT NULL) > 0
    """).fetchall()
    con.close()
    valid: list[float] = []
    malformed = 0
    for _, first_emit, last_result, complete_calls in rows:
        if first_emit is None or last_result is None:
            malformed += 1
            continue
        duration = (last_result - first_emit).total_seconds() * 1000.0
        if duration <= 0:
            malformed += 1
        else:
            valid.append(duration)
    retained = [value for value in valid if 100 <= value <= 3000]
    summary = {
        "dataset": "uw-syfi/TraceLab",
        "release": "v0.0.2",
        "asset": "syfi_coding_trace.duckdb",
        "sha256": actual_sha,
        "tool_phase_definition": "per round_pk: max(result_at) - min(emitted_at) over complete timestamped tool calls",
        "valid_tool_rounds_positive_duration": len(valid),
        "invalid_or_nonpositive_tool_rounds": malformed,
        "retained_100_3000_ms": len(retained),
        "retained_fraction": len(retained) / len(valid) if valid else 0.0,
        "full_valid_distribution_ms": qsummary(valid),
        "retained_distribution_ms": qsummary(retained),
        "extracted_at_utc": datetime.now(timezone.utc).isoformat(),
        "terminology": "TraceToolProxy / observed_tool_phase_ms; not resume slack or real deadline",
    }
    (OUT / "tracetool_proxy_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if not retained:
        raise RuntimeError("no retained TraceToolProxy durations")
    pauses = trace_tool_proxy_quantiles(retained, count=32, seed=SEED)
    # The helper applies one fixed permutation after deterministic quantile coverage.
    selected = []
    for session in sorted(__import__("csv").DictReader((OUT / "selected_sessions.csv").open(encoding="utf-8")),
                          key=lambda row: (row["config"], row["selection_hash"])):
        turns = list(__import__("csv").DictReader((OUT / "source_turns.csv").open(encoding="utf-8")))
        episode_turns = [r for r in turns if r["session_hash"] == session["session_hash"]]
        for turn in episode_turns[:4]:
            selected.append({"episode_id": session["episode_id"], "config": session["config"],
                             "session_hash": session["session_hash"], "boundary_index": int(turn["replay_turn_index"]),
                             "proxy_pause_ms": pauses[len(selected)], "assignment_seed": SEED})
    if len(selected) != 32:
        raise AssertionError(f"expected 32 boundaries, got {len(selected)}")
    write_csv(OUT / "pause_assignments.csv", selected)
    workload = []
    source_turns = list(csv.DictReader((OUT / "source_turns.csv").open(encoding="utf-8")))
    pause_map = {(r["session_hash"], int(r["boundary_index"])): r["proxy_pause_ms"] for r in selected}
    for row in source_turns:
        turn = int(row["replay_turn_index"])
        workload.append({
            "config": row["config"], "session_hash": row["session_hash"],
            "replay_turn_index": turn, "source_isl": row["source_isl"],
            "source_isl_new": row["source_isl_new"], "source_osl": row["source_osl"],
            "source_cached_tokens": row["source_cached_tokens"],
            "target_cached_tokens": row["target_cached_tokens"],
            "target_cached_blocks": row["target_cached_blocks"],
            "proxy_pause_ms": pause_map.get((row["session_hash"], turn), ""),
            "synthetic_seed": f"phase4c|{row['config']}|{row['session_hash']}|{turn}",
            "episode_id": row["episode_id"],
        })
    write_csv(OUT / "hybrid_workload.csv", workload)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
