"""Phase 4A read-only schema and actual-resume semantics gate.

The downstream opportunity and online-predictor stages must not run unless a
trace-observed next-LLM request/start timestamp is identified here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb


EXPECTED_SHA256 = "a7bab286bc640844560850965ccf47975cf66407154132abaab90f27ec9be744"
TABLES = ("rounds", "tool_calls", "timing_events", "trace_source")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def required_blocks(reusable_tokens: int) -> int:
    if reusable_tokens < 0:
        raise ValueError("reusable_tokens must not be negative")
    return (reusable_tokens + 15) // 16


def is_explicit_llm_start(event_type: str, source: str) -> bool:
    """Require an explicit request/start marker, never input-ready or output."""
    event = event_type.lower().replace("-", "_")
    origin = source.lower().replace("-", "_")
    return ("llm" in event or "model" in event or "llm" in origin or "model" in origin) and (
        "request_start" in event or "inference_start" in event or "request_start" in origin or "inference_start" in origin
    )


def event_semantics(event_types: list[dict]) -> dict:
    candidates = [row for row in event_types if is_explicit_llm_start(row["event_type"], row["source"])]
    return {
        "explicit_llm_request_start_event_types": candidates,
        "actual_resume_time": "UNOBSERVABLE" if not candidates else "REQUIRES_EVENT_LINK_VALIDATION",
        "tool_result_used_as_resume_time": False,
        "user_message_used_as_resume_time": False,
        "first_next_round_event_used_as_resume_time": False,
    }


def schema_snapshot(connection: duckdb.DuckDBPyConnection) -> dict:
    observed_tables = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
    if not set(TABLES).issubset(observed_tables):
        raise RuntimeError(f"Required tables missing: {sorted(set(TABLES) - observed_tables)}")
    tables = {}
    for name in TABLES:
        columns = connection.execute(f"DESCRIBE {name}").fetchall()
        tables[name] = {
            "row_count": connection.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0],
            "columns": [{"name": row[0], "type": row[1], "nullable": row[2]} for row in columns],
        }
    event_rows = connection.execute(
        "SELECT event_type, source, COUNT(*) AS n FROM timing_events GROUP BY 1, 2 ORDER BY n DESC, event_type, source"
    ).fetchall()
    first_rows = connection.execute(
        "SELECT event_type, source, COUNT(*) AS n FROM ("
        " SELECT event_type, source, ROW_NUMBER() OVER (PARTITION BY round_pk ORDER BY event_index) AS ordinal"
        " FROM timing_events) WHERE ordinal=1 GROUP BY 1, 2 ORDER BY n DESC, event_type, source"
    ).fetchall()
    return {
        "tables": tables,
        "identifiers": {"logical_session": "rounds.session_id", "round_order": "rounds.round_index", "round_join": "rounds.round_pk"},
        "timestamp_fields": ["tool_calls.emitted_at", "tool_calls.result_at", "timing_events.timestamp"],
        "token_fields": ["rounds.input_tokens_total", "rounds.prefix_tokens", "rounds.newly_append_tokens"],
        "event_type_counts": [{"event_type": kind, "source": source, "count": count} for kind, source, count in event_rows],
        "first_event_type_counts": [{"event_type": kind, "source": source, "count": count} for kind, source, count in first_rows],
    }


def render_schema(snapshot: dict, semantics: dict) -> str:
    lines = [
        "# TraceLab v0.0.2 schema and time-semantics audit",
        "",
        "The public dataset was inspected directly; all row counts and event frequencies are in `schema_snapshot.json`.",
        "",
        "## Tables and identifiers",
        "",
    ]
    for name, details in snapshot["tables"].items():
        lines.extend([f"### `{name}` ({details['row_count']:,} rows)", ""])
        lines.extend(f"- `{item['name']}`: `{item['type']}`" for item in details["columns"])
        lines.append("")
    lines.extend(
        [
            "## Mapping",
            "",
            "- Logical session: `rounds.session_id`; round order: `rounds.round_index`; round joins: `round_pk`.",
            "- Tool emission/completion: `tool_calls.emitted_at` / `result_at`.",
            "- Token accounting: `input_tokens_total = prefix_tokens + newly_append_tokens`. `prefix_tokens` is trace-observed cached-input tokens, so it is the most relevant reusable-prefix measurement; it is not an exact vLLM KV block identity.",
            "- `timing_events` contains input-ready (`user_message`, `tool_result`) and output-side (`reasoning`, `text`, `tool_call`, `usage_report`) observations, but no explicit next-LLM request/start event.",
            "- `rounds` has no request/start timestamp column. `tool_calls.result_at` is tool completion, not proof that the next LLM request started at that instant.",
            "",
            "## Actual resume time decision",
            "",
            f"`actual_resume_time = {semantics['actual_resume_time']}`. Accordingly `decision_time -> actual_resume_time` and `actual_slack_ms` cannot be measured without substituting a proxy forbidden by the Phase 4A protocol. No predictor or opportunity estimate is emitted.",
            "",
            "The [TraceLab v0.0.2 data-format documentation](https://github.com/uw-syfi/TraceLab/blob/v0.0.2/README.md#data-format) lists these event types and describes input-ready-to-tool-input as a *proxy*, not an LLM request-start timestamp. The [prompt-cache accounting note](https://github.com/uw-syfi/TraceLab/blob/v0.0.2/docs/prompt_cache_accounting.md) defines `prefix_tokens` as cache-read/cached-input tokens.",
            "",
        ]
    )
    return "\n".join(lines)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(dataset: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    observed_sha256 = sha256_file(dataset)
    git_start = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    source = {
        "repository": "uw-syfi/TraceLab",
        "release": "v0.0.2",
        "asset": dataset.name,
        "expected_sha256": EXPECTED_SHA256,
        "observed_sha256": observed_sha256,
        "sha256_validation": "PASS" if observed_sha256 == EXPECTED_SHA256 else "FAIL",
        "download_date": datetime.fromtimestamp(dataset.stat().st_mtime, timezone.utc).date().isoformat(),
        "dataset_file_size": dataset.stat().st_size,
        "git_commit_at_experiment_start": git_start,
    }
    write_json(output / "trace_source.json", source)
    if observed_sha256 != EXPECTED_SHA256:
        report = {"dataset_sha256_pass": False, "result": "EXPERIMENT_INVALID_DATASET"}
        write_json(output / "validation_report.json", report)
        return report
    connection = duckdb.connect(str(dataset), read_only=True)
    snapshot = schema_snapshot(connection)
    semantics = event_semantics(snapshot["event_type_counts"])
    write_json(output / "schema_snapshot.json", snapshot)
    (output / "trace_schema.md").write_text(render_schema(snapshot, semantics), encoding="utf-8")
    report = {
        "dataset_sha256_pass": True,
        "schema_audit_pass": True,
        "total_sessions": connection.execute("SELECT COUNT(DISTINCT session_id) FROM rounds").fetchone()[0],
        "total_tool_boundaries": connection.execute("SELECT COUNT(DISTINCT round_pk) FROM tool_calls").fetchone()[0],
        "valid_timing_boundaries": None,
        "replayable_boundaries": None,
        "actual_resume_semantics": semantics,
        "future_leakage_count": None,
        "predictor_evaluated_count": 0,
        "sessions_ge_1": None,
        "sessions_ge_2": None,
        "sessions_ge_3": None,
        "sessions_ge_4": None,
        "sessions_ge_5": None,
        "structural_trace_replay": "NO",
        "structural_trace_replay_reason": "A trace-faithful resume timeline cannot be constructed without a next-LLM-start timestamp; prefix-only reconstruction was not evaluated after the timing gate.",
        "promotion_service_model_status": "NOT_EVALUATED_AFTER_TIMING_GATE",
        "result": "RESULT_C_NO_VALID_RESUME_SEMANTICS" if semantics["actual_resume_time"] == "UNOBSERVABLE" else "REQUIRES_EVENT_LINK_VALIDATION",
    }
    write_json(output / "validation_report.json", report)
    write_json(output / "config.json", {"phase": "4A", "dataset_release": "v0.0.2", "block_size_tokens": 16, "reference_service_budgets_ms": [40, 70, 200, 240], "gpu_serving_benchmark": False})
    write_json(output / "environment.json", {"python": sys.version.split()[0], "duckdb": duckdb.__version__, "os": platform.platform(), "git_start": git_start})
    (output / "commands.sh").write_text(
        "# Dataset must be the SHA256-verified TraceLab v0.0.2 release asset.\n"
        "PYTHONPATH=/tmp/slackkv-phase4a-deps /media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python experiments/phase4a/run_phase4a.py external/tracelab_v002/syfi_coding_trace.duckdb results/phase4a\n"
        "PYTHONPATH=/tmp/slackkv-phase4a-deps /media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python -m pytest -q tests/test_phase4a.py\n",
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset, args.output), ensure_ascii=False, indent=2))
