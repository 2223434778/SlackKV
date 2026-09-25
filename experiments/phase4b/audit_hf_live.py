"""Audit a bounded sample of public HF rows without persisting trace text.

Run this from a login shell when the host requires its configured HTTPS proxy:
    python experiments/phase4b/audit_hf_live.py

Only schema names, sample counts, and token-count summaries are written. Row IDs
and all prompt/completion text stay in memory and are never emitted.
"""

from __future__ import annotations

import json
import statistics
import time
from http.client import HTTPException, IncompleteRead
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results" / "phase4b" / "live_hf_recheck.json"
DATASET = "netpreme/coding_agent_traces"
REVISION = "d56ae0d0bc80f3663a7a61a38c4ee60c26ce7b67"
SAMPLE_ROWS = 10
CONFIGS = ("gpt-oss-120b-pro", "gpt-oss-120b-verified")
BASE = "https://huggingface.co/api/datasets"
ROWS_API = "https://datasets-server.huggingface.co/rows"
TOKEN_FIELDS = ("isl", "isl_new", "osl")
TIMING_TOOL_MARKERS = ("time", "latency", "tool", "request", "duration", "timestamp")


def fetch_json(url: str) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urlopen(url, timeout=180) as response:
                return json.load(response)
        except (HTTPError, URLError, TimeoutError, HTTPException, IncompleteRead) as error:
            last_error = error
            if attempt < 2:
                time.sleep(attempt + 1)
    assert last_error is not None
    raise last_error


def summarize_rows(config: str, wrapped_rows: list[dict[str, Any]]) -> dict[str, Any]:
    sessions: list[dict[str, Any]] = []
    turns: list[dict[str, Any]] = []
    for wrapped in wrapped_rows:
        row = wrapped.get("row", {})
        row_turns = row.get("turns", [])
        if not isinstance(row_turns, list):
            raise ValueError(f"unexpected turns container for {config}")
        # Keep only counts; never copy IDs or text into the report.
        sessions.append({"turn_count": len(row_turns)})
        turns.extend(turn for turn in row_turns if isinstance(turn, dict))

    fields = sorted({key for turn in turns for key in turn})
    types = {
        key: sorted({type(turn[key]).__name__ for turn in turns if key in turn})
        for key in fields
    }
    token_stats: dict[str, dict[str, int | float | None]] = {}
    for key in TOKEN_FIELDS:
        values = sorted(
            value
            for turn in turns
            if isinstance((value := turn.get(key)), int) and not isinstance(value, bool)
        )
        token_stats[key] = {
            "count": len(values),
            "min": values[0] if values else None,
            "median": statistics.median(values) if values else None,
            "max": values[-1] if values else None,
        }
    timing_tool_fields = [
        key for key in fields if any(marker in key.lower() for marker in TIMING_TOOL_MARKERS)
    ]
    return {
        "config": config,
        "sample_rows_requested": SAMPLE_ROWS,
        "sample_rows_received": len(wrapped_rows),
        "sample_turns_received": len(turns),
        "sessions_ge_3_turns_in_sample": sum(item["turn_count"] >= 3 for item in sessions),
        "sessions_ge_5_turns_in_sample": sum(item["turn_count"] >= 5 for item in sessions),
        "turn_fields": fields,
        "turn_field_python_types": types,
        "timing_or_tool_named_fields_in_sample": timing_tool_fields,
        "token_count_stats": token_stats,
    }


def build_report(metadata: dict[str, Any], fetched: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    if metadata.get("sha") != REVISION:
        raise ValueError(f"HF revision moved: expected {REVISION}, got {metadata.get('sha')}")
    return {
        "dataset": DATASET,
        "revision": REVISION,
        "revision_verified_against_hub_api": True,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "hub_metadata": {
            "last_modified": metadata.get("lastModified"),
            "private": metadata.get("private"),
            "gated": metadata.get("gated"),
            "subset_metadata": metadata.get("cardData", {}).get("dataset_info", []),
        },
        "method": f"HF datasets-server rows API; fixed first {SAMPLE_ROWS} rows per GPT-OSS config",
        "raw_prompt_or_completion_text_persisted": False,
        "row_ids_persisted": False,
        "full_parquet_downloaded": False,
        "configs": [summarize_rows(config, fetched[config]) for config in CONFIGS],
        "interpretation_limit": (
            "Bounded row sample confirms the observed token/text-only turn schema; it is not an exhaustive scan of all rows."
        ),
    }


def main() -> None:
    metadata = fetch_json(f"{BASE}/{DATASET}/revision/{REVISION}")
    fetched: dict[str, list[dict[str, Any]]] = {}
    for config in CONFIGS:
        query = urlencode(
            {"dataset": DATASET, "config": config, "split": "train", "offset": 0, "length": SAMPLE_ROWS}
        )
        page = fetch_json(f"{ROWS_API}?{query}")
        fetched[config] = page.get("rows", [])
    report = build_report(metadata, fetched)
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
