"""Fetch the frozen HF row population and retain token structure only."""
from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from http.client import HTTPException, IncompleteRead
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

from selection import CONFIGS, REVISION, population_counts, select_sessions

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase4c"
DATASET = "netpreme/coding_agent_traces"
ROWS_URL = "https://datasets-server.huggingface.co/rows"
META_URL = f"https://huggingface.co/api/datasets/{DATASET}/revision/{REVISION}"


def fetch_json(url: str, retries: int = 5) -> dict[str, Any]:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urlopen(url, timeout=300) as response:
                return json.load(response)
        except (HTTPError, URLError, TimeoutError, HTTPException, IncompleteRead) as exc:
            last = exc
            if attempt + 1 < retries:
                time.sleep(attempt + 1)
    assert last is not None
    raise last


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    metadata = fetch_json(META_URL)
    if metadata.get("sha") != REVISION:
        raise RuntimeError(f"EXPERIMENT_INVALID_SOURCE:revision:{metadata.get('sha')}")
    records: list[tuple[str, str, list[dict[str, Any]]]] = []
    row_counts: dict[str, int] = {}
    for config in CONFIGS:
        row_counts[config] = 0
        for offset in range(0, 100, 10):
            query = urlencode({"dataset": DATASET, "config": config, "split": "train", "offset": offset, "length": 10})
            page = fetch_json(f"{ROWS_URL}?{query}")
            wrapped_rows = page.get("rows", [])
            if len(wrapped_rows) != 10:
                raise RuntimeError(f"EXPERIMENT_INVALID_SOURCE:row_count:{config}:{offset}:{len(wrapped_rows)}")
            row_counts[config] += len(wrapped_rows)
            for wrapped in wrapped_rows:
                raw = wrapped.get("row", {})
                iid = raw.get("instance_id")
                turns = raw.get("turns")
                if not isinstance(iid, str) or not isinstance(turns, list):
                    raise RuntimeError(f"EXPERIMENT_INVALID_SOURCE:schema:{config}")
                # Discard all raw text fields immediately; only integer count fields survive.
                token_turns = [{key: turn.get(key) for key in ("isl", "isl_new", "osl")} if isinstance(turn, dict) else None for turn in turns]
                records.append((config, iid, token_turns))
                del raw, iid, turns, token_turns
            del page, wrapped_rows
    population, selected_turns = select_sessions(records)
    if len(selected_turns) % 5:
        raise AssertionError(f"selected window is not five turns: {len(selected_turns)}")
    chosen_hashes = {(r["config"], r["session_hash"]) for r in selected_turns}
    for row in population:
        row["selected"] = (row["config"], row["session_hash"]) in chosen_hashes
    write_csv(OUT / "selection_population.csv", population)
    write_csv(OUT / "source_turns.csv", selected_turns)
    selected_sessions = []
    for episode_id in sorted({row["episode_id"] for row in selected_turns}):
        rows = [row for row in selected_turns if row["episode_id"] == episode_id]
        selected_sessions.append({
            "episode_id": episode_id,
            "config": rows[0]["config"],
            "session_hash": rows[0]["session_hash"],
            "selection_hash": next(r["selection_hash"] for r in population if r["config"] == rows[0]["config"] and r["session_hash"] == rows[0]["session_hash"]),
            "window_start_position": rows[0]["trace_turn_index"],
            "turn_count": len(rows),
        })
    write_csv(OUT / "selected_sessions.csv", selected_sessions)
    counts = population_counts(population)
    sparse = any(counts[config]["eligible"] < 4 for config in CONFIGS)
    manifest = {
        "dataset_id": DATASET,
        "revision": REVISION,
        "revision_verified": True,
        "configs": list(CONFIGS),
        "offsets": {config: {"start": 0, "length": 100} for config in CONFIGS},
        "rows_received": row_counts,
        "rows_api_returned_text_fields": True,
        "raw_text_persisted": False,
        "raw_instance_id_persisted": False,
        "full_dataset_downloaded": False,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "population": counts,
        "selected_sessions": len(selected_sessions),
        "selected_turns": len(selected_turns),
        "classification": "RESULT_B_NATIVE_FIT_SPARSE" if sparse else "CANDIDATES_READY",
    }
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (OUT / "trace_schema.md").write_text(
        "# Phase 4C source schema mapping\n\n"
        "The pinned HF rows API returns top-level `instance_id`, `dataset`, `model`, and ordered `turns`; nested turns expose `isl`, `isl_new`, `osl` and text fields. "
        "Only the three integer token-count fields and the in-memory row identifier were used. The identifier is immediately SHA256-hashed with revision/config and is not persisted. "
        "The ordered turn list is used as source turn order. This dataset supplies no tool-phase timing; timing proxies are sourced separately from TraceLab v0.0.2.\n",
        encoding="utf-8",
    )
    print(json.dumps({"population": manifest["population"], "selected_sessions": len(selected_sessions), "selected_turns": len(selected_turns), "classification": manifest["classification"]}, indent=2))


if __name__ == "__main__":
    main()
