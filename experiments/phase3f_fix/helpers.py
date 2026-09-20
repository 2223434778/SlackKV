"""Small result helpers for Phase 3F."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def percentile_linear(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def token_matches(token_id: int | None, control_id: int | None) -> bool:
    return token_id is not None and token_id == control_id
