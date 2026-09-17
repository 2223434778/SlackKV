"""Small analysis helpers for the Phase 2F promotion crossover."""
from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any


def crossover_orders() -> list[str]:
    """Return the frozen six-run AB/BA sequence."""
    return ["AB", "BA", "AB", "BA", "AB", "BA"]


def position_labels(order: str) -> list[dict[str, Any]]:
    if order not in {"AB", "BA"}:
        raise ValueError("order must be AB or BA")
    return [
        {"session": session, "position": position}
        for position, session in enumerate(order, start=1)
    ]


def valid_cache_evidence(external_tokens: float, local_tokens: float) -> bool:
    return (
        math.isfinite(external_tokens)
        and math.isfinite(local_tokens)
        and external_tokens == 8192.0
        and local_tokens == 0.0
    )


def group_statistics(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[float]] = {
        "A_first": [],
        "A_second": [],
        "B_first": [],
        "B_second": [],
        "all_first": [],
        "all_second": [],
    }
    for row in rows:
        if row.get("run_validity") != "PASS":
            raise ValueError("group statistics accept only cache-valid runs")
        session = str(row["session"])
        position = int(row["position"])
        if session not in {"A", "B"} or position not in {1, 2}:
            raise ValueError("invalid session or promotion position")
        value = float(row["promotion_request_ms"])
        if not math.isfinite(value) or value < 0:
            raise ValueError("promotion time must be finite and non-negative")
        pos_name = "first" if position == 1 else "second"
        groups[f"{session}_{pos_name}"].append(value)
        groups[f"all_{pos_name}"].append(value)
    output: dict[str, dict[str, float | int]] = {}
    for name, values in groups.items():
        if not values:
            raise ValueError(f"group {name} has no valid promotion samples")
        output[name] = {
            "count": len(values),
            "median_ms": statistics.median(values),
            "min_ms": min(values),
            "max_ms": max(values),
        }
    return output


def position_ratio(groups: Mapping[str, Mapping[str, float | int]]) -> float:
    first = float(groups["all_first"]["median_ms"])
    second = float(groups["all_second"]["median_ms"])
    if second <= 0:
        raise ValueError("second-position median must be positive")
    return first / second


def identity_differences_ms(groups: Mapping[str, Mapping[str, float | int]]) -> tuple[float, float]:
    return (
        float(groups["A_first"]["median_ms"]) - float(groups["B_first"]["median_ms"]),
        float(groups["A_second"]["median_ms"]) - float(groups["B_second"]["median_ms"]),
    )


def transfer_counter_delta(before: Mapping[Any, float], after: Mapping[Any, float],
                           *, time_counter: bool) -> dict[str, Any] | None:
    """Aggregate only CPU-to-GPU transfer counters, excluding *_created gauges."""
    series = []
    for (name, labels), end in after.items():
        label_map = dict(labels)
        lowered = name.lower()
        if lowered.endswith("_created"):
            continue
        direction = " ".join((name, *label_map.values())).lower().replace("-", "_").replace(" ", "_")
        if "cpu_to_gpu" not in direction:
            continue
        is_time = "time" in lowered or "duration" in lowered
        if time_counter != is_time or (not time_counter and "byte" not in lowered):
            continue
        unit = (
            "milliseconds" if "ms" in lowered else
            "seconds" if "second" in lowered or lowered.endswith("_time_total") else
            "unknown"
        ) if time_counter else "bytes"
        series.append({
            "metric_name": name,
            "labels": label_map,
            "delta": float(end) - float(before.get((name, labels), 0.0)),
            "unit": unit,
        })
    if not series:
        return None
    units = {item["unit"] for item in series}
    unit = next(iter(units)) if len(units) == 1 else "mixed"
    result: dict[str, Any] = {
        "series": series,
        "metric_names": sorted({item["metric_name"] for item in series}),
        "unit": unit,
        "delta": sum(item["delta"] for item in series),
    }
    if time_counter and unit == "seconds":
        result["delta_ms"] = result["delta"] * 1000.0
    elif time_counter and unit == "milliseconds":
        result["delta_ms"] = result["delta"]
    return result
