"""Privacy-preserving structure selection for the Phase 4C hybrid replay."""
from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Any, Iterable

REVISION = "d56ae0d0bc80f3663a7a61a38c4ee60c26ce7b67"
CONFIGS = ("gpt-oss-120b-pro", "gpt-oss-120b-verified")
BLOCK_SIZE = 16
MAX_CONTEXT = 32768
MAX_OSL = 512


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def session_digest(config: str, instance_id: str) -> str:
    return sha256_text(f"{REVISION}|{config}|{instance_id}")


def _valid_turn(turn: Any) -> bool:
    if not isinstance(turn, dict):
        return False
    values = (turn.get("isl"), turn.get("isl_new"), turn.get("osl"))
    if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
        return False
    isl, isl_new, osl = values
    return isl > 0 and osl > 0 and 0 <= isl_new <= isl


def aligned_prefix(isl: int, isl_new: int, block_size: int = BLOCK_SIZE) -> tuple[int, int, int]:
    if not isinstance(isl, int) or not isinstance(isl_new, int) or isl < 0 or not 0 <= isl_new <= isl or block_size <= 0:
        raise ValueError("invalid token accounting")
    source_cached = isl - isl_new
    blocks = source_cached // block_size
    target = blocks * block_size
    return source_cached, target, source_cached - target


def inspect_session(config: str, instance_id: str, turns: list[Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return one redacted population record and turns from the earliest valid window."""
    digest = session_digest(config, instance_id)
    row: dict[str, Any] = {
        "config": config,
        "session_hash": digest,
        "source_turn_count": len(turns),
        "has_valid_5turn_window": False,
        "selected_window_start": "",
        "max_isl_in_window": "",
        "max_osl_in_window": "",
        "max_isl_plus_osl": "",
        "invalid_token_accounting": False,
        "context_limit_violation": False,
        "osl_cap_violation": False,
        "prefix_feasibility_violation": False,
        "context_shrink": False,
        "eligible": False,
        "selected": False,
        "exclusion_reason": "",
    }
    if len(turns) < 5:
        row["exclusion_reason"] = "fewer_than_5_turns"
        return row, []

    seen_reasons: set[str] = set()
    for start in range(len(turns) - 4):
        window = turns[start : start + 5]
        reasons: set[str] = set()
        if not all(_valid_turn(turn) for turn in window):
            reasons.add("invalid_token_accounting")
        else:
            fields = [(int(t["isl"]), int(t["isl_new"]), int(t["osl"])) for t in window]
            if any(isl + osl > MAX_CONTEXT for isl, _, osl in fields):
                reasons.add("context_limit_violation")
            if any(osl > MAX_OSL for _, _, osl in fields):
                reasons.add("osl_cap_violation")
            if any(fields[i + 1][0] < fields[i][0] for i in range(4)):
                reasons.add("context_shrink")
            for i in range(4):
                next_isl, next_isl_new, _ = fields[i + 1]
                _, cached_aligned, _ = aligned_prefix(next_isl, next_isl_new)
                prev_isl, _, prev_osl = fields[i]
                if cached_aligned > prev_isl + prev_osl:
                    reasons.add("prefix_feasibility_violation")
        seen_reasons.update(reasons)
        if not reasons:
            row.update(
                has_valid_5turn_window=True,
                selected_window_start=start,
                max_isl_in_window=max(int(t["isl"]) for t in window),
                max_osl_in_window=max(int(t["osl"]) for t in window),
                max_isl_plus_osl=max(int(t["isl"]) + int(t["osl"]) for t in window),
                eligible=True,
                exclusion_reason="",
            )
            selected_turns = []
            for offset, t in enumerate(window):
                cached, target, loss = aligned_prefix(int(t["isl"]), int(t["isl_new"]))
                selected_turns.append({
                    "config": config,
                    "session_hash": digest,
                    "replay_turn_index": offset,
                    "trace_turn_index": start + offset,
                    "source_isl": int(t["isl"]),
                    "source_isl_new": int(t["isl_new"]),
                    "source_osl": int(t["osl"]),
                    "source_cached_tokens": cached,
                    "target_cached_tokens": target,
                    "target_cached_blocks": target // BLOCK_SIZE,
                    "alignment_loss_tokens": loss,
                })
            return row, selected_turns

    reason_order = ("invalid_token_accounting", "context_limit_violation", "osl_cap_violation", "context_shrink", "prefix_feasibility_violation")
    for key in reason_order:
        row[key] = key in seen_reasons
    row["exclusion_reason"] = ";".join(k for k in reason_order if k in seen_reasons) or "no_valid_5turn_window"
    return row, []


def select_sessions(records: Iterable[tuple[str, str, list[Any]]], per_config: int = 4) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    population: list[dict[str, Any]] = []
    candidates: dict[str, list[tuple[str, dict[str, Any], list[dict[str, Any]]]]] = {c: [] for c in CONFIGS}
    for config, instance_id, turns in records:
        rec, selected_turns = inspect_session(config, instance_id, turns)
        population.append(rec)
        if selected_turns:
            key = sha256_text(f"SlackKV-Phase4C|{REVISION}|{config}|{rec['session_hash']}")
            candidates[config].append((key, rec, selected_turns))
    chosen: list[dict[str, Any]] = []
    source_turns: list[dict[str, Any]] = []
    for config in CONFIGS:
        ranked = sorted(candidates[config], key=lambda item: item[0])
        for key, rec, turns in ranked[:per_config]:
            rec["selected"] = True
            rec["selection_hash"] = key
            chosen.append({"config": config, "session_hash": rec["session_hash"], "selection_hash": key, "window_start_position": rec["selected_window_start"]})
            source_turns.extend(turns)
    chosen.sort(key=lambda row: (row["config"], row["selection_hash"]))
    return population, [{**turn, "episode_id": f"E{i:02d}"} for i, session in enumerate(chosen) for turn in source_turns if turn["session_hash"] == session["session_hash"]]


def population_counts(population: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for config in CONFIGS:
        rows = [row for row in population if row["config"] == config]
        output[config] = {
            "sessions": len(rows),
            "ge_5_turns": sum(row["source_turn_count"] >= 5 for row in rows),
            "eligible": sum(bool(row["eligible"]) for row in rows),
            "selected": sum(bool(row["selected"]) for row in rows),
            "exclusions": dict(Counter(row["exclusion_reason"] for row in rows if row["exclusion_reason"])),
        }
    return output
