"""Fixed-work, serial background requests for Phase 4G-A-FIX."""
from __future__ import annotations

import threading
import time
from typing import Any, Callable

from experiments.phase2c_fix.helpers import counter_delta

BG_PROMPT_TOKENS = 9500
BG_CHUNK_OUTPUT_TOKENS = 2048
BG_CHUNKS = 4
BG_TOTAL_OUTPUT_TOKENS = BG_CHUNK_OUTPUT_TOKENS * BG_CHUNKS
BG_POST_FIRST_TOKENS = BG_CHUNKS * (BG_CHUNK_OUTPUT_TOKENS - 1)
MAX_IDLE_GAP_MS = 50.0
CALIBRATION_TAIL_MS = 30000.0


def local_source_delta(before: dict, after: dict) -> float | None:
    """Return the local-source token counter delta, or None if unavailable."""
    try:
        return counter_delta(
            before,
            after,
            "vllm:prompt_tokens_by_source_total",
            {"source": "local"},
        )
    except (KeyError, TypeError, ValueError):
        return None


def run_chunks(
    runtime: Any,
    prompt: list[int],
    on_first: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Run four non-overlapping requests with identical prompt token IDs."""
    if len(prompt) != BG_PROMPT_TOKENS:
        raise ValueError("background prompt length mismatch")
    prompt_copy = list(prompt)
    prompt_hash = __import__("hashlib").sha256(
        b"".join(int(token).to_bytes(8, "little", signed=False) for token in prompt_copy)
    ).hexdigest()
    chunks: list[dict[str, Any]] = []
    previous_done_ns: int | None = None
    first_chunk_before = runtime.source_samples(runtime.metric_text())
    metric_start_ns = time.monotonic_ns()
    tokens: list[int] = []
    for chunk_id in range(BG_CHUNKS):
        result = runtime.send(
            prompt_copy,
            BG_CHUNK_OUTPUT_TOKENS,
            stream=True,
            params={"max_offload_tokens": 0},
            on_first=(on_first if chunk_id == 0 else None),
        )
        count = len(result["tokens"])
        if count != BG_CHUNK_OUTPUT_TOKENS:
            raise RuntimeError(f"background chunk {chunk_id} generated {count} tokens")
        after = runtime.source_samples(runtime.metric_text())
        metric_end_ns = time.monotonic_ns()
        raw_local = local_source_delta(first_chunk_before, after)
        gap_ms = ((int(result["issue_ns"]) - previous_done_ns) / 1e6
                  if previous_done_ns is not None else 0.0)
        row = {
            "chunk_id": chunk_id,
            "prompt_tokens": len(prompt_copy),
            "generated_tokens": count,
            "issue_ns": int(result["issue_ns"]),
            "first_ns": int(result["first_ns"]),
            "done_ns": int(result["done_ns"]),
            "decode_ms": (int(result["done_ns"]) - int(result["first_ns"])) / 1e6,
            "TPOT_ms": (int(result["done_ns"]) - int(result["first_ns"])) / 1e6 / (count - 1),
            "previous_chunk_done_ns": previous_done_ns if previous_done_ns is not None else "NA",
            "inter_chunk_idle_gap_ms": gap_ms,
            "metric_window_start_ns": metric_start_ns,
            "metric_window_end_ns": metric_end_ns,
            "source_local_delta_raw_tokens": raw_local if raw_local is not None else "NA",
            "prompt_local_hit_tokens": raw_local if raw_local is not None else "NA",
            "prompt_sha256": prompt_hash,
        }
        chunks.append(row)
        tokens.extend(int(token) for token in result["tokens"])
        previous_done_ns = int(result["done_ns"])
        first_chunk_before = after
        metric_start_ns = metric_end_ns
    return {
        "chunks": chunks,
        "tokens": tokens,
        "issue_ns": int(chunks[0]["issue_ns"]),
        "first_ns": int(chunks[0]["first_ns"]),
        "done_ns": int(chunks[-1]["done_ns"]),
        "prompt_sha256": prompt_hash,
    }


def summarize_chunks(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    if len(chunks) != BG_CHUNKS:
        raise ValueError("background requires exactly four chunks")
    if any(int(row["generated_tokens"]) != BG_CHUNK_OUTPUT_TOKENS for row in chunks):
        raise ValueError("background chunk token count mismatch")
    if len({row["prompt_sha256"] for row in chunks}) != 1:
        raise ValueError("background chunks did not use the same prompt")
    gaps = [float(row["inter_chunk_idle_gap_ms"]) for row in chunks[1:]]
    decode_durations = [float(row["decode_ms"]) for row in chunks]
    weighted_tpot = sum(decode_durations) / sum(
        int(row["generated_tokens"]) - 1 for row in chunks
    )
    return {
        "chunks_complete": len(chunks),
        "total_generated_tokens": sum(int(row["generated_tokens"]) for row in chunks),
        "wall_ms": (int(chunks[-1]["done_ns"]) - int(chunks[0]["first_ns"])) / 1e6,
        "weighted_TPOT_ms": weighted_tpot,
        "max_inter_chunk_idle_gap_ms": max(gaps, default=0.0),
        "chunk_decode_ms": decode_durations,
    }


def background_tails(
    final_done_ns: int, last_first_ns: int, last_finish_ns: int
) -> tuple[float, float]:
    return (
        (final_done_ns - last_first_ns) / 1e6,
        (final_done_ns - last_finish_ns) / 1e6,
    )


def background_valid(
    chunks: list[dict[str, Any]], last_finish_ns: int, *, calibration: bool = False
) -> tuple[bool, float]:
    summary = summarize_chunks(chunks)
    tail = (int(chunks[-1]["done_ns"]) - last_finish_ns) / 1e6
    enough_tail = tail >= CALIBRATION_TAIL_MS if calibration else tail > 0
    continuous = summary["max_inter_chunk_idle_gap_ms"] <= MAX_IDLE_GAP_MS
    return summary["total_generated_tokens"] == BG_TOTAL_OUTPUT_TOKENS and enough_tail and continuous, tail


def adjust_local_hit_attribution(
    chunks: list[dict[str, Any]], source_events: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Subtract traced Agent local hits from each interval's aggregate counter."""
    adjusted = []
    for source in chunks:
        row = dict(source)
        raw = row.get("source_local_delta_raw_tokens", "NA")
        start = int(row["metric_window_start_ns"]) / 1e9
        end = int(row["metric_window_end_ns"]) / 1e9
        agent_tokens = sum(
            int(event.get("local_cached_tokens", 0))
            for event in source_events
            if start <= float(event.get("timestamp_monotonic", -1)) <= end
        )
        row["agent_local_tokens_during_metric_window"] = agent_tokens
        if raw == "NA":
            row["prompt_local_hit_tokens"] = "NA"
        else:
            residual = float(raw) - agent_tokens
            row["prompt_local_hit_tokens"] = max(0, int(round(residual)))
        adjusted.append(row)
    return adjusted
