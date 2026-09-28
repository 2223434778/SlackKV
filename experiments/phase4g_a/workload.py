"""Build the frozen Phase 4C inter-turn boundary manifest."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/phase4c/hybrid_workload.csv"
OUT = ROOT / "results/phase4g_a"
SOURCE_SHA = "9b381336a70e73b0efbddbb79d2c88608c19663247e819c1e7a364efd2971105"
BYTES_PER_TOKEN = 28672
WAVE_TOKENS = (168016, 172736, 179632, 185344)
WAVE_BLOCKS = (10501, 10796, 11227, 11584)
PAUSE_BINS = ((100, 200), (200, 500), (500, 1000), (1000, float("inf")))


def nearest_rank(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("empty observations")
    return sorted(values)[math.ceil(fraction * len(values)) - 1]


def load_source(path: Path = SOURCE) -> list[dict[str, str]]:
    if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError("EXPERIMENT_INVALID_WORKLOAD: source SHA mismatch")
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def build_manifest(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    episodes: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        episodes.setdefault(row["episode_id"], []).append(row)
    if len(episodes) != 8 or len(rows) != 40:
        raise ValueError("EXPERIMENT_INVALID_WORKLOAD: expected eight five-turn episodes")
    manifest: list[dict[str, object]] = []
    for episode_id in sorted(episodes):
        turns = sorted(episodes[episode_id], key=lambda row: int(row["replay_turn_index"]))
        if [int(turn["replay_turn_index"]) for turn in turns] != list(range(5)):
            raise ValueError(f"EXPERIMENT_INVALID_WORKLOAD: turn sequence {episode_id}")
        for boundary_index in range(4):
            previous, resumed = turns[boundary_index:boundary_index + 2]
            tokens = int(resumed["target_cached_tokens"])
            blocks = int(resumed["target_cached_blocks"])
            if tokens % 16 or blocks != tokens // 16:
                raise ValueError("EXPERIMENT_INVALID_WORKLOAD: block alignment")
            manifest.append({
                "config": resumed["config"],
                "session_hash": resumed["session_hash"],
                "episode_id": episode_id,
                "boundary_index": boundary_index,
                "boundary_id": f"{episode_id}:b{boundary_index}",
                "previous_turn_index": boundary_index,
                "resume_turn_index": boundary_index + 1,
                "proxy_pause_ms": float(previous["proxy_pause_ms"]),
                "required_prefix_tokens": tokens,
                "required_blocks": blocks,
                "resume_isl": int(resumed["source_isl"]),
                "resume_isl_new": int(resumed["source_isl_new"]),
                "resume_osl": int(resumed["source_osl"]),
                "lease_key": f"{episode_id}:turn:{boundary_index + 1}",
            })
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: list[dict[str, object]]) -> dict[str, object]:
    if len(manifest) != 32:
        raise ValueError("EXPERIMENT_INVALID_WORKLOAD: boundary count")
    tokens = [int(row["required_prefix_tokens"]) for row in manifest]
    blocks = [int(row["required_blocks"]) for row in manifest]
    pauses = [float(row["proxy_pause_ms"]) for row in manifest]
    wave_tokens = [sum(tokens[i] for i, row in enumerate(manifest) if row["boundary_index"] == b) for b in range(4)]
    wave_blocks = [sum(blocks[i] for i, row in enumerate(manifest) if row["boundary_index"] == b) for b in range(4)]
    bin_counts = [sum(lower <= pause < upper for pause in pauses) for lower, upper in PAUSE_BINS]
    if (sum(tokens), sum(blocks), wave_tokens, wave_blocks) != (705728, 44108, list(WAVE_TOKENS), list(WAVE_BLOCKS)):
        raise ValueError("EXPERIMENT_INVALID_WORKLOAD: reusable KV totals")
    if (min(pauses), statistics.median(pauses), nearest_rank(pauses, .95), max(pauses)) != (103, 473, 2374, 2764):
        raise ValueError("EXPERIMENT_INVALID_WORKLOAD: pause distribution")
    if bin_counts != [8, 8, 5, 11]:
        raise ValueError("EXPERIMENT_INVALID_WORKLOAD: pause bins")
    return {
        "sessions": 8, "turns": 40, "boundaries": 32,
        "reusable_tokens": sum(tokens), "reusable_blocks": sum(blocks),
        "one_full_crossing_sum_MiB": sum(tokens) * BYTES_PER_TOKEN / 2**20,
        "wave_tokens": wave_tokens, "wave_blocks": wave_blocks,
        "pause_min_ms": min(pauses), "pause_median_ms": statistics.median(pauses),
        "pause_P95_ms": nearest_rank(pauses, .95), "pause_max_ms": max(pauses),
        "pause_bin_counts": bin_counts, "source_sha256": SOURCE_SHA,
    }


def main() -> None:
    rows = load_source()
    manifest = build_manifest(rows)
    summary = validate_manifest(manifest)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "boundary_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    (OUT / "multi_turn_manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
