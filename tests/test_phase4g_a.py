from __future__ import annotations

import json
import time

import pytest

from experiments.phase4g_a.controller import BUDGET_BLOCKS, P_SAFE_MS, DynamicController, StructuredTraceReader
from experiments.phase4g_a.finalize_phase4g_a import (
    BYTES_PER_TOKEN, ORDER, POLICIES, bootstrap, classify, duplicate_bytes, pause_bin,
)
from experiments.phase4g_a.workload import (
    SOURCE, BYTES_PER_TOKEN as WORKLOAD_BYTES_PER_TOKEN,
    build_manifest, load_source, nearest_rank, validate_manifest,
)


def test_frozen_manifest_totals() -> None:
    manifest = build_manifest(load_source(SOURCE))
    assert validate_manifest(manifest) == {
        "sessions": 8, "turns": 40, "boundaries": 32,
        "reusable_tokens": 705728, "reusable_blocks": 44108,
        "one_full_crossing_sum_MiB": 19297.25,
        "wave_tokens": [168016, 172736, 179632, 185344],
        "wave_blocks": [10501, 10796, 11227, 11584],
        "pause_min_ms": 103, "pause_median_ms": 473,
        "pause_P95_ms": 2374, "pause_max_ms": 2764,
        "pause_bin_counts": [8, 8, 5, 11],
        "source_sha256": "9b381336a70e73b0efbddbb79d2c88608c19663247e819c1e7a364efd2971105",
    }


def test_boundary_mapping() -> None:
    source = load_source(SOURCE)
    manifest = build_manifest(source)
    by_turn = {(row["episode_id"], int(row["replay_turn_index"])): row for row in source}
    for boundary in manifest:
        episode = boundary["episode_id"]
        index = boundary["boundary_index"]
        previous = by_turn[episode, index]
        resumed = by_turn[episode, index + 1]
        assert boundary["proxy_pause_ms"] == float(previous["proxy_pause_ms"])
        assert boundary["required_prefix_tokens"] == int(resumed["target_cached_tokens"])
        assert boundary["lease_key"] == f"{episode}:turn:{index + 1}"


def test_nearest_rank_p95_of_32_is_second_worst() -> None:
    assert nearest_rank(list(range(32)), .95) == 30


def test_sha_rejects_modified_source(tmp_path) -> None:
    altered = tmp_path / "hybrid_workload.csv"
    altered.write_bytes(SOURCE.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="source SHA mismatch"):
        load_source(altered)


def test_exact_policy_and_paired_order() -> None:
    assert POLICIES == {"A": "OnDemand-K2", "D": "FullSlackKV+PSA1-K2"}
    assert ORDER == (("A", "D"), ("D", "A")) * 4
    assert [letter for pair in ORDER for letter in pair].count("A") == 8
    assert [letter for pair in ORDER for letter in pair].count("D") == 8


def test_frozen_parameters() -> None:
    assert P_SAFE_MS == 102
    assert BUDGET_BLOCKS == 3510
    assert BYTES_PER_TOKEN == WORKLOAD_BYTES_PER_TOKEN == 28672


@pytest.mark.parametrize("pause,expected", [
    (100, "100-199"), (199.99, "100-199"), (200, "200-499"),
    (499.99, "200-499"), (500, "500-999"), (999.99, "500-999"),
    (1000, ">=1000"), (2764, ">=1000"),
])
def test_pause_bins(pause: float, expected: str) -> None:
    assert pause_bin(pause) == expected


def test_one_crossing_is_not_a_minimum() -> None:
    manifest = build_manifest(load_source())
    assert sum(int(row["required_prefix_tokens"]) for row in manifest) == 705728
    assert sum(int(row["required_prefix_tokens"]) * BYTES_PER_TOKEN for row in manifest) / 2**20 == 19297.25
    assert duplicate_bytes(0, 0, 4096) == 0
    assert duplicate_bytes(4096 * BYTES_PER_TOKEN, 4096, 4096) == 4096 * BYTES_PER_TOKEN


def test_dynamic_registration_is_causal_and_unique(tmp_path) -> None:
    manifest = build_manifest(load_source())
    reader_path = tmp_path / "server.log"
    controller = DynamicController(manifest, reader_path, time.monotonic_ns())
    previous_finish = time.monotonic_ns() - 1_000_000
    boundary = manifest[0]
    key = str(boundary["boundary_id"])
    prefix = [42] * int(boundary["required_prefix_tokens"])
    deadline = controller.register(key, prefix, previous_finish)
    assert deadline == previous_finish + round(float(boundary["proxy_pause_ms"]) * 1e6)
    assert controller.registered[key]["registered_ns"] >= previous_finish
    with pytest.raises(RuntimeError, match="duplicate"):
        controller.register(key, prefix, previous_finish)


def test_dynamic_registration_rejects_wrong_prefix(tmp_path) -> None:
    manifest = build_manifest(load_source())
    controller = DynamicController(manifest, tmp_path / "server.log", time.monotonic_ns())
    with pytest.raises(ValueError, match="prefix length"):
        controller.register(str(manifest[0]["boundary_id"]), [1], time.monotonic_ns())


def test_exact_source_readiness_trace(tmp_path) -> None:
    path = tmp_path / "server.log"
    event = {"event": "SOURCE_READY", "boundary_id": "E01:b0", "timestamp_monotonic": 1.5}
    path.write_text("noise\nSLACKKV_OFFLOAD_TRACE " + json.dumps(event) + "\n", encoding="utf-8")
    reader = StructuredTraceReader(path)
    offload, leases = reader.read()
    assert offload == [event]
    assert leases == []
    assert reader.read() == ([], [])


def test_bootstrap_reproducible() -> None:
    assert bootstrap([-3.0, -2.0, 1.0, 2.0], 20260928) == bootstrap([-3.0, -2.0, 1.0, 2.0], 20260928)


def test_result_taxonomy() -> None:
    assert classify(True, True, True, True, -15, -10, -10) == "RESULT_A_MULTITURN_DIRECTIONAL_BENEFIT"
    assert classify(True, False, True, True, -5, -10, -10) == "RESULT_B_VALID_MULTITURN_MIXED_TRADEOFF"
    assert classify(True, False, True, True, -2, -2, -2) == "RESULT_C_MULTITURN_NO_MATERIAL_LATENCY_VALUE"
    assert classify(False, True, True, True, -20, -20, -20) == "EXPERIMENT_INVALID_MATRIX"
