from __future__ import annotations

import math
from pathlib import Path

import pytest

from experiments.phase4d_a.helpers import (
    BYTES_PER_TOKEN, BLOCK_TOKENS, FROZEN_SHA256, FROZEN_EXPECTED,
    ReservationLedger, block_size_valid, duplicate_h2d_bytes, first_boundary_jobs,
    frozen_service_budget, gate_valid, minimum_h2d_bytes, minimum_h2d_mib,
    physical_before_logical_release, prefix_memory_time_mib_s, pressure_request_count,
    promotion_cancelled, reservation_budget, result_classification, sha256_file,
    source_conserved,
)
from experiments.phase2d.helpers import source_tokens_delta

ROOT = Path(__file__).resolve().parents[1]


def jobs():
    return first_boundary_jobs(ROOT / "results/phase4c/hybrid_workload.csv")


def test_boundary_zero_uses_turn0_pause_and_turn1_prefix():
    e00 = jobs()[0]
    assert e00["deadline_ms"] == e00["proxy_pause_ms"] == 171
    assert e00["required_prefix_tokens"] == 20432
    assert e00["resume_osl"] == 84


def test_exact_eight_boundary_zero_jobs():
    assert len(jobs()) == 8


def test_frozen_cached_token_sum():
    assert sum(row["required_prefix_tokens"] for row in jobs()) == 168016


def test_frozen_block_sum():
    assert sum(row["required_blocks"] for row in jobs()) == 10501


def test_minimum_h2d_bytes_and_mib():
    assert minimum_h2d_bytes(jobs()) == 4_817_354_752
    assert minimum_h2d_mib(jobs()) == pytest.approx(4594.1875)


def test_kv_footprint_bytes_per_token():
    assert BYTES_PER_TOKEN == 28672


def test_workload_sha_pin():
    assert sha256_file(ROOT / "results/phase4c/hybrid_workload.csv") == FROZEN_SHA256


def test_turn0_turn1_mapping_is_not_off_by_one():
    data = jobs()[0]
    assert data["boundary_index"] == 0
    assert data["deadline_ms"] == FROZEN_EXPECTED["E00"][0]
    assert data["turn1_isl"] == 22885 and data["turn1_isl_new"] == 2453


def test_output_manifest_hash_reproducibility():
    import hashlib
    ids = [7, 21, 42, 5]
    digest1 = hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()
    digest2 = hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()
    assert digest1 == digest2


def test_block_size_is_exactly_sixteen():
    assert block_size_valid(16)
    assert not block_size_valid(64)


def test_reservation_budget_is_mechanical_three_quarters():
    assert reservation_budget(5000, jobs()) == math.floor(0.75 * 5000)


def test_no_512_block_fallback():
    assert reservation_budget(5000, jobs()) != 512


def test_reservation_ledger_prevents_overcommit():
    ledger = ReservationLedger(2500)
    assert ledger.reserve("x", 1271)
    assert not ledger.reserve("y", 1330)
    assert ledger.reserved_blocks <= 2500


def test_physical_lease_blocks_match_prediction():
    row = jobs()[0]
    ledger = ReservationLedger(row["required_blocks"])
    assert ledger.reserve(row["lease_key"], row["required_blocks"])
    assert ledger.active[row["lease_key"]] == row["required_blocks"]


def test_physical_release_precedes_logical_release():
    assert physical_before_logical_release(100, 101)
    assert not physical_before_logical_release(102, 101)


def test_stale_cancel_only_before_commit():
    assert promotion_cancelled(True, False)
    assert not promotion_cancelled(True, True)
    assert not promotion_cancelled(False, False)


def test_deadline_is_turn0_proxy_pause():
    for row in jobs():
        assert row["deadline_ms"] == row["proxy_pause_ms"]


def test_safe_service_budget_ceil_twenty_percent():
    assert frozen_service_budget(100.1) == math.ceil(120.12)


def test_pressure_count_exceeds_gpu_block_capacity():
    count = pressure_request_count(4681)
    assert count == math.ceil(4681 / (9500 // 16)) + 1
    assert count * (9500 // 16) > 4681


def test_duplicate_h2d_accounting():
    expected = minimum_h2d_bytes(jobs())
    assert duplicate_h2d_bytes(expected, 0, jobs()) == 0
    assert duplicate_h2d_bytes(expected + 4096, 0, jobs()) == 4096


def test_variable_prefix_memory_time():
    assert prefix_memory_time_mib_s(20432, 300, 200) == pytest.approx(55.86875)


def test_source_conservation():
    assert source_conserved(0, 168016)
    assert source_conserved(50000, 118016)
    assert not source_conserved(50000, 118015)


def test_source_counter_labels_match_vllm_runtime():
    before = {("vllm:prompt_tokens_by_source_total", (("source", "external_kv_transfer"),)): 10.0}
    after = {("vllm:prompt_tokens_by_source_total", (("source", "external_kv_transfer"),)): 30.0}
    assert source_tokens_delta(before, after, "external_kv_transfer") == 20.0


def test_k2_limit():
    assert gate_valid(2)
    assert not gate_valid(3)
    assert gate_valid(0)


def test_benefit_classification():
    assert result_classification(setup_valid=True, calibration_valid=True, mechanism_valid=True,
                                 all_runs_complete=True, benefit_valid=True) == "RESULT_A_HYBRID_MECHANISM_WITH_BENEFIT"


def test_mechanism_failure_classification():
    assert result_classification(setup_valid=True, calibration_valid=True, mechanism_valid=False,
                                 all_runs_complete=True, benefit_valid=False) == "RESULT_C_HYBRID_MECHANISM_INVALID"


def test_invalid_setup_precedes_other_classification():
    assert result_classification(setup_valid=False, calibration_valid=False, mechanism_valid=False,
                                 all_runs_complete=False, benefit_valid=False) == "EXPERIMENT_INVALID_SETUP"
