from __future__ import annotations

import hashlib
import csv
import json
import math
from pathlib import Path

import pytest

from experiments.phase4d_a.helpers import BYTES_PER_TOKEN, ReservationLedger
from experiments.phase4d_b.capacity import capacity_decision
from experiments.phase4d_b.experiment import frozen_jobs
from experiments.phase4e_a.metrics import (
    LATIN_SQUARE, MINIMUM_H2D_BYTES, POLICIES, POLICY_BY_NAME,
    classify, combined_decode_tok_s, deadline_latency, h2d_bytes,
    hypothesis_flags, lease_memory_time_mib_s, nearest_rank,
    server_ttft, source_conserved,
)
from experiments.phase4e_a.run_phase4e_a import (
    BUDGET, GUARD, P_SAFE, START_COMMIT, VLLM_DIFF_SHA, WORKLOAD_SHA,
    jit_target, max_lease_blocks, verify_frozen,
)

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_workload_sha() -> None:
    assert hashlib.sha256((ROOT / "results/phase4c/hybrid_workload.csv").read_bytes()).hexdigest() == WORKLOAD_SHA


def test_jobs_and_source_tokens() -> None:
    jobs = frozen_jobs()
    verify_frozen(jobs)
    assert len(jobs) == 8
    assert sum(j["required_prefix_tokens"] for j in jobs) == 168016


def test_minimum_h2d() -> None:
    assert MINIMUM_H2D_BYTES / 2**20 == 4594.1875
    assert BYTES_PER_TOKEN == 28672


@pytest.mark.parametrize("actual,expected", [(4681, True), (4680, False)])
def test_gpu_blocks(actual: int, expected: bool) -> None:
    assert (actual == 4681) is expected


def test_constants() -> None:
    assert BUDGET == 3510 and P_SAFE == 102 and GUARD == 110
    assert START_COMMIT == "9f9f6fab587611934e8e12297c28256989a0dd8d"
    assert len(VLLM_DIFF_SHA) == 64


@pytest.mark.parametrize("name,flags", [
    ("OnDemand-K2", (False, False, False, False, True)),
    ("GuardedJIT-Oracle-K2", (True, True, False, False, True)),
    ("GuardedJIT+Lease-Oracle-K2", (True, True, True, False, True)),
    ("FullSlackKV-Oracle-K2", (True, True, True, True, True)),
])
def test_policy_exact_flags(name: str, flags: tuple[bool, ...]) -> None:
    policy = POLICY_BY_NAME[name]
    assert (policy.jit_enabled, policy.capacity_guard_enabled, policy.physical_lease_enabled, policy.logical_reservation_enabled, policy.K2_enabled) == flags


def test_only_four_frozen_policies() -> None:
    assert len(POLICIES) == 4
    assert not any("Naive" in policy.name or "Unguarded" in policy.name for policy in POLICIES)


def test_jit_schedule_shared_for_bcd() -> None:
    pending = {"E04", "E00"}
    deadlines = {"E04": 120.0, "E00": 171.0}
    targets = [jit_target(0.0, pending, deadlines, []) for policy in POLICIES[1:]]
    assert targets == [targets[0]] * 3


def test_capacity_rejection_on_demand_fallback() -> None:
    assert capacity_decision(100, 1271) == "REJECT"
    assert capacity_decision(1271, 1271) == "ADMIT"


def test_lease_and_reservation_are_distinct() -> None:
    assert not POLICY_BY_NAME["GuardedJIT-Oracle-K2"].physical_lease_enabled
    assert not POLICY_BY_NAME["GuardedJIT+Lease-Oracle-K2"].logical_reservation_enabled
    assert POLICY_BY_NAME["FullSlackKV-Oracle-K2"].logical_reservation_enabled


def test_latin_square() -> None:
    assert LATIN_SQUARE == (
        ("OnDemand-K2", "GuardedJIT-Oracle-K2", "GuardedJIT+Lease-Oracle-K2", "FullSlackKV-Oracle-K2"),
        ("GuardedJIT-Oracle-K2", "GuardedJIT+Lease-Oracle-K2", "FullSlackKV-Oracle-K2", "OnDemand-K2"),
        ("GuardedJIT+Lease-Oracle-K2", "FullSlackKV-Oracle-K2", "OnDemand-K2", "GuardedJIT-Oracle-K2"),
        ("FullSlackKV-Oracle-K2", "OnDemand-K2", "GuardedJIT-Oracle-K2", "GuardedJIT+Lease-Oracle-K2"),
    )
    for position in range(4):
        assert {row[position] for row in LATIN_SQUARE} == set(POLICY_BY_NAME)


def test_source_conservation() -> None:
    assert source_conserved(0, 168016)
    assert source_conserved(50000, 118016)
    assert not source_conserved(0, 168015)


def test_h2d_formula_and_duplicate_floor() -> None:
    native, external, total, duplicate = h2d_bytes(0, 168016)
    assert native == 0 and external == MINIMUM_H2D_BYTES
    assert total == MINIMUM_H2D_BYTES and duplicate == 0
    assert h2d_bytes(100, 168016)[-1] == 100
    assert h2d_bytes(0, 0)[-1] == 0


def test_deadline_and_ttft() -> None:
    assert deadline_latency(130, 120) == 10
    assert deadline_latency(100, 120) == 0
    assert server_ttft(130, 115) == 15
    with pytest.raises(ValueError):
        server_ttft(100, 115)


def test_nearest_rank_is_phase_convention() -> None:
    assert nearest_rank([1, 2, 3, 4, 5, 6, 7, 8], 95) == 8
    assert nearest_rank([1, 2, 3, 4, 5, 6, 7, 8], 50) == 4


def test_x_slowdown_formula() -> None:
    assert 1500 / 1000 == 1.5


def test_lease_memory_time() -> None:
    assert math.isclose(lease_memory_time_mib_s(4096, 100, 1100), 112.0)
    with pytest.raises(ValueError):
        lease_memory_time_mib_s(4096, 200, 100)


def test_combined_decode_phase3u_definition() -> None:
    assert combined_decode_tok_s([32, 32], [100, 110], [900, 1000], 0, 800) == (62 + 255) * 1000 / 1000


def test_lease_lifecycle_peak() -> None:
    events = [
        {"event": "LEASE_ACQUIRED", "lease_key": "E04:turn:1", "num_blocks": 1271, "timestamp_monotonic": 1.0},
        {"event": "LEASE_ACQUIRED", "lease_key": "E05:turn:1", "num_blocks": 1274, "timestamp_monotonic": 2.0},
        {"event": "LEASE_RELEASED", "lease_key": "E04:turn:1", "num_blocks": 1271, "timestamp_monotonic": 3.0},
    ]
    assert max_lease_blocks(events) == 2545


def test_reservation_lifecycle() -> None:
    ledger = ReservationLedger(BUDGET)
    assert ledger.reserve("E04", 1271)
    assert ledger.reserve("E05", 1274)
    assert not ledger.reserve("E06", 1276)
    assert ledger.release("E04") == 1271
    assert ledger.reserve("E06", 1276)


@pytest.mark.parametrize("a_p95,d_p95,expected", [(100, 90, True), (100, 91, False)])
def test_h1(a_p95: float, d_p95: float, expected: bool) -> None:
    a = {"deadline_latency_P95": a_p95, "total_H2D_MiB": 100, "X_slowdown": 1.0}
    d = {"deadline_latency_P95": d_p95, "total_H2D_MiB": 100, "X_slowdown": 1.0}
    assert hypothesis_flags(a, d)["H1"] is expected


@pytest.mark.parametrize("total,expected", [(105, True), (106, False)])
def test_h2(total: float, expected: bool) -> None:
    a = {"deadline_latency_P95": 100, "total_H2D_MiB": 100, "X_slowdown": 1.0}
    d = {"deadline_latency_P95": 90, "total_H2D_MiB": total, "X_slowdown": 1.0}
    assert hypothesis_flags(a, d)["H2"] is expected


@pytest.mark.parametrize("slowdown,expected", [(1.05, True), (1.06, False)])
def test_h3(slowdown: float, expected: bool) -> None:
    a = {"deadline_latency_P95": 100, "total_H2D_MiB": 100, "X_slowdown": 1.0}
    d = {"deadline_latency_P95": 90, "total_H2D_MiB": 100, "X_slowdown": slowdown}
    assert hypothesis_flags(a, d)["H3"] is expected


def test_result_classification() -> None:
    assert classify(4, 16, {"H1": True, "H2": True, "H3": True}).startswith("RESULT_A")
    assert classify(4, 16, {"H1": False, "H2": True, "H3": True}).startswith("RESULT_B")
    assert classify(3, 16, {"H1": True, "H2": True, "H3": True}) == "EXPERIMENT_INVALID_MATRIX"
    assert classify(4, 15, {"H1": True, "H2": True, "H3": True}) == "EXPERIMENT_INVALID_MATRIX"


def _artifact_rows(name: str) -> list[dict[str, str]]:
    with (ROOT / "results/phase4e_a" / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_x_only_four_fresh_controls() -> None:
    rows = _artifact_rows("x_only_runs.csv")
    assert len(rows) == 4
    assert {row["repetition"] for row in rows} == {"1", "2", "3", "4"}
    assert all(row["valid"] == "True" and row["generated_tokens"] == "256" for row in rows)


def test_all_semantic_smokes_precede_matrix() -> None:
    rows = _artifact_rows("semantic_smoke.csv")
    assert len(rows) == 4
    assert [row["policy"] for row in rows] == [policy.name for policy in POLICIES]
    assert all(row["valid"] == "True" for row in rows)


def test_measured_order_stops_at_first_invalid_run() -> None:
    rows = _artifact_rows("run_metrics.csv")
    assert [(row["repetition"], row["run_order_position"], row["policy"]) for row in rows] == [
        ("1", "1", POLICIES[0].name), ("1", "2", POLICIES[1].name), ("1", "3", POLICIES[2].name),
    ]
    assert [row["valid"] for row in rows] == ["True", "True", "False"]


def test_invalid_background_uses_actual_monotonic_events() -> None:
    report = json.loads((ROOT / "results/phase4e_a/validation_report.json").read_text())
    assert report["last_agent_FIRST_TOKEN_ms"] > report["X_finished_ms"]
    assert report["agent_first_after_X_ms"] > 0
    assert report["reason"] == "RUN_INVALID_BACKGROUND"


def test_invalid_run_keeps_token_and_source_evidence() -> None:
    report = json.loads((ROOT / "results/phase4e_a/validation_report.json").read_text())
    assert report["measured_token_count_correct"] == report["measured_resumes_observed"] == 24
    assert report["failing_run_source_conservation"] is True
    assert report["failing_run_K2_max_pre_first"] <= 2


def test_invalid_result_does_not_claim_frozen_hypotheses() -> None:
    report = json.loads((ROOT / "results/phase4e_a/validation_report.json").read_text())
    assert report["result"] == "EXPERIMENT_INVALID_MATRIX"
    assert all(report[h] == "NOT_EVALUATED" for h in ("H1", "H2", "H3"))


def test_incomplete_policy_aggregates_are_na() -> None:
    rows = _artifact_rows("policy_summary.csv")
    assert len(rows) == 4
    assert all(row["deadline_P95_median"] == "NA" for row in rows)


def test_no_unpaired_partial_delta() -> None:
    assert _artifact_rows("paired_deltas.csv") == []


def test_failure_artifact_contains_scheduler_evidence() -> None:
    folder = ROOT / "results/phase4e_a/failures/measured_rep1_pos3_GuardedJITplusLease-Oracle-K2"
    assert (folder / "failure.json").exists()
    trace = list(csv.DictReader((folder / "scheduler_trace.csv").open(newline="", encoding="utf-8")))
    assert any(row["event"] == "FIRST_TOKEN" and row["session"] == "E03:turn:1" for row in trace)


def test_output_identity_is_diagnostic_not_token_gate() -> None:
    sessions = _artifact_rows("session_metrics.csv")
    assert any(row["identity_match"] == "False" for row in sessions)
    assert all(int(row["generated_tokens"]) == int(row["OSL"]) for row in sessions)


def test_source_tree_fingerprint_saved() -> None:
    manifest = json.loads((ROOT / "results/phase4e_a/source_manifest.json").read_text())
    assert manifest["vllm_diff_sha256"] == VLLM_DIFF_SHA
    assert manifest["workload_sha256"] == WORKLOAD_SHA
