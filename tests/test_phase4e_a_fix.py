from __future__ import annotations

import hashlib
import csv
import json
import math
from pathlib import Path

import pytest

from experiments.phase4d_a.helpers import BYTES_PER_TOKEN
from experiments.phase4d_b.experiment import frozen_jobs
from experiments.phase4e_a_fix.metrics import (
    LATIN_SQUARE, MINIMUM_H2D_BYTES, POLICIES, POLICY_BY_NAME,
    STAGE_A_MARGIN_MS, X_OUTPUT_TOKENS, X_PROMPT_TOKENS,
    classify, combined_decode_tok_s, deadline_latency, h2d_bytes,
    hypothesis_flags, lease_memory_time_mib_s, measured_background_valid,
    smoke_protocol_pass, source_conserved, x_output_valid, x_tail_margin_ms,
    x_tpot_ms,
)
from experiments.phase4e_a_fix.run_phase4e_a_fix import (
    BUDGET, GUARD, P_SAFE, START_COMMIT, VLLM_DIFF_SHA, WORKLOAD_SHA,
    jit_target, verify_frozen,
)

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_inputs() -> None:
    assert hashlib.sha256((ROOT / "results/phase4c/hybrid_workload.csv").read_bytes()).hexdigest() == WORKLOAD_SHA
    jobs = frozen_jobs()
    verify_frozen(jobs)
    assert len(jobs) == 8
    assert sum(j["required_prefix_tokens"] for j in jobs) == 168016
    assert sum(j["required_blocks"] for j in jobs) == 10501
    assert MINIMUM_H2D_BYTES / 2**20 == 4594.1875
    assert BYTES_PER_TOKEN == 28672
    assert (BUDGET, P_SAFE, GUARD) == (3510, 102, 110)
    assert START_COMMIT == "6c870afb2a47620a5b84be8aab0b3bf734ef5fc3"
    assert len(VLLM_DIFF_SHA) == 64


def test_x_protocol_constants_and_no_legacy_runner_formula() -> None:
    runner = (ROOT / "experiments/phase4e_a_fix/run_phase4e_a_fix.py").read_text()
    assert (X_PROMPT_TOKENS, X_OUTPUT_TOKENS, STAGE_A_MARGIN_MS) == (9500, 512, 1000)
    assert "/ 255" not in runner and "+ 255" not in runner
    assert "start_background_512" in runner
    assert "experiments.phase4e_a.run_phase4e_a" not in runner
    assert "results/phase4e_a/" not in runner


@pytest.mark.parametrize("actual,valid", [(512, True), (256, False), (511, False)])
def test_x_output_gate(actual: int, valid: bool) -> None:
    assert x_output_valid(actual) is valid


def test_x_tpot_uses_actual_count() -> None:
    assert x_tpot_ms(1022, 512) == 2
    with pytest.raises(ValueError):
        x_tpot_ms(1022, 256)


def test_combined_throughput_uses_x_output_count() -> None:
    assert combined_decode_tok_s([10, 20], [0, 0], [1000, 1000], 512, 0, 1000) == 539


def test_x_tail_margin_and_gates() -> None:
    assert x_tail_margin_ms(5000, 4665) == 335
    assert smoke_protocol_pass([1000, 1200, 1400, 1600])
    assert not smoke_protocol_pass([999, 1200, 1400, 1600])
    assert not smoke_protocol_pass([1000, 1200, 1400])
    assert measured_background_valid(512, 0.001)
    assert not measured_background_valid(512, 0)
    assert not measured_background_valid(256, 2000)


@pytest.mark.parametrize("name,flags", [
    ("OnDemand-K2", (False, False, False, False, True)),
    ("GuardedJIT-Oracle-K2", (True, True, False, False, True)),
    ("GuardedJIT+Lease-Oracle-K2", (True, True, True, False, True)),
    ("FullSlackKV-Oracle-K2", (True, True, True, True, True)),
])
def test_exact_policies(name: str, flags: tuple[bool, ...]) -> None:
    policy = POLICY_BY_NAME[name]
    assert (policy.jit_enabled, policy.capacity_guard_enabled, policy.physical_lease_enabled, policy.logical_reservation_enabled, policy.K2_enabled) == flags
    assert len(POLICIES) == 4


def test_same_jit_schedule() -> None:
    targets = [jit_target(0.0, {"E04", "E00"}, {"E04": 120.0, "E00": 171.0}, []) for _ in POLICIES[1:]]
    assert targets == [targets[0]] * 3


def test_frozen_latin_square() -> None:
    assert LATIN_SQUARE == (
        tuple(p.name for p in POLICIES),
        tuple(p.name for p in POLICIES[1:]) + (POLICIES[0].name,),
        tuple(p.name for p in POLICIES[2:]) + tuple(p.name for p in POLICIES[:2]),
        (POLICIES[3].name,) + tuple(p.name for p in POLICIES[:3]),
    )
    assert all({row[position] for row in LATIN_SQUARE} == set(POLICY_BY_NAME) for position in range(4))


def test_latency_and_h2d_accounting() -> None:
    assert deadline_latency(130, 120) == 10
    assert deadline_latency(100, 120) == 0
    assert source_conserved(0, 168016)
    assert not source_conserved(0, 168015)
    assert h2d_bytes(0, 168016) == (0, MINIMUM_H2D_BYTES, MINIMUM_H2D_BYTES, 0)
    assert h2d_bytes(100, 168016)[-1] == 100
    assert h2d_bytes(0, 0)[-1] == 0
    assert math.isclose(lease_memory_time_mib_s(4096, 100, 1100), 112)


def test_hypotheses_and_result_taxonomy() -> None:
    a = {"deadline_latency_P95": 100, "total_H2D_MiB": 100, "X_slowdown": 1.0}
    d = {"deadline_latency_P95": 90, "total_H2D_MiB": 105, "X_slowdown": 1.05}
    flags = hypothesis_flags(a, d)
    assert flags == {"H1": True, "H2": True, "H3": True}
    assert classify(4, 4, 1000, 16, flags) == "RESULT_A_FULL_SLACKKV_DIRECTIONAL_BENEFIT"
    assert classify(4, 4, 1000, 16, flags | {"H1": False}) == "RESULT_B_VALID_MATRIX_MIXED_TRADEOFF"
    assert classify(4, 4, 999, 0, flags) == "EXPERIMENT_INVALID_BACKGROUND_PROTOCOL"
    assert classify(4, 4, 1000, 15, flags) == "EXPERIMENT_INVALID_MATRIX"


def _artifact_rows(name: str) -> list[dict[str, str]]:
    path = ROOT / "results/phase4e_a_fix" / name
    if not path.exists():
        pytest.skip("full Phase4E-A-FIX artifacts not complete yet")
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _finished_runs() -> list[dict[str, str]]:
    rows = _artifact_rows("run_metrics.csv")
    if len(rows) != 16:
        pytest.skip("frozen 16-run matrix is still executing")
    return rows


def test_artifact_x_only_fresh_and_exact_count() -> None:
    rows = _artifact_rows("x_only_runs.csv")
    assert len(rows) == 4
    assert {row["repetition"] for row in rows} == {"1", "2", "3", "4"}
    assert all(row["valid"] == "True" and int(row["generated_tokens"]) == X_OUTPUT_TOKENS for row in rows)
    assert all(math.isclose(float(row["X_TPOT_ms"]), float(row["X_decode_ms"]) / (int(row["generated_tokens"]) - 1)) for row in rows)


def test_artifact_stage_a_semantics_and_frozen_margin() -> None:
    rows = _artifact_rows("semantic_smoke.csv")
    if len(rows) != 4:
        pytest.skip("Stage A is still executing")
    assert [row["policy"] for row in rows] == [policy.name for policy in POLICIES]
    assert all(row["valid"] == "True" and int(row["X_generated_tokens"]) == X_OUTPUT_TOKENS for row in rows)
    assert smoke_protocol_pass([float(row["X_tail_margin_ms"]) for row in rows])
    protocol = json.loads((ROOT / "results/phase4e_a_fix/background_protocol.json").read_text())
    assert protocol["protocol_pass"] and protocol["required_smoke_margin_ms"] == STAGE_A_MARGIN_MS


def test_artifact_matrix_exact_counterbalance() -> None:
    rows = _finished_runs()
    assert [(int(row["repetition"]), int(row["run_order_position"]), row["policy"]) for row in rows] == [
        (rep, pos, LATIN_SQUARE[rep - 1][pos - 1]) for rep in range(1, 5) for pos in range(1, 5)
    ]
    assert all(row["valid"] == "True" for row in rows)


def test_artifact_background_and_k2_all_runs() -> None:
    rows = _finished_runs()
    assert all(int(row["X_generated_tokens"]) == X_OUTPUT_TOKENS for row in rows)
    assert all(float(row["X_tail_margin_ms"]) > 0 for row in rows)
    assert all(int(row["K2_max_pre_first"]) <= 2 for row in rows)
    assert all(row["X_active_through_last_first"] == "True" for row in rows)


def test_artifact_agent_tokens_and_identity_diagnostic_only() -> None:
    _finished_runs()
    rows = [row for row in _artifact_rows("session_metrics.csv") if row["stage"] == "measured"]
    assert len(rows) == 128
    assert all(int(row["generated_tokens"]) == int(row["OSL"]) for row in rows)
    assert all(row["identity_match"] in {"True", "False"} for row in rows)


def test_artifact_source_conservation_and_h2d() -> None:
    rows = _finished_runs()
    assert all(int(row["local_source_tokens"]) + int(row["external_source_tokens"]) == 168016 for row in rows)
    assert all(math.isclose(float(row["total_H2D_MiB"]), float(row["promotion_H2D_MiB"]) + float(row["resume_H2D_MiB"])) for row in rows)
    assert all(math.isclose(float(row["duplicate_H2D_MiB"]), max(0, float(row["total_H2D_MiB"]) - MINIMUM_H2D_BYTES / 2**20)) for row in rows)
    assert all(float(row["total_H2D_MiB"]) == 4594.1875 for row in rows if row["policy"] == POLICIES[0].name)


def test_artifact_policy_component_separation() -> None:
    rows = _finished_runs()
    assert all(int(row["promotion_attempts"]) == 0 and row["lease_memory_time_MiB_s"] == "NA" for row in rows if row["policy"] == POLICIES[0].name)
    assert all(row["lease_memory_time_MiB_s"] == "NA" and row["peak_reserved_blocks"] == "NA" for row in rows if row["policy"] == POLICIES[1].name)
    assert all(row["peak_reserved_blocks"] == "NA" for row in rows if row["policy"] == POLICIES[2].name)
    assert all(int(row["peak_reserved_blocks"]) <= BUDGET for row in rows if row["policy"] == POLICIES[3].name)


def test_artifact_no_runtime_faults() -> None:
    rows = _finished_runs()
    assert all(row["server_crash"] == "False" and row["OOM"] == "False" for row in rows)
    assert all(int(row[key]) == 0 for row in rows for key in ("logical_leaks", "lease_leaks", "unsafe_releases"))


def test_artifact_service_overrun_is_not_invalidity() -> None:
    rows = _finished_runs()
    assert all(int(row["service_overrun_count"]) >= 0 for row in rows)
    assert all(row["valid"] == "True" for row in rows)


def test_no_old_matrix_or_next_phase_dependency() -> None:
    source = (ROOT / "experiments/phase4e_a_fix/finalize_phase4e_a_fix.py").read_text()
    assert 'OUT = ROOT / "results/phase4e_a_fix"' in source
    assert "results/phase4e_a/" not in source
    runner = (ROOT / "experiments/phase4e_a_fix/run_phase4e_a_fix.py").read_text()
    assert "phase4e_b" not in runner.lower()


def test_invalid_matrix_is_not_aggregated() -> None:
    result = ROOT / "results/phase4e_a_fix"
    report = json.loads((result / "validation_report.json").read_text())
    assert report["final_result"] == "EXPERIMENT_INVALID_MATRIX"
    assert report["X_only_valid"] == 4 and report["Stage_A_valid"] == 4
    assert report["Stage_B_expected"] == 16 and report["Stage_B_valid"] == 15
    assert all(report[key] == "NOT_EVALUATED" for key in ("H1", "H2", "H3"))
    with (result / "policy_summary.csv").open(newline="") as stream:
        policies = list(csv.DictReader(stream))
    assert len(policies) == 4
    assert all(row["deadline_P95_median"] == "NA" for row in policies)
    with (result / "paired_deltas.csv").open(newline="") as stream:
        assert list(csv.DictReader(stream)) == []


def test_failed_run_is_a_progress_fault_not_background_claim() -> None:
    report = json.loads((ROOT / "results/phase4e_a_fix/validation_report.json").read_text())
    failure = report["failed_run"]
    assert failure["policy"] == "GuardedJIT+Lease-Oracle-K2"
    assert (failure["resume_arrived"], failure["resume_admitted"], failure["first_tokens"], failure["request_finished"]) == (8, 0, 0, 0)
    assert (failure["capacity_admits"], failure["capacity_rejects"], failure["lease_acquired"]) == (3, 5, 3)
    assert failure["server_alive_until_forced_shutdown"]
    assert report["Stage_A_min_X_tail_margin_ms"] >= STAGE_A_MARGIN_MS
    prior = _artifact_rows("run_metrics.csv")
    assert len(prior) == 15
    assert min(float(row["X_tail_margin_ms"]) for row in prior) > 0
