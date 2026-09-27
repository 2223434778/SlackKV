from __future__ import annotations

import hashlib
import math
from collections import Counter
from pathlib import Path

import pytest

from experiments.phase4d_a.helpers import BYTES_PER_TOKEN, ReservationLedger
from experiments.phase4e_a_fix.metrics import combined_decode_tok_s
from experiments.phase4f_a.logic import progress_blocks, psa_decision, verified_jobs
from experiments.phase4f_b import metrics as m
from experiments.phase4f_b.finalize_phase4f_b import bootstrap_paired_median, paired_deltas
from experiments.phase4f_b.run_phase4f_b import BUDGET, P_SAFE, VLLM_DIFF_SHA, verify_frozen

ROOT = Path(__file__).resolve().parents[1]


def test_workload_sha_and_job_totals():
    path = ROOT / "results/phase4c/hybrid_workload.csv"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "9b381336a70e73b0efbddbb79d2c88608c19663247e819c1e7a364efd2971105"
    jobs = verified_jobs()
    assert len(jobs) == 8
    assert sum(int(j["required_prefix_tokens"]) for j in jobs) == 168016
    assert sum(int(j["required_blocks"]) for j in jobs) == 10501
    assert m.MINIMUM_H2D_BYTES / m.MIB == 4594.1875
    verify_frozen(jobs)


def test_frozen_runtime_and_policy_names():
    assert m.X_OUTPUT_TOKENS == 512
    assert P_SAFE == 102 and BUDGET == 3510
    assert len(VLLM_DIFF_SHA) == 64
    assert [p.name for p in m.POLICIES] == [
        "OnDemand-K2", "GuardedJIT-K2", "GuardedJIT+Lease+PSA1-K2", "FullSlackKV+PSA1-K2"]
    assert all(p.K2_enabled for p in m.POLICIES)
    assert not m.POLICIES[0].jit_enabled
    assert m.POLICIES[1].jit_enabled and m.POLICIES[1].capacity_guard_enabled
    assert not m.POLICIES[1].physical_lease_enabled and not m.POLICIES[1].psa_enabled
    assert m.POLICIES[2].physical_lease_enabled and m.POLICIES[2].psa_enabled
    assert not m.POLICIES[2].logical_reservation_enabled
    assert m.POLICIES[3].physical_lease_enabled and m.POLICIES[3].psa_enabled
    assert m.POLICIES[3].logical_reservation_enabled
    assert all(not (p.physical_lease_enabled and not p.psa_enabled) for p in m.POLICIES)


def test_williams_order():
    expected = ("ABDC", "BCAD", "CDBA", "DACB") * 2
    names = {p.name: chr(65 + i) for i, p in enumerate(m.POLICIES)}
    assert tuple("".join(names[n] for n in row) for row in m.WILLIAMS_ORDER) == expected
    assert len(m.WILLIAMS_ORDER) * 4 == 32
    for policy in names:
        assert Counter(row.index(policy) for row in m.WILLIAMS_ORDER) == {0: 2, 1: 2, 2: 2, 3: 2}


def test_psa_formula_is_frozen():
    assert progress_blocks(20434) == math.ceil((20434 + 1) / 16)
    assert psa_decision(2073, 1323, 1278, 0, False)[0] == "PROGRESS_REJECT"
    assert psa_decision(4086, 1271, 1278, 0, True)[0] == "ADMIT"


@pytest.mark.parametrize("local,external,expected", [(0, 168016, True), (168016, 0, True), (1, 168014, False)])
def test_source_conservation(local, external, expected):
    assert m.source_conserved(local, external) is expected


def test_h2d_and_duplicate_floor():
    assert BYTES_PER_TOKEN == 28672
    assert m.h2d_bytes(0, 168016)[2:] == (m.MINIMUM_H2D_BYTES, 0)
    assert m.h2d_bytes(100, 0)[3] == 0
    assert m.h2d_bytes(100, 168016)[3] == 100


def test_x_token_tpot_and_combined_throughput():
    assert m.x_output_valid(512) and not m.x_output_valid(256)
    assert m.x_tpot_ms(5110, 512) == 10
    with pytest.raises(ValueError):
        m.x_tpot_ms(2550, 256)
    assert combined_decode_tok_s([10, 20], [0, 0], [1000, 1000], 512, 0, 1000) == 539
    assert m.x_tail_margin_ms(8000, 6000) == 2000
    assert m.measured_background_valid(512, 0.1)
    assert not m.measured_background_valid(512, 0)


def test_latency_and_hypotheses():
    assert m.deadline_latency(99, 100) == 0
    assert m.deadline_latency(110, 100) == 10
    assert m.server_ttft(110, 100) == 10
    assert m.nearest_rank(list(range(1, 9)), 95) == 8
    a = {"deadline_latency_P95": 100, "total_H2D_MiB": 100, "X_slowdown": 1}
    d = {"deadline_latency_P95": 90, "total_H2D_MiB": 105, "X_slowdown": 1.05}
    assert all(m.hypothesis_flags(a, d).values())
    d["deadline_latency_P95"] = 91
    assert not m.hypothesis_flags(a, d)["H1"]


def test_exact_lease_and_reservation_cost():
    assert m.lease_memory_time_mib_s(4096, 100, 1100) == 112
    ledger = ReservationLedger(3510)
    assert ledger.reserve("E04", 1271)
    assert ledger.reserve("E00", 1277)
    assert not ledger.reserve("E01", 1330)
    assert ledger.release("E04") == 1271
    assert ledger.reserve("E01", 1330)


def test_classification_requires_complete_matrix():
    flags = {"H1": True, "H2": True, "H3": True}
    assert m.classify(8, 4, 1, 32, flags) == "RESULT_A_FULL_SLACKKV_DIRECTIONAL_BENEFIT"
    assert m.classify(8, 4, 1, 32, flags | {"H2": False}) == "RESULT_B_VALID_MATRIX_MIXED_TRADEOFF"
    assert m.classify(8, 4, 1, 31, flags) == "EXPERIMENT_INVALID_MATRIX"
    assert m.classify(7, 4, 1, 32, flags) == "EXPERIMENT_INVALID_MATRIX"
    assert m.classify(8, 4, -1, 32, flags) == "EXPERIMENT_INVALID_MATRIX"


def test_bootstrap_is_deterministic():
    sample = [float(i) for i in range(8)]
    assert bootstrap_paired_median(sample) == bootstrap_paired_median(sample)
    assert len(bootstrap_paired_median(sample)) == 2
    with pytest.raises(ValueError):
        bootstrap_paired_median(sample[:7])


def test_no_unsafe_policy_or_later_phase():
    joined = " ".join(p.name for p in m.POLICIES)
    assert "Lease-Oracle" not in joined
    source = (ROOT / "experiments/phase4f_b/run_phase4f_b.py").read_text()
    assert "phase4f_c" not in source.lower()
    assert "phase4f_b" in source
    assert "phase4e_a_fix.run_phase4e_a_fix" not in source
    assert "\"T0\", \"T1\", \"T6\", \"T7\"" in source


def test_paired_delta_and_full_eight_reps():
    rows = []
    for rep in range(1, 9):
        for i, policy in enumerate(m.POLICIES):
            rows.append({"repetition": str(rep), "policy": policy.name,
                         "deadline_latency_P95": str(100 + i), "TTFT_P95": str(50 + i),
                         "total_H2D_MiB": str(5000 + i), "duplicate_H2D_MiB": str(i),
                         "X_slowdown": str(1 + i / 100), "combined_tok_s": str(50 + i),
                         "lease_memory_time_MiB_s": str(i) if i >= 2 else "NA"})
    pairs = paired_deltas(rows)
    assert len(pairs) == 40
    assert len([r for r in pairs if r["comparison"] == "D_vs_A"]) == 8
    assert pairs[0]["P95_absolute_delta_ms"] == 3
