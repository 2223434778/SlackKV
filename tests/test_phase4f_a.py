from __future__ import annotations

from pathlib import Path

import pytest

from experiments.phase4f_a.logic import (
    BLOCK_SIZE, BUDGET_BLOCKS, EXPECTED_PROGRESS, GPU_BLOCKS, K, P_SAFE_MS,
    WORKLOAD_SHA, progress_blocks, protected_episode, psa_decision,
    psa_trace_valid, rejected_response_valid, verified_jobs,
)
from vllm.v1.core.sched.scheduler import _slackkv_psa_decision
from vllm.entrypoints.serve.dev.slackkv_prefetch.api_router import validate_prefetch_status
from experiments.phase4d_a.helpers import ReservationLedger
from experiments.phase4d_a.helpers import sha256_file
from experiments.phase4f_a.run_phase4f_a import unique_resume_counts


@pytest.mark.parametrize("episode,isl,expected", [
    ("E04", 20434, 1278), ("E00", 22885, 1431),
    ("E05", 20661, 1292), ("E02", 21235, 1328),
    ("E01", 21388, 1337), ("E06", 21135, 1321),
    ("E07", 23195, 1450), ("E03", 21728, 1359),
])
def test_progress_footprints(episode, isl, expected):
    assert progress_blocks(isl) == expected == EXPECTED_PROGRESS[episode]


def test_frozen_inputs():
    jobs = verified_jobs()
    assert len(jobs) == 8
    assert GPU_BLOCKS == 4681 and BUDGET_BLOCKS == 3510
    assert P_SAFE_MS == 102 and K == 2 and BLOCK_SIZE == 16
    assert WORKLOAD_SHA == "9b381336a70e73b0efbddbb79d2c88608c19663247e819c1e7a364efd2971105"
    assert sha256_file(Path("results/phase4c/hybrid_workload.csv")) == WORKLOAD_SHA


@pytest.mark.parametrize("free,required,progress,existing,matching,expected", [
    (100, 101, 1278, 0, False, "CAPACITY_REJECT"),
    (2073, 1323, 1278, 0, False, "PROGRESS_REJECT"),
    (2073, 1271, 1278, 0, True, "ADMIT"),
    (2000, 1000, 1278, 0, False, "PROGRESS_REJECT"),
    (3000, 1000, 1278, 0, False, "ADMIT"),
])
def test_psa_decision_matches_engine(free, required, progress, existing, matching, expected):
    assert psa_decision(free, required, progress, existing, matching) == _slackkv_psa_decision(
        free, required, progress, existing, matching
    )
    assert psa_decision(free, required, progress, existing, matching)[0] == expected


def test_candidate_aware_e04_reserve():
    decision, hypothetical, reserve = psa_decision(4680, 1271, 1278, 0, True)
    assert (decision, hypothetical, reserve) == ("ADMIT", 1271, 7)


def test_u3_e02_progress_rejected():
    assert psa_decision(2073, 1323, 1278, 0, False) == (
        "PROGRESS_REJECT", 0, 1278
    )
    assert psa_decision(1478, 1323, 1278, 0, False)[0] == "PROGRESS_REJECT"


def test_rejected_response_zero_work():
    row = {"status": "progress_rejected", "external_tokens_loaded": 0,
           "transfer_bytes": 0, "model_forward_count": 0,
           "generated_token_count": 0, "mixed_prefetch_model_step_count": 0}
    assert rejected_response_valid(row, "progress_rejected")
    assert not rejected_response_valid({**row, "transfer_bytes": 1}, "progress_rejected")
    validate_prefetch_status("progress_rejected", True, True)
    with pytest.raises(RuntimeError):
        validate_prefetch_status("progress_rejected", True, False)


def test_protected_frontier_includes_sent_but_not_first_token():
    jobs = verified_jobs()
    states = {job["episode_id"]: {"first_token_ns": None, "resume_started_ns": 1}
              for job in jobs}
    assert protected_episode(jobs, states) == "E04"
    states["E04"]["first_token_ns"] = 2
    assert protected_episode(jobs, states) == "E00"


@pytest.mark.parametrize("decision,free,required,reserve,valid", [
    ("ADMIT", 2000, 1000, 900, True),
    ("ADMIT", 1500, 1000, 900, False),
    ("PROGRESS_REJECT", 1500, 1000, 900, True),
    ("CAPACITY_REJECT", 500, 1000, 900, True),
    ("CAPACITY_REJECT", 2000, 1000, 900, False),
])
def test_trace_invariant(decision, free, required, reserve, valid):
    event = {"decision": decision, "free_blocks_before": free,
             "candidate_required_blocks": required,
             "free_blocks_after_candidate": free - required,
             "progress_reserve_blocks": reserve}
    assert psa_trace_valid(event) is valid


def test_scheduler_waiting_break_unmodified_in_delta():
    patch = Path("results/phase4f_a/vllm_phase4f_a_delta.patch").read_text()
    assert "_slackkv_psa_decision" in patch
    assert not any(line.startswith("-") and line.strip("- ") in ("break", "continue")
                   for line in patch.splitlines())


def test_no_performance_matrix_invocation():
    source = Path("experiments/phase4f_a/run_phase4f_a.py").read_text()
    assert "H1" not in source and "H2" not in source and "H3" not in source
    assert "run_matrix(" not in source


def test_progress_reject_releases_logical_reservation():
    ledger = ReservationLedger(BUDGET_BLOCKS)
    assert ledger.reserve("E02", 1323)
    response = {"status": "progress_rejected", "external_tokens_loaded": 0,
                "transfer_bytes": 0, "model_forward_count": 0,
                "generated_token_count": 0, "mixed_prefetch_model_step_count": 0}
    assert rejected_response_valid(response, "progress_rejected")
    assert ledger.release("E02") == 1323
    assert ledger.reserved_blocks == 0


def test_p4_requires_eight_repetitions():
    source = Path("experiments/phase4f_a/run_phase4f_a.py").read_text()
    assert "for repetition in range(1, 9):" in source
    assert 'run_jit("P4", repetition, logical_reservation=False' in source


def test_old_capacity_guard_unchanged_when_progress_disabled():
    source = Path("results/phase4f_a/vllm_phase4f_a_delta.patch").read_text()
    assert 'if params.get("progress_guard") is not True:' in source
    assert '"REJECT" if free_blocks < required_blocks else "ADMIT"' in source


def test_repeated_admission_after_preemption_counts_one_request():
    trace = [{"event": event, "request_id": "E07"} for event in (
        "RESUME_ARRIVED", "RESUME_ADMITTED", "FIRST_TOKEN", "RESUME_ADMITTED",
        "REQUEST_FINISHED")]
    assert unique_resume_counts(trace) == {
        "resume_arrived": 1, "resume_admitted": 1,
        "first_tokens": 1, "finished": 1,
    }
