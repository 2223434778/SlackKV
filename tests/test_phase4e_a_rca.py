from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.phase4e_a_rca.analysis import (
    CONDITIONS, EXPECTED_BLOCKS, GPU_BLOCKS, WORKLOAD_SHA, condition_blocks,
    final_result, first_e04_block, lease_unique_blocks, parse_structured_log,
    r1_result, should_emit_blocked, u3_nonprogress, verify_jobs,
)
from experiments.phase4e_a_rca.run_phase4e_a_rca import U3_OBSERVE_S, WATCHDOG_S


def test_frozen_workload_and_head():
    jobs = verify_jobs()
    assert len(jobs) == 8
    assert GPU_BLOCKS == 4681
    assert WORKLOAD_SHA == "9b381336a70e73b0efbddbb79d2c88608c19663247e819c1e7a364efd2971105"
    assert sum(int(job["required_blocks"]) for job in jobs) == 10501
    assert EXPECTED_BLOCKS["E04"] == 1271
    assert min(jobs, key=lambda job: float(job["deadline_ms"]))["episode_id"] == "E04"


@pytest.mark.parametrize("name,expected", [
    ("U3_UNLEASED_HEAD_LEASE3", 3930),
    ("H2_HEADROOM_LEASE2", 2607),
    ("N3_EVICTABLE_PREFETCH3", 3930),
    ("S3_HEAD_LEASED", 3822),
])
def test_condition_sets(name, expected):
    assert condition_blocks(name) == expected
    assert EXPECTED_BLOCKS["E04"] in (1271,)


def test_orthogonal_controls():
    assert CONDITIONS["U3_UNLEASED_HEAD_LEASE3"] == (("E00", "E01", "E02"), True)
    assert CONDITIONS["H2_HEADROOM_LEASE2"] == (("E00", "E01"), True)
    assert CONDITIONS["N3_EVICTABLE_PREFETCH3"] == (("E00", "E01", "E02"), False)
    assert CONDITIONS["S3_HEAD_LEASED"] == (("E04", "E00", "E05"), True)
    assert U3_OBSERVE_S == 12


def test_power_of_two_logging():
    assert [n for n in range(1, 10) if should_emit_blocked(n)] == [1, 2, 4, 8]


def test_trace_parser(tmp_path: Path):
    path = tmp_path / "log"
    event = {"event": "WAITING_ALLOCATE_NONE", "session": "E04:turn:1", "will_break_waiting_loop": True}
    path.write_text("x SLACKKV_WAITING_TRACE " + json.dumps(event) + "\nnot json\n")
    assert parse_structured_log(path, "SLACKKV_WAITING_TRACE") == [event]
    assert first_e04_block([event]) == event


def test_lease_block_dedup():
    assert lease_unique_blocks({"one": [1, 2], "two": [2, 3]}) == 3


def _u3_evidence():
    waiting = [{"event": "WAITING_ALLOCATE_NONE", "session": "E04:turn:1",
                "matching_lease": False, "will_break_waiting_loop": True,
                "free_blocks": 751, "active_lease_unique_blocks": 3930}]
    resume = [{"event": "RESUME_ARRIVED", "session": "E04:turn:1"}]
    lease = [{"event": "LEASE_ACQUIRED", "lease_key": f"{episode}:turn:1"}
             for episode in ("E00", "E01", "E02")]
    return waiting, resume, lease


def test_u3_direct_nonprogress_and_invalid_crash():
    waiting, resume, lease = _u3_evidence()
    kwargs = dict(alive=True, oom=False, server_error=False, waiting=waiting,
                  resume=resume, lease=lease, final_elapsed_s=12.0)
    assert u3_nonprogress(**kwargs)
    assert not u3_nonprogress(**{**kwargs, "alive": False})
    assert not u3_nonprogress(**{**kwargs, "oom": True})
    assert not u3_nonprogress(**{**kwargs, "final_elapsed_s": 5})
    assert not u3_nonprogress(**{**kwargs, "lease": lease + [{"event": "LEASE_RELEASED"}]})
    assert not u3_nonprogress(**{**kwargs, "waiting": [{**waiting[0], "will_break_waiting_loop": False}]})


def test_result_gate():
    names = list(CONDITIONS)
    expected = {names[0]: ["LEASE_HOL_NONPROGRESS"] * 2,
                **{name: ["PROGRESS"] * 2 for name in names[1:]}}
    assert r1_result(expected) == "R1_EXPECTED_PATTERN"
    assert final_result("R1_EXPECTED_PATTERN", ["LEASE_HOL_NONPROGRESS"] * 3,
                        ["PROGRESS"] * 3) == "RESULT_A_LEASE_HOL_CIRCULAR_WAIT_CONFIRMED"
    assert r1_result({names[0]: ["PROGRESS"] * 2}) == "RESULT_C_HYPOTHESIS_NOT_SUPPORTED"
    assert r1_result({**expected, names[1]: ["OTHER_NONPROGRESS"] * 2}) == "RESULT_B_ROOT_CAUSE_PARTIAL"


def test_trace_patch_is_additive():
    patch = Path("results/phase4e_a_rca/vllm_rca_trace_only.patch").read_text()
    changed = [line for line in patch.splitlines() if line.startswith("-") and not line.startswith("---")]
    assert not changed, "RCA scheduler patch must not remove or replace control flow"
    assert not any(line.lstrip("+ ").strip() in ("break", "continue")
                   for line in patch.splitlines() if line.startswith("+"))
    assert "WAITING_ALLOCATE_NONE" in patch
    assert "will_break_waiting_loop" in patch


def test_no_performance_matrix_or_progress_fix():
    source = Path("experiments/phase4e_a_rca/run_phase4e_a_rca.py").read_text()
    assert "run_matrix(" not in source
    assert "latest_feasible_start(" not in source
    assert "progress_reserve" not in source
    assert "capacity_prefetch(" in source


def test_sequential_precondition_before_t0_and_no_later_promotion():
    source = Path("experiments/phase4e_a_rca/run_phase4e_a_rca.py").read_text()
    precondition = source.index("for episode in selected:")
    promotion = source.index("result = capacity_prefetch(", precondition)
    resume_t0 = source.index("t0_s = time.monotonic()", promotion)
    launch = source.index("resume_pool = ThreadPoolExecutor", resume_t0)
    assert precondition < promotion < resume_t0 < launch
    assert source.find("capacity_prefetch(", resume_t0) == -1


def test_observation_checkpoints_and_watchdog():
    source = Path("experiments/phase4e_a_rca/run_phase4e_a_rca.py").read_text()
    assert "wait_until(t0_s + 5)" in source
    assert "wait_until(t0_s + U3_OBSERVE_S)" in source
    assert U3_OBSERVE_S == 12
    assert WATCHDOG_S == 20


@pytest.mark.parametrize("stage,condition,repetitions", [
    ("R1", "U3_UNLEASED_HEAD_LEASE3", (1, 2)),
    ("R1", "H2_HEADROOM_LEASE2", (1, 2)),
    ("R1", "N3_EVICTABLE_PREFETCH3", (1, 2)),
    ("R1", "S3_HEAD_LEASED", (1, 2)),
    ("R2", "U3_UNLEASED_HEAD_LEASE3", (1, 2, 3)),
    ("R2", "S3_HEAD_LEASED", (1, 2, 3)),
])
def test_frozen_repetition_plan(stage, condition, repetitions):
    assert stage in ("R1", "R2")
    assert condition in CONDITIONS
    assert len(repetitions) == (2 if stage == "R1" else 3)


def test_u3_must_have_direct_waiting_break():
    waiting, resume, lease = _u3_evidence()
    waiting[0]["session"] = "E00:turn:1"
    assert not u3_nonprogress(alive=True, oom=False, server_error=False,
                              waiting=waiting, resume=resume, lease=lease,
                              final_elapsed_s=12)


def test_result_a_requires_all_three_r2_repetitions():
    assert final_result("R1_EXPECTED_PATTERN", ["LEASE_HOL_NONPROGRESS"] * 2,
                        ["PROGRESS"] * 3) != "RESULT_A_LEASE_HOL_CIRCULAR_WAIT_CONFIRMED"


def test_final_artifact_gate():
    root = Path("results/phase4e_a_rca")
    gate = json.loads((root / "stage_gate.json").read_text())
    assert gate["R1"] == "R1_EXPECTED_PATTERN"
    assert gate["R2_U3"] == ["LEASE_HOL_NONPROGRESS"] * 3
    assert gate["R2_S3"] == ["PROGRESS"] * 3
    assert gate["final_result"] == "RESULT_A_LEASE_HOL_CIRCULAR_WAIT_CONFIRMED"
    import csv
    with (root / "liveness_runs.csv").open(newline="") as stream:
        runs = list(csv.DictReader(stream))
    assert len(runs) == 14
    assert all(row["server_alive"] == "True" and row["OOM"] == "False"
               and row["server_error"] == "False" for row in runs)
    u3 = [row for row in runs if row["condition"] == "U3_UNLEASED_HEAD_LEASE3"]
    assert len(u3) == 5
    assert all(row["first_resume_session"] == "E04:turn:1"
               and row["resume_arrived_final"] == "8"
               and row["resume_admitted_final"] == "0"
               and row["first_tokens_final"] == "0"
               and row["finished_final"] == "0"
               and row["lease_released_count"] == "0" for row in u3)
    s3 = [row for row in runs if row["condition"] == "S3_HEAD_LEASED"]
    assert len(s3) == 5
    assert all(row["finished_final"] == "8"
               and row["lease_released_count"] == "3" for row in s3)
