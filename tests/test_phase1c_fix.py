"""Focused regressions for Phase 1C-Fix's closed-loop promotion loop."""
from pathlib import Path

from experiments.phase1c_fix.run_phase1c_fix import run_sessions
from slackkv.simulator.admission import admit, slack_ratio
from slackkv.simulator.transfer_model import TransferModel


def model():
    return (
        TransferModel("d2h", "gpu0_to_pinned_cpu", {1: 0.2}, "test"),
        TransferModel("h2d", "pinned_cpu_to_gpu0", {1: 0.5}, "test"),
    )


def event(release, pause):
    return {"release_epoch_s": release, "pause_s": pause, "reusable_tokens": 1}


def test_ready_promotion_wakes_before_later_session_release():
    records, _ = run_sessions(
        {"first": [event(0.0, 2.0)], "later": [event(0.0, 2.0)]},
        {"first": 0.0, "later": 10.0}, 1, "OffloadAll-QueueJIT", model()
    )
    first = next(row for row in records if row["sid"] == "first")
    assert first["completion"] <= first["deadline"] + 1e-12


def test_ready_job_completes_without_any_future_session_release():
    records, _ = run_sessions({"only": [event(0.0, 2.0)]}, {"only": 0.0}, 1, "OffloadAll-QueueJIT", model())
    assert len(records) == 1
    assert records[0]["completion"] <= records[0]["deadline"] + 1e-12


def test_multiple_ready_jobs_use_safe_idle_then_edf():
    records, _ = run_sessions(
        {"early": [event(0.0, 1.2)], "late": [event(0.0, 1.5)]},
        {"early": 0.0, "late": 0.0}, 1, "OffloadAll-QueueJIT", model()
    )
    assert all(row["completion"] <= row["deadline"] + 1e-12 for row in records)
    assert records[0]["sid"] == "early"


def test_d2h_and_h2d_directions_are_distinct():
    down, up = model()
    assert down.direction == "gpu0_to_pinned_cpu"
    assert up.direction == "pinned_cpu_to_gpu0"
    assert down.direction != up.direction


def test_c0_rejects_every_isolated_infeasible_slackratio_admission():
    for gamma in (1.0, 1.25, 1.5, 2.0, 3.0, 4.0):
        policy = f"SlackRatio-{gamma:g}"
        assert not admit(policy, 0.69, 0.2, 0.5)
        assert slack_ratio(0.69, 0.2, 0.5) < 1.0
