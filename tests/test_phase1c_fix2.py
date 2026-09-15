"""Deterministic event-timing and residency regressions for Phase 1C-Fix2."""
from experiments.phase1c_fix2.run_phase1c_fix2 import run_sessions
from slackkv.simulator.selective_offload import local_metrics
from slackkv.simulator.transfer_model import TransferModel


def model(down_times={1: 0.2, 2: 1.0}, up_times={1: 0.5, 2: 0.5}):
    return (
        TransferModel("d2h", "gpu0_to_pinned_cpu", down_times, "test"),
        TransferModel("h2d", "pinned_cpu_to_gpu0", up_times, "test"),
    )


def event(release, pause, tokens=1):
    return {"release_epoch_s": release, "pause_s": pause, "reusable_tokens": tokens}


def single(event_value, transfer_model=None):
    return run_sessions({"only": [event_value]}, {"only": 0.0}, 1, "OffloadAll-QueueJIT", transfer_model or model())


def test_a_isolated_ready_job_idles_to_latest_safe_start():
    records, diagnostics = single(event(0.0, 2.0))
    assert records[0]["ready_time"] == 0.2
    assert records[0]["start"] == 1.5
    assert records[0]["completion"] == 2.0
    assert diagnostics["safe_idle_expiry_wakeups"] == 1


def test_b_new_d2h_completion_interrupts_intentional_idle():
    records, diagnostics = run_sessions(
        {"a": [event(0.0, 3.0, 1)], "b": [event(0.0, 1.8, 2)]},
        {"a": 0.0, "b": 0.0}, 1, "OffloadAll-QueueJIT", model()
    )
    by_session = {record["sid"]: record for record in records}
    assert diagnostics["promotion_ready_wakeups"] >= 2
    # B's arrival at 1.0 s wakes the loop; recomputed cumulative slack permits
    # another 0.3 s of intentional idle before EDF dispatch.
    assert by_session["b"]["start"] == 1.3
    assert by_session["b"]["completion"] == 1.8


def test_c_session_release_interrupts_intentional_idle():
    records, diagnostics = run_sessions(
        {"a": [event(0.0, 3.0, 1)], "b": [event(0.0, 4.0, 2)]},
        {"a": 0.0, "b": 1.0}, 1, "OffloadAll-QueueJIT", model()
    )
    assert diagnostics["session_release_wakeups"] >= 1
    assert next(record for record in records if record["sid"] == "a")["start"] > 1.0


def test_d_safe_idle_expiry_wins_without_external_event():
    records, diagnostics = single(event(0.0, 2.0))
    assert diagnostics["intentional_idle_events"] == 1
    assert diagnostics["intentional_idle_time_total"] == 1.3
    assert records[0]["start"] == 1.5


def test_e_multiple_ready_jobs_use_prefix_slack_and_edf():
    records, _ = run_sessions(
        {"early": [event(0.0, 1.2)], "late": [event(0.0, 1.5)]},
        {"early": 0.0, "late": 0.0}, 1, "OffloadAll-QueueJIT", model({1: 0.2}, {1: 0.5})
    )
    assert [record["sid"] for record in records] == ["early", "late"]
    assert records[0]["start"] == 0.5
    assert records[1]["start"] == 1.0
    assert all(record["completion"] <= record["deadline"] for record in records)


def test_f_closed_loop_stall_shifts_only_its_session():
    records, _ = run_sessions(
        {"delayed": [event(0.0, 0.4), event(2.0, 1.0)], "other": [event(0.0, 1.0)]},
        {"delayed": 0.0, "other": 0.0}, 1, "OffloadAll-QueueJIT", model({1: 0.2}, {1: 0.5})
    )
    delayed = [record for record in records if record["sid"] == "delayed"]
    other = next(record for record in records if record["sid"] == "other")
    assert abs(delayed[0]["stall"] - 0.3) < 1e-12
    assert abs(delayed[1]["release"] - 2.3) < 1e-12
    assert other["release"] == 0.0


def test_d2h_residency_is_not_truncated_at_deadline():
    metrics = local_metrics([{"release": 0.0, "deadline": 1.0, "pause": 1.0, "size_gb": 2.0,
                               "size": 2, "down": 1.5, "up": 0.5, "offload": True,
                               "completion": 2.0, "stall": 1.0}])
    assert metrics["local_paused_vram_time_gbs_total"] == 3.0
