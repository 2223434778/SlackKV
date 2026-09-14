from experiments.phase1b_fix.run_phase1b_fix import closed_loop
from slackkv.simulator.policies import Policy
from slackkv.simulator.transfer_model import TransferModel


def event(release, pause):
    return {"release_epoch_s": release, "pause_s": pause, "reusable_tokens": 1}


def test_closed_loop_propagates_a_session_stall_to_its_next_release():
    model = TransferModel("cpu", "pinned_cpu_to_gpu0", {1: .5}, "test")
    result = closed_loop({"s": [event(0, .1), event(1, .1)]}, {"s": 0.0}, 1, Policy.ON_DEMAND, model)
    assert result.outcomes[0].stall_s == .5
    assert result.outcomes[1].job.release_time_s == 1.5


def test_piecewise_model_chunks_payloads_larger_than_the_largest_knot():
    model = TransferModel("cpu", "pinned_cpu_to_gpu0", {10: 1.0}, "test")
    assert model.piecewise_service_time_s(25) == 2.5
