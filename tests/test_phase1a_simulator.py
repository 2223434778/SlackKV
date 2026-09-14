from slackkv.simulator.jobs import PromotionJob
from slackkv.simulator.policies import Policy
from slackkv.simulator.simulator import aggregate, simulate
from slackkv.simulator.transfer_model import TransferModel
from experiments.phase1a.generate_workloads import stochastic_stream


def job(name, release, deadline, service):
    return PromotionJob(name, release, deadline, 256 << 20, service)


def outcomes(jobs, policy):
    return {outcome.job.job_id: outcome for outcome in simulate(jobs, policy).outcomes}


def test_one_job_queuejit_starts_at_latest_start():
    result = outcomes([job("a", 0.0, 1.0, .25)], Policy.QUEUE_JIT)["a"]
    assert abs(result.start_time_s - .75) < 1e-12
    assert abs(result.completion_time_s - 1.0) < 1e-12


def test_two_nonconflicting_jobs_finish_before_deadline():
    result = outcomes([job("a", 0, 1, .2), job("b", 0, 2, .2)], Policy.QUEUE_JIT)
    assert result["a"].completion_time_s <= 1 and result["b"].completion_time_s <= 2


def test_queuejit_avoids_feasible_naive_latest_start_collision():
    jobs = [job("a", 0, .8, .4), job("b", 0, .9, .4)]
    naive = aggregate(simulate(jobs, Policy.NAIVE_JIT))
    queue = aggregate(simulate(jobs, Policy.QUEUE_JIT))
    assert naive["deadline_miss_count"] == 1
    assert queue["deadline_miss_count"] == 0


def test_infeasible_queuejit_records_miss_not_impossible_timestamps():
    result = simulate([job("a", 0, .5, .8), job("b", 0, .6, .8)], Policy.QUEUE_JIT)
    assert aggregate(result)["deadline_miss_count"] == 2
    assert all(outcome.completion_time_s >= outcome.start_time_s for outcome in result.outcomes)


def test_no_jobs_exits_cleanly():
    assert simulate([], Policy.QUEUE_JIT).outcomes == ()


def test_determinism_for_identical_inputs():
    model = TransferModel("cpu", "pinned_cpu_to_gpu0", {256 << 20: .01, 1 << 30: .04, 4 << 30: .16}, "test")
    first = stochastic_stream(20, .5, 7, model, 1.0)
    second = stochastic_stream(20, .5, 7, model, 1.0)
    assert first == second
    assert aggregate(simulate(first, Policy.QUEUE_JIT)) == aggregate(simulate(second, Policy.QUEUE_JIT))
