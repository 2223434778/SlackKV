import pytest

from experiments.autonomous_capacity.capacity import validate_pool
from experiments.autonomous_capacity.continuation import ContinuationCredit
from collections import deque
from types import SimpleNamespace


class Queue(deque):
    def remove_request(self, request):
        self.remove(request)

    def prepend_request(self, request):
        self.appendleft(request)


def request(name, arrival, *, resume=False, status="WAITING", computed=0, preemptions=0):
    return SimpleNamespace(request_id=name, arrival_time=arrival,
                           kv_transfer_params={"slackkv_trace_resume": resume},
                           status=SimpleNamespace(name=status), num_output_tokens=0,
                           num_computed_tokens=computed, num_preemptions=preemptions)


def choose(policy, queue, protected=False):
    policy.select(queue, active_waiting_ids={r.request_id for r in queue}, protected=protected)


def test_one_credit_across_steps_and_async_load_then_admission():
    head, a, b = request("cold", 0), request("a", 1, resume=True), request("b", 2, resume=True)
    queue = Queue([head, a, b])
    policy = ContinuationCredit(True)
    choose(policy, queue)
    assert queue[0] is a
    # Repeated unsuccessful allocation checks do not spend credit.
    choose(policy, queue)
    assert policy.used == set()
    assert policy.admitted("a") == "cold"
    assert policy.admitted("a") is None
    queue.remove_request(a)
    choose(policy, queue)
    assert queue[0] is head
    assert policy.used == {"cold"}


def test_cancel_clears_pending_and_used_state():
    queue = Queue([request("cold", 0), request("a", 1, resume=True)])
    policy = ContinuationCredit(True)
    choose(policy, queue)
    policy.admitted("a")
    choose(policy, Queue())
    assert not policy.pending and not policy.used


@pytest.mark.parametrize("enabled,protected", [(False, False), (True, True)])
def test_default_and_lease_path_keep_queue_order(enabled, protected):
    head = request("cold", 0)
    queue = Queue([head, request("a", 1, resume=True)])
    choose(ContinuationCredit(enabled), queue, protected)
    assert queue[0] is head


@pytest.mark.parametrize("options", [{"status": "WAITING_FOR_REMOTE_KVS"},
                                     {"computed": 16}, {"preemptions": 1}])
def test_blocked_inflight_and_preempted_requests_are_not_promoted(options):
    head = request("cold", 0)
    queue = Queue([head, request("a", 1, resume=True, **options)])
    choose(ContinuationCredit(True), queue)
    assert queue[0] is head


def test_success_invalidates_other_pending_credits_for_same_head():
    policy = ContinuationCredit(True)
    policy.pending = {"a": "cold", "b": "cold"}
    assert policy.admitted("a") == "cold"
    assert policy.admitted("b") is None
    head = request("cold", 0)
    queue = Queue([request("b", 2, resume=True), head])
    choose(policy, queue)
    assert queue[0] is head


def test_actual_input_duplicate_is_rejected(tmp_path):
    import json
    from experiments.autonomous_capacity.summary import input_hashes
    path = tmp_path / "inputs.jsonl"
    rows = [{"session": f"E{i:02}:turn:{j}", "prompt": [i, j]}
            for i in range(8) for j in range(5)]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    assert len(input_hashes(path)) == 40
    rows[-1] = rows[0]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="duplicate"):
        input_hashes(path)


from experiments.autonomous_capacity.epochs import WorkingSetEpochs


def epoch_request(key, name, *, tokens=160, output=16, continues=True):
    return SimpleNamespace(request_id=name, num_prompt_tokens=tokens, max_tokens=output,
                           kv_transfer_params={"slackkv_epoch_program": key,
                                               "slackkv_epoch_continues": continues})


def allowed(policy, req, now=0, pool=24):
    return policy.allowed(req, now=now, pool_blocks=pool, background_blocks=2, max_model_len=32768)


def test_epoch_capacity_and_background_accounting_and_progress_escape():
    policy = WorkingSetEpochs(True)
    a, b = epoch_request("a", "a0"), epoch_request("b", "b0")
    assert allowed(policy, a)
    policy.admitted(a, max_model_len=32768)
    assert not allowed(policy, b)  # 11 + 11 > 24 - 1 - 2
    assert allowed(policy, b, pool=25)
    assert allowed(policy, epoch_request(None, "background"))
    policy.finished(a, now=0, cancelled=True)
    assert allowed(policy, epoch_request("big", "big", tokens=50000))


def test_two_calls_idle_timeout_and_async_double_allocation():
    policy = WorkingSetEpochs(True)
    a = epoch_request("a", "a0")
    assert policy.admitted(a, max_model_len=32768)
    assert not policy.admitted(a, max_model_len=32768)
    policy.finished(a, now=0, cancelled=False)
    assert policy.entries["a"].completed == 1
    policy.expire(.999)
    assert "a" in policy.entries
    b = epoch_request("a", "a1", tokens=320)
    assert allowed(policy, b, pool=12)
    policy.admitted(b, max_model_len=32768)
    policy.expire(100)
    assert "a" in policy.entries  # Busy requests must not lose epoch state.
    policy.finished(b, now=100, cancelled=False)
    assert not policy.entries


@pytest.mark.parametrize("cancelled,continues", [(True, True), (False, False)])
def test_epoch_terminal_and_cancel_release(cancelled, continues):
    policy = WorkingSetEpochs(True)
    a = epoch_request("a", "a0", continues=continues)
    policy.admitted(a, max_model_len=32768)
    policy.finished(a, now=0, cancelled=cancelled)
    assert not policy.entries


def test_epoch_expired_missing_continuation_and_disabled_path():
    policy = WorkingSetEpochs(True)
    a = epoch_request("a", "a0")
    policy.admitted(a, max_model_len=32768)
    policy.finished(a, now=0, cancelled=False)
    policy.expire(1)
    assert not policy.entries
    disabled = WorkingSetEpochs()
    assert allowed(disabled, a, pool=1)
    assert not disabled.admitted(a, max_model_len=32768)


def test_missing_counter_is_unknown_and_not_implicit_zero(tmp_path):
    from experiments.autonomous_capacity.audit import counter
    path = tmp_path / "metrics.prom"
    path.write_text('transfer{type="D2H"} 16\n')
    assert counter(path, "transfer", 'type="H2D"') is None
    assert counter(path, "transfer", 'type="D2H"') == 16


def test_round_two_raw_evidence_and_rejected_candidates():
    from pathlib import Path
    from experiments.autonomous_capacity.summary import summarize
    from experiments.autonomous_capacity.audit import audit
    directory = Path(__file__).resolve().parents[1] / "results/autonomous_capacity"
    result = summarize(directory)
    assert len(result["runs"]) == 8
    assert result["gpu_hours_conservative"] < 4
    assert all(not p["eligible_for_replication"] for p in result["algorithm_pairs"].values())
    assert result["algorithm_pairs"]["credit_2"]["gates"]["p95"]
    assert not result["algorithm_pairs"]["credit_2"]["gates"]["session_median"]
    evidence = audit(directory)
    assert evidence["archive_verified_files"] == 87
    assert len(evidence["runs"]) == 7
    assert all(r["compiled_execution_log_present"] for r in evidence["runs"])
    by_name = {r["name"]: r for r in evidence["runs"]}
    assert by_name["epoch-candidate-2"]["final_epoch_programs"] == 0
    assert by_name["credit-candidate-2"]["H2D_actual_minus_source_bytes"] == 16 * 28672
    assert by_name["capacity-8"]["H2D_bytes_actual"] is None


def test_capacity_is_derived_from_pinned_layout_not_rounded_gib():
    for gib, blocks in ((2, 4681), (4, 9362), (8, 18724)):
        assert validate_pool(gib * 2**30, 16, blocks)["total_gpu_kv_blocks"] == blocks


def test_wrong_pool_or_layout_is_rejected():
    for size, blocks in ((16, 4681), (32, 9362), (16, 9363)):
        with pytest.raises(ValueError, match="mismatch"):
            validate_pool(4 * 2**30, size, blocks)
