"""Exercise actual scheduler gate branches without importing the GPU runtime."""
import ast
import copy
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(os.environ.get("SLACKKV_SCHEDULER_SOURCE", "/media/a6000/D/ZhaoBinHe/slackkv-phase2c-fix-sourceprobe/vllm/v1/core/sched/scheduler.py"))


class Request:
    def __init__(self, request_id="A", outputs=0):
        self.request_id = request_id
        self._output_token_ids = [7] * outputs
        self.kv_transfer_params = {"slackkv_trace_resume": True,
                                   "slackkv_first_token_gate": True,
                                   "slackkv_first_token_gate_limit": 2}
        self.client_index = 0

    @property
    def num_output_tokens(self):
        return len(self._output_token_ids)

    def is_finished(self):
        return True


@pytest.fixture
def production():
    tree = ast.parse(SOURCE.read_text())
    helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {
        "_is_slackkv_trace_resume", "_is_slackkv_prefetch_only",
        "_is_slackkv_first_token_gate", "_slackkv_first_token_gate_limit",
        "_slackkv_first_token_gate_can_admit"}]
    scheduler = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
    methods = {n.name: n for n in scheduler.body if isinstance(n, ast.FunctionDef)}
    ns = {"Request": Request, "Any": object}
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *copy.deepcopy(helpers), copy.deepcopy(methods["_free_request"])], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), ns)
    acquire = next(n for n in ast.walk(methods["schedule"]) if isinstance(n, ast.If)
                   and "_is_slackkv_first_token_gate" in ast.unparse(n.test)
                   and "_slackkv_ft_gate_owners.add" in ast.unparse(n))
    waiting = next(n for n in ast.walk(methods["schedule"]) if isinstance(n, ast.If)
                   and "_is_slackkv_first_token_gate" in ast.unparse(n.test)
                   and "GATE_BLOCKED" in ast.unparse(n))
    first = next(n for n in ast.walk(methods["update_from_output"]) if isinstance(n, ast.If)
                 and "num_output_tokens_before" in ast.unparse(n.test)
                 and "_slackkv_ft_gate_owners.remove" in ast.unparse(n))

    def execute(node, state, request, **extra):
        scope = dict(ns, self=state, request=request, req_id=request.request_id, **extra)
        code = ast.Module(body=[copy.deepcopy(node)], type_ignores=[])
        exec(compile(ast.fix_missing_locations(code), str(SOURCE), "exec"), scope)

    return ns, acquire, waiting, first, execute


def state():
    return SimpleNamespace(
        _slackkv_ft_gate_owners=set(), _slackkv_ft_gate_limit=2,
        _slackkv_ft_first_token_seen=set(), _slackkv_trace=lambda *args: None,
        _inflight_prefills=set(), _connector_finished=lambda r: (False, None),
        encoder_cache_manager=SimpleNamespace(free=lambda r: None),
        finished_req_ids=set(), finished_req_ids_dict=None,
        _free_blocks=lambda r: None)


@pytest.mark.parametrize("outputs", [6, 17])
def test_post_first_preemption_does_not_reacquire(production, outputs):
    ns, acquire, waiting, first, execute = production
    s, r = state(), Request()
    execute(acquire, s, r)
    assert s._slackkv_ft_gate_owners == {"A"}
    r._output_token_ids.append(7)
    execute(first, s, r, num_output_tokens_before=0, new_token_ids=[7])
    assert not s._slackkv_ft_gate_owners
    r._output_token_ids = [7] * outputs
    execute(acquire, s, r)
    assert not s._slackkv_ft_gate_owners
    ns["_free_request"](s, r)
    assert not s._slackkv_ft_first_token_seen


def test_post_first_request_bypasses_full_gate(production):
    ns, _, waiting, _, _ = production
    s, r = state(), Request(outputs=6)
    s._slackkv_ft_gate_owners.update({"B", "C"})
    assert not eval(compile(ast.Expression(waiting.test), str(SOURCE), "eval"), dict(ns, self=s, request=r))


def test_pre_first_owner_can_reenter_when_gate_full(production):
    ns, acquire, _, _, execute = production
    s, r = state(), Request()
    s._slackkv_ft_gate_owners.update({"A", "B"})
    assert ns["_slackkv_first_token_gate_can_admit"](s._slackkv_ft_gate_owners, 2, r)[0]
    execute(acquire, s, r)
    assert s._slackkv_ft_gate_owners == {"A", "B"}
    assert not ns["_slackkv_first_token_gate_can_admit"](s._slackkv_ft_gate_owners, 2, Request("C"))[0]


@pytest.mark.parametrize("delayed", [False, True])
def test_terminal_cleanup_is_idempotent_even_with_delayed_kv_free(production, delayed):
    ns, *_ = production
    s, r = state(), Request(outputs=42)
    s._slackkv_ft_gate_owners.add(r.request_id)
    s._slackkv_ft_first_token_seen.add(r.request_id)
    ns["_free_request"](s, r, delay_free_blocks=delayed)
    ns["_free_request"](s, r, delay_free_blocks=delayed)
    assert not s._slackkv_ft_gate_owners
    assert not s._slackkv_ft_first_token_seen


def test_unmarked_request_and_mixed_limits(production):
    ns, acquire, _, _, execute = production
    s, r = state(), Request()
    r.kv_transfer_params = {}
    execute(acquire, s, r)
    assert not s._slackkv_ft_gate_owners
    with pytest.raises(AssertionError, match="mixed"):
        ns["_slackkv_first_token_gate_can_admit"](set(), 1, Request())


def test_two_post_first_readmissions_leave_capacity_for_next_turn(production):
    ns, acquire, _, _, execute = production
    s = state()
    for episode, outputs in [("E03:turn:1", 17), ("E06:turn:1", 6)]:
        r = Request(episode, outputs)
        s._slackkv_ft_first_token_seen.add(episode)
        execute(acquire, s, r)
        ns["_free_request"](s, r)
    assert not s._slackkv_ft_gate_owners
    execute(acquire, s, Request("E03:turn:2"))
    execute(acquire, s, Request("E06:turn:2"))
    assert len(s._slackkv_ft_gate_owners) == 2
