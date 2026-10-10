from types import SimpleNamespace as NS

import pytest

from experiments.autonomous_round4.retention import RetentionControl


class Pool:
    def touch(self, blocks):
        for b in blocks:
            b.ref_cnt += 1

    def free_blocks(self, blocks):
        for b in blocks:
            assert b.ref_cnt > 0
            b.ref_cnt -= 1


def block(i, refs=1, hashed=True, null=False):
    return NS(block_id=i, ref_cnt=refs, block_hash="hash" if hashed else None, is_null=null)


def request(key="a", continues=True, status="FINISHED_LENGTH_CAPPED", arrival=10):
    return NS(request_id=key + ":r", arrival_time=arrival, status=NS(name=status),
              kv_transfer_params={"slackkv_epoch_program": key, "slackkv_epoch_continues": continues})


def control(pin=True):
    return RetentionControl(Pool(), NS(info=lambda *a: None), enabled=True, pin=pin)


def test_pin_unique_full_blocks_and_expire_without_model_work():
    ctl, r = control(), request()
    ctl.arrived(r, 0)
    a, partial, null = block(1), block(2, hashed=False), block(3, null=True)
    ctl.finished(r, [a, a, partial, null], 0)
    assert a.ref_cnt == 2 and partial.ref_cnt == null.ref_cnt == 1
    ctl.pool.free_blocks([a])  # native request free
    assert ctl.delay(.5) == .5
    ctl.expire(1)
    assert a.ref_cnt == 0 and not ctl.entries and ctl.delay(1) is None


def test_partial_reuse_handoff_does_not_drop_real_or_dma_references():
    ctl, r = control(), request()
    ctl.arrived(r, 0)
    a, unused = block(1), block(2)
    ctl.finished(r, [a, unused], 0)
    ctl.pool.free_blocks([a, unused])
    ctl.pool.touch([a])  # actual next request owns only the matching prefix
    ctl.admitted(r)
    assert a.ref_cnt == 1 and unused.ref_cnt == 0 and not ctl.entries


@pytest.mark.parametrize("status,continues", [("FINISHED_ABORTED", True),
                                              ("FINISHED_ERROR", True),
                                              ("FINISHED_LENGTH_CAPPED", False)])
def test_cancel_error_and_terminal_turn_never_pin(status, continues):
    ctl, r, b = control(), request(status=status, continues=continues), block(1)
    ctl.arrived(r, 0)
    ctl.finished(r, [b], 0)
    assert b.ref_cnt == 1 and not ctl.entries


def test_pressure_reclaim_preserves_dma_and_is_idempotent():
    ctl, r, b = control(), request(), block(1, refs=2)
    ctl.arrived(r, 0)
    ctl.finished(r, [b], 0)
    ctl.pool.free_blocks([b])
    assert ctl.reclaim("DECODE") and b.ref_cnt == 1
    assert not ctl.reclaim("DECODE")


def test_age_guard_overrides_sticky_program_and_pin_priority():
    class Queue(list):
        def remove_requests(self, rs):
            self.clear()
        def add_request(self, r):
            self.append(r)
    ctl, old, returning, fresh = control(), request("old", arrival=2), request("return", arrival=3), request("new", arrival=4)
    for r, t in ((old, 0), (returning, 1.1), (fresh, 1.2)):
        ctl.arrived(r, t)
    ctl.finished(returning, [block(1)], 1)
    ctl.arrived(returning, 1.1)
    q = Queue([fresh, returning, old])
    ctl.reorder(q, 1.2)
    assert q == [old, returning, fresh]


def test_priority_only_control_adds_no_physical_refs():
    ctl, r, b = control(False), request(), block(1)
    ctl.arrived(r, 0)
    ctl.finished(r, [b], 0)
    assert b.ref_cnt == 1 and not ctl.entries


def test_delayed_old_d2h_free_does_not_repin_after_terminal_turn():
    ctl, old, new, b = control(), request(), request(continues=False), block(1)
    new.request_id = "a:new"
    ctl.arrived(old, 0)
    ctl.arrived(new, .1)
    ctl.finished(new, [], .2)
    ctl.finished(old, [b], .3)
    assert b.ref_cnt == 1 and not ctl.entries and not ctl.latest_request and not ctl.outstanding


def test_cancel_before_delayed_free_cannot_touch_newer_retention():
    ctl, old, new, b = control(), request(status="FINISHED_ABORTED"), request(), block(1)
    new.request_id = "a:new"
    ctl.arrived(old, 0)
    ctl.cancelled(old)
    ctl.arrived(new, .1)
    ctl.finished(new, [b], .2)
    ctl.finished(old, [], .3)
    assert b.ref_cnt == 2 and "a" in ctl.entries
    ctl.expire(1.2)
    assert b.ref_cnt == 1 and not ctl.entries


def test_mature_head_cannot_wait_for_retained_consumer_behind_it():
    ctl, owner, cold, b = control(), request(), request("cold"), block(1)
    ctl.arrived(owner, 0)
    ctl.finished(owner, [b], 0)
    ctl.pool.free_blocks([b])
    assert ctl.resolve_blocked(cold, [cold, owner])
    assert b.ref_cnt == 0 and not ctl.entries
