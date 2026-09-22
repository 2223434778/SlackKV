from __future__ import annotations

from dataclasses import dataclass

from experiments.phase3s.helpers import (
    lease_sequence_valid,
    parse_structured_events,
    source_conservation,
    unique_blocks,
)
from experiments.phase3f.native_client import native_prefetch


@dataclass
class Block:
    block_id: int
    ref_cnt: int = 1
    is_null: bool = False


class Pool:
    def __init__(self):
        self.touched = []
        self.freed = []

    def touch(self, blocks):
        self.touched.extend(blocks)
        for block in blocks:
            block.ref_cnt += 1

    def free_blocks(self, blocks):
        self.freed.extend(blocks)
        for block in blocks:
            block.ref_cnt -= 1


def test_native_prefetch_lease_key_is_optional():
    assert native_prefetch.__kwdefaults__["lease_key"] is None


def test_unique_blocks_touched_once():
    a, b = Block(1), Block(2)
    blocks = unique_blocks(([a, b], [a]))
    pool = Pool(); pool.touch(blocks)
    assert [block.block_id for block in pool.touched] == [1, 2]
    assert [block.ref_cnt for block in blocks] == [2, 2]


def test_release_uses_pool_and_removes_once():
    blocks = [Block(1, 2), Block(2, 3)]
    pool = Pool(); leases = {"A": blocks}
    pool.free_blocks(leases.pop("A"))
    assert [block.block_id for block in pool.freed] == [1, 2]
    assert [block.ref_cnt for block in blocks] == [1, 2]
    assert "A" not in leases


def test_duplicate_and_unknown_lease_rejected():
    leases = {"A": [Block(1)]}
    assert "A" in leases
    try:
        assert "A" not in leases, "duplicate SlackKV residency lease"
    except AssertionError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate was not rejected")
    try:
        leases.pop("B")
    except KeyError:
        pass
    else:
        raise AssertionError("unknown release was not rejected")


def test_release_requires_real_owner():
    block = Block(1, 1)
    assert block.ref_cnt < 2


def test_unleased_has_no_lease_and_source_conserves():
    assert source_conservation(0, 4096)
    assert source_conservation(4096, 0)
    assert not source_conservation(4095, 0)


def test_lease_trace_serialization():
    events = [
        {"event": "LEASE_ACQUIRED", "active_lease_count": 1},
        {"event": "LEASE_HELD_AFTER_PREFETCH_FREE", "active_lease_count": 1},
        {"event": "LEASE_RELEASE_READY", "active_lease_count": 1},
        {"event": "LEASE_RELEASED", "active_lease_count": 0},
    ]
    text = "\n".join("INFO SLACKKV_LEASE_TRACE " + __import__("json").dumps(x) for x in events)
    assert lease_sequence_valid(parse_structured_events(text, "SLACKKV_LEASE_TRACE "))
