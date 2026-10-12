"""Logical allocation tests; real native lifecycle is audited on Ubuntu."""
import pytest

from slackkv.m1.controller import PrefixController, max_min


def path(name, n):
    return tuple(f'{name}-{i}'.encode() for i in range(n))


def test_short_session_lends_capacity_without_an_extra_full_mode():
    result, _ = max_min({('short', 0): 1, ('long', 0): 10}, 10)
    assert result == {('short', 0): 1, ('long', 0): 9}
    result, _ = max_min({('short', 0): 1, ('long', 0): 10}, 20)
    assert sum(result.values()) == 11


def test_rotating_integer_remainder_and_zero_capacity():
    demand = {(s, 0): 5 for s in ('a', 'b', 'c')}
    cursor = 0
    total = {k: 0 for k in demand}
    for _ in range(3):
        shares, cursor = max_min(demand, 5, cursor)
        assert sum(shares.values()) == 5 and sorted(shares.values()) == [1, 2, 2]
        for key, value in shares.items():
            total[key] += value
    assert len(set(total.values())) == 1
    assert all(n == 0 for n in max_min(demand, 0)[0].values())


def test_frozen_old_ten_block_plan_cannot_overwrite_new_five_block_share():
    c = PrefixController(10)
    a = c.observe('a0', 'a', 0, path('a', 10))
    assert a.ceiling == 10
    b = c.observe('b0', 'b', 0, path('b', 10))
    assert a.ceiling == 10 and c.allowed(a) == c.allowed(b) == 5
    c.end('b', 0)
    assert c.allowed(a) == 10 and c.allowed(b) == 0


def test_old_smaller_ceiling_does_not_grow_until_a_new_request():
    c = PrefixController(10)
    c.observe('a0', 'a', 0, path('a', 10))
    b = c.observe('b0', 'b', 0, path('b', 10))
    c.end('a', 0)
    assert c.members[('b', 0)].target == 10 and c.allowed(b) == 5
    c.finish('b0')
    assert c.observe('b1', 'b', 0, path('b', 10)).ceiling == 10


@pytest.mark.parametrize('change', ['shorten', 'rewrite', 'parallel'])
def test_unsupported_identity_changes_stop_optional_writes(change):
    c = PrefixController(10)
    a = c.observe('first', 'a', 0, path('a', 4))
    if change != 'parallel':
        c.finish('first')
    new = path('a', 2) if change == 'shorten' else path('x', 4) if change == 'rewrite' else path('a', 4)
    later = c.observe('later', 'a', 0, new)
    assert later.ceiling == c.allowed(a) == 0
    assert later.reason is not None


def test_sharing_does_not_create_duplicate_physical_owner_or_hidden_full_fallback():
    c = PrefixController(10)
    p = path('common', 3)
    c.observe('a', 'a', 0, p)
    b = c.observe('b', 'b', 0, p)
    assert b.reason == 'shared_prefix' and b.ceiling == 0
    assert set(c.key_owner.values()) == {('a', 0)}


def test_reused_generation_waits_for_retired_identity_without_sticky_shared_fallback():
    c = PrefixController(10)
    p = path('same', 3)
    old = c.observe('old', 'a', 0, p)
    c.finish('old')
    new = c.observe('new', 'a', 1, p)
    assert new.reason == 'retired_identity_pending' and c.allowed(old) == 0
    c.collect(('a', 0))  # Physical adapter has confirmed old keys gone.
    c.finish('new')
    assert c.observe('next', 'a', 1, p).ceiling == 3
    assert c.observe('late', 'a', 0, p).reason == 'stale_generation'


@pytest.mark.parametrize('value', [-1, 1.5, True])
def test_invalid_capacity_is_rejected(value):
    with pytest.raises(ValueError):
        PrefixController(value)
