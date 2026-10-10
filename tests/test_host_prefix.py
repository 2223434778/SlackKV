import pytest
from experiments.host_prefix.model import Cache, Tree
from experiments.host_prefix.run import replay


@pytest.mark.parametrize('policy', ['block_lru', 'prefix_lru', 'rr_tail', 'cost_tail'])
def test_capacity_and_identity(policy):
    tree = Tree()
    paths = [tree.path([1, 2, 3]), tree.path([1, 2, 4]), tree.path([9, 2, 3])]
    cache = Cache(4, policy, tree)
    for path in paths:
        cache.store(path, chunk=2)
        cache.validate()
    assert cache.hit(tree.path([7, 2, 3])) == 0
    assert cache.hit(tree.path([9, 3, 2])) <= 1


@pytest.mark.parametrize('policy', ['prefix_lru', 'rr_tail', 'cost_tail'])
def test_oversized_path_stays_prefix_closed(policy):
    tree = Tree()
    path = tree.path([1, 2, 3, 4, 5, 6])
    cache = Cache(3, policy, tree)
    cache.store(path)
    cache.validate()
    assert 0 < cache.hit(path) <= 3


def test_native_reverse_touch_and_atomic_rejection():
    tree = Tree()
    path = tree.path([1, 2, 3])
    cache = Cache(2, 'block_lru', tree)
    cache.store(path, chunk=3)
    assert not cache.blocks and cache.rejected_batches == 1
    cache.store(path[:2], chunk=2)
    assert list(cache.blocks) == list(reversed(path[:2]))


def test_no_ready_hit_before_completion():
    req = lambda t, d: dict(t=t, api_time=d, hash_ids=[1, 2])
    rows, _ = replay([req(0, 10), req(1, 2), req(11, 1)], 128, 'prefix_lru')
    assert [r['hit_proxy_tokens'] for r in rows] == [0, 0, 128]


def test_topology_does_not_leak_future_lengths():
    tree = Tree()
    short = tree.path([1, 2])
    tree.path([1, 2, 3, 4, 5])
    tree.observe(short)
    assert tree.extent[short[-1]] == 2


def test_zero_duration_cannot_publish_before_arrival():
    with pytest.raises(ValueError):
        replay([dict(t=0, api_time=0, hash_ids=[1])], 64, 'prefix_lru')


def test_correctness_oracle_salt_is_top_level():
    from experiments.host_prefix.oracle import cold_payload
    payload = cold_payload('m', [1, 2])
    assert payload['cache_salt'] == 'host-prefix-independent-cold'
    assert 'cache_salt' not in payload['kv_transfer_params']
    assert payload['kv_transfer_params']['max_offload_tokens'] == 0


def test_missing_counter_is_unknown_and_decreases_fail():
    from experiments.host_prefix.analyze_gpu import delta
    assert delta('', '', 'metric') is None
    with pytest.raises(ValueError):
        delta('metric{x="y"} 2', 'metric{x="y"} 1', 'metric')


def test_single_path_frontier_is_not_a_multi_session_quota():
    tree = Tree()
    a, b = tree.path([1, 2, 3, 4, 5, 6]), tree.path([9, 8, 7, 6, 5, 4])
    cache = Cache(4, 'block_lru', tree)
    cache.store(a[:4], chunk=4)
    assert cache.hit(a) == 4
    cache.store(b[:4], chunk=4)
    assert cache.hit(a) == 0  # The last path owns the whole tier; no fairness claim.


def test_appending_future_request_does_not_change_past_hits():
    req = lambda t, hashes: dict(t=t, api_time=1, hash_ids=hashes)
    old = [req(0, [1, 2]), req(2, [1, 2, 3])]
    first, _ = replay(old, 128, 'cost_tail')
    later, _ = replay(old + [req(100, [1, 2, 3, 4, 5, 6])], 128, 'cost_tail')
    assert first == later[:len(first)]
