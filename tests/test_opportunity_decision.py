import pytest
from experiments.opportunity_decision.analyze import Prefixes, analyze_trace, flatten


def req(t, hashes, model='m', duration=1):
    return {'t': t, 'hash_ids': hashes, 'model': model, 'in': len(hashes)*64,
            'out': 2, 'api_time': duration, 'type': 's'}


def trace(requests):
    return {'block_size': 64, 'hash_id_scope': 'local', 'requests': requests}


def test_prefix_not_unordered_block_membership():
    p = Prefixes()
    p.insert([1, 2, 3], 'a')
    assert p.match([1, 3, 2]) == 1
    assert p.match([2, 3]) == 0
    assert p.match([1, 2], outside='a') == 0
    p.insert([1, 2, 4], 'b')
    assert p.match([1, 2, 3], outside='a') == 2


def test_ongoing_and_cross_model_not_completed_cache():
    rows, _ = analyze_trace(trace([req(0, [1, 2], duration=10), req(1, [1, 2, 3]),
                                   req(12, [1, 2, 4], model='other'), req(13, [1, 2, 4])]), 0)
    assert rows[1]['completed_history_lcp_tokens'] == 0
    assert rows[1]['issued_history_lcp_tokens'] == 128
    assert rows[2]['completed_history_lcp_tokens'] == 0
    assert rows[3]['completed_history_lcp_tokens'] == 128


def test_child_times_absolute_and_cross_container_match():
    child = {'type': 'subagent', 't': 2, 'duration_ms': 1000,
             'requests': [req(2, [1, 2, 3])]}
    rows, _ = analyze_trace(trace([req(0, [1, 2]), child]), 0)
    assert rows[1]['t_s'] == 2
    assert rows[1]['other_container_completed_lcp_tokens'] == 128
    child['requests'][0]['t'] = 1
    with pytest.raises(ValueError):
        flatten(trace([child]))


def test_reject_cross_trace_hash_scope_and_bad_accounting():
    t = trace([req(0, [1])])
    t['hash_id_scope'] = 'global'
    with pytest.raises(ValueError):
        flatten(t)
    t['hash_id_scope'] = 'local'
    t['requests'][0]['in'] = 65
    with pytest.raises(ValueError):
        flatten(t)
