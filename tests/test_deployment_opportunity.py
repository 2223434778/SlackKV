from experiments.deployment_opportunity.analyze import footprint, kv_bytes_per_token


def req(t, duration, hashes, container='root', out=0):
    return dict(t=t, api_time=duration, hash_ids=hashes, container=container, out=out)


def test_shared_active_prefix_and_half_open_end():
    result = footprint([req(0, 2, [1, 2]), req(1, 1, [1, 3]), req(2, 1, [4])])
    assert result['peak_active_input_proxy_tokens'] == 192
    assert result['peak_endpoint_inflight_requests'] == 2
    assert result['active_input_proxy_token_seconds'] == 384
    assert result['events'][-1]['union_proxy_tokens'] == 0


def test_future_retention_and_intervening_footprint():
    result = footprint([req(0, 1, [1, 2]), req(2, 1, [3], 'other'), req(4, 1, [1, 2])])
    assert result['peak_active_input_proxy_tokens'] == 128
    assert result['peak_active_and_future_needed_union_proxy_tokens'] == 192
    assert result['same_container_intervening_issued_unique_input_proxy_tokens']['max'] == 64
    assert result['events'][1]['completed_future_needed_proxy_tokens'] == 128


def test_contextual_identity_not_hash_alone():
    result = footprint([req(0, 1, [1, 9]), req(0, 1, [2, 9])])
    assert result['cumulative_unique_input_proxy_tokens'] == 256


def test_zero_duration_and_output_reserve():
    result = footprint([req(0, 0, [1]), req(1, 1, [1], out=16)])
    assert result['peak_active_input_plus_full_output_reserve_tokens'] == 80
    assert result['peak_completed_future_needed_proxy_tokens'] == 64
    assert result['active_input_proxy_token_seconds'] == 64


def test_no_future_retention_for_overlapping_only_reuse():
    result = footprint([req(0, 3, [1]), req(1, 1, [1])])
    assert result['peak_completed_future_needed_proxy_tokens'] == 0


def test_real_kv_geometry():
    config = dict(hidden_size=1536, num_attention_heads=12, num_hidden_layers=28, num_key_value_heads=2)
    assert kv_bytes_per_token(config) == 28672
    config.update(hidden_size=5120, num_attention_heads=40, num_hidden_layers=64, num_key_value_heads=8)
    assert kv_bytes_per_token(config) == 262144


def test_namespace_isolation():
    from experiments.deployment_opportunity.analyze import analyze_trace
    trace = dict(block_size=64, hash_id_scope='local', requests=[
        dict(req(0, 1, [1]), type='request', model=model, **{'in': 64})
        for model in ('model_a', 'model_b')])
    rows = analyze_trace(trace, 9)
    assert len(rows) == 2
    assert all(row['peak_active_input_proxy_tokens'] == 64 and row['trace_index'] == 9 for row in rows)


def test_event_sweep_matches_independent_interval_reference():
    import random
    random.seed(774421)
    requests = [req(random.randrange(8), random.randrange(4),
                    [random.randrange(3), random.randrange(3)], str(i % 3)) for i in range(20)]
    result = footprint(requests)
    paths = [[tuple(request['hash_ids'][:depth]) for depth in (1, 2)] for request in requests]
    for row in result['events']:
        t = row['t_s']
        active = {block for request, path in zip(requests, paths)
                  if request['t'] <= t < request['t'] + request['api_time'] for block in path}
        completed = {block for request, path in zip(requests, paths)
                     if request['t'] + request['api_time'] <= t for block in path}
        future = {block for request, path in zip(requests, paths) if request['t'] > t for block in path}
        assert row['active_input_proxy_tokens'] == len(active) * 64
        assert row['completed_future_needed_proxy_tokens'] == len(completed & future) * 64
        assert row['union_proxy_tokens'] == len(active | (completed & future)) * 64


def test_pin_probe_selects_idle_device_without_disrupting_busy_peer():
    import pytest
    from experiments.deployment_opportunity.pin_probe import selected_gpu_idle
    assert selected_gpu_idle('321, 0\n19, 0\n', 1)
    assert not selected_gpu_idle('321, 0\n19, 0\n', 0)
    assert not selected_gpu_idle('19, 1\n', 0)
    with pytest.raises(ValueError):
        selected_gpu_idle('19, 0\n', 1)


def test_pin_budget_refuses_missing_history_and_preserves_continuity(tmp_path):
    import json
    import pytest
    from experiments.deployment_opportunity.pin_probe import budget
    with pytest.raises(RuntimeError, match='ledger unavailable'):
        budget(tmp_path)
    for family in ('autonomous_fork', 'autonomous_round4', 'host_prefix_cost_20261010', 'host_prefix_followup_20261010'):
        path = tmp_path / 'results' / family / 'run' / 'execution.json'
        path.parent.mkdir(parents=True)
        field = 'gpu_hours_conservative' if family in ('autonomous_fork', 'autonomous_round4') else 'gpu_hours'
        path.write_text(json.dumps(dict(status='COMPLETE', **{field: .2})))
    assert sum(budget(tmp_path).values()) == pytest.approx(.8)
    path = tmp_path/'results/large_model_20261012/tp2/execution.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(status='COMPLETE', gpu_hours=.4)))
    assert sum(budget(tmp_path).values()) == pytest.approx(1.2)
    path.write_text(json.dumps(dict(status='RUNNING')))
    with pytest.raises(RuntimeError, match='unresolved'):
        budget(tmp_path)
