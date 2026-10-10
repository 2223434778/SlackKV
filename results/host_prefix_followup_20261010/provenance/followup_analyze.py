"""Independent raw-payload and source accounting for the three-question screen."""
import argparse
import hashlib
import json
import lzma
from pathlib import Path
import re
import statistics
from experiments.autonomous_fork.analyze import counter, read as legacy_read
from experiments.host_prefix.analyze_gpu import delta
from experiments.host_prefix.followup_screen import TOKEN_BYTES, cap, shares


def read(path):
    compressed = Path(str(path)+'.xz')
    if not path.exists() and compressed.exists():
        return lzma.decompress(compressed.read_bytes())
    return legacy_read(path)


def transfer(before, after, direction, external):
    name = 'vllm:kv_offload_total_bytes_total'
    label = f'transfer_type="{direction}"'
    start, end = counter(before, name, label), counter(after, name, label)
    # Only known-empty fresh connector with no external reads admits absent labels.
    if start is None and f'# HELP {name} ' in before:
        start = 0
    if end is None and external == 0 and f'# HELP {name} ' in after:
        end = 0
    if start is None or end is None:
        return None
    if end < start:
        raise ValueError('transfer counter decreased')
    return end-start


def analyze(directory):
    execution = json.loads(read(directory/'execution.json'))
    if execution['status'] == 'RUNNING':
        raise ValueError('unfinished run: budget and measurements remain unresolved')
    if execution['status'] != 'COMPLETE':
        return dict(run=directory.name, status=execution['status'], gpu_hours=execution['gpu_hours'])
    cfg = json.loads(read(directory/'workload.json'))
    rows = json.loads(read(directory/'requests.json'))
    inputs = json.loads(read(directory/'inputs.json'))
    quotas = shares(cfg['cpu_bytes'], cfg['sessions'])
    assert cfg['quotas_tokens'] == quotas
    identities = {}
    for item in inputs:
        key = (item['role'], item['session'], item['turn'])
        assert key not in identities
        identities[key] = item['prompt']
    recorded = {}
    for row in rows:
        key = (row['role'], row['session'], row['turn'])
        assert key not in recorded
        prompt = identities[key]
        assert len(prompt) == row['length']
        assert hashlib.sha256(json.dumps(prompt).encode()).hexdigest() == row['prompt_sha256']
        assert row['issue_ns'] <= row['first_ns'] <= row['done_ns']
        assert row['output_count'] == len(row['output_tokens']) == (16 if row['role'] == 'agent' else 256)
        expected_limit = cap(cfg['policy'], row['length'], quotas[row['session']]) if row['role'] == 'agent' else 0
        assert row['max_offload_tokens'] == expected_limit
        recorded[key] = row
    assert len(identities) == len(recorded)
    agents = [r for r in rows if r['role'] == 'agent']
    bg = sorted((r for r in rows if r['role'] == 'bg'), key=lambda r: r['turn'])
    assert len(agents) == cfg['sessions']*6
    assert len(bg) == (cfg['sessions']*6 if cfg['reset_gpu'] else 12)
    roots = [identities[('agent', s, 0)][:16] for s in range(cfg['sessions'])]
    assert len({tuple(p) for p in roots}) == cfg['sessions']
    assert all(identities[('bg', -1, r['turn'])][:16] not in roots for r in bg)
    session_metrics = []
    for session in range(cfg['sessions']):
        own = sorted((r for r in agents if r['session'] == session), key=lambda r: r['turn'])
        assert [r['turn'] for r in own] == list(range(6))
        for turn in range(1, 6):
            prev = identities[('agent', session, turn-1)]
            assert identities[('agent', session, turn)][:len(prev)] == prev
            assert own[turn]['issue_ns'] >= own[turn-1]['done_ns']
        ttfts = [(r['first_ns']-r['issue_ns'])/1e6 for r in own[1:]]
        service = [(r['done_ns']-r['issue_ns'])/1e6 for r in own]
        session_metrics.append(dict(session=session, resume_p95_ms=max(ttfts),
            mean_service_ms=statistics.mean(service),
            closed_loop_e2e_ms=(own[-1]['done_ns']-own[0]['issue_ns'])/1e6))
    if cfg['reset_gpu']:
        for a in agents:
            b = next(r for r in bg if r['turn'] == a['turn']*cfg['sessions']+a['session'])
            assert b['first_ns'] <= a['issue_ns'] <= b['done_ns']
    else:
        assert all(r['issue_ns'] >= bg[0]['first_ns'] for r in agents)
        assert all(bg[i]['issue_ns'] >= bg[i-1]['done_ns'] for i in range(1, len(bg)))
    before, after = (read(directory/f).decode() for f in ('before.prom', 'after.prom'))
    sources = {source: delta(before, after, 'vllm:prompt_tokens_by_source_total', f'source="{source}"')
               for source in ('local_compute', 'local_cache_hit', 'external_kv_transfer')}
    warmup_seen = counter(before, 'vllm:prompt_tokens_by_source_total', 'source="local_compute"')
    assert warmup_seen is not None and 0 <= warmup_seen <= 256
    assert all(v is not None and v >= 0 for v in sources.values())
    sources['local_compute'] -= 256-warmup_seen
    assert sum(sources.values()) == sum(r['length'] for r in rows)
    h2d = transfer(before, after, 'CPU_to_GPU', sources['external_kv_transfer'])
    d2h = transfer(before, after, 'GPU_to_CPU', sources['external_kv_transfer'])
    assert h2d is not None and d2h is not None
    assert h2d == sources['external_kv_transfer']*TOKEN_BYTES
    if cfg['reset_gpu']:
        assert sources['local_cache_hit'] == 0
    if cfg['policy'] == 'nohost':
        assert h2d == d2h == sources['external_kv_transfer'] == 0
    log = read(directory/'server_logs'/f'{directory.name}.log').decode()
    pool = re.search(r'GPU KV cache size: ([\d,]+) tokens', log)
    assert pool is not None
    assert counter(after, 'vllm:num_requests_running') == counter(after, 'vllm:num_requests_waiting') == 0
    assert execution['server_exit_code'] == 0
    result = dict(run=directory.name, status='VALID', config=cfg, sources=sources,
                  h2d_bytes=h2d, d2h_bytes=d2h, total_dma_bytes=h2d+d2h,
                  actual_gpu_pool_tokens=int(pool.group(1).replace(',', '')),
                  session_metrics=session_metrics,
                  session_e2e_median_ms=statistics.median(r['closed_loop_e2e_ms'] for r in session_metrics),
                  background_median_ms=statistics.median((r['done_ns']-r['issue_ns'])/1e6 for r in bg),
                  background_makespan_ms=(bg[-1]['done_ns']-bg[0]['issue_ns'])/1e6,
                  input_identity={':'.join(map(str, k)): v['prompt_sha256'] for k, v in sorted(recorded.items())},
                  total_inputs=sum(r['length'] for r in rows), total_outputs=sum(r['output_count'] for r in rows),
                  gpu_hours=execution['gpu_hours'],
                  per_request_source_deltas='concurrent snapshots overlap; no per-request attribution claim')
    if cfg['reset_gpu'] and cfg['policy'] == 'fair':
        expected = 5*sum(quotas)
        observed = [dict(session=r['session'], turn=r['turn'],
                        external_tokens=delta(r['metrics_before'], r['metrics_after'],
                            'vllm:prompt_tokens_by_source_total', 'source="external_kv_transfer"')) for r in agents]
        # Only one Agent is active at a time here; backgrounds never offload or
        # share its prefix. Concurrent stage C cannot use this attribution.
        result['serial_agent_external_tokens'] = observed
        result['expected_fair_external_tokens'] = expected
        result['all_quotas_restored'] = sources['external_kv_transfer'] == expected and all(
            r['external_tokens'] == (quotas[r['session']] if r['turn'] else 0) for r in observed)
    return result


def compare(base, candidate):
    assert base['status'] == candidate['status'] == 'VALID'
    assert base['input_identity'] == candidate['input_identity']
    assert base['total_outputs'] == candidate['total_outputs']
    assert base['actual_gpu_pool_tokens'] == candidate['actual_gpu_pool_tokens']
    for field in ('sessions', 'seed', 'cpu_bytes', 'gpu_bytes', 'reset_gpu'):
        assert base['config'][field] == candidate['config'][field]
    change = lambda a, b: 100*(b/a-1)
    session_changes = [dict(session=a['session'], **{
        key: change(a[key], b[key]) for key in ('resume_p95_ms', 'mean_service_ms', 'closed_loop_e2e_ms')})
        for a, b in zip(base['session_metrics'], candidate['session_metrics'])]
    return dict(baseline=base['run'], candidate=candidate['run'],
                percent={key: change(base[key], candidate[key]) for key in
                    ('session_e2e_median_ms', 'background_median_ms', 'background_makespan_ms')},
                per_session_percent=session_changes,
                reset_screen_pass=all(r['resume_p95_ms'] <= -10 and r['mean_service_ms'] <= 5 for r in session_changes)
                    and change(base['background_median_ms'], candidate['background_median_ms']) <= 5,
                natural_screen_pass=change(base['session_e2e_median_ms'], candidate['session_e2e_median_ms']) <= -10
                    and all(r['closed_loop_e2e_ms'] <= 5 for r in session_changes)
                    and change(base['background_makespan_ms'], candidate['background_makespan_ms']) <= 5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--directory', type=Path, required=True)
    args = p.parse_args()
    runs = {d.name: analyze(d) for d in sorted(args.directory.iterdir()) if (d/'execution.json').exists()}
    pairs = []
    for base, candidate in [('a1nohost', 'a1fair'), ('a2nohost', 'a2fair'),
                            ('bnohost', 'bfair'), ('bfull', 'bfair')] + [
                                (regime+base, regime+'fair') for regime in ('hot', 'pressure', 'ample') for base in ('nohost', 'full')]:
        if base in runs and candidate in runs and runs[base]['status'] == runs[candidate]['status'] == 'VALID':
            pairs.append(compare(runs[base], runs[candidate]))
    result = dict(runs=runs, pairs=pairs, gpu_hours=sum(r['gpu_hours'] for r in runs.values()),
                  qualification='small synthetic screens, no statistical or production claim')
    (args.directory/'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(pairs=pairs, gpu_hours=result['gpu_hours']), indent=2))


if __name__ == '__main__':
    main()
