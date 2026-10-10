"""Independent raw input, work and transfer accounting; missing means UNKNOWN."""
import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path
from experiments.autonomous_fork.analyze import counter, read


def delta(before, after, name, label=None):
    a, b = counter(before, name, label), counter(after, name, label)
    if a is None or b is None:
        return None
    if b < a:
        raise ValueError('counter decreased')
    return b - a


def checkpoint(directory):
    rows = json.loads(read(directory / 'requests.json'))
    bg = json.loads(read(directory / 'backgrounds.json'))
    inputs = json.loads(read(directory / 'inputs.json'))
    assert len(rows) == len(bg) == 6 and len(inputs) == 12
    actual = {(r['turn'], r['role']): r for r in rows + bg}
    for r in inputs:
        saved = actual[(r['turn'], r['role'])]
        assert hashlib.sha256(json.dumps(r['prompt']).encode()).hexdigest() == saved['prompt_sha256']
        assert len(r['prompt']) == saved['length']
        assert saved['output_count'] == len(saved['output_tokens']) == (16 if r['role'] == 'agent' else 256)
        assert saved['issue_ns'] <= saved['first_ns'] <= saved['done_ns']
        assert abs(saved['ttft_ms'] - (saved['first_ns'] - saved['issue_ns'])/1e6) < 1e-9
        assert abs(saved['e2e_ms'] - (saved['done_ns'] - saved['issue_ns'])/1e6) < 1e-9
    input_map = {(r['turn'], r['role']): r['prompt'] for r in inputs}
    for turn in range(6):
        assert input_map[(turn, 'bg')][:16] != input_map[(turn, 'agent')][:16]
        if turn:
            previous = input_map[(turn-1, 'agent')]
            assert input_map[(turn, 'agent')][:len(previous)] == previous
    for a, b in zip(rows, bg):
        assert b['first_ns'] <= a['issue_ns'] <= b['done_ns']
    before, after = read(directory / 'before.prom').decode(), read(directory / 'after.prom').decode()
    # The fresh process's only prior call explicitly disables offload. Dynamic
    # Prometheus transfer labels do not exist until the first job, unlike source
    # counters. This known initial state permits0 here, not for arbitrary absence.
    assert counter(before, 'vllm:prompt_tokens_by_source_total', 'source="external_kv_transfer"') == 0
    transfers = {}
    for key, tag in (('h2d_bytes', 'CPU_to_GPU'), ('d2h_bytes', 'GPU_to_CPU')):
        label = f'transfer_type="{tag}"'
        initial = counter(before, 'vllm:kv_offload_total_bytes_total', label)
        assert initial in (None, 0)
        transfers[key] = counter(after, 'vllm:kv_offload_total_bytes_total', label)
        if transfers[key] is None:
            raise ValueError('missing final transfer counter remains UNKNOWN')
    external = delta(before, after, 'vllm:prompt_tokens_by_source_total', 'source="external_kv_transfer"')
    local = delta(before, after, 'vllm:prompt_tokens_by_source_total', 'source="local_cache_hit"')
    computed = delta(before, after, 'vllm:prompt_tokens_by_source_total', 'source="local_compute"')
    # Native stats can publish the fixed256-token warmup after the before snapshot.
    warmup_seen = counter(before, 'vllm:prompt_tokens_by_source_total', 'source="local_compute"')
    assert warmup_seen is not None and 0 <= warmup_seen <= 256
    pending_warmup = 256 - warmup_seen
    if computed is not None:
        computed -= pending_warmup
    accounted = (external is not None and computed is not None and local == 0 and transfers['h2d_bytes'] == external * 28672
                 and external + local + computed == sum(r['length'] for r in rows + bg))
    if not accounted:
        raise ValueError('real source accounting not verified')
    resumes = [r['ttft_ms'] for r in rows if r['turn'] > 0]
    log = read(directory / 'server_logs' / (directory.name + '.log')).decode()
    capacity = re.search(r'GPU KV cache size: ([\d,]+) tokens', log)
    if capacity is None:
        raise ValueError('actual GPU pool capacity missing')
    final_running = counter(after, 'vllm:num_requests_running')
    final_waiting = counter(after, 'vllm:num_requests_waiting')
    assert final_running == final_waiting == 0
    return dict(run=directory.name, source_accounting='PASS', actual_external_tokens=external,
                actual_local_tokens=local, computed_tokens=computed, **transfers,
                agent_computed_input_tokens=computed-sum(r['length'] for r in bg),
                unavoidable_first_and_growth_tokens=rows[-1]['length'],
                repeated_history_prefill_tokens=computed-sum(r['length'] for r in bg)-rows[-1]['length'],
                warmup_tokens_published_after_before=pending_warmup,
                actual_gpu_pool_tokens=int(capacity.group(1).replace(',', '')),
                final_running=final_running, final_waiting=final_waiting,
                resume_p95_ms=max(resumes), resume_median_ms=statistics.median(resumes),
                background_median_ms=statistics.median(r['e2e_ms'] for r in bg),
                background_overlap='PASS', paired_inputs={(f"{r['turn']}:{r['role']}"): r['prompt_sha256'] for r in rows + bg})


def calibration(directory):
    rows = json.loads(read(directory / 'requests.json'))
    assert len(rows) == 36 and all(r['output_count'] == len(r['output_tokens']) == 16 for r in rows)
    identities = {(r['length'], r['rep']): set() for r in rows}
    for r in rows:
        identities[(r['length'], r['rep'])].add(r['prompt_sha256'])
    assert all(len(v) == 1 for v in identities.values())
    expected = sum(r['prefix'] - (16 if r['prefix'] == r['length'] else 0) for r in rows)
    observed = counter(rows[-1]['metrics_after'], 'vllm:prefix_cache_hits_total')
    assert observed == expected
    groups = []
    for length in (8192, 16384, 28672):
        for fraction in (0, .5, .875, 1):
            selected = [r for r in rows if r['length'] == length and r['prefix'] == length * fraction]
            groups.append(dict(length=length, prefix=int(length*fraction), n=len(selected),
                               median_ttft_ms=statistics.median(r['ttft_ms'] for r in selected)))
    return dict(measured_calls=36, matched_full_prompt_groups=9, actual_gpu_hit_tokens=observed,
                expected_gpu_hit_tokens=expected, cumulative_hit_accounting='PASS', rows=groups,
                qualification='isolated serial cost diagnostic, no candidate speedup')


def oracle(directory):
    result = json.loads(read(directory / 'oracle.json'))
    before, after, final = (read(directory / name).decode() for name in ('before.prom', 'after_restore.prom', 'after_cold.prom'))
    restore_external = delta(before, after, 'vllm:prompt_tokens_by_source_total', 'source="external_kv_transfer"')
    restore_local = delta(before, after, 'vllm:prompt_tokens_by_source_total', 'source="local_cache_hit"')
    cold_external = delta(after, final, 'vllm:prompt_tokens_by_source_total', 'source="external_kv_transfer"')
    cold_local = delta(after, final, 'vllm:prompt_tokens_by_source_total', 'source="local_cache_hit"')
    assert restore_external == 9360 and restore_local == cold_external == cold_local == 0
    assert len(result['restored']['tokens']) == 16 and result['restored']['tokens'] == result['cold']['tokens']
    return dict(tokens=16, token_match='PASS', restore_external_tokens=restore_external,
                independent_cold_external_tokens=cold_external, local_tokens=0,
                qualification='one invariant input; timing excluded; not general KV quality proof')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--directory', type=Path, required=True)
    args = p.parse_args()
    runs = {name: checkpoint(args.directory / name) for name in ('full1', 'frontier1', 'full2', 'frontier2')}
    pairs = []
    for suffix in ('1', '2'):
        base, candidate = runs['full' + suffix], runs['frontier' + suffix]
        assert base['paired_inputs'] == candidate['paired_inputs']
        assert base['actual_gpu_pool_tokens'] == candidate['actual_gpu_pool_tokens']
        assert candidate['actual_external_tokens'] == 5 * 9360
        change = {k: 100*(candidate[k]/base[k]-1) for k in ('d2h_bytes', 'resume_p95_ms', 'resume_median_ms', 'background_median_ms')}
        change['total_transfer_bytes'] = 100*((candidate['h2d_bytes']+candidate['d2h_bytes'])/(base['h2d_bytes']+base['d2h_bytes'])-1)
        passed = change['d2h_bytes'] <= -50 and change['total_transfer_bytes'] <= -40 and change['resume_p95_ms'] <= 5 and change['background_median_ms'] <= 5
        pairs.append(dict(pair=suffix, changes_percent=change, mechanism_gate_passed=passed))
    executions = [json.loads(f.read_text()) for f in args.directory.glob('*/execution.json')]
    assert all(r['status'] != 'RUNNING' for r in executions)
    result = dict(runs=runs, pairs=pairs, calibration=calibration(args.directory/'calibration2'),
                  oracle=oracle(args.directory/'oracle2'),
                  gpu_hours=sum(r['gpu_hours'] for r in executions),
                  scope='synthetic single-path host-overflow forced-GPU-cold mechanism screen',
                  production_speedup='UNKNOWN', multi_session_fairness='UNKNOWN',
                  gpu_owned_memory_time='UNKNOWN')
    (args.directory / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('pairs', 'gpu_hours', 'oracle')}, indent=2))


if __name__ == '__main__':
    main()
