"""Validate actual TP2 feasibility evidence; never infer algorithm speedup."""
import argparse
import hashlib
import json
from pathlib import Path

from experiments.host_prefix.analyze_gpu import delta
from experiments.host_prefix.followup_analyze import read


def analyze(path):
    execution = json.loads((path/'execution.json').read_text())
    if execution['status'] == 'RUNNING':
        raise ValueError('unresolved GPU execution')
    if execution['status'] != 'COMPLETE':
        return dict(status=execution['status'], gpu_hours=execution['gpu_hours'])
    requests = json.loads(read(path/'requests.json'))
    inputs = json.loads(read(path/'inputs.json'))
    assert len(requests) == len(inputs) == 5
    assert [row['label'] for row in requests] == ['short', 'long', 'gpu_reuse', 'cpu_store', 'cpu_resume']
    assert inputs[1]['prompt'] == inputs[2]['prompt']
    assert inputs[3]['prompt'] == inputs[4]['prompt']
    result = []
    for row, payload in zip(requests, inputs):
        assert row['label'] == payload['label'] and len(payload['prompt']) == row['length']
        assert row['prompt_sha256'] == hashlib.sha256(json.dumps(payload['prompt']).encode()).hexdigest()
        response = row['response']
        assert response['issue_ns'] <= response['first_ns'] <= response['done_ns']
        assert len(response['tokens']) == 16
        sources = {source: delta(row['before'], row['after'], 'vllm:prompt_tokens_by_source_total', f'source="{source}"')
                   for source in ('local_compute', 'local_cache_hit', 'external_kv_transfer')}
        result.append(dict(label=row['label'], ttft_ms=(response['first_ns']-response['issue_ns'])/1e6,
                           e2e_ms=(response['done_ns']-response['issue_ns'])/1e6, sources=sources))
    # Do not interpret absent/unflushed counters as cache reuse evidence.
    reuse = result[2]['sources']['local_cache_hit']
    restore = result[4]['sources']['external_kv_transfer']
    valid_reuse = reuse is not None and reuse > 0 and restore is not None and restore > 0
    pool = min(execution['actual_gpu_pool_tokens'])
    cold_seconds = result[1]['e2e_ms'] / 1000
    return dict(status='VALID_EXECUTION_REUSE_OBSERVED' if valid_reuse else 'EXECUTION_COMPLETE_SOURCE_UNKNOWN',
                metrics=result, actual_pool_tokens=pool, actual_pool_kv_total_bytes=pool*262144,
                ready_seconds=execution['ready_seconds'], gpu_hours=execution['gpu_hours'],
                sequential_cold_21_sessions_3_turns_seconds=63*cold_seconds,
                output_equality_diagnostic=requests[1]['response']['tokens'] == requests[2]['response']['tokens'],
                cpu_restore_output_equality_diagnostic=requests[3]['response']['tokens'] == requests[4]['response']['tokens'],
                caveats=['Sequential extrapolation is a time-planning estimate, not mixed throughput.',
                         'Output equality diagnostic is not an independent cold/batch-invariant correctness oracle.',
                         'No matched policy comparison or speedup claim.'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.directory)
    (args.directory/'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
