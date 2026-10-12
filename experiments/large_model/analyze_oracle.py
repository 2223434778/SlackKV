"""Reproduce the sensitive partial-prefix oracle from saved raw evidence."""
import argparse
import hashlib
import json
from pathlib import Path

from experiments.host_prefix.analyze_gpu import delta
from experiments.host_prefix.followup_analyze import read


def analyze(path):
    execution = json.loads((path/'execution.json').read_text())
    if execution['status'] != 'COMPLETE' or not execution['batch_invariant']:
        raise ValueError('oracle requires completed batch-invariant execution')
    assert execution['gpu_release_verified'] and not execution['owned_live_pids_after_cleanup']
    rows = json.loads(read(path/'requests.json'))
    inputs = json.loads(read(path/'inputs.json'))
    labels = [f'{stage}_{key}' for stage in ('store', 'resume', 'independent_cold') for key in ('a', 'b')]
    assert [r['label'] for r in rows] == [r['label'] for r in inputs] == labels
    requests = {r['label']: r for r in rows}
    payloads = {r['label']: r['prompt'] for r in inputs}
    for row in rows:
        prompt = payloads[row['label']]
        assert len(prompt) == row['length'] == 28672
        assert hashlib.sha256(json.dumps(prompt).encode()).hexdigest() == row['prompt_sha256']
        response = row['response']
        assert len(response['tokens']) == 16
        assert response['issue_ns'] <= response['first_ns'] <= response['done_ns']
    cases = {}
    for key, marker in [('a', 'CEDAR17'), ('b', 'MAPLE42')]:
        store, resume, cold = [requests[f'{stage}_{key}'] for stage in ('store', 'resume', 'independent_cold')]
        assert store['cache_salt'] == resume['cache_salt'] == f'stored_{key}'
        assert cold['cache_salt'] == f'independent_{key}'
        assert store['max_offload_tokens'] == resume['max_offload_tokens'] == 12480
        assert cold['max_offload_tokens'] == 0
        assert store['prompt_sha256'] == resume['prompt_sha256'] == cold['prompt_sha256']
        assert store['response']['tokens'] == resume['response']['tokens'] == cold['response']['tokens']
        for row in (store, resume, cold):
            assert row['response']['expected_marker'] == marker
            assert row['response']['decoded'].startswith(marker)
        sources = {}
        for stage, row in [('resume', resume), ('independent_cold', cold)]:
            source = {s: delta(row['before'], row['after'], 'vllm:prompt_tokens_by_source_total', f'source="{s}"')
                      for s in ('local_compute', 'local_cache_hit', 'external_kv_transfer')}
            expected_external = 12480 if stage == 'resume' else 0
            assert source == dict(local_compute=28672-expected_external, local_cache_hit=0,
                                  external_kv_transfer=expected_external)
            sources[stage] = source
        cases[key] = dict(marker=marker, **sources, tokens_equal=True)
    assert requests['store_a']['response']['tokens'] != requests['store_b']['response']['tokens']
    return dict(status='PASS_SENSITIVE_PARTIAL_PREFIX_ORACLE', cases=cases,
                caveat='two meaningful inputs, not all lifecycle/quality cases')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.directory)
    (args.directory/'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
