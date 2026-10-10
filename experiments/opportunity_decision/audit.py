"""Direct sampled prefix comparisons, independent of the trie implementation."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from experiments.opportunity_decision.analyze import flatten


def lcp(a, b):
    for index, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return index
    return min(len(a), len(b))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    summary = json.loads((args.results / 'structure.json').read_text())
    assert hashlib.sha256(raw).hexdigest() == summary['source_prefix_sha256']
    with (args.results / 'requests.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == summary['requests']
    by_key = {(int(r['trace_index']), int(r['request_index'])): r for r in rows}
    checks = []
    for trace_index, line in enumerate(raw.splitlines()):
        requests, _ = flatten(json.loads(line))
        for index, request in enumerate(requests):
            if index % 37 and index != len(requests) - 1:
                continue
            candidates = [r for r in requests[:index] if r['model'] == request['model']
                          and r['t'] + r['api_time'] <= request['t']]
            best = max((lcp(request['hash_ids'], r['hash_ids']) for r in candidates), default=0) * 64
            cross = max((lcp(request['hash_ids'], r['hash_ids']) for r in candidates
                         if r['container'] != request['container']), default=0) * 64
            saved = by_key[(trace_index, index)]
            assert best == int(saved['completed_history_lcp_tokens'])
            assert cross == int(saved['other_container_completed_lcp_tokens'])
            checks.append({'trace_index': trace_index, 'request_index': index,
                           'direct_lcp_tokens': best, 'cross_container_tokens': cross})
    for key in ('completed_history_lcp_tokens', 'issued_history_lcp_tokens', 'other_container_completed_lcp_tokens'):
        fraction = sum(int(r[key]) for r in rows) / sum(int(r['proxy_input_tokens']) for r in rows)
        assert abs(fraction - summary['structural_input_token_fractions'][key]) < 1e-14
    for key in ('proxy_input_tokens', 'output_tokens'):
        values = sorted(int(r[key]) for r in rows)
        assert statistics.median(values) == summary['distributions'][key]['median']
        assert values[math.ceil(len(values)*.95) - 1] == summary['distributions'][key]['p95']
    result = {'csv_rows': len(rows), 'coverage_and_quantile_checks': 'PASS', 'direct_lcp_checks': len(checks),
              'selection': 'every37th request and final request in each of15traces', 'checks': checks}
    (args.results / 'audit.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'checks'}))


if __name__ == '__main__':
    main()
