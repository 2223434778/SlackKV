"""Fixed public request-time replay, completion-time writes, no latency claims."""
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
from experiments.host_prefix.model import Cache, Tree
from experiments.opportunity_decision.analyze import flatten

POLICIES = ('block_lru', 'prefix_lru', 'rr_tail', 'cost_tail')
CAPACITIES = (16384, 65536, 131072, 262144, 524288, 1048576, 2097152)


def replay(requests, capacity, policy):
    if any(r['api_time'] <= 0 for r in requests):
        raise ValueError('zero-duration completion ordering unsupported')
    tree = Tree()
    cache = Cache(capacity // 64, policy, tree)
    paths = [tree.path(r['hash_ids']) for r in requests]
    events = []
    for index, r in enumerate(requests):
        events.extend(((r['t'], 1, index), (r['t'] + r['api_time'], 0, index)))
    rows = []
    for _, kind, index in sorted(events):
        path = paths[index]
        if kind:
            hit = cache.hit(path) * 64
            rows.append({'request_index': index, 'input_proxy_tokens': len(path) * 64,
                         'hit_proxy_tokens': hit, 'uncached_proxy_tokens': len(path) * 64 - hit,
                         'attention_work_proxy': (len(path) * 64) ** 2 - hit ** 2})
            cache.touch(path)
        else:
            cache.store(path)
            cache.validate()
    return rows, {'writes_proxy_tokens': cache.writes * 64,
                  'evictions_proxy_tokens': cache.evictions * 64,
                  'rejected_batches': cache.rejected_batches}


def quantile(values, percentile):
    values = sorted(values)
    return values[math.ceil(len(values) * percentile) - 1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    raw = args.input.read_bytes()
    records, summaries = [], []
    for trace_index, line in enumerate(raw.splitlines()):
        rr, _ = flatten(json.loads(line))
        for model in sorted({r['model'] for r in rr}):
            requests = [r for r in rr if r['model'] == model]
            for capacity in CAPACITIES:
                for policy in POLICIES:
                    rows, counters = replay(requests, capacity, policy)
                    base = dict(trace_index=trace_index, model=model, capacity_proxy_tokens=capacity, policy=policy)
                    summaries.append(dict(base, requests=len(rows), **counters,
                        uncached_proxy_tokens=sum(r['uncached_proxy_tokens'] for r in rows),
                        uncached_p95=quantile([r['uncached_proxy_tokens'] for r in rows], .95),
                        attention_work_proxy=sum(r['attention_work_proxy'] for r in rows)))
                    records.extend(dict(base, **r) for r in rows)
        print(f'trace {trace_index} complete', flush=True)
    for name, rows in (('requests.csv.gz', records), ('namespace_metrics.csv', summaries)):
        opener = gzip.open if name.endswith('.gz') else open
        with opener(args.output / name, 'wt', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, rows[0].keys(), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    pooled = []
    for capacity in CAPACITIES:
        for policy in POLICIES:
            rows = [r for r in records if r['capacity_proxy_tokens'] == capacity and r['policy'] == policy]
            counters = [r for r in summaries if r['capacity_proxy_tokens'] == capacity and r['policy'] == policy]
            pooled.append(dict(capacity_proxy_tokens=capacity, policy=policy,
                uncached_proxy_tokens=sum(r['uncached_proxy_tokens'] for r in rows),
                uncached_p95=quantile([r['uncached_proxy_tokens'] for r in rows], .95),
                uncached_p99=quantile([r['uncached_proxy_tokens'] for r in rows], .99),
                attention_work_proxy=sum(r['attention_work_proxy'] for r in rows),
                writes_proxy_tokens=sum(r['writes_proxy_tokens'] for r in counters)))
    (args.output / 'summary.json').write_text(json.dumps(dict(
        source_sha256=hashlib.sha256(raw).hexdigest(), pooled=pooled,
        gpu_hours=0, qualification='host-only structural counterfactual, not latency or actual misses'), indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
