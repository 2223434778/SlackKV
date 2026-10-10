"""Apply frozen structural gates; keep every capacity and namespace."""
import argparse
import csv
import gzip
import json
from pathlib import Path
from experiments.host_prefix.run import CAPACITIES, POLICIES, quantile


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--directory', type=Path, required=True)
    args = p.parse_args()
    with gzip.open(args.directory / 'requests.csv.gz', 'rt') as f:
        records = list(csv.DictReader(f))
    with (args.directory / 'namespace_metrics.csv').open() as f:
        namespaces = list(csv.DictReader(f))
    summary = json.loads((args.directory / 'summary.json').read_text())
    pooled = {(r['capacity_proxy_tokens'], r['policy']): r for r in summary['pooled']}
    paired = {}
    for r in records:
        n, h, m, q = (int(r[k]) for k in ('input_proxy_tokens', 'hit_proxy_tokens',
                                          'uncached_proxy_tokens', 'attention_work_proxy'))
        assert 0 <= h <= n and m == n - h and q == n * n - h * h
        key = (r['trace_index'], r['model'], r['request_index'])
        observed = paired.setdefault(key, [])
        observed.append(n)
    assert all(len(v) == len(CAPACITIES) * len(POLICIES) and len(set(v)) == 1 for v in paired.values())
    rows = []
    for cap in CAPACITIES:
        for policy in POLICIES:
            selected = [r for r in records if int(r['capacity_proxy_tokens']) == cap and r['policy'] == policy]
            saved = pooled[(cap, policy)]
            assert len(selected) == len(paired)
            assert sum(int(r['uncached_proxy_tokens']) for r in selected) == saved['uncached_proxy_tokens']
            assert quantile([int(r['uncached_proxy_tokens']) for r in selected], .95) == saved['uncached_p95']
        base, rr, candidate = (pooled[(cap, s)] for s in ('block_lru', 'rr_tail', 'cost_tail'))
        reference = {(r['trace_index'], r['model']): r for r in namespaces
                     if int(r['capacity_proxy_tokens']) == cap and r['policy'] == 'block_lru'}
        current = [r for r in namespaces if int(r['capacity_proxy_tokens']) == cap and r['policy'] == 'cost_tail']
        nonregressing = sum(int(r['uncached_proxy_tokens']) <= 1.1 * int(reference[(r['trace_index'], r['model'])]['uncached_proxy_tokens']) for r in current)
        ratios = dict(uncached_vs_lru=candidate['uncached_proxy_tokens'] / base['uncached_proxy_tokens'],
                      p95_vs_lru=candidate['uncached_p95'] / base['uncached_p95'],
                      uncached_vs_rr=candidate['uncached_proxy_tokens'] / rr['uncached_proxy_tokens'],
                      p95_vs_rr=candidate['uncached_p95'] / rr['uncached_p95'],
                      writes_vs_lru=candidate['writes_proxy_tokens'] / base['writes_proxy_tokens'])
        passed = (ratios['uncached_vs_lru'] <= .9 and ratios['p95_vs_lru'] <= .9 and
                  ratios['uncached_vs_rr'] <= 1.05 and ratios['p95_vs_rr'] <= 1.05 and
                  ratios['writes_vs_lru'] <= 1.1 and nonregressing >= 20)
        rows.append(dict(capacity_proxy_tokens=cap, **ratios,
                         nonregressing_namespaces=nonregressing, namespaces=len(current), passed=passed))
    go = any(a['passed'] and b['passed'] for a, b in zip(rows, rows[1:]))
    result = dict(rows=rows, matched_requests=len(paired), per_request_arithmetic='PASS',
                  verdict='STRUCTURAL_GO_NOT_RUNTIME_GO' if go else 'REJECT_COST_TAIL',
                  qualification='fixed-event host-only model, not actual misses, SLO or E2E')
    (args.directory / 'assessment.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
