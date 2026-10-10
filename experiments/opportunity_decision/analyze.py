"""Analyze explicit Weka containers without inferring a ground-truth agent DAG."""
import argparse
import csv
import hashlib
import heapq
import json
import math
import statistics
from collections import Counter
from pathlib import Path


class Prefixes:
    def __init__(self):
        self.children = [{}]
        self.containers = [set()]

    def insert(self, hashes, container):
        node = 0
        for value in hashes:
            child = self.children[node].get(value)
            if child is None:
                child = len(self.children)
                self.children[node][value] = child
                self.children.append({})
                self.containers.append(set())
            node = child
            self.containers[node].add(container)

    def match(self, hashes, *, outside=None):
        node, length = 0, 0
        for value in hashes:
            child = self.children[node].get(value)
            if child is None or (outside is not None and not self.containers[child] - {outside}):
                break
            node = child
            length += 1
        return length


def distribution(values):
    if not values:
        return None
    values = sorted(values)
    return {'n': len(values), 'min': values[0], 'median': statistics.median(values),
            'p95': values[math.ceil(len(values) * .95) - 1], 'max': values[-1]}


def flatten(trace):
    if trace['block_size'] != 64 or trace['hash_id_scope'] != 'local':
        raise ValueError('unsupported hash identity semantics')
    entries, groups = [], []
    for index, entry in enumerate(trace['requests']):
        if entry['type'] == 'subagent':
            groups.append(entry)
            container = f'child:{index}'
            requests = entry['requests']
            if any(r['t'] + 1e-6 < entry['t'] for r in requests):
                raise ValueError('producer absolute timestamp assumption violated')
        else:
            container, requests = 'root', [entry]
        for position, request in enumerate(requests):
            for key in ('t', 'api_time'):
                if not isinstance(request[key], (float, int)) or not math.isfinite(request[key]) or request[key] < 0:
                    raise ValueError('invalid timestamp/duration')
            if request['in'] != len(request['hash_ids']) * 64 or request['out'] < 0:
                raise ValueError('invalid block/input accounting')
            entries.append({**request, 'container': container, 'entry': index, 'position': position})
    return sorted(entries, key=lambda r: (r['t'], r['entry'], r['position'])), groups


def analyze_trace(trace, index):
    requests, groups = flatten(trace)
    ready, arrived, pending, previous, records = {}, {}, [], {}, []
    for sequence, request in enumerate(requests):
        while pending and pending[0][0] <= request['t']:
            _, _, finished = heapq.heappop(pending)
            ready.setdefault(finished['model'], Prefixes()).insert(finished['hash_ids'], finished['container'])
        model, container, hashes = request['model'], request['container'], request['hash_ids']
        completed = ready.setdefault(model, Prefixes())
        issued = arrived.setdefault(model, Prefixes())
        predecessor = previous.get((container, model))
        adjacent = 0
        if predecessor:
            for a, b in zip(hashes, predecessor['hash_ids']):
                if a != b:
                    break
                adjacent += 1
        records.append({'trace_index': index, 'request_index': sequence,
                        'container': container, 'model': model, 't_s': request['t'],
                        'api_time_s': request['api_time'], 'ttft_s': request.get('ttft'),
                        'reported_think_time_s': request.get('think_time'),
                        'proxy_input_tokens': request['in'], 'output_tokens': request['out'],
                        'completed_history_lcp_tokens': completed.match(hashes) * 64,
                        'issued_history_lcp_tokens': issued.match(hashes) * 64,
                        'other_container_completed_lcp_tokens': completed.match(hashes, outside=container) * 64,
                        'container_adjacent_lcp_tokens': adjacent * 64 if predecessor else None,
                        'container_input_delta': request['in'] - predecessor['in'] if predecessor else None,
                        'container_previous_end_gap_s': request['t'] - predecessor['t'] - predecessor['api_time'] if predecessor else None,
                        'block_path_sha256': hashlib.sha256(json.dumps(hashes, separators=(',', ':')).encode()).hexdigest()})
        previous[(container, model)] = request
        # Simultaneous arrivals are not earlier history; this insertion is only an
        # optimistic structural envelope, never a physical readiness assertion.
        issued.insert(hashes, container)
        heapq.heappush(pending, (request['t'] + request['api_time'], sequence, request))
    summary = {'trace_index': index, 'trace_sha256': hashlib.sha256(json.dumps(trace, sort_keys=True).encode()).hexdigest(),
               'request_count': len(requests), 'subagent_groups': len(groups),
               'models': sorted({r['model'] for r in requests}),
               'span_s': max(r['t'] + r['api_time'] for r in requests) if requests else None,
               'group_duration_ms': distribution([g['duration_ms'] for g in groups]),
               'nested_requests': sum(len(g['requests']) for g in groups)}
    summary['all_history_unique_input_blocks_by_model'] = {
        model: len(prefixes.children) - 1 for model, prefixes in arrived.items()}
    return records, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records, traces = [], []
    raw = args.input.read_bytes()
    for index, line in enumerate(raw.splitlines()):
        rr, ss = analyze_trace(json.loads(line), index)
        records.extend(rr)
        traces.append(ss)
    if not records:
        raise ValueError('no complete source requests')
    total = sum(r['proxy_input_tokens'] for r in records)
    result = {'source_prefix_sha256': hashlib.sha256(raw).hexdigest(), 'traces': traces,
              'requests': len(records), 'models': dict(Counter(r['model'] for r in records)),
              'subagent_groups': sum(t['subagent_groups'] for t in traces),
              'nested_requests': sum(t['nested_requests'] for t in traces),
              'per_trace_model_unique_input_tokens': distribution([
                  blocks * 64 for t in traces for blocks in t['all_history_unique_input_blocks_by_model'].values()]),
              'distributions': {k: distribution([r[k] for r in records if r[k] is not None]) for k in
                                ('proxy_input_tokens', 'output_tokens', 'api_time_s', 'ttft_s', 'reported_think_time_s',
                                 'container_input_delta', 'container_previous_end_gap_s')},
              'structural_input_token_fractions': {k: sum(r[k] for r in records) / total for k in
                                                  ('completed_history_lcp_tokens', 'issued_history_lcp_tokens',
                                                   'other_container_completed_lcp_tokens')},
              'over32768_requests': sum(r['proxy_input_tokens'] > 32768 for r in records),
              'over131072_requests': sum(r['proxy_input_tokens'] > 131072 for r in records),
              'negative_container_end_gaps': sum(r['container_previous_end_gap_s'] is not None
                                                and r['container_previous_end_gap_s'] < -1e-6 for r in records),
              'nonadjacent_better_than_container_predecessor': sum(r['container_adjacent_lcp_tokens'] is not None and
                  r['completed_history_lcp_tokens'] > r['container_adjacent_lcp_tokens'] for r in records),
              'caveats': ['first file prefix, producer-filtered; not random population estimate',
                          'input lengths are proxy hash blocks, not real target-model tokenization',
                          'completed input-only history assumes infinite cache; no actual hit/miss observation',
                          'API time is recorded endpoint wall time, not new hardware compute cost',
                          'explicit containers are not proven agent chains; no inferred true dependency DAG',
                          'other-container reuse does not prove fork or simultaneous sharing',
                          'tool-only latency, host misses, owned memory-time, cross-session identity UNKNOWN']}
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'requests.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, records[0].keys(), lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)
    (args.output / 'structure.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'traces'}, indent=2))


if __name__ == '__main__':
    main()
