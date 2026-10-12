"""CPU-only input-prefix footprint audit; endpoint timelines are not GPU traces."""
import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from experiments.opportunity_decision.analyze import distribution, flatten


def footprint(requests):
    """One trace/model namespace, with shared prefix identity and half-open times.

    Future-needed keeps every completed input block with a later arrival. It is
    an offline no-eviction retention target, not a serving policy or tight bound.
    Generated output has no hash identity and is excluded from shared input KV.
    """
    children = [{}]
    paths, events = [], defaultdict(lambda: {'start': [], 'end': []})
    arrivals = defaultdict(list)
    for i, request in enumerate(requests):
        node, path = 0, []
        for value in request['hash_ids']:
            child = children[node].get(value)
            if child is None:
                child = len(children)
                children[node][value] = child
                children.append({})
            node = child
            path.append(node)
            arrivals[node].append(request['t'])
        paths.append(path)
        events[request['t']]['start'].append(i)
        events[request['t'] + request['api_time']]['end'].append(i)
    for times in arrivals.values():
        times.sort()
    active, ready = Counter(), set()
    peak_active = peak_retained = peak_combined = peak_concurrency = 0
    active_requests, output_reserve = set(), 0
    peak_input_plus_output = 0
    input_memory_token_seconds = retained_memory_token_seconds = 0.0
    previous_time = None
    rows = []
    for t, event in sorted(events.items()):
        if previous_time is not None:
            elapsed = t - previous_time
            input_memory_token_seconds += len(active) * 64 * elapsed
            retained_memory_token_seconds += len(ready | active.keys()) * 64 * elapsed
        # Expire at the last arrival. Completing at exactly an arrival time is
        # considered ready, but no idle retention interval is needed there.
        for i in event['start']:
            for node in paths[i]:
                times = arrivals[node]
                if times[-1] <= t:
                    ready.discard(node)
        nonzero_ends = [i for i in event['end'] if requests[i]['api_time'] > 0]
        for i in nonzero_ends:
            active_requests.remove(i)
            output_reserve -= requests[i]['out']
            for node in paths[i]:
                active[node] -= 1
                if active[node] == 0:
                    del active[node]
                if arrivals[node][-1] > t:
                    ready.add(node)
        for i in event['start']:
            if requests[i]['api_time'] == 0:
                for node in paths[i]:
                    if arrivals[node][-1] > t:
                        ready.add(node)
                continue
            active_requests.add(i)
            output_reserve += requests[i]['out']
            active.update(paths[i])
        union = len(ready | active.keys())
        peak_active = max(peak_active, len(active))
        peak_retained = max(peak_retained, len(ready))
        peak_combined = max(peak_combined, union)
        peak_concurrency = max(peak_concurrency, len(active_requests))
        peak_input_plus_output = max(peak_input_plus_output, len(active) * 64 + output_reserve)
        rows.append({'t_s': t, 'active_requests': len(active_requests),
                     'active_input_proxy_tokens': len(active) * 64,
                     'completed_future_needed_proxy_tokens': len(ready) * 64,
                     'union_proxy_tokens': union * 64})
        previous_time = t
    if active or active_requests or ready or output_reserve:
        raise AssertionError('unreleased interval state')
    # Exact intervening *issued input* union until the next same-container call.
    # This is a diagnostic distance, not an actual LRU eviction measurement.
    previous, distances = {}, []
    ordered = sorted(range(len(requests)), key=lambda i: requests[i]['t'])
    start_times = [requests[i]['t'] for i in ordered]
    for i in ordered:
        request = requests[i]
        container = request['container']
        if container in previous:
            predecessor = requests[previous[container]]
            end = predecessor['t'] + predecessor['api_time']
            if end <= request['t']:
                intervening = set()
                for j in ordered[bisect_left(start_times, end):bisect_left(start_times, request['t'])]:
                    intervening.update(paths[j])
                distances.append(len(intervening) * 64)
        previous[container] = i
    return {'requests': len(requests), 'cumulative_unique_input_proxy_tokens': (len(children) - 1) * 64,
            'peak_active_input_proxy_tokens': peak_active * 64,
            'peak_completed_future_needed_proxy_tokens': peak_retained * 64,
            'peak_active_and_future_needed_union_proxy_tokens': peak_combined * 64,
            'peak_active_input_plus_full_output_reserve_tokens': peak_input_plus_output,
            'peak_endpoint_inflight_requests': peak_concurrency,
            'active_input_proxy_token_seconds': input_memory_token_seconds,
            'active_and_future_needed_union_proxy_token_seconds': retained_memory_token_seconds,
            'same_container_intervening_issued_unique_input_proxy_tokens': distribution(distances),
            'events': rows}


def kv_bytes_per_token(config, dtype_bytes=2):
    head_dim = config.get('head_dim', config['hidden_size'] // config['num_attention_heads'])
    return 2 * config['num_hidden_layers'] * config['num_key_value_heads'] * head_dim * dtype_bytes


def analyze_trace(trace, trace_index):
    requests, _ = flatten(trace)
    grouped = defaultdict(list)
    for request in requests:
        grouped[request['model']].append(request)
    return [{'trace_index': trace_index, 'model': model, **footprint(subset)}
            for model, subset in sorted(grouped.items())]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.expected_sha256:
        raise ValueError('source hash mismatch')
    namespaces = []
    for trace_index, line in enumerate(raw.splitlines()):
        namespaces.extend(analyze_trace(json.loads(line), trace_index))
    if not namespaces:
        raise ValueError('no namespaces')
    keys = ('cumulative_unique_input_proxy_tokens', 'peak_active_input_proxy_tokens',
            'peak_active_and_future_needed_union_proxy_tokens', 'peak_endpoint_inflight_requests')
    result = {'source_sha256': args.expected_sha256, 'gpu_hours_added': 0,
              'retention_peak_increased_namespaces': sum(
                  row['peak_active_and_future_needed_union_proxy_tokens'] > row['peak_active_input_proxy_tokens']
                  for row in namespaces),
              'namespace_distributions': {key: distribution([row[key] for row in namespaces]) for key in keys},
              'namespaces': namespaces,
              'caveats': ['Independent trace/model namespaces are never superposed.',
                          '64-token proxy blocks are not target-model tokens.',
                          'Endpoint wall-time inflight intervals are not A6000 compute intervals.',
                          'Future-needed is an offline no-eviction input-only retention target, not a policy.',
                          'Output full reserve is a conservative numerical token-count diagnostic; input/output tokenizers differ.',
                          'Intervening input union excludes output growth, cache policy and concurrent compute effects.']}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'footprints.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result['namespace_distributions'], indent=2))


if __name__ == '__main__':
    main()
