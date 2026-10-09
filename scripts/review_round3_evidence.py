"""CPU-only spot check of committed Round 3 evidence; no serving or GPU work."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics


def read(path: Path) -> bytes:
    if path.exists():
        return path.read_bytes()
    return gzip.decompress(Path(str(path) + '.gz').read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    fork = args.root / 'results/autonomous_fork'
    run = fork / 'fork-decode-control-4'
    paths = {
        'requests': run / 'requests.json',
        'episodes': run / 'episodes.json',
        'log': run / 'server_logs/fork-decode-control-4.log',
        'metrics': run / 'metrics_after.prom',
        'native_scheduler': fork / 'native_offloading_scheduler.py',
        'runtime_scheduler': fork / 'runtime_scheduler_final.py',
    }
    data = {key: read(path) for key, path in paths.items()}
    hashes = {key: hashlib.sha256(value).hexdigest() for key, value in data.items()}
    assert hashes['native_scheduler'] == 'bc883cf47d882d4406d7708a8d3520abe2752f545985705126f41fca66558bf0'
    assert hashes['runtime_scheduler'] == '5c0ef2cd7cc65a452d86c4751e43971efebf5d97066cb5f8d8126f3f5a8e43c8'
    requests = json.loads(data['requests'])
    episodes = json.loads(data['episodes'])
    log = data['log'].decode()

    def events(marker: str) -> list[dict]:
        return [json.loads(line.split(marker + ' ', 1)[1])
                for line in log.splitlines() if marker + ' ' in line]

    source = [e for e in events('SLACKKV_SOURCE_TRACE') if e['session'].startswith('F')]
    trace = events('SLACKKV_RESUME_TRACE')
    event_map: dict[str, dict] = {}
    for event in trace:
        per_request = event_map.setdefault(event['session'], {})
        assert event['event'] not in per_request, 'This run should have no repeated lifecycle events'
        per_request[event['event']] = event['timestamp_monotonic']
    assert len(requests) == 51 and len({r['session'] for r in requests}) == 51
    branches = [r for r in requests if r['session'].startswith('F')]
    background = [r for r in requests if r['session'].startswith('BG')]
    assert len(branches) == 48 and len(background) == 3
    assert all(len(r['tokens']) == 128 for r in branches)
    assert all(len(r['tokens']) == 1024 for r in background)
    assert len(source) == 48
    actual_h2d = sum(float(line.rsplit(' ', 1)[1]) for line in data['metrics'].decode().splitlines()
                     if line.startswith('vllm:kv_offload_total_bytes_total{') and 'transfer_type="CPU_to_GPU"' in line)
    source_bytes = sum(e['external_cached_tokens'] for e in source) * 28672
    assert actual_h2d == source_bytes == 3 * 24576 * 28672
    per_episode = []
    for ep in range(3):
        calls = [r for r in branches if r['session'].startswith(f'F{ep}:')]
        duration = (max(r['done_ns'] for r in calls) - min(r['issue_ns'] for r in calls)) / 1e6
        assert abs(duration - episodes[ep]['workflow_ms']) < .001
        first_sources = [e for e in source if e['session'].startswith(f'F{ep}:') and e['session'].endswith(':0')]
        assert sum(e['external_cached_tokens'] == 24576 for e in first_sources) == 1
        assert sum(e['local_cached_tokens'] == 24576 for e in first_sources) == 7
        per_episode.append({'episode': ep, 'workflow_ms': duration,
                            'external_root_loads': 1, 'local_root_hits': 7})
    decompositions = []
    for r in branches:
        e = event_map[r['session']]
        components = {
            'client_to_engine_ms': (e['RESUME_ARRIVED'] - r['issue_ns'] / 1e9) * 1000,
            'pre_admit_ms': (e['RESUME_ADMITTED'] - e['RESUME_ARRIVED']) * 1000,
            'admit_to_engine_first_ms': (e['FIRST_TOKEN'] - e['RESUME_ADMITTED']) * 1000,
            'engine_first_to_client_ms': (r['first_ns'] / 1e9 - e['FIRST_TOKEN']) * 1000,
        }
        assert all(value >= 0 for value in components.values())
        assert abs(sum(components.values()) - r['ttft_ms']) < .001
        decompositions.append({'session': r['session'], 'components_ms': components,
                               'own_remote_wait_event': 'WAITING_FOR_REMOTE_KVS' in e})
    medians = {}
    for turn in (0, 1):
        rows = [r['components_ms'] for r in decompositions if r['session'].endswith(f':{turn}')]
        medians[str(turn)] = {key: statistics.median(r[key] for r in rows) for key in rows[0]}
    output = {
        'reviewed_commit': '61139e310a5976cf4311e8bbc4badf82f56439af',
        'scope': 'One representative run and two saved runtime sources; not a full 101-file audit or GPU rerun.',
        'uncompressed_input_sha256': hashes,
        'run': 'fork-decode-control-4',
        'branch_calls': len(branches), 'background_calls': len(background),
        'actual_h2d_bytes': actual_h2d, 'source_restore_bytes': source_bytes,
        'workflow_median_ms': statistics.median(e['workflow_ms'] for e in per_episode),
        'branch_ttft_p95_ms': sorted(r['ttft_ms'] for r in branches)[math.ceil(.95 * len(branches)) - 1],
        'remote_wait_event_count': sum(r['own_remote_wait_event'] for r in decompositions),
        'component_medians_by_turn_ms': medians,
        'episodes': per_episode,
        'per_request_decomposition': decompositions,
        'qualification': 'Same-host monotonic clocks assumed by the saved runner. Components are observed intervals, not pure DMA/kernel/network times. Component medians are not additive. Lookup deferral duration is uninstrumented; follower wait cannot be assigned wholly to DMA. No causal speedup inferred.',
    }
    rendered = json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding='utf-8')
    print(json.dumps({k: v for k, v in output.items() if k != 'per_request_decomposition'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
