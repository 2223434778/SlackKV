"""Independent raw arithmetic and workload audit, CPU only."""
import json
import gzip
import math
import statistics
from pathlib import Path

root = Path('results/autonomous_round4')
def read(path):
    return path.read_bytes() if path.exists() else gzip.decompress(Path(str(path) + '.gz').read_bytes())
summary = json.loads((root / 'round_summary.json').read_text())
control = json.loads((root / 'control_summary.json').read_text())
reference_inputs = None
reference_outputs = None
reference_bg = None
records = []
for path in sorted(root.glob('*/execution.json')):
    config = json.loads(path.read_text())
    if config['status'] != 'VALID' or path.parent.name == 'retention-smoke':
        continue
    run_paths = list(path.parent.glob('runs/*/run.json'))
    run_paths += [Path(str(p)[:-3]) for p in path.parent.glob('runs/*/run.json.gz')
                  if not Path(str(p)[:-3]).exists()]
    run_path, = run_paths
    run = json.loads(read(run_path))
    turns = run['turn_rows']
    assert len(turns) == 40 and run['turns_complete'] == 40
    inputs = {t['episode_id'] + ':turn:' + str(t['turn_index']): t['prompt_sha256'] for t in turns}
    outputs = {t['episode_id'] + ':turn:' + str(t['turn_index']): t['output_tokens'] for t in turns}
    bg = [(c['prompt_tokens'], c['generated_tokens'], c['prompt_sha256']) for c in run['background_chunks']]
    if reference_inputs is None:
        reference_inputs, reference_outputs, reference_bg = inputs, outputs, bg
    assert inputs == reference_inputs and outputs == reference_outputs and bg == reference_bg
    assert run['BG_total_generated_tokens'] == 8192 and run['BG_tail_request_finish_ms'] > 0
    assert not run['OOM'] and run['lease_leaks'] == run['logical_leaks'] == 0
    session = {key: (max(t['finish_ns'] for t in turns if t['episode_id'] == key)
                     - min(t['issue_ns'] for t in turns if t['episode_id'] == key)) / 1e6
               for key in {t['episode_id'] for t in turns}}
    p95 = sorted((t['first_ns'] - t['issue_ns']) / 1e6 for t in turns if t['turn_index'] > 0)[math.ceil(32 * .95) - 1]
    record = summary['runs'].get(path.parent.name, control['runs'].get(path.parent.name))
    assert record is not None
    metrics = record['metrics']
    assert abs(p95 - metrics['resume_p95_ms']) < 1e-6
    assert abs(statistics.median(session.values()) - metrics['session_e2e_median_ms']) < 1e-6
    assert session == metrics['session_e2e_ms']
    records.append({'name': path.parent.name, 'session_e2e_median_ms': statistics.median(session.values()),
                    'resume_p95_ms': p95, 'foreground_outputs': sum(outputs.values()), 'background': bg})
log = root / 'resource-on-repaired/runs/measured_rep1_a/server_logs/measured_rep1_a.log'
events = [json.loads(line.split('SLACKKV_ROUND4_RESOURCES ', 1)[1]) for line in read(log).decode().splitlines()
          if 'SLACKKV_ROUND4_RESOURCES ' in line]
failures = [e for e in events if e['event'] == 'ALLOCATE_NONE' and e['last_allocator_check']
            and e['last_allocator_check']['fit'] is False]
assert len(failures) == 175
assert all(len(e['busy_ref_counts']) + e['pool_free_blocks'] == 4680 for e in failures)
assert all(e['last_allocator_check']['available'] == e['pool_free_blocks'] for e in failures)
resumes = {e['target_session'] for e in failures if e['target_session']
           and e['target_session'].startswith('E') and not e['target_session'].endswith(':turn:0')}
assert len(resumes) == 30
result = {'matched_complete_performance_runs': len(records), 'records': records,
          'observed_resume_capacity_failures': len(resumes), 'pool_partition_checks': len(failures),
          'claims': 'independent arithmetic/work-count audit; no semantic quality or continuous cause claim'}
(root / 'integration_audit.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'records'}, indent=2))
