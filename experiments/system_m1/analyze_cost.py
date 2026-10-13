"""Existing-event diagnosis and explicit equivalent-fix cohort accounting."""
import argparse
import hashlib
import json
from pathlib import Path

from experiments.system_m1.analyze_contention import analyze, counter, load, read


def timing_counters(path):
    before = read(path/'measurement_start.prom')
    after = read(path/'measurement_end.prom')
    result = {}
    for direction in ('GPU_to_CPU','CPU_to_GPU'):
        label = f'transfer_type="{direction}"'
        a = counter(before,'vllm:kv_offload_total_time_total',label)
        b = counter(after,'vllm:kv_offload_total_time_total',label)
        result[direction] = None if b is None else dict(seconds=b-(a or 0), initial=a, final=b)
    return result


def existing_events(base, unused_evidence=None):
    path = base/'m1_shared_content4'
    states = load(path/'state_events.json')
    rows = load(path/'requests.json')
    groups = []
    for i, state in enumerate(states):
        if state['event'] == 'observe':groups.append([i, state])
        else:groups[-1].extend([i, state])
    # Five warmups, three salted independent cold references, then 20 saved rows.
    if len(groups) != len(rows)+8:raise ValueError('unexpected event/request alignment')
    previous = states[groups[8][0]-1]
    seen = set(); max_written_depth = {}; details = []
    for row, group in zip(rows,groups[8:]):
        before = previous; after = group[-1]; indices = group[::2]
        admitted = after['metrics'].get('admitted_store_blocks',0)-before['metrics'].get('admitted_store_blocks',0)
        reclaimed = after['metrics'].get('reclaimed_blocks',0)-before['metrics'].get('reclaimed_blocks',0)
        sid = row['session']
        before_depth = next((m['ready_prefix_blocks'] for m in before['members_physical'] if m['session']==sid),0)
        store_states = [s for s in group[1::2] if s['event']=='complete_store']
        after_depth = max([before_depth]+[m['ready_prefix_blocks'] for s in store_states
                                       for m in s['members_physical'] if m['session']==sid])
        rewritten = max(0,min(max_written_depth.get(sid,0),after_depth)-before_depth) if admitted else 0
        phase = 'no_new_write'
        if admitted:phase = 'initial_fill' if sid not in seen else 'expanded_quota_write'
        if admitted and sid in seen and rewritten:phase = 'expanded_quota_rewrite_or_extension'
        if sid:seen.add(sid)
        max_written_depth[sid] = max(max_written_depth.get(sid,0),after_depth)
        details.append(dict(label=row['label'], session=sid, state_indices=indices,
            admitted_blocks=admitted, reclaimed_blocks=reclaimed, phase=phase,
            ready_depth_before=before_depth, ready_depth_after_store=after_depth,
            prefix_depth_inferred_rewritten_blocks=rewritten,
            source_trace=row['source_trace']))
        previous = after
    if sum(d['admitted_blocks'] for d in details) != 1228:raise ValueError('store event accounting changed')
    if sum(d['reclaimed_blocks'] for d in details) != 1228:raise ValueError('reclaim accounting changed')
    byte_per_block = 458752
    decomposition = {}
    for name in ('m1_shared_m11','m1_shared_fair1','m1_shared_full1'):
        run_path = base/name
        run_rows = load(run_path/'requests.json')
        request_sum = sum(r['e2e_ms'] for r in run_rows)
        elapsed = (run_rows[-1]['done_ns']-run_rows[0]['issue_ns'])/1e6
        e = load(run_path/'execution.json')
        decomposition[name] = dict(request_sequence_ms=elapsed, request_e2e_sum_ms=request_sum,
            pressure_request_e2e_sum_ms=sum(r['e2e_ms'] for r in run_rows if r['session'] is None),
            between_requests_ms=elapsed-request_sum,
            configured_gap_ms=(len(run_rows)-1)*e['config']['tool_gap_seconds']*1000,
            transfer_measured_time=timing_counters(run_path))
    return dict(requests=details, admitted_blocks=1228, reclaimed_blocks=1228,
        written_mib=1228*byte_per_block/2**20,
        prefix_depth_inferred_rewritten_mib=sum(d['prefix_depth_inferred_rewritten_blocks'] for d in details)*byte_per_block/2**20,
        known_unused_write_cases=load(unused_evidence or base.parent/'system_m1_shared/unused_write_evidence.json'),
        evidence_limit='Aggregate state/order and unchanged private prefixes imply depths; no per-key time/identity trace. Rewritten and unused categories overlap; do not add them.',
        online_signals=dict(initial_A='No earlier host reuse at A0; future B/S joins are unknown.',
            last_B='B has already demonstrated host reuse by expansion; reuse-history gating would not exclude the observed final unused tail.',
            decision='No demonstrated signal that safely removes both cases without sacrificing future hits. No new policy GPU trial.'),
        original_screen_cost_decomposition=decomposition,
        unknown=['GPU critical-path CPU time/call count in old runs', 'blocking/overlap of individual DMA jobs',
                 'causal E2E cost of 201.25 MiB', 'production reuse/survival probability'])


def cohort(base):
    names = ['m1_cost_ref1','m1_cost_opt1','m1_cost_fair1']
    reference = None; reports = {}; signatures = {}
    for name in names:
        path = base/name; e = load(path/'execution.json')
        signature = (e['config'],e['plan'],load(path/'inputs.json'))
        if reference is None:reference = signature
        if signature != reference:raise ValueError('unmatched configuration/plan/inputs')
        individual = analyze(base,'shared_v1',name)
        if list(individual['runs']) != [name]:raise ValueError('explicit run unavailable')
        reports[name] = dict(**individual['runs'][name], transfer_measured_time=timing_counters(path))
        signatures[name] = e['sources']
    for file in ('controller.py','spec.py','__init__.py'):
        if len({s[file] for s in signatures.values()}) != 1:raise ValueError('non-native source changed')
    if signatures[names[0]]['native.py'] == signatures[names[1]]['native.py']:
        raise ValueError('reference and optimized source identical')
    if signatures[names[1]] != signatures[names[2]]:raise ValueError('fair stack differs')
    if reports[names[0]]['source_tokens'] != reports[names[1]]['source_tokens']:
        raise ValueError('equivalent policy changed prompt sources')
    if reports[names[0]]['transfer_bytes'] != reports[names[1]]['transfer_bytes']:
        raise ValueError('equivalent policy changed transferred bytes')
    changes = {}
    opt = reports[names[1]]
    for name in (names[0],names[2]):
        r = reports[name]
        changes[name] = {metric:100*(opt[metric]/r[metric]-1) for metric in
                        ('median_session_e2e_ms','request_sequence_ms','resume_p95_nearest_rank_ms')}
    return dict(runs=reports, optimized_change_percent=changes, source_hashes=signatures,
        additional_gpu_hours=individual['additional_gpu_hours'],continuous_gpu_hours=individual['continuous_gpu_hours'],
        executions=individual['executions'],
        qualification='One serial screen each. Equivalent CPU fix, no policy novelty or stable speedup claim. Transfer time sums are worker operation measurements, not blocked request time.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--events-only',action='store_true')
    parser.add_argument('--unused-evidence',type=Path,help='explicit committed evidence path when remote raw tree lacks the report artifact')
    args = parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    result = dict(existing_event_diagnosis=existing_events(args.base,args.unused_evidence))
    if not args.events_only:result['cohort'] = cohort(args.base)
    result['analyzer_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':main()
