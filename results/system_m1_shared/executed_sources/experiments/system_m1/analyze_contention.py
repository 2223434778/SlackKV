"""Independent raw accounting for the single frozen contention screen."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

def read(path):
    if path.exists():return path.read_text(encoding='utf-8')
    with gzip.open(str(path)+'.gz','rt',encoding='utf-8') as f:return f.read()


def load(path):return json.loads(read(path))


def counter(raw,name,label):
    values=[float(line.rsplit(' ',1)[1]) for line in raw.splitlines()
            if line.startswith(name+'{') and label in line]
    return sum(values) if values else None


def analyze(base,variant='contention_v1'):
    runs={};reference=None
    executions={p.parent.name:load(p) for p in base.glob('*/execution.json')}
    spent=sum(e['gpu_hours'] for e in executions.values())
    if spent>.5 or any(e['status']=='RUNNING' or not e['cleanup_verified'] for e in executions.values()):
        raise ValueError('unresolved ledger or budget exceeded')
    for name,e in executions.items():
        if e['status']!='COMPLETE' or e['mode']!='mechanism' or e['config'].get('workload_variant')!=variant:continue
        path=base/name;inputs=load(path/'inputs.json');rows=load(path/'requests.json');warm=load(path/'warmup.json')
        sequence=e['plan']['sequence']
        if len(rows)!=len(sequence) or [r['label'] for r in rows]!=[s[0] for s in sequence]:raise ValueError('sequence differs')
        if e['config'].get('workload_variant')=='shared_v1' and load(path/'state_events.json'):
            raise ValueError('detailed state logging is forbidden in shared performance screen')
        if reference is None:reference=(e['config'],inputs)
        if reference!=(e['config'],inputs):raise ValueError('unmatched resources/inputs')
        if '--enforce-eager' in e['command']:raise ValueError('not compiled baseline')
        for row in rows+warm:
            if row['prompt_sha256']!=hashlib.sha256(json.dumps(inputs[row['key']]).encode()).hexdigest():raise ValueError('input mismatch')
            if len(row['tokens'])!=16 or not row['issue_ns']<=row['first_ns']<=row['done_ns']:raise ValueError('work/time mismatch')
        log=read(path/'server.log');pool=re.findall(r'GPU KV cache size: ([\d,]+) tokens',log)
        if not pool or len(set(pool))!=1:raise ValueError('actual GPU pool unknown')
        before=read(path/'measurement_start.prom');after=read(path/'measurement_end.prom')
        sources={};traffic={};presence={}
        for source in ('local_compute','local_cache_hit','external_kv_transfer'):
            label=f'source="{source}"';a=counter(before,'vllm:prompt_tokens_by_source_total',label);b=counter(after,'vllm:prompt_tokens_by_source_total',label)
            # Dynamic zero labels are absent until first use. All warmup requests
            # are proven fresh by their actual admission source traces below.
            if any(not r['source_trace'] or any(t['local_cached_tokens'] or t['external_cached_tokens'] for t in r['source_trace']) for r in warm):
                raise ValueError('warmup initial sources not independently fresh')
            if b is None:raise ValueError('final prompt source counter absent')
            sources[source]=b-(a or 0)
            presence[source]=dict(initial=a,final=b)
        for direction in ('GPU_to_CPU','CPU_to_GPU'):
            label=f'transfer_type="{direction}"';a=counter(before,'vllm:kv_offload_total_bytes_total',label);b=counter(after,'vllm:kv_offload_total_bytes_total',label)
            if a is not None and a!=0:raise ValueError('warmup host traffic nonzero')
            traffic[direction]=None if b is None else b-(a or 0)
        expected=sum(r['length'] for r in rows)
        if sum(sources.values())!=expected:raise ValueError('sources do not conserve prompt work')
        if traffic['CPU_to_GPU']!=sources['external_kv_transfer']*e['config']['expected_kv_bytes_per_token']:
            raise ValueError('H2D does not match restored token bytes')
        by={r['label']:r for r in rows}
        sessions={s:(by[s+'_end']['done_ns']-by[s+'0']['issue_ns'])/1e6 for s in ('a','b','s','d')}
        resumes=[r['ttft_ms'] for r in rows if r['session'] and r['label']!=r['session']+'0']
        ordered=sorted(resumes);p95=ordered[max(0,math.ceil(.95*len(ordered))-1)]
        states=load(path/'state_events.json');final=states[-1] if states else None
        runs[name]=dict(policy=e['policy'],actual_gpu_tokens=int(pool[0].replace(',','')),startup_seconds=e['startup_seconds'],
            warmup_seconds=load(path/'measurement_initial_state.json')['warmup_seconds'],
            session_e2e_ms=sessions,median_session_e2e_ms=statistics.median(sessions.values()),
            resume_ttft_ms=resumes,resume_p95_nearest_rank_ms=p95,
            first_request_e2e_ms=rows[0]['e2e_ms'],request_sequence_ms=(rows[-1]['done_ns']-rows[0]['issue_ns'])/1e6,
            prompt_tokens=expected,output_tokens=sum(len(r['tokens']) for r in rows),
            source_tokens=sources,source_counter_presence=presence,transfer_bytes=traffic,
            pressure_e2e_ms=[r['e2e_ms'] for r in rows if r['session'] is None],
            terminal_write_bytes=[r['transfer_bytes_delta']['GPU_to_CPU'] for r in rows if r['end']],
            native_final_state=final,
            requests=[{k:r[k] for k in ('label','ttft_ms','e2e_ms','source_trace')} for r in rows])
    if len({r['actual_gpu_tokens'] for r in runs.values()})>1:raise ValueError('actual GPU pool mismatch')
    policies={r['policy']:r for r in runs.values()};comparisons={}
    if 'm1' in policies:
        m=policies['m1']
        for p in ('fair','full'):
            if p not in policies:continue
            b=policies[p]
            comparisons[p]=dict(session_e2e_change_percent={s:100*(m['session_e2e_ms'][s]/b['session_e2e_ms'][s]-1) for s in m['session_e2e_ms']},
                median_session_e2e_change_percent=100*(m['median_session_e2e_ms']/b['median_session_e2e_ms']-1),
                total_sequence_change_percent=100*(m['request_sequence_ms']/b['request_sequence_ms']-1),
                resume_p95_change_percent=100*(m['resume_p95_nearest_rank_ms']/b['resume_p95_nearest_rank_ms']-1))
    return dict(runs=runs,m1_change_percent=comparisons,additional_gpu_hours=spent,
        continuous_gpu_hours=4.191988019727527+spent,
        executions={k:{f:e.get(f) for f in ('status','policy','mode','gpu_hours','cleanup_verified')} for k,e in executions.items()},
        qualification='One serial synthetic resource-contention screen per method, equivalent disjoint shape warmup, no GPU clearing; common terminal gate. Shared_v1 has no detailed state logging. No production, statistical or simultaneous-background-fairness claim.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--variant',choices=('contention_v1','shared_v1'),default='contention_v1')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.write_text(json.dumps(analyze(args.base,args.variant),indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
