"""Raw matched smoke accounting, including failures and first-request cost."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

from experiments.autonomous_fork.analyze import counter, read


def load(path):
    return json.loads(read(path))


def analyze(base):
    executions={p.parent.name:load(p) for p in base.glob('*/execution.json')}
    if any(e['status']=='RUNNING' or not e['cleanup_verified'] for e in executions.values()):
        raise ValueError('unresolved execution/cleanup')
    if sum(e['gpu_hours'] for e in executions.values()) > .5:
        raise ValueError('approved additional budget exceeded')
    runs={};reference=None
    for name,e in executions.items():
        if e['status']!='COMPLETE' or e['mode']!='mechanism':continue
        if e['config'].get('workload_variant') in ('contention_v1','shared_v1'):continue
        if '--enforce-eager' in e['command']:raise ValueError('weak eager baseline')
        inputs=load(base/name/'inputs.json')
        if reference is None:reference=(e['config'],inputs)
        if reference != (e['config'],inputs):raise ValueError('model/resource/input mismatch')
        rows=load(base/name/'requests.json');by={r['label']:r for r in rows}
        if len(by)!=12 or len(rows)!=12:raise ValueError('request count differs')
        for r in rows:
            if len(r['tokens'])!=16 or r['prompt_sha256']!=hashlib.sha256(json.dumps(inputs[r['key']]).encode()).hexdigest():
                raise ValueError('input/output work mismatch')
            if not r['issue_ns']<=r['first_ns']<=r['done_ns']:raise ValueError('invalid timing')
        sessions={key:(by[end]['done_ns']-by[start]['issue_ns'])/1e6 for key,start,end in
                  [('long','long0','long_end'),('short','short0','short_end')]}
        final=read(base/name/'long_new_after.prom').decode()
        sources={s:counter(final,'vllm:prompt_tokens_by_source_total',f'source="{s}"') for s in
                 ['local_compute','local_cache_hit','external_kv_transfer']}
        traffic={d:counter(final,'vllm:kv_offload_total_bytes_total',f'transfer_type="{d}"') for d in ['GPU_to_CPU','CPU_to_GPU']}
        expected=sum(r['length'] for r in rows)
        if any(v is None for v in sources.values()) or sum(sources.values())!=expected:
            raise ValueError('prompt source accounting incomplete/mismatched')
        if e['policy']!='nohost' and traffic['CPU_to_GPU']!=sources['external_kv_transfer']*e['config']['expected_kv_bytes_per_token']:
            raise ValueError('actual H2D bytes/source tokens disagree')
        boundary=read(base/name/'long_end_after.prom').decode()
        before_end={d:counter(boundary,'vllm:kv_offload_total_bytes_total',f'transfer_type="{d}"') for d in ['GPU_to_CPU','CPU_to_GPU']}
        pressure=[r for r in rows if r['key'].startswith('pressure')]
        runs[name]=dict(policy=e['policy'],startup_seconds=e['startup_seconds'],session_e2e_ms=sessions,
                       median_session_e2e_ms=statistics.median(sessions.values()),first_request_e2e_ms=rows[0]['e2e_ms'],
                       request_sequence_ms=(rows[-1]['done_ns']-rows[0]['issue_ns'])/1e6,
                       prompt_tokens=sum(r['length'] for r in rows),output_tokens=sum(len(r['tokens']) for r in rows),
                       source_counters=sources,transfer_total_bytes=traffic,
                       transfer_at_session_end_bytes=before_end,
                       pressure_e2e_ms=[r['e2e_ms'] for r in pressure],
                       pressure_first_to_done_ms=[(r['done_ns']-r['first_ns'])/1e6 for r in pressure],
                       request_details=[dict(label=r['label'],ttft_ms=r['ttft_ms'],e2e_ms=r['e2e_ms'],source_trace=r['source_trace']) for r in rows])
    pairs={}
    for index in (1,2):
        a,b=f'm1_mechanism{index}',f'fair_mechanism{index}'
        if a in runs and b in runs:
            pairs[str(index)]={k:100*(runs[a]['session_e2e_ms'][k]/runs[b]['session_e2e_ms'][k]-1) for k in ('long','short')}
            pairs[str(index)]['median_change_percent']=100*(runs[a]['median_session_e2e_ms']/runs[b]['median_session_e2e_ms']-1)
    return dict(runs=runs,paired_m1_vs_fair_change_percent=pairs,
                executions={k:{f:e.get(f) for f in ['status','policy','mode','gpu_hours','cleanup_verified']} for k,e in executions.items()},
                additional_gpu_hours=sum(e['gpu_hours'] for e in executions.values()),
                historical_gpu_hours=4.191988019727527,
                continuous_gpu_hours=4.191988019727527+sum(e['gpu_hours'] for e in executions.values()),
                qualification='Synthetic serial two-session smoke; first requests retained; no production/steady-state/statistical claim. Missing metrics remain null; pressure first-to-done is a proxy, not true TPOT.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.write_text(json.dumps(analyze(args.base),indent=2)+'\n',encoding='utf-8',newline='\n')


if __name__=='__main__':main()
