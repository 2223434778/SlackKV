"""CPU-only independent archive/work/source/cleanup checks; no model imports."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path


def read(path):
    if path.exists():return path.read_bytes()
    return gzip.decompress(Path(str(path)+'.gz').read_bytes())


def load(path):return json.loads(read(path))


def metric(raw,name,label):
    rows=[float(line.rsplit(' ',1)[1]) for line in raw.splitlines() if line.startswith(name+'{') and label in line]
    return math.fsum(rows) if rows else None


def verify(base):
    manifest=load(base/'manifest_e2.json')
    for item in manifest:
        raw=read(base/item['source'])
        assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256'],item['source']
    all_executions={p.parent.name:load(p) for p in base.glob('runs/*/execution.json')}
    assert all(e['cleanup_verified'] and e['status']!='RUNNING' and not e['owned_live_pids_after'] for e in all_executions.values())
    assert all(0<e['gpu_hours'] and e['wall_seconds_including_startup_cleanup']<=e['wall_limit_seconds'] for e in all_executions.values())
    spent=math.fsum(e['gpu_hours'] for e in all_executions.values());assert spent<=.5
    cases={}
    for scenario in ('hot','host-enough','host-limited'):
        records=[];signature=None
        for policy in ('nohost','full','fair'):
            p=base/'runs'/('e2_'+scenario+'_'+policy+'1')
            e=load(p/'execution.json');result=load(p/'benchmark.json');plan=load(p/'benchmark_plan.json');rows=load(p/'benchmark_requests.json')
            assert e['status']=='COMPLETE' and result['status']=='PASS' and result['policy']==policy
            config=dict(e['config']);config.pop('policy')
            assert config==plan['config_signature']
            actual=(config,plan['inputs'],result['actual_gpu_tokens'])
            if signature is None:signature=actual
            assert actual==signature
            assert len(rows)==(6 if scenario=='hot' else 8)
            for r in rows:
                ids=plan['inputs'][r['key']]
                assert len(ids)==r['prompt_tokens']==r['usage']['prompt_tokens']
                assert r['prompt_sha256']==hashlib.sha256(json.dumps(ids).encode()).hexdigest()
                assert len(r['tokens'])==r['usage']['completion_tokens']==16
                assert r['issue_ns']<=r['first_ns']<=r['done_ns']
            before=read(p/'metrics_before.prom').decode();after=read(p/'metrics_after.prom').decode()
            sources={s:metric(after,'vllm:prompt_tokens_by_source_total',f'source="{s}"')-
                     metric(before,'vllm:prompt_tokens_by_source_total',f'source="{s}"')
                     for s in ('local_compute','local_cache_hit','external_kv_transfer')}
            assert sources==result['source_tokens'] and sum(sources.values())==sum(r['prompt_tokens'] for r in rows)
            for direction in ('GPU_to_CPU','CPU_to_GPU'):
                if policy=='nohost':
                    assert '--kv-transfer-config' not in e['command']
                    assert result['transfer_bytes'][direction]==0
                else:
                    a=metric(before,'vllm:kv_offload_total_bytes_total',f'transfer_type="{direction}"')
                    b=metric(after,'vllm:kv_offload_total_bytes_total',f'transfer_type="{direction}"')
                    assert a in (None,0) and b==result['transfer_bytes'][direction]
            agent=[r for r in rows if r['session_id']!='pressure']
            overlap=any(a['session_id']!=b['session_id'] and max(a['issue_ns'],b['issue_ns'])<min(a['done_ns'],b['done_ns']) for a in agent for b in agent)
            assert overlap and result['cross_session_overlap']
            for s,ms in result['session_e2e_ms'].items():
                own=[r for r in agent if r['session_id']==s]
                assert ms==(max(r['done_ns'] for r in own)-min(r['issue_ns'] for r in own))/1e6
            assert result['sequence_ms']==(max(r['done_ns'] for r in rows)-min(r['issue_ns'] for r in rows))/1e6
            for name,expected in e['project_source_sha256'].items():
                assert hashlib.sha256((base/'provenance/e2_source'/name).read_bytes()).hexdigest()==expected,name
            records.append(dict(policy=policy,sequence_ms=result['sequence_ms'],session_e2e_ms=result['session_e2e_ms'],
                source_tokens=sources,transfer_bytes=result['transfer_bytes'],actual_gpu_tokens=result['actual_gpu_tokens'],
                input_tokens=sum(r['prompt_tokens'] for r in rows),output_tokens=16*len(rows)))
        cases[scenario]=records
    reliability=load(base/'runs/e2_reliability1/reliability.json')
    assert reliability['status']=='PASS' and reliability['failed_history_messages']==1
    assert reliability['new_generation']==reliability['old_generation']+1
    assert reliability['observed_running']==reliability['observed_waiting']==0
    assert load(base/'e2_reliability_stop.json')==dict(status='STOPPED',cleanup_verified=True,owner_exit_code=0)
    return dict(status='PASS',archived_raw_files=len(manifest),gpu_hours_engineering=spent,
        gpu_hours_e2=math.fsum(e['gpu_hours'] for n,e in all_executions.items() if n.startswith('e2_')),
        continuous_gpu_hours=4.570877168576147+spent,runs=len(all_executions),cases=cases,
        reliability='real disconnect, atomic history, new generation, idle and external owner stop verified; abort acknowledgement UNKNOWN')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    result=verify(args.directory)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))


if __name__=='__main__':main()
