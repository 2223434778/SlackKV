"""Tiny frozen-work engineering scenarios, separate from real chat/quality."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import time
import uuid

from slackkv.infra.config import layout
from slackkv.infra.http import get,post
from slackkv.infra.report import counter,delta
from slackkv.infra.runtime import digest,save
from slackkv.infra.workloads import settled_metrics,tokenizer


def signature(config):
    value=config.snapshot();value.pop('policy');return value


def prepare(config,scenario,out):
    if scenario not in ('hot','host-enough','host-limited'):raise ValueError('unknown scenario')
    if config.session_slots!=2 or config.batch_invariant:raise ValueError('fixed bench requires two slots/non-invariant mode')
    if scenario=='host-enough' and config.host_kv_bytes<2*8320*layout(config)['kv_bytes_per_token']:
        raise ValueError('host-enough requires capacity for both complete growing prefixes')
    if scenario=='host-limited' and config.host_kv_bytes>=2*8192*layout(config)['kv_bytes_per_token']:
        raise ValueError('host-limited needs disclosed host pressure, no silent budget change')
    tok=tokenizer(config);n=1024 if scenario=='hot' else 8192;inputs={}
    for name in ('cedar','maple'):
        base=tok.encode(f'Archive {name}. '+('Ordinary engineering notes about caches and processes. '*5000),add_special_tokens=False)[:n]
        extension=tok.encode(' Additional factual record.'*256,add_special_tokens=False)
        for turn in range(3):inputs[f'{name}{turn}']=base+extension[:64*turn]
    if scenario!='hot':inputs['pressure']=tok.encode('Independent pressure ledger. '*10000,add_special_tokens=False)[:16384]
    if max(map(len,inputs.values()))+16>config.max_model_len:raise ValueError('fixed inputs exceed configured context')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    plan=dict(scenario=scenario,config_signature=signature(config),inputs=inputs,
        output_tokens=16,rounds=3,sessions=2,tool_gap_seconds=0,
        model_config_sha256=digest(Path(config.model)/'config.json'),runner_sha256=digest(__file__),
        workload='synthetic frozen token inputs, prescribed16 outputs ignoring EOS; not a true chat/quality test',
        gate='same terminal signal all methods; no required performance improvement',
        pressure='two serial16K cache-pressure requests between rounds; no GPU reset' if scenario!='hot' else 'none')
    save(out/'plan.json',plan)
    return plan


def run(config,plan_dir,out):
    plan=json.loads((Path(plan_dir)/'plan.json').read_text(encoding='utf-8'))
    if plan['config_signature']!=signature(config) or plan['runner_sha256']!=digest(__file__):raise ValueError('frozen plan/config/source mismatch')
    if plan['model_config_sha256']!=digest(Path(config.model)/'config.json'):raise ValueError('model changed')
    inputs=plan['inputs'];out=Path(out);rows=[];warm=[];salts={k:uuid.uuid4().hex for k in ('cedar','maple','pressure')}
    quotas=layout(config)['quota_tokens']
    def request(label,key,sid,turn,warmup=False):
        pressure=sid=='pressure';terminal=pressure or warmup or turn==2
        cap=0 if pressure or warmup else quotas[0 if sid=='cedar' else 1]
        params={'slackkv_terminal':dict(independent=True,end_after_request=terminal)}
        if config.policy=='fair' or pressure or warmup:params['max_offload_tokens']=cap
        body=dict(model='slackkv',prompt=inputs[key],max_tokens=16,temperature=0,stream=True,
            ignore_eos=True,skip_special_tokens=False,return_token_ids=True,stream_options={'include_usage':True},
            cache_salt=uuid.uuid4().hex if warmup else salts[sid])
        if config.policy!='nohost':body['kv_transfer_params']=params
        result=post(config.url+'/v1/completions',body,config.request_timeout_seconds)
        if len(result.get('tokens') or [])!=16 or result.get('usage',{}).get('completion_tokens')!=16 or result['usage']['prompt_tokens']!=len(inputs[key]):
            raise ValueError('prescribed output work not verified')
        return dict(label=label,key=key,session_id=sid,turn=turn,terminal=terminal,warmup=warmup,
            prompt_tokens=len(inputs[key]),prompt_sha256=hashlib.sha256(json.dumps(inputs[key]).encode()).hexdigest(),**result)
    # Warm every submitted shape with isolated salts, zero optional host writes.
    for key in ('cedar0','cedar1','cedar2')+(() if plan['scenario']=='hot' else ('pressure',)):
        warm.append(request('warm-'+key,key,'pressure',0,True))
    save(out/'warmup.json',warm)
    initial=settled_metrics(config,sum(r['prompt_tokens'] for r in warm))
    (out/'metrics_before.prom').write_text(initial,encoding='utf-8')
    save(out/'benchmark_plan.json',plan)
    for turn in range(3):
        pool=ThreadPoolExecutor(max_workers=2)
        try:
            futures=[pool.submit(request,sid+str(turn),sid+str(turn),sid,turn) for sid in ('cedar','maple')]
            rows.extend(f.result() for f in futures)
        finally:pool.shutdown(wait=False,cancel_futures=True)
        save(out/'benchmark_requests.json',rows)
        if turn<2 and plan['scenario']!='hot':
            # Fresh salt each pressure request, so it cannot be a hot repeat.
            salts['pressure']=uuid.uuid4().hex
            rows.append(request('pressure'+str(turn),'pressure','pressure',turn))
            save(out/'benchmark_requests.json',rows)
    final=settled_metrics(config,sum(r['prompt_tokens'] for r in warm+rows))
    (out/'metrics_after.prom').write_text(final,encoding='utf-8')
    sources={s:delta(initial,final,'vllm:prompt_tokens_by_source_total',f'source="{s}"') for s in ('local_compute','local_cache_hit','external_kv_transfer')}
    if any(v is None for v in sources.values()) or sum(sources.values())!=sum(r['prompt_tokens'] for r in rows):raise ValueError('actual prompt work does not conserve')
    traffic={}
    for direction in ('GPU_to_CPU','CPU_to_GPU'):
        label=f'transfer_type="{direction}"'
        a=counter(initial,'vllm:kv_offload_total_bytes_total',label)
        b=counter(final,'vllm:kv_offload_total_bytes_total',label)
        # Warmup has explicit zero store caps and unique identities on fresh service.
        if a not in (None,0):raise ValueError('warmup unexpectedly transferred host KV')
        traffic[direction]=0 if config.policy=='nohost' else b
    if traffic['CPU_to_GPU'] is not None and traffic['CPU_to_GPU']!=sources['external_kv_transfer']*layout(config)['kv_bytes_per_token']:
        raise ValueError('host bytes do not match source tokens')
    agent=[r for r in rows if r['session_id']!='pressure']
    session_ms={s:(max(r['done_ns'] for r in agent if r['session_id']==s)-min(r['issue_ns'] for r in agent if r['session_id']==s))/1e6 for s in ('cedar','maple')}
    overlap=any(a['session_id']!=b['session_id'] and max(a['issue_ns'],b['issue_ns'])<min(a['done_ns'],b['done_ns']) for a in agent for b in agent)
    pools=re.findall(r'GPU KV cache size: ([\d,]+) tokens',(out/'server.log').read_text(encoding='utf-8'))
    if not pools or len(set(pools))!=1:raise ValueError('actual GPU KV pool unknown')
    table=[dict(label=r['label'],ttft_first_content_ms=(r['first_ns']-r['issue_ns'])/1e6,
                response_ms=(r['done_ns']-r['issue_ns'])/1e6,first_content_to_end_mean_ms=(r['done_ns']-r['first_ns'])/15e6) for r in rows]
    result=dict(status='PASS',scenario=plan['scenario'],policy=config.policy,source_tokens=sources,
        actual_gpu_tokens=int(pools[0].replace(',','')),request_metrics=table,
        transfer_bytes=traffic,session_e2e_ms=session_ms,cross_session_overlap=overlap,
        total_requests=len(rows),input_tokens=sum(r['prompt_tokens'] for r in rows),output_tokens=16*len(rows),
        sequence_ms=(max(r['done_ns'] for r in rows)-min(r['issue_ns'] for r in rows))/1e6,
        prompt_identity={r['label']:r['prompt_sha256'] for r in rows},
        qualification='one frozen-work engineering smoke, no statistical speedup or task-quality claim; Full/Fair same connector, NoHost absent connector; null traffic UNKNOWN, NoHost0 by config')
    save(out/'benchmark.json',result)
    import csv
    with (out/'benchmark_requests.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    (out/'benchmark_summary.md').write_text('# Frozen-work engineering smoke\n\n```json\n'+json.dumps(result,indent=2)+'\n```\n',encoding='utf-8')
    return result


def compare(directories,out):
    rows=[];reference=None
    for directory in map(Path,directories):
        result=json.loads((directory/'benchmark.json').read_text(encoding='utf-8'))
        plan=json.loads((directory/'benchmark_plan.json').read_text(encoding='utf-8'))
        execution=json.loads((directory/'execution.json').read_text(encoding='utf-8'))
        if execution['status']!='COMPLETE' or not execution['cleanup_verified']:raise ValueError('unsettled run')
        identity=(plan,result['prompt_identity'],result['input_tokens'],result['output_tokens'],result['actual_gpu_tokens'])
        if reference is None:reference=identity
        if identity!=reference:raise ValueError('unmatched frozen work/resources')
        rows.append(dict(run=directory.name,**result))
    if {r['policy'] for r in rows}!={'fair','full','nohost'} or len(rows)!=3:raise ValueError('exactly one NoHost/Full/Fair run required')
    save(out,dict(runs=rows,qualification='matched single engineering cohort; no stability claim'))
    return rows
