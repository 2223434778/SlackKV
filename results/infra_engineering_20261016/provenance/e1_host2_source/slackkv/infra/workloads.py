"""Real history demo; controlled host-content test is a separate workload."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time
import uuid

from slackkv.infra.config import layout
from slackkv.infra.http import get,post
from slackkv.infra.runtime import save
from slackkv.infra.sessions import SessionPool


def tokenizer(config):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(config.model,local_files_only=True,trust_remote_code=False)


def chat_tokens(tok,messages):
    # Transformers5 defaults to BatchEncoding; len(mapping) is not token count.
    ids=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=False)
    if not isinstance(ids,list) or not all(type(i) is int and i>=0 for i in ids):
        raise ValueError('expected a single flat chat token-ID sequence')
    return ids


def settled_metrics(config,expected_total,timeout=8):
    from slackkv.infra.report import counter
    until=time.monotonic()+timeout
    while True:
        raw=get(config.url+'/metrics')
        values=[counter(raw,'vllm:prompt_tokens_by_source_total',f'source="{s}"')
                for s in ('local_compute','local_cache_hit','external_kv_transfer')]
        if all(v is not None for v in values) and sum(values)>=expected_total:return raw
        if time.monotonic()>=until:raise RuntimeError('source telemetry did not settle; do not fill missing work with zero')
        time.sleep(.1)


def run_demo(config,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'requests.json').exists():raise FileExistsError('demo already recorded')
    tok=tokenizer(config)
    count=lambda m:len(chat_tokens(tok,m))
    pool=SessionPool(config,layout(config)['quota_tokens'],token_count=count)
    docs=json.loads(Path(__file__).with_name('documents.json').read_text())
    if config.session_slots<2:raise ValueError('demo requires at least two session slots')
    rows=[];tools=[];lock=threading.Lock();barrier=threading.Barrier(2)
    save(out/'config.json',config.snapshot())
    (out/'metrics_before.prom').write_text(get(config.url+'/metrics'),encoding='utf-8')
    def agent(key):
        s=pool.open(key);barrier.wait(timeout=10)
        first=pool.send(key,f'我们正在检查{key}项目。请用一句话说明你会根据提供的项目资料回答。')
        with lock:rows.append(first);save(out/'requests.json',rows)
        started=time.monotonic_ns();value=docs[key];ended=time.monotonic_ns()
        with lock:
            tools.append(dict(session_id=key,tool='local_json_lookup',arguments=key,result=value,
                              started_ns=started,done_ns=ended,selection='client-driven, not autonomous model tool choice'))
            save(out/'tools.json',tools)
        second=pool.send(key,'客户端本地工具查到资料：'+json.dumps(value,ensure_ascii=False)+'。项目负责人和发布日期是什么？请简短回答。')
        with lock:rows.append(second);save(out/'requests.json',rows)
        third=pool.send(key,'根据刚才的资料，用一句话总结这个项目的负责人、发布日期和当前状态。',terminal=True)
        with lock:rows.append(third);save(out/'requests.json',rows)
        return dict(session_id=key,generation=s.generation,state=s.state,messages=s.messages,
                    started_ns=s.started_ns,done_ns=s.done_ns)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures=[executor.submit(agent,key) for key in ('cedar','maple')]
        sessions=[f.result() for f in futures]
    save(out/'sessions.json',sessions)
    (out/'metrics_after.prom').write_text(get(config.url+'/metrics'),encoding='utf-8')
    return dict(status='PASS',requests=len(rows),sessions=len(sessions),history='actual assistant text included next turn',
        tool='client-selected local JSON lookup',physical_close='terminal skips new optional stores; physical cached data follows native LRU')


def host_inputs(config):
    """CPU preflight before reserving GPU for this fixed pressure workload."""
    if config.policy=='nohost':raise ValueError('host-content check requires Full or StaticFair')
    if not config.batch_invariant:raise ValueError('use batch_invariant=true for content oracle, not mixed performance')
    tok=tokenizer(config)
    limit=min(layout(config)['quota_tokens'])
    if limit<512:raise ValueError('oracle requires at least512 retained tokens')
    context=chat_tokens(tok,[dict(role='system',content='You are a precise assistant.'),
        dict(role='user',content='The secret password is CEDAR17. Remember it.\n'+
             ('Inspection log: ordinary records without another password.\n'*1600)+
             '\nWhat is the secret password? Reply with the exact code only.')])
    # Preserve the real question/chat suffix while selecting an explicit stress size.
    prompt=context[:8064]+context[-128:]
    if len(prompt)+16>config.max_model_len:raise ValueError('oracle needs8192+16 context')
    pressure_len=min(config.max_model_len-16,16384)
    if (pressure_len+len(prompt))*layout(config)['kv_bytes_per_token']<=config.gpu_kv_bytes:
        raise ValueError('configured GPU too ample for this fixed pressure check; do not silently resize it')
    filler=tok.encode('Unrelated shipping ledger. '*20000,add_special_tokens=False)[:pressure_len]
    if len(prompt)!=8192 or len(filler)!=pressure_len:raise ValueError('oracle input dimensions differ')
    return prompt,filler,limit


def verify_host(config,out,prepared=None):
    """Independent cold oracle, real cache pressure, no reset or fake close call."""
    prompt,filler,limit=prepared or host_inputs(config)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'oracle.json').exists():raise FileExistsError('oracle already recorded')
    salt=uuid.uuid4().hex
    def request(prompt,salt,cap):
        body=dict(model='slackkv',prompt=prompt,max_tokens=16,temperature=0,stream=False,
                  cache_salt=salt,return_token_ids=True,
                  kv_transfer_params=dict(max_offload_tokens=cap))
        return post(config.url+'/v1/completions',body,config.request_timeout_seconds)
    (out/'before.prom').write_text(get(config.url+'/metrics'),encoding='utf-8')
    store=request(prompt,salt,limit)
    pressure=request(filler,uuid.uuid4().hex,0)
    (out/'before_resume.prom').write_text(settled_metrics(config,len(prompt)+len(filler)),encoding='utf-8')
    resumed=request(prompt,salt,0)
    (out/'after_resume.prom').write_text(settled_metrics(config,2*len(prompt)+len(filler)),encoding='utf-8')
    cold=request(prompt,uuid.uuid4().hex,0)
    (out/'after_cold.prom').write_text(settled_metrics(config,3*len(prompt)+len(filler)),encoding='utf-8')
    record=dict(store=store,pressure=pressure,resumed=resumed,cold=cold,prompt_tokens=len(prompt),
        retained_cap_tokens=limit,config=config.snapshot(),scope='controlled pressure/content; not real-chat performance')
    save(out/'oracle.json',record)
    from slackkv.infra.report import counter,delta
    start=(out/'before.prom').read_text();before=(out/'before_resume.prom').read_text();after=(out/'after_resume.prom').read_text()
    external=delta(before,after,'vllm:prompt_tokens_by_source_total','source="external_kv_transfer"')
    moved=delta(before,after,'vllm:kv_offload_total_bytes_total','transfer_type="CPU_to_GPU"')
    no_prior_load=(counter(start,'vllm:prompt_tokens_by_source_total','source="external_kv_transfer"')==0 and
                   counter(before,'vllm:prompt_tokens_by_source_total','source="external_kv_transfer"')==0)
    initial_h2d=counter(before,'vllm:kv_offload_total_bytes_total','transfer_type="CPU_to_GPU"')
    if moved is None and initial_h2d is None and no_prior_load:
        # This workload owns a fresh service with no prefetch and no prior host
        # reads. Dynamic H2D labels first appear on this one verified restore.
        moved=counter(after,'vllm:kv_offload_total_bytes_total','transfer_type="CPU_to_GPU"')
    written=counter(after,'vllm:kv_offload_total_bytes_total','transfer_type="GPU_to_CPU"')
    record['physical_evidence']=dict(external_resume_tokens=external,h2d_resume_bytes=moved,d2h_total_bytes=written,
        initial_h2d_label=initial_h2d,initial_zero_proof=no_prior_load,
        qualification='isolated fresh native service, no prefetch, zero external source before first restore; no generic absent-label zero substitution')
    save(out/'oracle.json',record)
    if external is None or external<=0 or moved is None or moved!=external*layout(config)['kv_bytes_per_token'] or written is None or written<=0:
        raise AssertionError('actual host source/transfer evidence missing or inconsistent')
    cold_metrics=(out/'after_cold.prom').read_text()
    cold_sources={s:delta(after,cold_metrics,'vllm:prompt_tokens_by_source_total',f'source="{s}"')
                  for s in ('local_compute','local_cache_hit','external_kv_transfer')}
    record['cold_sources']=cold_sources;save(out/'oracle.json',record)
    if cold_sources!={'local_compute':len(prompt),'local_cache_hit':0,'external_kv_transfer':0}:
        raise AssertionError('cold reference was not independent full prefill')
    if not resumed['tokens'] or resumed['tokens']!=cold['tokens']:raise AssertionError('full output IDs missing or differ from independent cold')
    answer=cold['text'].split('<|im_end|>',1)[0].strip()
    if answer!='CEDAR17':raise AssertionError('cold oracle does not recover discriminative marker: '+repr(answer))
    return record
