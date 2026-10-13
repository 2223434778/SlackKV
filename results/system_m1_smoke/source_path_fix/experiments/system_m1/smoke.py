"""CPU preparation by default; explicitly authorized, owned single-GPU smoke."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import time
import traceback
import urllib.request

FAMILY = 'system_m1_smoke'


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def budget(base):
    rows = [json.loads(p.read_text()) for p in base.glob('*/execution.json')]
    if any(r['status'] == 'RUNNING' or not r.get('cleanup_verified', False) for r in rows):
        raise RuntimeError('unresolved owned execution: reconcile before another launch')
    if any(not math.isfinite(r['gpu_hours']) or r['gpu_hours'] < 0 for r in rows):
        raise ValueError('invalid GPU cost in continuous ledger')
    return sum(r['gpu_hours'] for r in rows)


def wall_limit(config, allowance, spent):
    if not math.isfinite(allowance) or not 0 < allowance <= config['requested_additional_gpu_hours']:
        raise ValueError('explicit applicable additional allowance required, maximum 0.5 GPUh')
    limit = min(config['max_run_seconds_including_cleanup'], int((allowance-spent)*3600)-2)
    if limit < config['cleanup_reserve_seconds']+180:
        raise RuntimeError('insufficient remaining allowance for startup/work/cleanup')
    return limit


def resources(config, model):
    shape = json.loads((model/'config.json').read_text())
    if shape.get('use_sliding_window', False):
        raise ValueError('full attention required')
    token_bytes = (shape['num_hidden_layers']*2*shape['num_key_value_heads']*
                   (shape['hidden_size']//shape['num_attention_heads'])*2)
    if token_bytes != config['expected_kv_bytes_per_token']:
        raise ValueError('frozen FP16 TP1 KV layout differs')
    if max(config['pressure_tokens'], *config['session_tokens'].values())+config['output_tokens'] > config['max_model_len']:
        raise ValueError('prescribed prompt/output exceeds model limit')
    if config['max_model_len']*token_bytes > config['gpu_kv_bytes']:
        raise ValueError('configured model limit exceeds physical GPU KV budget')
    return dict(kv_bytes_per_token=token_bytes,
                host_bytes=config['host_tokens']*token_bytes,
                host_blocks=config['host_tokens']//config['block_tokens'])


def command(config, policy, mode, host_bytes):
    argv = [str(Path(config['env_root'])/'bin/vllm'), 'serve', config['model'],
            '--dtype', 'float16', '--tensor-parallel-size', '1', '--enable-prefix-caching',
            '--kv-cache-memory-bytes', str(config['gpu_kv_bytes']), '--max-model-len', str(config['max_model_len']),
            '--max-num-seqs', '4', '--max-num-batched-tokens', '8192',
            '--host', '127.0.0.1', '--port', str(config['port'])]
    if policy != 'nohost':
        extra = dict(block_size=config['block_tokens'], cpu_bytes_to_use=host_bytes)
        if policy == 'm1':
            extra.update(spec_name='SlackKVM1Spec', spec_module_path='slackkv.m1.spec',
                         slackkv_m1_trace=mode == 'content')
        argv += ['--kv-transfer-config', json.dumps(dict(kv_connector='OffloadingConnector',
                                                       kv_role='kv_both', kv_connector_extra_config=extra))]
    return argv


def prepare(config, out):
    from transformers import AutoTokenizer
    model = Path(config['model'])
    layout = resources(config, model)
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    filler = tokenizer.encode('Inspection log: read completed, checksum verified, status normal.\n', add_special_tokens=False)
    def text(label, count):
        head = tokenizer.encode(f'{label}: archive record for this independent project. The initial code is CEDAR17.\n', add_special_tokens=False)
        tail = tokenizer.encode('\nRetrieve the initial code. Reply with the code only.', add_special_tokens=False)
        n = count-len(head)-len(tail)
        if n <= 0 or not filler:raise ValueError('invalid meaningful-text length')
        return head+(filler*(n//len(filler)+1))[:n]+tail
    inputs = dict(long=text('Cedar project', config['session_tokens']['long']),
                  short=text('Maple dossier', config['session_tokens']['short']))
    for i in range(3):inputs[f'pressure{i}'] = text(f'Pressure-{i} standalone audit', config['pressure_tokens'])
    if len({tuple(p[:config['block_tokens']]) for p in inputs.values()}) != len(inputs):
        raise ValueError('independent mechanism inputs share first block; do not silently salt them')
    out.mkdir(parents=True, exist_ok=False)
    save(out/'inputs.json', inputs)
    save(out/'plan.json', dict(status='GPU_NOT_STARTED', gpu_hours=0, config=config, layout=layout,
                              config_model_sha256=digest(model/'config.json'), inputs_sha256=digest(out/'inputs.json'),
                              model_revision=model.name, runner_sha256=digest(Path(__file__)),
                              commands={f'{p}_{m}':command(config,p,m,layout['host_bytes']) for p,m in
                                        [('m1','content'),('m1','mechanism'),('fair','mechanism'),('full','mechanism'),('nohost','mechanism')]},
                              allowance='0.5 additional GPUh requested, NOT granted by configuration'))


def events(log, marker):
    decoder = json.JSONDecoder()
    return [decoder.raw_decode(line.split(marker+' ',1)[1].lstrip())[0]
            for line in log.splitlines() if marker+' ' in line]


def metric(raw, name, label):
    rows = [line for line in raw.splitlines() if line.startswith(name+'{') and label in line]
    return sum(float(line.rsplit(' ',1)[1]) for line in rows) if rows else None


def transfer_delta(before, after):
    result = {}
    for direction in ('GPU_to_CPU', 'CPU_to_GPU'):
        a = metric(before,'vllm:kv_offload_total_bytes_total',f'transfer_type="{direction}"')
        b = metric(after,'vllm:kv_offload_total_bytes_total',f'transfer_type="{direction}"')
        if a is not None and b is not None and b < a:raise ValueError('counter decreased')
        result[direction] = b-a if a is not None and b is not None else None
    return result  # Missing labels remain UNKNOWN, never synthetic zero.


def workload(config, plan, policy, mode, out):
    inputs = json.loads((plan/'inputs.json').read_text())
    rows = []
    base = f'http://127.0.0.1:{config["port"]}'
    def rawmetrics():
        with urllib.request.urlopen(base+'/metrics', timeout=5) as r:return r.read().decode()
    def request(label, key, generation=0, end=False, cold=False):
        prompt = inputs[key]
        params = dict(slackkv_trace_resume=True, slackkv_session=label)
        if key.startswith('pressure') or cold:params['max_offload_tokens'] = 0
        elif policy == 'm1':params['slackkv_m1'] = dict(session_id=key,generation=generation,end_after_request=end)
        elif policy == 'fair':params['max_offload_tokens'] = config['host_tokens']//len(config['static_fair_members'])
        elif policy == 'full':params['max_offload_tokens'] = len(prompt)
        payload = dict(model=config['model'],prompt=prompt,max_tokens=config['output_tokens'],temperature=0,
                       ignore_eos=True,stream=True,return_token_ids=True)
        if policy != 'nohost':payload['kv_transfer_params'] = params
        if cold:payload['cache_salt'] = 'm1-independent-cold-'+out.name+'-'+label
        before = rawmetrics();(out/f'{label}_before.prom').write_text(before)
        issue = time.monotonic_ns();first=None;tokens=[]
        req = urllib.request.Request(base+'/v1/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=120) as response:
            for line in response:
                if not line.startswith(b'data: ') or line[6:].strip() == b'[DONE]':continue
                choice = (json.loads(line[6:]).get('choices') or [{}])[0]
                ids = choice.get('token_ids') or []
                if ids and first is None:first=time.monotonic_ns()
                tokens.extend(ids)
        done = time.monotonic_ns()
        if first is None or len(tokens) != config['output_tokens']:raise ValueError('prescribed model output count differs')
        rows.append(dict(label=label,key=key,generation=generation,end=end,cold=cold,params=params,
                         cache_salt=payload.get('cache_salt'),length=len(prompt),prompt_sha256=hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
                         issue_ns=issue,first_ns=first,done_ns=done,tokens=tokens,ttft_ms=(first-issue)/1e6,e2e_ms=(done-issue)/1e6))
        save(out/'requests.json',rows)
        time.sleep(config['tool_gap_seconds'])
        after=rawmetrics();(out/f'{label}_after.prom').write_text(after)
        rows[-1]['transfer_bytes_delta']=transfer_delta(before,after);save(out/'requests.json',rows)
    sequence=[('long0','long',0,False),('short0','short',0,False),('pressure0','pressure0',0,False),
              ('long1','long',0,False),('short_end','short',0,True),('pressure1','pressure1',0,False),
              ('long2_expand','long',0,False),('pressure2','pressure2',0,False),('long3','long',0,False),
              ('long_end','long',0,True),('long_late','long',0,False),('long_new','long',1,True)]
    for label,key,gen,end in sequence:request(label,key,gen,end)
    if mode == 'content':request('cold_reference','long',cold=True)
    time.sleep(2)
    log = (out/'server.log').read_text(errors='replace')
    sources=events(log,'SLACKKV_SOURCE_TRACE');states=events(log,'SLACKKV_M1_STATE')
    for row in rows:row['source_trace']=[e for e in sources if e['session']==row['label']]
    save(out/'requests.json',rows);save(out/'state_events.json',states)
    validation = dict(status='COMPLETE',gpu_content='NOT_RUN',dynamic_benefit='UNKNOWN',
                      sources=sources,not_covered=['explicit GPU in-flight cancellation/forced end collision',
                                                  'background fairness','production workload','formal performance significance'])
    if mode == 'content':
        cold=rows[-1];restore=next(r for r in rows if r['label']=='long1')
        if not cold['source_trace'] or any(e['local_cached_tokens'] or e['external_cached_tokens'] for e in cold['source_trace']):
            raise AssertionError('independent cold reference not proven cold')
        if not restore['source_trace'] or not any(e['external_cached_tokens']>0 for e in restore['source_trace']):
            raise AssertionError('ordinary capacity pressure did not exercise nonzero host restore')
        if restore['tokens'] != cold['tokens']:raise AssertionError('restored token IDs differ from invariant cold prefill')
        if not states:raise AssertionError('native state trace absent')
        for s in states:
            if s['physical_ready_blocks']+s['physical_pending_store_blocks']+s['physical_free_blocks'] != s['capacity_blocks']:
                raise AssertionError('physical pool not conserved')
            if s['physical_reserved_bytes'] != resources(config,Path(config['model']))['host_bytes']:
                raise AssertionError('actual native host pool differs from frozen layout')
        if not any(s['metrics'].get('admitted_store_blocks',0)>0 for s in states):raise AssertionError('no actual host store')
        if not any(s['metrics'].get('fallback_ended_generation_requires_new_generation',0)>0 for s in states):
            raise AssertionError('ended late generation did not take denied-store fallback')
        final=states[-1]
        if final['members_physical'] or final['pending_store_batches'] or final['pending_load_batches']:
            raise AssertionError('ended sessions still physically present after settlement')
        validation.update(gpu_content='PASS',native_final_state=final)
    save(out/'validation.json',validation)


def owned_group(pgid):
    result=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdecimal():continue
        try:
            fields=(path/'stat').read_text().rsplit(')',1)[1].split()
            if int(fields[2])==pgid and fields[0]!='Z':result.append(int(path.name))
        except (OSError,ValueError,IndexError):pass
    return result


def server_environment(config, mode, gpu):
    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
         'PYTHONPATH':config['native_source']+':'+str(Path(__file__).resolve().parents[2]),
         'VLLM_BATCH_INVARIANT':str(int(mode=='content')),'VLLM_SERVER_DEV_MODE':'0'}
    envroot=Path(config['env_root'])
    env['PATH']=f'{envroot}/bin:'+env.get('PATH','')
    env['LD_LIBRARY_PATH']=f'{envroot}/targets/x86_64-linux/lib:{envroot}/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:{envroot}/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:'+env.get('LD_LIBRARY_PATH','')
    for key in ('SLACKKV_CONTINUATION_CREDIT','SLACKKV_WORKING_SET_EPOCHS','SLACKKV_CAPACITY_BACKFILL_LIMIT',
                'SLACKKV_RETENTION_CONTROL','SLACKKV_RETENTION_PIN','SLACKKV_FORK_TRACE','SLACKKV_ROUND4_RESOURCE_TRACE'):env[key]='0'
    return env


def execute(config, plan, root, policy, mode, gpu, name, allowance):
    if root.resolve() != Path(config['ledger_root']).resolve():
        raise ValueError('frozen ledger root required; renaming output cannot reset accounting')
    if mode == 'content' and policy != 'm1':raise ValueError('content oracle starts with M1 only')
    if Path(name).name != name or name in ('.','..'):raise ValueError('plain unique run name required')
    base=root/'results'/FAMILY;base.mkdir(parents=True,exist_ok=True)
    # This family-wide lock/ledger covers every policy, startup, failure and exit.
    import fcntl
    with (base/'budget.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        spent=budget(base);limit=wall_limit(config,allowance,spent)
        prepared=json.loads((plan/'plan.json').read_text())
        if prepared['runner_sha256'] != digest(Path(__file__)):
            raise ValueError('runner changed since frozen CPU preparation')
        if prepared['config']!=config or prepared['inputs_sha256']!=digest(plan/'inputs.json') or prepared['config_model_sha256']!=digest(Path(config['model'])/'config.json'):
            raise ValueError('frozen preparation/model differs')
        if policy != 'm1' or mode != 'content':
            passed=[json.loads(p.read_text()) for p in base.glob('*/validation.json')]
            if not any(v.get('gpu_content')=='PASS' for v in passed):raise RuntimeError('M1 content gate has not passed')
        with socket.socket() as probe:
            if probe.connect_ex(('127.0.0.1',config['port']))==0:raise RuntimeError('occupied port, do not disturb service')
        def gpu_state():
            return subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
        before=gpu_state()
        if int(before.split(',')[0])>512 or int(before.split(',')[1])!=0:raise RuntimeError('selected GPU is occupied')
        env=server_environment(config,mode,gpu)
        ninja=shutil.which('ninja',path=env['PATH'])
        if ninja is None:raise RuntimeError('ninja unavailable in server PATH; refuse GPU startup')
        out=base/name;out.mkdir(exist_ok=False)
        argv=command(config,policy,mode,prepared['layout']['host_bytes'])
        metadata=dict(status='RUNNING',gpu_count=1,gpu_hours=0,cleanup_verified=False,started_unix=time.time(),
                      additional_allowance_gpu_hours=allowance,family_spent_before=spent,wall_limit_seconds=limit,
                      historical_gpu_hours_before_new_allowance=4.191988019727527,
                      policy=policy,mode=mode,gpu=gpu,command=argv,config=config,plan=prepared,ninja_path=ninja,
                      runner_sha256=digest(Path(__file__)),sources={p.name:digest(p) for p in (Path(__file__).parents[2]/'slackkv/m1').glob('*.py')})
        save(out/'execution.json',metadata);save(out/'inputs.json',json.loads((plan/'inputs.json').read_text()))
        started=time.monotonic();proc=None
        def timeout(*_):raise TimeoutError('work watchdog: cleanup reserve included')
        signal.signal(signal.SIGALRM,timeout);signal.alarm(limit-config['cleanup_reserve_seconds'])
        try:
            with (out/'server.log').open('w') as log:
                proc=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                metadata['owned_pgid']=proc.pid;save(out/'execution.json',metadata)
                while True:
                    if proc.poll() is not None:raise RuntimeError('server exited before ready')
                    try:
                        with urllib.request.urlopen(f'http://127.0.0.1:{config["port"]}/health',timeout=2) as r:
                            if r.status==200:break
                    except OSError:pass
                    time.sleep(1)
                metadata['startup_seconds']=time.monotonic()-started
                workload(config,plan,policy,mode,out)
                metadata['status']='COMPLETE'
        except BaseException:
            metadata['status']='FAILED';(out/'failure.txt').write_text(traceback.format_exc());raise
        finally:
            signal.alarm(0)
            if proc is not None:
                for sig in (signal.SIGTERM,signal.SIGKILL):
                    if not owned_group(proc.pid):break
                    try:os.killpg(proc.pid,sig)
                    except ProcessLookupError:pass
                    deadline=time.monotonic()+(60 if sig==signal.SIGTERM else 5)
                    while owned_group(proc.pid) and time.monotonic()<deadline:time.sleep(.2)
                proc.poll()
            deadline=time.monotonic()+15
            while True:
                after=gpu_state();released=int(after.split(',')[0])<=512
                if released or time.monotonic()>=deadline:break
                time.sleep(.5)
            metadata['cleanup_verified']=released and (proc is None or not owned_group(proc.pid))
            metadata['gpu_after']=after
            if not metadata['cleanup_verified']:metadata['status']='RUNNING'
            metadata['wall_seconds_including_startup_cleanup']=time.monotonic()-started
            metadata['gpu_hours']=(metadata['wall_seconds_including_startup_cleanup']+1)/3600
            metadata['additional_allowance_spent']=spent+metadata['gpu_hours']
            metadata['continuous_gpu_hours']=metadata['historical_gpu_hours_before_new_allowance']+metadata['additional_allowance_spent']
            metadata['finished_unix']=time.time();save(out/'execution.json',metadata)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('smoke_config.json'))
    parser.add_argument('--prepare',type=Path)
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--root',type=Path)
    parser.add_argument('--name')
    parser.add_argument('--gpu',type=int,choices=(0,1),default=0)
    parser.add_argument('--policy',choices=('m1','fair','full','nohost'),default='m1')
    parser.add_argument('--mode',choices=('content','mechanism'),default='content')
    parser.add_argument('--authorized-additional-gpu-hours',type=float)
    args=parser.parse_args();config=json.loads(args.config.read_text())
    if args.prepare and not args.execute:prepare(config,args.prepare);return
    if not args.execute or args.prepare or not all((args.root,args.plan,args.name,args.authorized_additional_gpu_hours)):
        parser.error('CPU --prepare OUT, or explicitly authorized --execute --root ROOT --plan PLAN --name UNIQUE --authorized-additional-gpu-hours VALUE')
    execute(config,args.plan,args.root,args.policy,args.mode,args.gpu,args.name,args.authorized_additional_gpu_hours)


if __name__=='__main__':main()
