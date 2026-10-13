"""One frozen independent-prefix contention screen; reuses the shared GPU ledger."""
import hashlib
import json
from pathlib import Path
import time
import urllib.request

from experiments.system_m1.smoke import command, digest, events, resources, save, transfer_delta


SEQUENCE = [('a0','a',0,False),('b0','b',0,False),('s0','s',0,False),
            ('pressure0','pressure0',0,False),('a1','a',1,False),('b1','b',1,False),
            ('s_end','s',1,True),('d0','d',0,False),('pressure1','pressure1',0,False),
            ('a2','a',2,False),('b2','b',2,False),('d1','d',1,False),
            ('d_end','d',2,True),('pressure2','pressure2',0,False),
            ('a3','a',3,False),('b_end','b',3,True),('a_end','a',4,True)]

SHARED_SEQUENCE = SEQUENCE[:15]+[('a_end','a',4,True),('pressure3','pressure3',0,False),
                                ('b_after_a_end','b',3,False),('b_end','b',4,True)]


def store_cap(config,policy,key,length,end=False):
    if key.startswith(('pressure','warmup','cold')):return 0
    if policy=='fair':return config['host_tokens']//len(config['static_fair_members'])
    return length


def prepare_inputs(config,out):
    from transformers import AutoTokenizer
    model=Path(config['model']);layout=resources(config,model)
    tok=AutoTokenizer.from_pretrained(str(model),local_files_only=True)
    filler=tok.encode('Inspection log: checksum verified, status normal.\n',add_special_tokens=False)
    if not {'<|im_start|>','<|im_end|>'}.issubset(tok.get_vocab()):
        raise ValueError('frozen Qwen chat delimiters unavailable')
    tail=tok.encode('\nWhat is the secret password at the beginning? Output that exact password only.<|im_end|>\n<|im_start|>assistant\n',add_special_tokens=False)
    if config.get('shared_prefix_blocks',0):
        tail=tok.encode('\nChoose the INITIAL private password: CEDAR17 or MAPLE42. Reply with only the chosen code.<|im_end|>\n<|im_start|>assistant\n',add_special_tokens=False)
    next_head=tok.encode('Acknowledged.<|im_end|>\n<|im_start|>user\nAdditional inspection records:\n',add_special_tokens=False)
    extension=next_head+(filler*64)[:64-len(next_head)-len(tail)]+tail
    if len(extension)!=64:raise ValueError('invalid append segment')
    inputs={};markers={}
    shared_blocks=config.get('shared_prefix_blocks',0)
    common=[]
    if shared_blocks:
        head=tok.encode('<|im_start|>system\nYou are a precise assistant. Remember the exact password in each private user history. System inspection rules:\n',add_special_tokens=False)
        end=tok.encode('<|im_end|>\n',add_special_tokens=False)
        count=shared_blocks*config['block_tokens']-len(head)-len(end)
        if count<0:raise ValueError('shared root too short for frozen system text')
        common=head+(filler*(count//len(filler)+1))[:count]+end
    def text(label,n,marker,shared=False):
        head=tok.encode(f'<|im_start|>user\n{label}: independent archive. The secret password is {marker}. Remember the exact uppercase password while reading the inspection records.\n',add_special_tokens=False)
        if shared:head=common+head
        free=n-len(head)-len(tail)
        if free<0:raise ValueError('prompt too short')
        return head+(filler*(free//len(filler)+1))[:free]+tail
    for key,n in config['session_tokens'].items():
        marker='CEDAR17' if key in ('a','s') else 'MAPLE42'
        root=text(dict(a='Cedar project',b='Maple dossier',s='Spruce notes',d='Willow records')[key],n,marker,shared=bool(shared_blocks))
        for turn in range(5):
            inputs[f'{key}{turn}']=root+extension*turn;markers[f'{key}{turn}']=marker
    pressure_count=4 if shared_blocks else 3
    for i in range(pressure_count):inputs[f'pressure{i}']=text(f'Pressure-{i} audit',config['pressure_tokens'],'CEDAR17')
    for i,n in enumerate(sorted(set(config['session_tokens'].values())|{config['pressure_tokens']})):
        inputs[f'warmup{i}']=text(f'Warmup-{i} calibration',n,'CEDAR17' if i%2==0 else 'MAPLE42')
    roots=[inputs[k+'0'] for k in config['session_tokens']]+[inputs[f'pressure{i}'] for i in range(pressure_count)]+[v for k,v in inputs.items() if k.startswith('warmup')]
    mechanism_roots=roots[:len(config['session_tokens'])]
    if shared_blocks:
        if any(v[:len(common)]!=common for v in mechanism_roots):raise ValueError('shared root differs')
        if len({tuple(v[len(common):len(common)+config['block_tokens']]) for v in mechanism_roots})!=len(mechanism_roots):
            raise ValueError('private suffix first blocks not distinct')
        independent_roots=roots[len(config['session_tokens']):]+[common]
    else:independent_roots=roots
    if len({tuple(v[:config['block_tokens']]) for v in independent_roots})!=len(independent_roots):
        raise ValueError('independent roots overlap; no automatic salts')
    if max(map(len,inputs.values()))+config['output_tokens']>config['max_model_len']:
        raise ValueError('grown request exceeds frozen model limit')
    out.mkdir(parents=True,exist_ok=False);save(out/'inputs.json',inputs)
    save(out/'plan.json',dict(status='GPU_NOT_STARTED',config=config,layout=layout,
        config_model_sha256=digest(model/'config.json'),inputs_sha256=digest(out/'inputs.json'),
        runner_sha256=digest(Path(__file__).with_name('smoke.py')),
        additional_runner_sources={'contention.py':digest(Path(__file__))},markers=markers,
        model_revision=model.name,sequence=SHARED_SEQUENCE if shared_blocks else SEQUENCE,warmup_keys=[k for k in inputs if k.startswith('warmup')],
        resource_model=dict(host_tokens=config['host_tokens'],initial_known_demand=15360,
                            all_configured_nominal_demand=19456,after_short_exit_then_join_demand=18432,
                            shared_root_tokens=len(common),
                            distinct_initial_demand=15360-2*len(common),
                            kind='synthetic bounded shared-root/private-tail resource contention' if common else 'synthetic independent-prefix resource contention'),
        acceptance='Screen only: examine all session E2E, aggregate completion, resume TTFT, compute/traffic and actual prefix changes. No stable superiority claim from one run.',
        commands={f'{p}_{m}':command(config,p,m,layout['host_bytes']) for p,m in
                  [('m1','content'),('m1','mechanism'),('fair','mechanism'),('full','mechanism')]}))


def workload(config,plan,policy,mode,out):
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(config['model'],local_files_only=True)
    prepared=json.loads((plan/'plan.json').read_text());inputs=json.loads((plan/'inputs.json').read_text())
    base=f'http://127.0.0.1:{config["port"]}';rows=[];warm=[];cold=[]
    def metrics():
        with urllib.request.urlopen(base+'/metrics',timeout=5) as r:return r.read().decode()
    def call(label,key,sid=None,end=False,reference=False,phase='measured'):
        prompt=inputs[key];params=dict(slackkv_trace_resume=True,slackkv_session=label,
            slackkv_terminal=dict(independent=True,end_after_request=end))
        if policy=='m1' and (sid is not None or phase=='warmup'):
            params['slackkv_m1']=dict(session_id=sid or label,generation=0,end_after_request=end)
        if label==prepared['warmup_keys'][-1]:params['slackkv_checkpoint_tag']='warmup_complete'
        if label==prepared['sequence'][-1][0]:params['slackkv_checkpoint_tag']='measurement_complete'
        if phase!='measured' or sid is None:params['max_offload_tokens']=0
        elif policy!='m1':params['max_offload_tokens']=store_cap(config,policy,sid,len(prompt),end)
        payload=dict(model=config['model'],prompt=prompt,max_tokens=config['output_tokens'],temperature=0,
                     ignore_eos=True,stream=True,return_token_ids=True)
        if policy!='nohost':payload['kv_transfer_params']=params
        if reference:payload['cache_salt']='independent-reference-'+out.name+'-'+label
        before=metrics();(out/f'{label}_before.prom').write_text(before)
        issue=time.monotonic_ns();first=None;tokens=[]
        req=urllib.request.Request(base+'/v1/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=120) as response:
            for line in response:
                if not line.startswith(b'data: ') or line[6:].strip()==b'[DONE]':continue
                ids=((json.loads(line[6:]).get('choices') or [{}])[0]).get('token_ids') or []
                if ids and first is None:first=time.monotonic_ns()
                tokens.extend(ids)
        done=time.monotonic_ns()
        if first is None or len(tokens)!=config['output_tokens']:raise AssertionError('output work differs')
        row=dict(label=label,key=key,session=sid,end=end,cold=reference,phase=phase,
            params=payload.get('kv_transfer_params'),length=len(prompt),
            prompt_sha256=hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
            issue_ns=issue,first_ns=first,done_ns=done,tokens=tokens,text=tok.decode(tokens),
            ttft_ms=(first-issue)/1e6,e2e_ms=(done-issue)/1e6)
        target=warm if phase=='warmup' else cold if phase=='cold' else rows
        target.append(row);save(out/('warmup.json' if phase=='warmup' else 'cold.json' if phase=='cold' else 'requests.json'),target)
        time.sleep(config['tool_gap_seconds']);after=metrics();(out/f'{label}_after.prom').write_text(after)
        row['transfer_bytes_delta']=transfer_delta(before,after)
        return row
    for key in prepared['warmup_keys']:call(key,key,end=True,phase='warmup')
    # No clear-cache API: distinct inputs remain in GPU; no warmup host writes.
    time.sleep(2);p0=metrics();time.sleep(.5);p1=metrics()
    if any(v not in (None,0) for v in transfer_delta(p0,p1).values()):raise AssertionError('warmup DMA not settled')
    save(out/'warmup.json',warm);(out/'measurement_start.prom').write_text(p1)
    prelog=(out/'server.log').read_text(errors='replace');prestates=events(prelog,'SLACKKV_M1_STATE')
    checkpoints=events(prelog,'SLACKKV_POOL_CHECKPOINT')
    warmcheck=[c for c in checkpoints if c['tag']=='warmup_complete']
    if not warmcheck or warmcheck[-1]['ready_blocks'] or warmcheck[-1]['pending_store_blocks'] or warmcheck[-1]['load_protected_blocks']:
        raise AssertionError('warmup physical pool not verified empty/settled')
    if policy=='m1' and warmcheck[-1]['retained_members']!=0:raise AssertionError('warmup retained M1 membership')
    save(out/'measurement_initial_state.json',dict(native_state=prestates[-1] if prestates else None,
         gpu_caches='not cleared; disjoint warmup roots and identical shape coverage',warmup_requests=len(warm),
         checkpoint=warmcheck[-1],warmup_seconds=(warm[-1]['done_ns']-warm[0]['issue_ns'])/1e9))
    if mode=='content':
        for sid in ('a','b'):call('cold_'+sid,sid+'1',reference=True,phase='cold')
        if config.get('shared_prefix_blocks',0):call('cold_b_after_a_end','b3',reference=True,phase='cold')
        for oracle in cold:
            marker=prepared['markers'][oracle['key']]
            if not oracle['text'].lstrip().startswith(marker):
                raise AssertionError('cold model cannot distinguish marker '+marker)
    for label,sid,turn,end in prepared['sequence']:
        key=sid if sid.startswith('pressure') else f'{sid}{turn}'
        call(label,key,None if sid.startswith('pressure') else sid,end)
    if mode=='content':
        # Same hot-GPU generation boundary as the earlier smoke, now gated.
        call('new_terminal','a0','fresh-terminal',end=True)
    time.sleep(2);log=(out/'server.log').read_text(errors='replace')
    sources=events(log,'SLACKKV_SOURCE_TRACE');states=events(log,'SLACKKV_M1_STATE')
    checkpoints=events(log,'SLACKKV_POOL_CHECKPOINT')
    save(out/'checkpoints.json',checkpoints)
    finalchecks=[c for c in checkpoints if c['tag']=='measurement_complete']
    if not finalchecks or finalchecks[-1]['pending_store_blocks'] or finalchecks[-1]['load_protected_blocks']:
        raise AssertionError('final physical transfers not settled')
    if policy=='m1' and (finalchecks[-1]['retained_members'] or finalchecks[-1]['free_blocks']!=finalchecks[-1]['capacity_blocks']):
        raise AssertionError('final M1 membership/physical cleanup incomplete')
    for row in rows+warm+cold:row['source_trace']=[e for e in sources if e['session']==row['label']]
    save(out/'requests.json',rows);save(out/'warmup.json',warm);save(out/'cold.json',cold);save(out/'state_events.json',states)
    validation=dict(status='COMPLETE',gpu_content='NOT_RUN',dynamic_benefit='UNKNOWN',
        initial_state=prestates[-1] if prestates else None,sources=sources,
        not_covered=['GPU in-flight forced cancellation','shared-prefix ownership','simultaneous compute/background fairness','production trace','statistical significance'])
    if states:
        for s in states:
            if s['physical_ready_blocks']+s['physical_pending_store_blocks']+s['physical_free_blocks']!=s['capacity_blocks']:
                raise AssertionError('physical pool not conserved')
        final=states[-1]
        if final['members_physical'] or final['pending_store_batches'] or final['pending_load_batches']:
            raise AssertionError('final physical cleanup incomplete')
        validation['native_final_state']=final
    if mode=='content':
        for sid,oracle in zip(('a','b'),cold):
            restored=next(r for r in rows if r['label']==sid+'1')
            marker=prepared['markers'][sid+'1']
            if not oracle['text'].lstrip().startswith(marker):raise AssertionError('cold model cannot distinguish marker '+marker)
            if not oracle['source_trace'] or any(e['local_cached_tokens'] or e['external_cached_tokens'] for e in oracle['source_trace']):
                raise AssertionError('reference not cold')
            if not any(e['external_cached_tokens']>0 for e in restored['source_trace']):raise AssertionError('no actual host recovery '+sid)
            if restored['tokens']!=oracle['tokens']:raise AssertionError('restored content differs '+sid)
        if cold[0]['tokens']==cold[1]['tokens']:raise AssertionError('oracle not discriminative')
        fresh=next(r for r in rows if r['label']=='new_terminal')
        if fresh['transfer_bytes_delta']['GPU_to_CPU']!=0:raise AssertionError('new terminal wrote unused host data')
        if not final['metrics'].get('terminal_no_retention_requests'):raise AssertionError('terminal demand gate not exercised')
        if shared_blocks:=config.get('shared_prefix_blocks',0):
            if not any(s['physical_shared_blocks']==shared_blocks for s in states):raise AssertionError('actual shared physical root not observed')
            if any(s['metrics'].get('fallback_shared_prefix',0) or s['metrics'].get('fallback_unsupported_shared_shape',0) for s in states):
                raise AssertionError('supported shared shape fell back')
            resumed=next(r for r in rows if r['label']=='b_after_a_end');oracle=cold[2]
            if not any(e['external_cached_tokens']>shared_blocks*config['block_tokens'] for e in resumed['source_trace']):
                raise AssertionError('B did not restore private tail after A ended')
            if any(e['local_cached_tokens'] or e['external_cached_tokens'] for e in oracle['source_trace']) or not oracle['source_trace']:
                raise AssertionError('post-end B reference not cold')
            if resumed['tokens']!=oracle['tokens'] or not oracle['text'].lstrip().startswith('MAPLE42'):
                raise AssertionError('post-end B private identity mismatch')
        if final['metrics'].get('fallback_ended_generation_requires_new_generation',0):
            raise AssertionError('ordinary shared session unexpectedly ended early')
        validation['gpu_content']='PASS'
    (out/'measurement_end.prom').write_text(metrics());save(out/'validation.json',validation)
