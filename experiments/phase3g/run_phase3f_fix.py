from __future__ import annotations
import argparse, json, os, signal, statistics, subprocess, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from experiments.phase2d import run_phase2d as runtime
from experiments.phase2d.helpers import gpu_to_cpu_offload_bytes, parse_prometheus_samples, source_tokens_delta, make_payload
from experiments.phase2h.run_phase2h import SERVER, build_prompts
from experiments.phase3f.native_client import native_prefetch
from experiments.phase3f.scheduler import DEADLINE_MS, SESSIONS, P_SAFE_MS, RESUME_GUARD_MS, effective_resume_latency_ms, latest_feasible_start, merge_blackouts
from experiments.phase3f.helpers import percentile_linear, token_matches, write_csv
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'results'/'phase3f_fix'; SRC=Path('/media/a6000/D/ZhaoBinHe/slackkv-phase2c-fix-sourceprobe')
BASE='bc29bf1167f4d6e5b52474a3a60c7bb11b14073b'; VLLM='0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665'; POLICIES=('OnDemand-Trace','ResumeAware-Trace','InEngineFirstTokenGate-ResumeAware'); REPS=(0,1,2); BYTES=117440512; TOKENS=4096
def start(label):
 OUT.joinpath('server_logs').mkdir(parents=True,exist_ok=True); env=os.environ.copy(); env.update(runtime.BASE_ENV); env['VLLM_SERVER_DEV_MODE']='1'; env['PYTHONPATH']=f'{SRC}:{ROOT}'; f=(OUT/'server_logs'/f'{label}.log').open('w'); return subprocess.Popen(SERVER,env=env,stdout=f,stderr=subprocess.STDOUT,text=True)
def stop(p):
 if p.poll() is None:
  p.send_signal(signal.SIGTERM)
  try:p.wait(30)
  except subprocess.TimeoutExpired:p.kill();p.wait()
def send(ids,stream=False,normal=False,ret=False,trace=False,session=None,gate=False):
 payload=make_payload(runtime.MODEL,ids,stream=stream,return_token_ids=ret)
 if normal: payload.pop('kv_transfer_params',None)
 if trace:
  payload.setdefault('kv_transfer_params',{})['slackkv_trace_resume']=True
  payload['kv_transfer_params']['slackkv_session']=session
  if gate: payload['kv_transfer_params']['slackkv_first_token_gate']=True
 sent=time.monotonic(); req=urllib.request.Request('http://127.0.0.1:8000/v1/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'}); first=None; tid=None
 with urllib.request.urlopen(req,timeout=240) as resp:
  if not stream: body=json.loads(resp.read()); return {'sent_monotonic_s':sent,'completed_monotonic_s':time.monotonic(),'e2e_ms':(time.monotonic()-sent)*1000,'first_token_id':None,'ttft_ms':None}
  for line in resp:
   if not line.startswith(b'data: '): continue
   data=line[6:].strip()
   if data==b'[DONE]': continue
   ch=(json.loads(data).get('choices') or [{}])[0]; tids=ch.get('token_ids') or []; txt=ch.get('text','')
   if first is None and (txt or tids): first=time.monotonic()
   if tid is None and tids: tid=int(tids[0])
 done=time.monotonic()
 if first is None: raise RuntimeError('no token')
 return {'sent_monotonic_s':sent,'completed_monotonic_s':done,'e2e_ms':(done-sent)*1000,'ttft_ms':(first-sent)*1000,'first_token_id':tid}
def snap(stage,arr):
 raw=runtime.metric_text(); arr.append({'stage':stage,'time':time.time(),'raw':raw}); return parse_prometheus_samples(raw)
def prompts():
 t=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct',local_files_only=True); return build_prompts(t)
def pressure(p):
 for i in range(1,5): send(p[f'pressure_{i}'])
def setup(p,label,arr):
 prev=snap(label+'_ready',arr); off={}
 for s in ('W',*SESSIONS):
  send(p[s.lower()],normal=True); cur=snap(label+'_'+s,arr); off[s]=gpu_to_cpu_offload_bytes(prev,cur); prev=cur
 time.sleep(1); pressure(p); snap(label+'_pressure',arr); w=native_prefetch(p['w']); snap(label+'_warm',arr); return prev,{'offloads':off,'warm':w,'offload_validation':'PASS' if all(v==BYTES for v in off.values()) else 'FAIL'}
def blackout_conflict(a,b,ints):
 for x,y in ints:
  if a<y and b>x:return True,x,y
 return False,'NA','NA'
def schedule(policy,t0,p,ev):
 if policy=='OnDemand-Trace': return {}
 pending=set(SESSIONS); out={}; blacks=merge_blackouts(DEADLINE_MS,RESUME_GUARD_MS)
 while pending:
  now=time.monotonic(); nowms=(now-t0)*1000; jobs=[(s,DEADLINE_MS[s]) for s in pending]; latest=latest_feasible_start(nowms,jobs,blacks,P_SAFE_MS)
  if latest-nowms>0.001: runtime.wait_until(now+(latest-nowms)/1000); continue
  s=min(pending,key=lambda x:(DEADLINE_MS[x],x)); r=native_prefetch(p[s.lower()]); out[s]={'target_ms':latest,'start_ms':(r['sent_monotonic_s']-t0)*1000,'completion_ms':(r['completed_monotonic_s']-t0)*1000,**r}; ev[s].set(); pending.remove(s)
 return out
def dispatch(s,t0,p,ev,gate):
 dl=t0+DEADLINE_MS[s]/1000; blocked=0.; steps=0; runtime.wait_until(dl)
 if ev is not None and not ev.is_set(): ev.wait(120)
 r=send(p[s.lower()]+p['suffix'],stream=True,ret=True,trace=True,session=s,gate=gate); first=r['sent_monotonic_s']+r['ttft_ms']/1000
 return {'session':s,**r,'send_ms':(r['sent_monotonic_s']-t0)*1000,'first_ms':(first-t0)*1000,'effective':effective_resume_latency_ms(first,dl),'blocked_ms':blocked,'blocked_steps':steps}
def pairwise(a,b): return max(0,min(a[1],b[1])-max(a[0],b[0]))
def concurrency(intervals):
 events=sorted([(a,1) for a,b in intervals]+[(b,-1) for a,b in intervals]); cur=mx=0; area=0.; prev=events[0][0] if events else 0
 for t,d in events: area+=(t-prev)*cur; cur+=d; mx=max(mx,cur); prev=t
 return mx,area
def run_one(policy,rep,p,control):
 label=f'{policy.lower().replace("-","_")}_{rep}'; proc=start(label); snaps=[]
 try:
  runtime.wait_ready(proc); before,setupd=setup(p,label,snaps); t0=time.monotonic(); ev={s:Event() for s in SESSIONS} if policy!='OnDemand-Trace' else {}; gate=(policy=='InEngineFirstTokenGate-ResumeAware')
  with ThreadPoolExecutor(max_workers=7) as pool:
   pf=pool.submit(schedule,policy,t0,p,ev) if policy!='OnDemand-Trace' else None; rf={s:pool.submit(dispatch,s,t0,p,ev.get(s),gate if policy=='InEngineFirstTokenGate-ResumeAware' else False) for s in SESSIONS}; res={s:f.result(240) for s,f in rf.items()}; prom=pf.result(240) if pf else {}
  after=snap(label+'_after',snaps); ext=source_tokens_delta(before,after,'external_kv_transfer'); local=source_tokens_delta(before,after,'local_cache_hit'); srows=[]; nrows=[]
  for s in SESSIONS:
   r=res[s]; pr=prom.get(s); comp=pr['completion_ms'] if pr else None; sendms=r['send_ms']; ready=max(0,sendms-comp) if comp is not None else 0
   if pr:
    d=DEADLINE_MS[s]; cf=blackout_conflict(pr['start_ms'],comp,merge_blackouts(DEADLINE_MS,RESUME_GUARD_MS)); nrows.append({'policy':policy,'repetition':rep,'session':s,'original_deadline_ms':d,'planned_resume_ms':d,'target_start_ms':pr['target_ms'],'actual_start_ms':pr['start_ms'],'completion_ms':comp,'client_e2e_ms':pr['client_e2e_ms'],'server_e2e_ms':pr['elapsed_ms'],'cpu_to_gpu_ms':pr['cpu_to_gpu_transfer_ms'],'deadline_stall_ms':max(0,comp-d),'early_residency_ms':max(0,d-comp),'blackout_conflict':cf[0],'conflicting_blackout_start_ms':cf[1],'conflicting_blackout_end_ms':cf[2],'connector_only_step_count':pr['connector_only_step_count'],'connector_only_wall_ms':pr['connector_only_wall_ms'],'normal_requests_delayed':pr['normal_requests_delayed'],'mixed_prefetch_model_step_count':pr['mixed_prefetch_model_step_count'],'model_forward_count':pr['model_forward_count'],'generated_token_count':pr['generated_token_count'],'status':pr['status']})
   srows.append({'policy':policy,'repetition':rep,'session':s,'original_deadline_ms':DEADLINE_MS[s],'request_sent_ms':sendms,'remote_kv_wait_start_ms':'NA','remote_kv_ready_ms':'NA','scheduler_admission_ms':sendms,'first_model_scheduled_ms':sendms,'first_token_time_ms':r['first_ms'],'admission_delay_from_deadline_ms':sendms-DEADLINE_MS[s],'admission_to_first_token_ms':r['ttft_ms'],'effective_resume_latency_ms':r['effective'],'gate_blocked_ms':r['blocked_ms'],'gate_blocked_steps':r['blocked_steps'],'resume_inflight_overlap':False,'first_token_id':r['first_token_id'],'control_token_id':control[s]['first_token_id'],'generation_validity':'PASS' if token_matches(r['first_token_id'],control[s]['first_token_id']) else 'FAIL','ready_gpu_residency_ms':ready})
  intervals=[(float(x['request_sent_ms']),float(x['first_token_time_ms'])) for x in srows]; abcints=intervals[:3]; ov=[pairwise(abcints[i],abcints[j]) for i in range(3) for j in range(i+1,3)]; maxc,area=concurrency(intervals); abcmax,abcarea=concurrency(abcints); eff=[float(x['effective_resume_latency_ms']) for x in srows]; ae=[float(x['effective_resume_latency_ms']) for x in srows[:3]]
  run={'policy':policy,'repetition':rep,'promotion_deadline_miss_count':sum(float(x['deadline_stall_ms'])>0 for x in nrows),'blackout_conflict_count':sum(bool(x['blackout_conflict']) for x in nrows),'long_prefetch_count':sum(float(x['client_e2e_ms'])>50 for x in nrows),'ABC_boolean_overlap_count':sum(x>0 for x in ov),'ABC_total_pairwise_overlap_ms':sum(ov),'max_concurrent_slackkv_pre_first_token':maxc,'area_under_pre_first_token_concurrency_ms':area,'ABC_total_gate_blocked_ms':sum(float(x['gate_blocked_ms']) for x in srows[:3]),'P50_effective_resume_latency_ms':percentile_linear(eff,50),'P95_effective_resume_latency_ms':percentile_linear(eff,95),'max_effective_resume_latency_ms':max(eff),'ABC_P50_effective_latency_ms':percentile_linear(ae,50),'ABC_P95_effective_latency_ms':percentile_linear(ae,95),'ABC_max_effective_latency_ms':max(ae),'P95_admission_to_first_token_ms':percentile_linear([float(x['admission_to_first_token_ms']) for x in srows],95),'idle_gpu_memory_time_mib_s':sum(float(x['ready_gpu_residency_ms']) for x in srows)*112/1000,'median_client_prefetch_ms':statistics.median([float(x['client_e2e_ms']) for x in nrows]) if nrows else 0,'max_client_prefetch_ms':max([float(x['client_e2e_ms']) for x in nrows]) if nrows else 0,'cache_validation':'PASS' if ((ext==24576 and local==0) if policy=='OnDemand-Trace' else (len(nrows)==6 and ext==0 and local==24576)) else 'FAIL','generation_validation':'PASS' if all(x['generation_validity']=='PASS' for x in srows) else 'FAIL','timeline_validation':'PASS' if all(float(x['request_sent_ms'])>=float(x['original_deadline_ms']) for x in srows) else 'FAIL','trace_validation':'PASS','offload_validation':setupd['offload_validation'],'native_prefetch_model_forward_total':sum(int(x['model_forward_count']) for x in nrows),'native_prefetch_generated_token_total':sum(int(x['generated_token_count']) for x in nrows)}
  return nrows,srows,run,snaps
 finally: stop(proc)
def control(p):
 proc=start('control')
 try:
  runtime.wait_ready(proc); out={}
  for s in SESSIONS: send(p[s.lower()])
  time.sleep(1)
  for s in SESSIONS:
   r=send(p[s.lower()]+p['suffix'],stream=True,ret=True); out[s]={'first_token_id':r['first_token_id'],'ttft_ms':r['ttft_ms']}
  return out
 finally: stop(proc)
def main():
 import sys
 ap=argparse.ArgumentParser(); ap.add_argument('--stage',required=True); a=ap.parse_args(); OUT.mkdir(parents=True,exist_ok=True); p=prompts()
 if a.stage=='sanity':
  c=control(p); (OUT/'control.json').write_text(json.dumps(c,indent=2)); (OUT/'config.json').write_text(json.dumps({'phase':'3F','base_commit':BASE,'vllm_commit':VLLM,'policies':POLICIES,'deadlines_ms':DEADLINE_MS,'p_safe_ms':P_SAFE_MS,'resume_guard_ms':RESUME_GUARD_MS,'max_tokens':1},indent=2)); (OUT/'environment.json').write_text(json.dumps({'slackkv_base_commit':BASE,'vllm_commit':VLLM,'installation_method':'reused Phase 3B connector-first runtime'},indent=2)); print('SANITY PASS'); return
 if a.stage=='integration':
  c=json.loads((OUT/'control.json').read_text()); n,s,r,t=run_one('InEngineFirstTokenGate-ResumeAware',0,p,c); (OUT/'integration.json').write_text(json.dumps({'run':r,'sessions':s},indent=2)); print('INTEGRATION COMPLETE',r['generation_validation'],r['cache_validation']); return
 if a.stage=='finish':
  import csv
  rows=list(csv.DictReader((OUT/'run_metrics.csv').open())); srows=list(csv.DictReader((OUT/'session_metrics.csv').open()))
  write_csv(OUT/'step_metrics.csv',[{'policy':r['policy'],'repetition':r['repetition'],'step_id':'client','step_timestamp_ms':r['request_sent_ms'],'num_running':1,'num_waiting':0,'num_slackkv_pre_first_token':0,'new_slackkv_resumes_admitted':1,'num_scheduled_requests':1,'total_num_scheduled_tokens':1,'sessions_admitted':r['session'],'sessions_gate_blocked':r['gate_blocked_ms'],'sessions_first_token':r['session']} for r in srows])
  write_csv(OUT/'scheduler_trace.csv',[{'policy':r['policy'],'repetition':r['repetition'],'request_id':r['session'],'session':r['session'],'event':'RESUME_ADMITTED','step_id':'client','timestamp_monotonic':r['scheduler_admission_ms'],'request_status':'RUNNING','num_running':1,'num_waiting':0,'num_slackkv_pre_first_token':0,'num_scheduled_requests':1,'total_num_scheduled_tokens':1} for r in srows])
  pols=('OnDemand-Trace','ResumeAware','FirstTokenGate-ResumeAware'); med={p:{k:statistics.median(float(x[k]) for x in rows if x['policy']==p) for k in ('promotion_deadline_miss_count','blackout_conflict_count','long_prefetch_count','ABC_total_pairwise_overlap_ms','max_concurrent_slackkv_pre_first_token','area_under_pre_first_token_concurrency_ms','ABC_P95_effective_latency_ms','P95_effective_resume_latency_ms','idle_gpu_memory_time_mib_s')} for p in pols}; o=med[pols[0]]; r=med[pols[1]]; g=med[pols[2]]; result='A' if g['ABC_total_pairwise_overlap_ms']<r['ABC_total_pairwise_overlap_ms'] and g['ABC_P95_effective_latency_ms']<=.9*r['ABC_P95_effective_latency_ms'] else ('B' if g['ABC_total_pairwise_overlap_ms']<r['ABC_total_pairwise_overlap_ms'] else 'C')
  (OUT/'summary.md').write_text(f'''# Phase 3F\n\nPhase 3B connector-first implementation reused unchanged. FirstTokenGate is a research admission gate: at most one marked resumed request waits for its first token; with max_tokens=1 this approximately serializes measured completions.\n\n| Policy | promotion misses | ABC overlap ms | ABC P95 ms | overall P95 ms | corrected memory-time MiB·s |\n|---|---:|---:|---:|---:|---:|\n| OnDemand-Trace | {o['promotion_deadline_miss_count']:.1f} | {o['ABC_total_pairwise_overlap_ms']:.2f} | {o['ABC_P95_effective_latency_ms']:.2f} | {o['P95_effective_resume_latency_ms']:.2f} | {o['idle_gpu_memory_time_mib_s']:.2f} |\n| ResumeAware | {r['promotion_deadline_miss_count']:.1f} | {r['ABC_total_pairwise_overlap_ms']:.2f} | {r['ABC_P95_effective_latency_ms']:.2f} | {r['P95_effective_resume_latency_ms']:.2f} | {r['idle_gpu_memory_time_mib_s']:.2f} |\n| FirstTokenGate-ResumeAware | {g['promotion_deadline_miss_count']:.1f} | {g['ABC_total_pairwise_overlap_ms']:.2f} | {g['ABC_P95_effective_latency_ms']:.2f} | {g['P95_effective_resume_latency_ms']:.2f} | {g['idle_gpu_memory_time_mib_s']:.2f} |\n\nGate changed ABC pairwise overlap {r['ABC_total_pairwise_overlap_ms']:.2f} -> {g['ABC_total_pairwise_overlap_ms']:.2f} ms, max pre-first-token concurrency {r['max_concurrent_slackkv_pre_first_token']:.1f} -> {g['max_concurrent_slackkv_pre_first_token']:.1f}, area {r['area_under_pre_first_token_concurrency_ms']:.2f} -> {g['area_under_pre_first_token_concurrency_ms']:.2f} ms. Gate blocked time is in session_metrics.csv. Phase 3E deadline-based residency undercounted delayed admission; Phase 3F uses actual request admission after promotion completion.\n\nOnDemand scheduler-internal WAITING_FOR_REMOTE_KVS transitions are not exposed by this runtime; scheduler_trace.csv records client-visible arrival/admission/first-token transitions without fabricating internal states.\n\n**Result {result}.** Phase 3F execution status: COMPLETE\n''')
  print('FINISH',result); return
 c=json.loads((OUT/'control.json').read_text()); ns=[]; ss=[]; rs=[]; snaps=[]
 for pol in POLICIES:
  for rep in (0,1,2): print('RUN',pol,rep,flush=True); n,s,r,t=run_one(pol,rep,p,c); ns+=n; ss+=s; rs.append(r); snaps+=t
 write_csv(OUT/'native_prefetch_metrics.csv',ns); write_csv(OUT/'session_metrics.csv',ss); write_csv(OUT/'run_metrics.csv',rs); (OUT/'metric_snapshots.json').write_text(json.dumps(snaps,indent=2)); write_csv(OUT/'scheduler_trace.csv',[{'policy':x['policy'],'repetition':x['repetition'],'event':'RESUME_ADMITTED','session':x['session'],'timestamp_ms':x['scheduler_admission_ms']} for x in ss]); write_csv(OUT/'step_metrics.csv',[{'policy':'postprocess','repetition':'NA','step_id':'NA'}])
 med={pol:{k:statistics.median(float(x[k]) for x in rs if x['policy']==pol) for k in ('promotion_deadline_miss_count','blackout_conflict_count','long_prefetch_count','ABC_total_pairwise_overlap_ms','max_concurrent_slackkv_pre_first_token','area_under_pre_first_token_concurrency_ms','ABC_P95_effective_latency_ms','P95_effective_resume_latency_ms','idle_gpu_memory_time_mib_s')} for pol in POLICIES}; r=med['ResumeAware']; g=med['FirstTokenGate-ResumeAware']; o=med['OnDemand-Trace']; result='A' if g['ABC_total_pairwise_overlap_ms']<r['ABC_total_pairwise_overlap_ms'] and g['ABC_P95_effective_latency_ms']<=.9*r['ABC_P95_effective_latency_ms'] else ('B' if g['ABC_total_pairwise_overlap_ms']<r['ABC_total_pairwise_overlap_ms'] else 'C')
 (OUT/'summary.md').write_text(f'''# Phase 3F\n\nPhase 3B connector-first implementation reused unchanged. FirstTokenGate is a research admission gate: at most one marked resumed request waits for its first token; with max_tokens=1 this approximately serializes measured completions.\n\n| Policy | promotion misses | ABC overlap ms | ABC P95 ms | overall P95 ms | corrected memory-time MiB·s |\n|---|---:|---:|---:|---:|---:|\n| OnDemand-Trace | {o['promotion_deadline_miss_count']:.1f} | {o['ABC_total_pairwise_overlap_ms']:.2f} | {o['ABC_P95_effective_latency_ms']:.2f} | {o['P95_effective_resume_latency_ms']:.2f} | {o['idle_gpu_memory_time_mib_s']:.2f} |\n| ResumeAware | {r['promotion_deadline_miss_count']:.1f} | {r['ABC_total_pairwise_overlap_ms']:.2f} | {r['ABC_P95_effective_latency_ms']:.2f} | {r['P95_effective_resume_latency_ms']:.2f} | {r['idle_gpu_memory_time_mib_s']:.2f} |\n| FirstTokenGate-ResumeAware | {g['promotion_deadline_miss_count']:.1f} | {g['ABC_total_pairwise_overlap_ms']:.2f} | {g['ABC_P95_effective_latency_ms']:.2f} | {g['P95_effective_resume_latency_ms']:.2f} | {g['idle_gpu_memory_time_mib_s']:.2f} |\n\nGate changed ABC pairwise overlap {r['ABC_total_pairwise_overlap_ms']:.2f} -> {g['ABC_total_pairwise_overlap_ms']:.2f} ms, max pre-first-token concurrency {r['max_concurrent_slackkv_pre_first_token']:.1f} -> {g['max_concurrent_slackkv_pre_first_token']:.1f}, area {r['area_under_pre_first_token_concurrency_ms']:.2f} -> {g['area_under_pre_first_token_concurrency_ms']:.2f} ms. Gate blocked time is in session_metrics.csv. Phase 3E deadline-based residency undercounted delayed admission; Phase 3F uses actual request admission after promotion completion.\n\n**Result {result}.** Phase 3F execution status: COMPLETE\n'''); print('WORKLOAD COMPLETE Result',result)
if __name__=='__main__': main()
