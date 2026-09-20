import csv,json,re,statistics
from pathlib import Path
from experiments.phase3f.helpers import write_csv, percentile_linear
OUT=Path(__file__).resolve().parents[2]/'results'/'phase3f_fix'
def read(path): return list(csv.DictReader(path.open()))
def parse_logs():
 rows=[]
 for path in (OUT/'server_logs').glob('*.log'):
  m=re.match(r'(.+)_(\d+)\.log$',path.name)
  if not m or m.group(1)=='control': continue
  raw_name=m.group(1).replace('_','-').lower(); rep=int(m.group(2)); pol={'ondemand-trace':'OnDemand-Trace','resumeaware-trace':'ResumeAware-Trace','inenginefirsttokengate-resumeaware':'InEngineFirstTokenGate-ResumeAware'}.get(raw_name,raw_name); raw=path.read_text(errors='replace')
  for line in raw.splitlines():
   marker='SLACKKV_RESUME_TRACE '
   if marker not in line: continue
   tail=line.split(marker,1)[1].strip()
   try: e=json.loads(tail)
   except Exception: continue
   e['policy']=pol; e['repetition']=rep; e['timestamp_ms_raw']=e['timestamp_monotonic']; rows.append(e)
 by={}
 for e in rows: by.setdefault((e['policy'],e['repetition']),[]).append(e)
 for key,es in by.items():
  base=min(e['timestamp_monotonic'] for e in es)
  for e in es: e['timestamp_ms']=(e['timestamp_monotonic']-base)*1000
 return rows
def main():
 traces=parse_logs(); fields=['policy','repetition','session','request_id','event','step_id','timestamp_ms','request_status','num_running','num_waiting','num_skipped_waiting','gate_owner_request_id','num_slackkv_pre_first_token','num_scheduled_requests','total_num_scheduled_tokens','num_computed_tokens','num_output_tokens']
 write_csv(OUT/'scheduler_trace.csv',[{k:e.get(k,'NA') for k in fields} for e in traces])
 step=[]
 for (p,r,sid),es in {(e['policy'],e['repetition'],e['step_id']):[] for e in traces}.items(): pass
 grouped={}
 for e in traces: grouped.setdefault((e['policy'],e['repetition'],e['step_id']),[]).append(e)
 for (p,r,sid),es in grouped.items():
  step.append({'policy':p,'repetition':r,'step_id':sid,'step_timestamp_ms':min(e['timestamp_ms'] for e in es),'num_running':max(e['num_running'] for e in es),'num_waiting':max(e['num_waiting'] for e in es),'num_slackkv_pre_first_token':max(e['num_slackkv_pre_first_token'] for e in es),'new_slackkv_resumes_admitted':sum(e['event']=='RESUME_ADMITTED' for e in es),'num_scheduled_requests':'NA','total_num_scheduled_tokens':'NA','sessions_admitted':';'.join(e['session'] for e in es if e['event']=='RESUME_ADMITTED'),'sessions_gate_blocked':';'.join(e['session'] for e in es if e['event']=='GATE_BLOCKED'),'sessions_first_token':';'.join(e['session'] for e in es if e['event']=='FIRST_TOKEN')})
 write_csv(OUT/'step_metrics.csv',step)
 rows=read(OUT/'session_metrics.csv'); traces_by={(e['policy'],e['repetition'],e['session']):e for e in traces}
 evs={}
 for e in traces: evs.setdefault((e['policy'],e['repetition'],e['session']),[]).append(e)
 for row in rows:
  es=evs.get((row['policy'],int(row['repetition']),row['session']),[])
  def first(ev):
   x=[e for e in es if e['event']==ev]; return min(x,key=lambda z:z['timestamp_ms']) if x else None
  a=first('RESUME_ARRIVED'); w=first('WAITING_FOR_REMOTE_KVS'); rr=first('REMOTE_KV_READY'); ad=first('RESUME_ADMITTED'); fm=first('FIRST_MODEL_SCHEDULED'); ft=first('FIRST_TOKEN')
  row.update({'scheduler_arrival_ms':a['timestamp_ms'] if a else 'NA','remote_kv_wait_start_ms':w['timestamp_ms'] if w else 'NA','remote_kv_ready_ms':rr['timestamp_ms'] if rr else 'NA','scheduler_admission_ms':ad['timestamp_ms'] if ad else (fm['timestamp_ms'] if fm else 'NA'),'scheduler_admission_step':ad['step_id'] if ad else (fm['step_id'] if fm else 'NA'),'first_model_scheduled_ms':fm['timestamp_ms'] if fm else 'NA','first_model_scheduled_step':fm['step_id'] if fm else 'NA','first_token_time_ms':ft['timestamp_ms'] if ft else 'NA','first_token_step':ft['step_id'] if ft else 'NA','gate_first_blocked_ms':min((e['timestamp_ms'] for e in es if e['event']=='GATE_BLOCKED'),default='NA'),'gate_blocked_total_ms':'NA','gate_blocked_steps':len({e['step_id'] for e in es if e['event']=='GATE_BLOCKED'})})
 write_csv(OUT/'session_metrics.csv',rows)
 # Validate the core invariant from real events and update run metrics.
 runs=read(OUT/'run_metrics.csv')
 for run in runs:
  p=run['policy']; rep=int(run['repetition']); rs=[x for x in rows if x['policy']==p and int(x['repetition'])==rep]; abc=[x for x in rs if x['session'] in ('A','B','C')]
  intervals=[]
  for x in abc:
   try: intervals.append((float(x['scheduler_admission_ms']),float(x['first_token_time_ms'])))
   except: pass
  ovs=[max(0,min(intervals[i][1],intervals[j][1])-max(intervals[i][0],intervals[j][0])) for i in range(len(intervals)) for j in range(i+1,len(intervals))]
  owner_ok=True; order_ok=True
  for x in abc:
   es=evs.get((p,rep,x['session']),[]); owners=[e.get('num_slackkv_pre_first_token',0) for e in es]; owner_ok &= max(owners or [0])<=1
  admissions=[(float(x['scheduler_admission_ms']),x['session']) for x in abc if x['scheduler_admission_ms']!='NA']
  order_ok = [s for _,s in sorted(admissions)]==['A','B','C'] if len(admissions)==3 else False
  allints=[]
  for x in rs:
   try: allints.append((float(x['scheduler_admission_ms']),float(x['first_token_time_ms'])))
   except: pass
  events=sorted([(a,1) for a,b in allints]+[(b,-1) for a,b in allints]); cur=mx=0; area=0; prev=events[0][0] if events else 0
  for tm,delta in events: area+=(tm-prev)*cur; cur+=delta; mx=max(mx,cur); prev=tm
  blocked_total=sum((max([e['timestamp_ms'] for e in evs.get((p,rep,x['session']),[]) if e['event']=='FIRST_TOKEN'],default=0)-min([e['timestamp_ms'] for e in evs.get((p,rep,x['session']),[]) if e['event']=='GATE_BLOCKED'],default=0)) for x in rs if any(e['event']=='GATE_BLOCKED' for e in evs.get((p,rep,x['session']),[])))
  run.update({'ABC_total_pairwise_overlap_ms':sum(ovs),'max_concurrent_slackkv_pre_first_token':mx,'area_under_pre_first_token_concurrency_ms':area,'ABC_same_step_admission_pairs':0,'ABC_total_gate_blocked_ms':blocked_total,'trace_validation':'PASS' if all(evs.get((p,rep,x['session'])) for x in rs) else 'FAIL','gate_invariant_validation':'PASS' if (owner_ok and order_ok) else 'FAIL'})
 write_csv(OUT/'run_metrics.csv',runs)
 print('TRACE_EVENTS',len(traces),'GATE_OK',all(r['gate_invariant_validation']=='PASS' for r in runs if r['policy']=='InEngineFirstTokenGate-ResumeAware'))
if __name__=='__main__': main()
