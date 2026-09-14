#!/usr/bin/env python3
"""Phase 1C transfer-normalized selective offload simulator."""
import csv,heapq,json,math,random,statistics,sys
from types import SimpleNamespace
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from experiments.phase1b_fix.run_phase1b_fix import MAX_CONTEXT,extract_events,session_workload,poisson_workload,write_csv,quantiles
from slackkv.simulator.admission import admit,slack_ratio
from slackkv.simulator.jobs import PromotionJob
from slackkv.simulator.policies import Policy,queue_safe_idle_s
from slackkv.simulator.simulator import EPS,JobOutcome,percentile
from slackkv.simulator.selective_offload import local_metrics
from slackkv.simulator.transfer_model import TransferModel

KIB=1024; PRIMARY=56*KIB; STRESS=128*KIB
POLICIES=['KeepAll','OffloadAll-QueueJIT','FeasibilityOracle']+[f'FixedTTL-{x}' for x in (.1,.25,.5,1,2,5,10)]+[f'SlackRatio-{x}' for x in (1,1.25,1.5,2,3,4)]
PREFIX=('workload','footprint_bytes_per_token','concurrency','rho','seed','policy')

def run_sessions(groups,offsets,footprint,policy,model):
    """Closed-loop replay: an offloaded miss delays only that session's future."""
    states={}; future=[]; serial=0; records=[]; pending=[]; outcomes=[]; busy=0.
    for i,(sid,evs) in enumerate(sorted(groups.items())):
        evs=[dict(e) for e in evs]; evs[0]['actual']=offsets[sid]
        states[sid]={'e':evs,'i':0,'origin':evs[0]['release_epoch_s'],'offset':offsets[sid],'shift':0.}
        heapq.heappush(future,(offsets[sid],i,sid))
    now=future[0][0]
    def release(sid):
        nonlocal serial
        st=states[sid]; e=st['e'][st['i']]; size=e['reusable_tokens']*footprint; down=model[0].piecewise_service_time_s(size);up=model[1].piecewise_service_time_s(size);ratio=slack_ratio(e['pause_s'],down,up);off=admit(policy,e['pause_s'],down,up)
        r={'sid':sid,'idx':st['i'],'release':e['actual'],'deadline':e['actual']+e['pause_s'],'pause':e['pause_s'],'size':size,'size_gb':size/1e9,'down':down,'up':up,'ratio':ratio,'offload':off,'completion':e['actual']+e['pause_s'],'stall':0.}
        records.append(r)
        if off: heapq.heappush(pending,(r['release']+down,serial,r));serial+=1
        else: advance(sid,r,r['deadline'])
    def advance(sid,r,completion):
        st=states[sid]; stall=max(0.,completion-r['deadline']);st['shift']+=stall;st['i']+=1
        if st['i']>=len(st['e']):return
        e=st['e'][st['i']]; nominal=st['offset']+e['release_epoch_s']-st['origin']+st['shift'];e['actual']=max(nominal,completion);heapq.heappush(future,(e['actual'],len(future),sid))
    while future or pending:
        while future and future[0][0]<=now+EPS: _,_,sid=heapq.heappop(future);release(sid)
        elig=[x for x in pending if x[0]<=now+EPS]; ready=[x[2] for x in elig]
        if not ready:
            times=([future[0][0]] if future else [])+([min(x[0] for x in pending)] if pending else [])
            if not times:break
            now=min(times);continue
        safe=queue_safe_idle_s(now,[SimpleNamespace(deadline_s=x['deadline'],service_time_s=x['up'],job_id=x['sid']+':'+str(x['idx'])) for x in ready])
        nxt=future[0][0] if future else math.inf
        if safe>EPS and nxt<now+safe-EPS:now=nxt;continue
        r=min(ready,key=lambda x:(x['deadline'],x['release'],x['sid'],x['idx'])); pending.remove(next(x for x in pending if x[2] is r));start=now;finish=start+r['up'];r['completion']=finish;r['stall']=max(0.,finish-r['deadline']);outcomes.append(r);busy+=r['up'];now=finish;advance(r['sid'],r,finish)
    metrics=local_metrics(records); span=max(max(r['deadline'] for r in records)-min(r['release'] for r in records),EPS);metrics['promotion_queue_utilization']=busy/span
    return records,metrics

def static(events,footprint,policy,model):
    groups={'p':[]}; offsets={'p':0.}
    for e in events: groups['p'].append({**e,'release_epoch_s':e['release_time_s']})
    return run_sessions(groups,offsets,footprint,policy,model)

def decisions(records,policy):
    c=Counter()
    for r in records:
        if policy.startswith('SlackRatio-'):
            g=float(policy.split('-',1)[1]); c['intrinsically_nonoffloadable' if r['ratio']<1 else 'kept_by_safety_margin' if r['ratio']<g else 'admitted_for_offload']+=1
        elif policy.startswith('FixedTTL-'):c['admitted_by_ttl' if r['offload'] else 'kept_by_ttl']+=1
        else:c['admitted_for_offload' if r['offload'] else 'kept']+=1
    return c

def main():
 out=ROOT/'results/phase1c_fix';out.mkdir(parents=True,exist_ok=True);events,skips,_=extract_events(ROOT/'data/tracelab/syfi_coding_trace.duckdb');events=[e for e in events if 0<e['reusable_tokens']<=MAX_CONTEXT];model=(TransferModel.from_phase0_direction(ROOT,'gpu0_to_pinned_cpu'),TransferModel.from_phase0_direction(ROOT,'pinned_cpu_to_gpu0'));rows=[];admitrows=[];miss=[]
 for footprint,seeds,ms in ((PRIMARY,range(10),(4,8,16,32,64)),(STRESS,range(5),(16,32,64))):
  for m in ms:
   for seed in seeds:
    groups,offs=session_workload(events,m,seed)
    for p in POLICIES:
     rec,met=run_sessions(groups,offs,footprint,p,model);ctx={'workload':'closed_loop_session','footprint_bytes_per_token':footprint,'concurrency':m,'rho':'','seed':seed,'policy':p};rows.append({**ctx,**met});
     if p=='FeasibilityOracle' or p.startswith('SlackRatio-'): assert not any(r['offload'] and r['ratio'] < 1-EPS for r in rec)
     for reason,n in decisions(rec,p).items():admitrows.append({**ctx,'decision':reason,'count':n})
     for r in rec:
      if r['stall']>EPS and r['offload']:
       reason='transfer_model_boundary_or_rounding' if abs(r['ratio']-1)<1e-9 else 'large_KV_blocking' if r['size']>=4*(1024**3) else 'promotion_queue_collision';miss.append({**ctx,'reason':reason,'count':1})
  for rho in (.25,.5,.75,.9):
   for seed in range(10):
    replay=poisson_workload(events,rho,1000+seed,footprint,model[1])
    for p in POLICIES:
     rec,met=static(replay,footprint,p,model);ctx={'workload':'poisson_tuple','footprint_bytes_per_token':footprint,'concurrency':'','rho':rho,'seed':1000+seed,'policy':p};rows.append({**ctx,**met});
     if p=='FeasibilityOracle' or p.startswith('SlackRatio-'): assert not any(r['offload'] and r['ratio'] < 1-EPS for r in rec)
     for reason,n in decisions(rec,p).items():admitrows.append({**ctx,'decision':reason,'count':n})
 fields=PREFIX+tuple(k for k in rows[0] if k not in PREFIX);write_csv(out/'aggregate_metrics.csv',fields,rows);write_csv(out/'admission_metrics.csv',PREFIX+('decision','count'),admitrows);write_csv(out/'miss_analysis.csv',PREFIX+('reason','count'),miss)
 char=[]
 for fp in (PRIMARY,STRESS):
  vals={'pause_ms':[e['pause_s']*1000 for e in events],'kv_gb':[e['reusable_tokens']*fp/1e9 for e in events],'d2h_ms':[model[0].piecewise_service_time_s(e['reusable_tokens']*fp)*1000 for e in events],'h2d_ms':[model[1].piecewise_service_time_s(e['reusable_tokens']*fp)*1000 for e in events]};vals['roundtrip_ms']=[a+b for a,b in zip(vals['d2h_ms'],vals['h2d_ms'])];vals['slack_ratio']=[p/(r/1000) for p,r in zip(vals['pause_ms'],vals['roundtrip_ms'])]
  for name,v in vals.items():char += [{'footprint_bytes_per_token':fp,'measure':name,'statistic':k,'value':x} for k,x in quantiles(v).items()]
  if fp==PRIMARY:
   for lo,hi,label in ((0,1,'<1'),(1,1.25,'1-1.25'),(1.25,1.5,'1.25-1.5'),(1.5,2,'1.5-2'),(2,3,'2-3'),(3,4,'3-4'),(4,math.inf,'>=4')):char.append({'footprint_bytes_per_token':fp,'measure':'slack_ratio_fraction','statistic':label,'value':sum(lo<=x<hi for x in vals['slack_ratio'])/len(events)})
 write_csv(out/'characterization.csv',('footprint_bytes_per_token','measure','statistic','value'),char)
 # C0 is an event-level invariant asserted above.  C1/C2 are intentionally
 # computed by the separate cross-seed aggregation after all rows are written.
 c0=True
 sanity=[]
 for size in (256<<20,1<<30,4<<30,max(e['reusable_tokens'] for e in events)*PRIMARY): sanity.append({'size_bytes':size,'d2h_ms':model[0].piecewise_service_time_s(size)*1000,'h2d_ms':model[1].piecewise_service_time_s(size)*1000,'roundtrip_ms':(model[0].piecewise_service_time_s(size)+model[1].piecewise_service_time_s(size))*1000})
 write_csv(out/'transfer_sanity.csv',('size_bytes','d2h_ms','h2d_ms','roundtrip_ms'),sanity)
 config={'phase0_calibration_commit':model[0].phase0_commit,'d2h_direction':model[0].direction,'h2d_direction':model[1].direction,'retained_events':len(events),'context_cap':MAX_CONTEXT,'footprints':[PRIMARY,STRESS],'policies':POLICIES,'c0':c0,'d2h_assumption':'immediate and non-contended; optimistic','skips':dict(skips)};(out/'config.json').write_text(json.dumps(config,indent=2)+'\n');print('rows',len(rows),'C0',c0)
 from experiments.phase1c_fix.analyze_phase1c_fix import main as analyze
 analyze()
if __name__=='__main__':main()
