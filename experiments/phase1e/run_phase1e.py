#!/usr/bin/env python3
"""Small prediction-noise validation using the corrected Fix2 event loop."""
import csv, heapq, json, math, random, statistics, sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from experiments.phase1b_fix.run_phase1b_fix import MAX_CONTEXT,extract_events,session_workload,write_csv
from slackkv.simulator.policies import queue_safe_idle_s
from slackkv.simulator.simulator import EPS,percentile
from slackkv.simulator.transfer_model import TransferModel
KIB=1024; FP=56*KIB; SIGMAS=(0.,.1,.25,.5); POLICIES=('KeepAll','OffloadAll-QueueJIT','FixedTTL-0.5','FixedTTL-1','SlackRatio-1.25','SlackRatio-2')
def predicted(pause,sigma,seed,sid,index):
    rng=random.Random(f'phase1d:{seed}:{sid}:{index}:{sigma}')
    return max(pause*(1+rng.gauss(0,sigma)),.001)
def allow(policy,pause,down,up):
    if policy=='KeepAll': return False
    if policy=='OffloadAll-QueueJIT': return True
    if policy.startswith('FixedTTL-'): return pause>=float(policy.split('-',1)[1])
    return pause/(down+up)>=float(policy.split('-',1)[1])
def run(groups,offs,policy,sigma,seed,model):
    future=[];states={};pending=[];rows=[];serial=0;now=math.inf
    for order,(sid,events) in enumerate(sorted(groups.items())):
        es=[dict(x) for x in events];es[0]['actual']=offs[sid];states[sid]={'e':es,'i':0,'o':es[0]['release_epoch_s'],'off':offs[sid],'shift':0.,'order':order};heapq.heappush(future,(offs[sid],order,sid));now=min(now,offs[sid])
    def advance(sid,r,completion):
        s=states[sid];r['completion']=completion;r['stall']=max(0.,completion-r['actual_deadline']);s['shift']+=r['stall'];s['i']+=1
        if s['i']==len(s['e']):return
        e=s['e'][s['i']];e['actual']=max(s['off']+e['release_epoch_s']-s['o']+s['shift'],completion);heapq.heappush(future,(e['actual'],s['order'],sid))
    def release(sid):
        nonlocal serial
        s=states[sid];e=s['e'][s['i']];size=e['reusable_tokens']*FP;d,u=model[0].piecewise_service_time_s(size),model[1].piecewise_service_time_s(size);p=predicted(e['pause_s'],sigma,seed,sid,s['i']);off=allow(policy,p,d,u)
        r={'sid':sid,'idx':s['i'],'release':e['actual'],'actual_deadline':e['actual']+e['pause_s'],'pred_deadline':e['actual']+p,'pause':e['pause_s'],'pred_pause':p,'size_gb':size/1e9,'down':d,'up':u,'offload':off,'ready':e['actual']+d,'completion':e['actual']+e['pause_s'],'stall':0.};rows.append(r)
        if off:heapq.heappush(pending,(r['ready'],serial,r));serial+=1
        else:advance(sid,r,r['actual_deadline'])
    while future or pending:
        while future and future[0][0]<=now+EPS:release(heapq.heappop(future)[2])
        ready=[x[2] for x in pending if x[0]<=now+EPS]
        if not ready:
            nxt=([future[0][0]] if future else [])+([pending[0][0]] if pending else [])
            if not nxt: break
            now=min(nxt);continue
        safe=queue_safe_idle_s(now,[SimpleNamespace(deadline_s=x['pred_deadline'],service_time_s=x['up'],job_id=f"{x['sid']}:{x['idx']}") for x in ready])
        if safe>EPS and now+safe>now:
            events=[now+safe]+([future[0][0]] if future and future[0][0]>now+EPS else [])+[x[0] for x in pending if x[0]>now+EPS];now=min(events);continue
        r=min(ready,key=lambda x:(x['pred_deadline'],x['release'],x['sid'],x['idx']));pending.remove(next(x for x in pending if x[2] is r));heapq.heapify(pending);advance(r['sid'],r,now+r['up']);now=r['completion']
    stalls=[r['stall']*1000 for r in rows];keep=sum(r['size_gb']*(r['actual_deadline']-r['release']) for r in rows);local=0.
    for r in rows:
        if not r['offload']: local+=r['size_gb']*(r['actual_deadline']-r['release'])
        else: local+=r['size_gb']*r['down']+r['size_gb']*max(0.,r['actual_deadline']-r['completion'])
    return {'job_count':len(rows),'offload_fraction':sum(r['offload'] for r in rows)/len(rows),'deadline_miss_rate':sum(x>EPS for x in [r['stall'] for r in rows])/len(rows),'stall_ms_p95':percentile(stalls,.95),'stall_ms_p99':percentile(stalls,.99),'normalized_local_memory_time':local/keep},rows
def main():
    out=ROOT/'results/phase1d';out.mkdir(parents=True,exist_ok=True);events,_,_=extract_events(ROOT/'data/tracelab/syfi_coding_trace.duckdb');events=[e for e in events if 0<e['reusable_tokens']<=MAX_CONTEXT];model=(TransferModel.from_phase0_direction(ROOT,'gpu0_to_pinned_cpu'),TransferModel.from_phase0_direction(ROOT,'pinned_cpu_to_gpu0'));raw=[]
    for seed in range(5):
        g,o=session_workload(events,32,seed)
        for sigma in SIGMAS:
            for p in POLICIES:
                m,_=run(g,o,p,sigma,seed,model);raw.append({'seed':seed,'sigma':sigma,'policy':p,**m})
    write_csv(out/'aggregate_metrics.csv',tuple(raw[0]),raw);agg=[]
    for sigma in SIGMAS:
        for p in POLICIES:
            rs=[r for r in raw if r['sigma']==sigma and r['policy']==p];q=lambda k:statistics.median(float(r[k]) for r in rs);agg.append({'sigma':sigma,'policy':p,**{k:q(k) for k in raw[0] if k not in ('seed','sigma','policy')}})
    write_csv(out/'aggregated_metrics.csv',tuple(agg[0]),agg);(out/'config.json').write_text(json.dumps({'base_commit':'2586d3ac1a0c99a09b76d21de564614e2eb74ae8','seeds':list(range(5)),'sigmas':SIGMAS,'policies':POLICIES,'footprint_bytes_per_token':FP,'M':32},indent=2)+'\n');(out/'commands.sh').write_text('python experiments/phase1d/run_phase1d.py\npython scripts/plot_phase1d.py\n');print(len(raw))
if __name__=='__main__':main()
