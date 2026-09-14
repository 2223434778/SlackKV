"""Selective D2H/H2D lifecycle metrics used by Phase 1C."""
from statistics import mean
from .simulator import EPS, percentile

def local_metrics(records):
    intervals=[]; keep=0.; d2h=0.; h2d=0.; stalls=[]
    for r in records:
        p=r['pause']; size=r['size_gb']; keep += size*p; stalls.append(r['stall']*1000)
        if not r['offload']: intervals.append((r['release'],r['deadline'],size)); continue
        d2h += r['size']; h2d += r['size']; down_end=min(r['deadline'],r['release']+r['down'])
        intervals.append((r['release'],down_end,size))
        if r['completion'] < r['deadline']-EPS: intervals.append((r['completion'],r['deadline'],size))
    first=min(r['release'] for r in records); last=max(r['deadline'] for r in records); events=[]
    for a,b,s in intervals:
        if b>a: events += [(a,1,s),(b,0,-s)]
    level=0.; prev=first; weighted=[]
    for t,_,delta in sorted(events):
        if t>prev: weighted.append((level,t-prev))
        level+=delta;prev=t
    if last>prev: weighted.append((level,last-prev))
    total=sum(v*d for v,d in weighted); span=max(last-first,EPS); cumulative=0.; p95=0.
    for v,d in sorted(weighted):
        cumulative+=d
        if cumulative/span>=.95: p95=v;break
    n=len(records); off=sum(r['offload'] for r in records)
    return {'job_count':n,'keep_count':n-off,'offload_count':off,'keep_fraction':(n-off)/n,'offload_fraction':off/n,
      'D2H_bytes':d2h,'H2D_bytes':h2d,'total_transfer_bytes':d2h+h2d,'D2H_transfer_time_total':sum(r['down'] for r in records if r['offload']),'H2D_transfer_time_total':sum(r['up'] for r in records if r['offload']),
      'deadline_miss_count':sum(x>EPS for x in [r['stall'] for r in records]),'deadline_miss_rate':sum(x>EPS for x in [r['stall'] for r in records])/n,'stall_ms_mean':mean(stalls),'stall_ms_p50':percentile(stalls,.5),'stall_ms_p95':percentile(stalls,.95),'stall_ms_p99':percentile(stalls,.99),'max_stall_ms':max(stalls),
      'local_paused_vram_time_gbs_total':total,'normalized_local_memory_time':total/keep if keep else 0.,'mean_local_paused_GB':total/span,'P95_local_paused_GB':p95,'peak_local_paused_GB':max([v for v,_ in weighted] or [0.])}
