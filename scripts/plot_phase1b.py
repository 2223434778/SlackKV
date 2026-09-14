#!/usr/bin/env python3
"""Regenerate Phase 1B plots from compact committed CSVs."""
import csv, statistics
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
OUT=Path('results/phase1b'); PLOTS=OUT/'plots'; POLICIES=('OnDemand','ASAP-EDF','NaiveJIT','SlackKV-QueueJIT')
def read(name):
 with open(OUT/name,newline='') as f:return list(csv.DictReader(f))
def median(rows,key):return statistics.median(float(r[key]) for r in rows)
def base(rows):return [r for r in rows if r['workload']=='session' and r['footprint_bytes_per_token']=='131072']
def points(rows,policy,key):
 return [(m,median([r for r in rows if r['policy']==policy and int(r['concurrency'])==m],key)) for m in (4,8,16,32,64)]
def line(rows,key,name,ylabel):
 plt.figure(figsize=(8,5))
 for p in POLICIES:
  x=points(rows,p,key);plt.plot(*zip(*x),marker='o',label=p)
 if key=='stall_ms_p95': plt.yscale('log')
 plt.xlabel('Multiplexed real sessions (M)');plt.ylabel(ylabel);plt.legend();plt.grid(alpha=.3);plt.tight_layout();plt.savefig(PLOTS/name,dpi=160);plt.close()
def main():
 PLOTS.mkdir(parents=True,exist_ok=True); agg=base(read('aggregate_metrics.csv'))
 cdf=read('trace_cdf.csv')
 for measure,name,xlabel in [('pause_duration_s','trace_pause_duration_cdf.png','Pause duration (s)'),('kv_size_gb','trace_kv_size_cdf.png','Derived KV size (GB), 128 KiB/token')]:
  r=[x for x in cdf if x['measure']==measure];plt.figure(figsize=(7,5));plt.plot([float(x['value']) for x in r],[float(x['cdf']) for x in r]);plt.xscale('log');plt.xlabel(xlabel);plt.ylabel('Empirical CDF');plt.grid(alpha=.3);plt.tight_layout();plt.savefig(PLOTS/name,dpi=160);plt.close()
 line(agg,'deadline_miss_rate','miss_rate_vs_concurrency.png','Deadline miss rate');line(agg,'stall_ms_p95','p95_stall_vs_concurrency.png','P95 resume stall (ms)');line(agg,'prefetch_vram_time_gbs_total','vram_time_vs_concurrency.png','Prefetch VRAM-time (GB·s)');line(agg,'peak_prefetched_gb','peak_prefetch_memory_vs_concurrency.png','Peak early-prefetched memory (GB)')
 plt.figure(figsize=(7,5))
 offsets={'OnDemand':(5,8),'ASAP-EDF':(5,5),'NaiveJIT':(5,-15),'SlackKV-QueueJIT':(5,-30)}
 for p in POLICIES:
  r=[x for x in agg if x['policy']==p and int(x['concurrency'])==32];x=median(r,'prefetch_to_keep_ratio');y=median(r,'stall_ms_p95');plt.scatter(x,y,s=60);plt.annotate(p,(x,y),xytext=offsets[p],textcoords='offset points')
 plt.scatter(1,0,marker='x',s=70,label='KeepReference');plt.xlabel('Prefetch VRAM-time / KeepReference');plt.ylabel('P95 stall (ms)');plt.legend();plt.grid(alpha=.3);plt.tight_layout();plt.savefig(PLOTS/'stall_vs_vram_time_tradeoff.png',dpi=160);plt.close()
 plt.figure(figsize=(8,5))
 for p in ('NaiveJIT','SlackKV-QueueJIT'):
  x=points(agg,p,'deadline_miss_rate');plt.plot(*zip(*x),marker='o',label=p)
 plt.xlabel('Multiplexed real sessions (M)');plt.ylabel('Deadline miss rate');plt.legend();plt.grid(alpha=.3);plt.tight_layout();plt.savefig(PLOTS/'queuejit_vs_naivejit_real_trace.png',dpi=160);plt.close()
 miss=defaultdict(float)
 for r in read('miss_analysis.csv'):miss[r['reason']]+=float(r['miss_count'])
 plt.figure(figsize=(7,5));plt.bar(list(miss),list(miss.values()));plt.xticks(rotation=20,ha='right');plt.ylabel('QueueJIT missed deadlines (count)');plt.tight_layout();plt.savefig(PLOTS/'miss_reason_breakdown.png',dpi=160);plt.close()
 s=read('trace_scatter_sample.csv');plt.figure(figsize=(7,5))
 for p in sorted(set(r['provider'] for r in s)):
  r=[x for x in s if x['provider']==p];plt.scatter([float(x['kv_size_gb']) for x in r],[float(x['pause_duration_s']) for x in r],s=5,alpha=.4,label=p)
 plt.xscale('log');plt.yscale('log');plt.xlabel('Derived KV size (GB), 128 KiB/token');plt.ylabel('Pause duration (s)');plt.legend();plt.grid(alpha=.3);plt.tight_layout();plt.savefig(PLOTS/'kv_size_vs_pause_duration.png',dpi=160);plt.close()
if __name__=='__main__':main()
