import csv,statistics
from pathlib import Path
import matplotlib.pyplot as plt
O=Path('results/phase1c');P=O/'plots';P.mkdir(parents=True,exist_ok=True)
R=list(csv.DictReader(open(O/'aggregate_metrics.csv')))
def med(x,k):return statistics.median(float(r[k]) for r in x)
def q(m=32):return [r for r in R if r['workload']=='closed_loop_session' and r['footprint_bytes_per_token']=='57344' and r['concurrency']==str(m)]
def pts(data,key):
 out=[]
 for p in sorted(set(r['policy'] for r in data)):
  x=[r for r in data if r['policy']==p];out.append((p,med(x,'normalized_local_memory_time'),med(x,key)))
 return out
def scatter(name,key,y):
 plt.figure(figsize=(8,5))
 for p,x,v in pts(q(),key):plt.scatter(x,v);plt.annotate(p,(x,v),fontsize=7)
 plt.xlabel('Normalized local paused-KV memory-time');plt.ylabel(y);plt.grid(alpha=.3);plt.tight_layout();plt.savefig(P/name,dpi=140);plt.close()
scatter('miss_rate_vs_memory_time.png','deadline_miss_rate','Deadline miss rate');scatter('p95_stall_vs_memory_time.png','stall_ms_p95','P95 stall (ms)');scatter('fixedttl_vs_slackratio_pareto.png','stall_ms_p95','P95 stall (ms)')
for name,key,y in [('memory_time_vs_concurrency.png','normalized_local_memory_time','Normalized memory-time'),('p95_stall_vs_concurrency.png','stall_ms_p95','P95 stall (ms)'),('peak_paused_kv_vs_concurrency.png','peak_local_paused_GB','Peak paused KV (GB)')]:
 plt.figure(figsize=(8,5))
 for p in ['KeepAll','OffloadAll-QueueJIT','SlackRatio-1','SlackRatio-2','SlackRatio-3','SlackRatio-4']:
  xs=[];ys=[]
  for m in (4,8,16,32,64):
   x=[r for r in q(m) if r['policy']==p]
   if x:xs.append(m);ys.append(med(x,key))
  plt.plot(xs,ys,marker='o',label=p)
 plt.xlabel('M');plt.ylabel(y);plt.legend(fontsize=7);plt.grid(alpha=.3);plt.tight_layout();plt.savefig(P/name,dpi=140);plt.close()
C=list(csv.DictReader(open(O/'characterization.csv')));x=[r for r in C if r['footprint_bytes_per_token']=='57344' and r['measure']=='slack_ratio_fraction'];plt.figure();plt.bar([r['statistic'] for r in x],[float(r['value']) for r in x]);plt.xticks(rotation=30);plt.tight_layout();plt.savefig(P/'slack_ratio_cdf.png',dpi=140);plt.close()
plt.figure();a=[r for r in q() if r['policy'].startswith('SlackRatio')];plt.plot([r['policy'] for r in a],[float(r['offload_fraction']) for r in a],'o-');plt.xticks(rotation=30);plt.tight_layout();plt.savefig(P/'offload_fraction_vs_gamma.png',dpi=140);plt.close()
M=list(csv.DictReader(open(O/'miss_analysis.csv')));from collections import Counter;c=Counter();[c.update({r['reason']:int(r['count'])}) for r in M];plt.figure();plt.bar(c.keys(),c.values());plt.xticks(rotation=25);plt.tight_layout();plt.savefig(P/'miss_reason_breakdown.png',dpi=140);plt.close()
plt.figure();
for fp in ['57344','131072']:
 x=[r for r in R if r['workload']=='closed_loop_session' and r['footprint_bytes_per_token']==fp and r['concurrency']=='32' and r['policy'].startswith('SlackRatio')];plt.plot([r['policy'] for r in x],[float(r['deadline_miss_rate']) for r in x],'o-',label=fp)
plt.legend();plt.xticks(rotation=30);plt.tight_layout();plt.savefig(P/'qwen_vs_largekv_sensitivity.png',dpi=140);plt.close()
