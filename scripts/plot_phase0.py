#!/usr/bin/env python3
"""Aggregate Phase 0 CSVs and regenerate every committed PNG from those CSVs."""
import csv, math, statistics
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt

OUT=Path("results/phase0"); PLOTS=OUT/"plots"
MiB = 1 << 20
def num(v): return float(v) if v not in ("",None) else None
def pctl(a,p):
    a=sorted(a); x=(len(a)-1)*p; lo,hi=int(x),math.ceil(x)
    return a[lo] if lo==hi else a[lo]+(a[hi]-a[lo])*(x-lo)
def read(path):
    with open(path,newline="") as f:return list(csv.DictReader(f))
def main():
 PLOTS.mkdir(parents=True,exist_ok=True); raw=read(OUT/"metrics.csv"); groups=defaultdict(list)
 for r in raw:
  if r["success"]=="true" and r["latency_ms"]: groups[(r["direction"],r["size_bytes"],r["size_mib"],r["p2p_available"])].append(r)
 fields=["backend","direction","size_bytes","size_mib","p2p_available","count","latency_ms_mean","latency_ms_std","latency_ms_p50","latency_ms_p95","latency_ms_min","latency_ms_max","bandwidth_GBps_p50","bandwidth_GBps_mean","bandwidth_GBps_p95_latency_derived"]
 summary=[]
 for (d,b,m,p),rs in sorted(groups.items(),key=lambda x:(x[0][0],int(x[0][1]))):
  lat=[float(r["latency_ms"]) for r in rs]; bw=[float(r["bandwidth_GBps"]) for r in rs]
  summary.append(dict(backend="native_cuda",direction=d,size_bytes=b,size_mib=m,p2p_available=p,count=len(rs),latency_ms_mean=statistics.mean(lat),latency_ms_std=statistics.stdev(lat) if len(lat)>1 else 0,latency_ms_p50=pctl(lat,.5),latency_ms_p95=pctl(lat,.95),latency_ms_min=min(lat),latency_ms_max=max(lat),bandwidth_GBps_p50=pctl(bw,.5),bandwidth_GBps_mean=statistics.mean(bw),bandwidth_GBps_p95_latency_derived=float(b)/(pctl(lat,.95)*1e6)))
 with open(OUT/"summary_metrics.csv","w",newline="") as f: w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader();w.writerows(summary)
 def lines(metric,name,ylabel):
  plt.figure(figsize=(8,5))
  for d in sorted({x["direction"] for x in summary}):
   rs=[x for x in summary if x["direction"]==d]; plt.plot([float(x["size_mib"]) for x in rs],[float(x[metric]) for x in rs],marker="o",label=d)
  plt.xscale("log",base=2);plt.xlabel("Payload size (MiB)");plt.ylabel(ylabel);plt.legend();plt.grid(True,which="both",alpha=.3);plt.tight_layout();plt.savefig(PLOTS/name,dpi=160);plt.close()
 lines("bandwidth_GBps_p50","transfer_bandwidth_vs_size.png","P50 bandwidth (GB/s, decimal)")
 lines("latency_ms_p50","transfer_latency_vs_size.png","P50 latency (ms)")
 lookup={(x["direction"],x["size_bytes"]):float(x["bandwidth_GBps_p50"]) for x in summary}; ratios=[]
 for peer,cpu,label in [("gpu0_to_gpu1","gpu0_to_pinned_cpu","GPU0 outgoing"),("gpu1_to_gpu0","pinned_cpu_to_gpu0","GPU0 incoming"),("gpu1_to_gpu0","gpu1_to_pinned_cpu","GPU1 outgoing"),("gpu0_to_gpu1","pinned_cpu_to_gpu1","GPU1 incoming")]:
  for s in sorted({int(x["size_bytes"]) for x in summary}):
   if (peer,str(s)) in lookup and (cpu,str(s)) in lookup: ratios.append((label,s/MiB,lookup[(peer,str(s))]/lookup[(cpu,str(s))]))
 plt.figure(figsize=(9,5));
 for label in sorted({x[0] for x in ratios}):
  rs=[x for x in ratios if x[0]==label];plt.plot([x[1] for x in rs],[x[2] for x in rs],marker="o",label=label)
 plt.axhline(1,color="black",lw=.8);plt.xscale("log",base=2);plt.xlabel("Payload size (MiB)");plt.ylabel("Peer / pinned-CPU P50 bandwidth ratio");plt.legend();plt.grid(True,which="both",alpha=.3);plt.tight_layout();plt.savefig(PLOTS/"peer_vs_cpu_bandwidth_ratio.png",dpi=160);plt.close()
 over=read(OUT/"overlap_summary_metrics.csv")
 for metric,name,ylabel in [("compute_slowdown_p50","compute_slowdown.png","Compute slowdown (fraction)"),("overlap_efficiency_p50","overlap_efficiency.png","Overlap efficiency")]:
  plt.figure(figsize=(9,5))
  for kind in sorted({x["transfer_type"] for x in over}):
   rs=[x for x in over if x["transfer_type"]==kind and x["workload"]=="gemm_fp16"];plt.plot([float(x["size_mib"]) for x in rs],[float(x[metric]) for x in rs],marker="o",label=kind)
  plt.xscale("log",base=2);plt.xlabel("Transfer payload size (MiB)");plt.ylabel(ylabel);plt.legend();plt.grid(True,which="both",alpha=.3);plt.tight_layout();plt.savefig(PLOTS/name,dpi=160);plt.close()
if __name__=="__main__":main()
