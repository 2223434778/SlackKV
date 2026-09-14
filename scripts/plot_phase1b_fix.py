#!/usr/bin/env python3
"""Regenerate Phase 1B-Fix plots from compact aggregate CSVs."""
import csv
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

OUT = Path("results/phase1b_fix")
PLOTS = OUT / "plots"

def rows():
    with (OUT / "aggregate_metrics.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))

def median(items, key):
    return statistics.median(float(item[key]) for item in items)

def line(data, footprint, workload, metric, name, ylabel):
    plt.figure(figsize=(8,5))
    for policy in ("ASAP-EDF","NaiveJIT","SlackKV-QueueJIT"):
        xs=[]; ys=[]
        for m in (4,8,16,32,64):
            group=[r for r in data if r["footprint_bytes_per_token"]==str(footprint) and r["workload"]==workload and r["concurrency"]==str(m) and r["policy"]==policy]
            xs.append(m); ys.append(median(group,metric))
        plt.plot(xs,ys,marker="o",label=policy)
    plt.xlabel("Multiplexed sessions (M)"); plt.ylabel(ylabel); plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(PLOTS/name,dpi=160); plt.close()

def main():
    PLOTS.mkdir(parents=True,exist_ok=True); data=rows()
    line(data,57344,"closed_loop_session","deadline_miss_rate","qwen_closed_loop_miss_rate.png","Deadline miss rate")
    line(data,57344,"closed_loop_session","stall_ms_p95","qwen_closed_loop_p95_stall.png","P95 resume stall (ms)")
    plt.figure(figsize=(8,5))
    for footprint,label in ((57344,"56 KiB/token"),(131072,"128 KiB/token")):
        for workload,style in (("open_loop_session_audit","--"),("closed_loop_session","-")):
            xs=[]; ys=[]
            for m in (4,8,16,32,64):
                group=[r for r in data if r["footprint_bytes_per_token"]==str(footprint) and r["workload"]==workload and r["concurrency"]==str(m) and r["policy"]=="SlackKV-QueueJIT"]
                xs.append(m); ys.append(median(group,"stall_ms_p95"))
            plt.plot(xs,ys,style,marker="o",label=f"{label}, {workload.split('_')[0]}")
    plt.yscale("log"); plt.xlabel("Multiplexed sessions (M)"); plt.ylabel("QueueJIT P95 stall (ms)"); plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(PLOTS/"open_vs_closed_loop_queuejit.png",dpi=160); plt.close()
    plt.figure(figsize=(8,5))
    for footprint,label in ((57344,"56 KiB/token"),(131072,"128 KiB/token")):
        xs=[]; ys=[]
        for rho in (.25,.5,.75,.9):
            group=[r for r in data if r["footprint_bytes_per_token"]==str(footprint) and r["workload"]=="poisson_tuple" and r["rho"]==str(rho) and r["policy"]=="SlackKV-QueueJIT"]
            xs.append(rho); ys.append(median(group,"deadline_miss_rate"))
        plt.plot(xs,ys,marker="o",label=label)
    plt.xlabel("Offered load rho"); plt.ylabel("QueueJIT miss rate"); plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(PLOTS/"poisson_footprint_sensitivity.png",dpi=160); plt.close()
if __name__=="__main__": main()
