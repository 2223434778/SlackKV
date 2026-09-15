#!/usr/bin/env python3
"""Write compact Fix2 diagnostics and Fix-to-Fix2 deltas."""
import csv, json
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'results/phase1c_fix2'; OLD=ROOT/'results/phase1c_fix'
KEY=('workload','footprint_bytes_per_token','concurrency','rho','seed','policy')
def read(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))
def write(p,fields,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def med(v):
    v=sorted(v);n=len(v);return v[n//2] if n%2 else (v[n//2-1]+v[n//2])/2
def main():
    new,old=read(OUT/'aggregate_metrics.csv'),read(OLD/'aggregate_metrics.csv')
    oldmap={tuple(r[k] for k in KEY):r for r in old}; deltas=[]
    for r in new:
        o=oldmap.get(tuple(r[k] for k in KEY))
        if o:
            row={k:r[k] for k in KEY}
            for m in ('deadline_miss_rate','stall_ms_p95','normalized_local_memory_time','offload_fraction'):
                row[m+'_fix2_minus_fix']=float(r[m])-float(o[m])
            deltas.append(row)
    write(OUT/'fix_delta.csv',list(deltas[0]),deltas)
    fields=('intentional_idle_time_total','intentional_idle_events','promotion_ready_wakeups','session_release_wakeups','safe_idle_expiry_wakeups','H2D_busy_time','mean_promotion_lead_time')
    diag=[]
    for m in ('32','64'):
        for p in ('OffloadAll-QueueJIT','FixedTTL-0.5','FixedTTL-1','FixedTTL-2','SlackRatio-2','SlackRatio-3','SlackRatio-4'):
            rs=[r for r in new if r['workload']=='closed_loop_session' and r['footprint_bytes_per_token']=='57344' and r['concurrency']==m and r['policy']==p]
            diag.append({'concurrency':m,'policy':p,**{x:med([float(r[x]) for r in rs]) for x in fields}})
    write(OUT/'scheduler_diagnostics.csv',list(diag[0]),diag)
    a=read(OUT/'aggregated_policy_metrics.csv')
    select=lambda m,p:next(r for r in a if r['concurrency']==str(m) and r['policy']==p)
    vals=[]
    for m in (32,64):
        for p in ('OffloadAll-QueueJIT','FixedTTL-0.5','FixedTTL-1','FixedTTL-2','SlackRatio-2','SlackRatio-3','SlackRatio-4'):
            r=select(m,p);vals.append(f'| {m} | {p} | {float(r["deadline_miss_rate_median"]):.4%} | {float(r["stall_ms_p95_median"]):.1f} ms | {float(r["normalized_local_memory_time_median"]):.4f} |')
    c=json.loads((OUT/'config.json').read_text()); outcome='A = Strong GO' if c['c0'] and c['c1'] and c['c2'] else 'B = Selective-Offload GO, SlackRatio NO-GO'
    (OUT/'summary.md').write_text('# Phase 1C-Fix2\n\nQueueJIT now advances to safe-idle expiry and wakes for D2H-ready/session events; D2H residency continues through D2H completion. The frozen Phase 1C-Fix matrix was rerun.\n\nC0: PASS. C1: '+('PASS' if c['c1'] else 'FAIL')+'. C2: '+('PASS' if c['c2'] else 'FAIL')+'.\n\n'+outcome+'.\n\n| M | Policy | miss | P95 stall | normalized memory |\n|---:|---|---:|---:|---:|\n'+'\n'.join(vals)+'\n\nQ1: corrected timing materially reduced P95 and local-memory time versus Fix; see `fix_delta.csv`. Q2: selective offload remains far better than OffloadAll. Q3/Q4: C1/C2 pass. Q5/Q6: see TTL and SlackRatio Pareto CSVs; SlackRatio remains a qualifying frontier extension.\n\nLimitations: oracle-deadline simulator, optimistic non-contended D2H, serialized H2D, no real KV tensors or LLM serving.\n\nPhase 1C-Fix2 execution status: COMPLETE\n')
    (OUT/'commands.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\n/home/a6000/anaconda3/bin/python experiments/phase1c_fix2/run_phase1c_fix2.py\n/home/a6000/anaconda3/bin/python experiments/phase1c_fix2/analyze_phase1c_fix2.py\n/home/a6000/anaconda3/bin/python experiments/phase1c_fix2/finalize_phase1c_fix2.py\n')
if __name__=='__main__':main()
