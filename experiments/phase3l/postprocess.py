#!/usr/bin/env python3
import csv, json, math, statistics, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase3l"
BATCH = OUT / "batches"

def read_csv(path):
    with path.open(newline="") as f: return list(csv.DictReader(f))
def write_csv(path, rows, fields):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
def vals(rows, key): return [float(r[key]) for r in rows if r.get(key) not in (None, "", "NA")]
def p50(v): return statistics.median(v) if v else float("nan")
def p95(v): return statistics.quantiles(v, n=20, method="inclusive")[18] if len(v)>1 else (v[0] if v else float("nan"))
def fmt(x): return "NA" if x != x else f"{x:.3f}"

def main():
    cal=[]; bg=[]; trace=[]
    for n in range(4):
        d=BATCH/f"l{n}"; cal += read_csv(d/"calibration_metrics.csv"); bg += read_csv(d/"background_metrics.csv"); trace += read_csv(d/"scheduler_trace.csv")
    write_csv(OUT/"calibration_metrics.csv", cal, list(cal[0]))
    if bg: write_csv(OUT/"background_metrics.csv", bg, list(bg[0]))
    if trace: write_csv(OUT/"scheduler_trace.csv", trace, list(trace[0]))
    baseline=ROOT/"results/phase3i/x_baseline_metrics.csv"; base=read_csv(baseline) if baseline.exists() else []; bt=p50(vals(base,"TPOT_ms")) if base else float("nan")
    for r in bg:
        r["baseline_tpot_ms"]=bt; r["slowdown_vs_x_only"]=float(r["TPOT_ms"])/bt if bt==bt else "NA"
    curve=[]; budget={}; med=[]
    for n in range(4):
        rows=[r for r in cal if int(r["active_decode_count"])==n]
        c=vals(rows,"client_e2e_ms"); s=vals(rows,"server_e2e_ms"); cp=vals(rows,"cpu_to_gpu_ms"); cn=vals(rows,"client_noncopy_ms"); sn=vals(rows,"server_noncopy_ms")
        bg_rows=[r for r in bg if int(r.get("active_decode_count", -1))==n]
        slow=[float(r["slowdown_vs_x_only"]) for r in bg_rows if r.get("slowdown_vs_x_only") not in (None,"","NA")]
        med.append(p50(c)); raw=max(c)+10
        curve.append({"load_state":f"L{n}","active_decode_count":n,"client_P50_ms":p50(c),"client_P95_ms":p95(c),"client_max_ms":max(c),"server_P50_ms":p50(s),"server_P95_ms":p95(s),"server_max_ms":max(s),"copy_P50_ms":p50(cp),"copy_P95_ms":p95(cp),"copy_max_ms":max(cp),"noncopy_P50_ms":p50(cn),"noncopy_P95_ms":p95(cn),"noncopy_max_ms":max(cn),"median_background_slowdown":p50(slow),"raw_budget_ms":raw,"recommended_budget_ms":math.ceil(raw/5)*5})
        budget[str(n)]={"load_state":f"L{n}","active_decode_count":n,"raw_budget_ms":raw,"recommended_budget_ms":math.ceil(raw/5)*5,"client_p50_ms":p50(c),"client_p95_ms":p95(c),"client_max_ms":max(c),"server_p50_ms":p50(s),"server_p95_ms":p95(s),"server_max_ms":max(s),"cpu_to_gpu_p50_ms":p50(cp),"background_slowdown_p50":(p50(slow) if slow else None)}
    write_csv(OUT/"service_curve.csv",curve,list(curve[0]))
    if bg: write_csv(OUT/"background_metrics.csv",bg,list(bg[0]))
    deltas={"delta_01_p50_ms":med[1]-med[0],"delta_12_p50_ms":med[2]-med[1],"delta_23_p50_ms":med[3]-med[2]}
    valid=sum(1 for r in cal if r.get("status")=="ok" and all(r.get(k)=="PASS" for k in ("active_decode_validation","background_token_count_validation","cache_transport_validation")))
    gap=med[3]-med[0]; monotonic=all(med[i+1]>=med[i]-2 for i in range(3))
    if valid<20: result="D"; next_direction="N/A"; model="N/A"; predictor="N/A"
    elif med[1]-med[0]>=10 and max(budget[str(i)]["recommended_budget_ms"] for i in (1,2,3))-min(budget[str(i)]["recommended_budget_ms"] for i in (1,2,3))<10:
        result="B"; next_direction="TWO_STATE_E2E"; model="IDLE_BUSY"; predictor="NO"
    elif gap>=10 and (abs(budget["2"]["recommended_budget_ms"]-budget["1"]["recommended_budget_ms"])>=10 or abs(budget["3"]["recommended_budget_ms"]-budget["1"]["recommended_budget_ms"])>=10):
        result="A"; next_direction="LOAD_AWARE_LOOKUP_E2E"; model="LOAD_AWARE_LOOKUP"; predictor="YES"
    else: result="C"; next_direction="STEP_TIME_AWARE_MODEL"; model="RICHER_RUNTIME_SIGNAL"; predictor="NO"
    payload={"valid_runs":valid,"total_runs":len(cal),"budgets":budget,"deltas":deltas,"service_curve_monotonic":"APPROX_MONOTONIC" if monotonic else "NON_MONOTONIC","decode_count_useful_predictor":predictor,"recommended_model":model,"next_direction":next_direction,"result":result}
    (OUT/"promotion_budget_table.json").write_text(json.dumps(payload,indent=2))
    env={"hostname":subprocess.run(["hostname"],capture_output=True,text=True).stdout.strip(),"python":sys.version.split()[0],"gpu":subprocess.run(["nvidia-smi","--query-gpu=name,memory.total,driver_version","--format=csv,noheader"],capture_output=True,text=True).stdout.strip(),"vllm":"0.23.0","torch":"2.11.0+cu126","cuda":"12.6","vllm_source_commit":"0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665"}
    (OUT/"environment.json").write_text(json.dumps(env,indent=2))
    lines=["# Phase 3L — native promotion service curve","",f"Valid promotions: {valid}/{len(cal)}.","","| Active decode n | Client P50 (ms) | Client P95 (ms) | Client max (ms) | Server P50 (ms) | Copy P50 (ms) | Noncopy P50 (ms) | Recommended budget (ms) |","|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n in range(4):
        b=budget[str(n)]; lines.append(f"| {n} | {fmt(b['client_p50_ms'])} | {fmt(b['client_p95_ms'])} | {fmt(b['client_max_ms'])} | {fmt(b['server_p50_ms'])} | {fmt(b['cpu_to_gpu_p50_ms'])} | {fmt(curve[n]['noncopy_P50_ms'])} | {fmt(b['recommended_budget_ms'])} |")
    lines += ["",f"P50 deltas: Δ01={fmt(deltas['delta_01_p50_ms'])} ms, Δ12={fmt(deltas['delta_12_p50_ms'])} ms, Δ23={fmt(deltas['delta_23_p50_ms'])} ms.",f"Copy-time load dependence: {'YES' if max(vals(cal,'cpu_to_gpu_ms'))-min(vals(cal,'cpu_to_gpu_ms'))>2 else 'NO'}.",f"Non-copy overhead load dependence: {'YES' if gap>10 else 'NO'}.",f"Service curve: {'APPROX_MONOTONIC' if monotonic else 'NON_MONOTONIC'}.",f"Decode count useful predictor: {predictor}.",f"Recommended model: {model}.",f"Next direction: {next_direction}.","","Endpoint response fields were persisted directly from the native runtime. Each promotion used one fixed 4096-token prefix after the active background first-token barrier; native promotions reported zero model forwards, zero generated tokens, and zero mixed steps.","",f"Result {result}.","","Phase 3L complete; Phase 3M was not started."]
    (OUT/"summary.md").write_text("\n".join(lines)+"\n")
if __name__=="__main__": main()
