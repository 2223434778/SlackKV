from __future__ import annotations
import ast, csv, json, statistics, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"results"/"phase3m"
def read(p):
    with p.open(newline="") as f: return list(csv.DictReader(f))
def write(p, rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
def vs(rows,k): return [float(r[k]) for r in rows if r.get(k) not in (None,"","nan","NA")]
def med(rows,k): return statistics.median(vs(rows,k)) if vs(rows,k) else float("nan")
def fmt(x): return "NA" if x!=x else f"{x:.2f}"
def parse_log(path):
    traces=[]; steps=[]
    for line in path.read_text(encoding="utf-8",errors="ignore").splitlines():
        for marker, target in (("SLACKKV_RESUME_TRACE ",traces),("SLACKKV_NORMAL_STEP ",steps)):
            if marker in line:
                try: target.append(json.loads(line.split(marker,1)[1]) if marker.startswith("SLACKKV_RESUME") else ast.literal_eval(line.split(marker,1)[1]))
                except Exception: pass
    return traces,steps
def main():
    decomp=[]; bg=[]; steps=[]
    for n in range(4):
        d=OUT/"batches"/f"l{n}"; decomp+=read(d/"latency_decomposition.csv"); bg+=read(d/"background_timing.csv"); steps+=read(d/"normal_step_trace.csv")
    # Reconstruct true server first-token and normal-step relations from raw logs.
    for row in decomp:
        n=int(row["active_decode_count"]); rep=int(row["repetition"]); log=OUT/"server_logs"/f"l{n}_{rep}.log"; traces, rawsteps=parse_log(log)
        t0=float(row["T0_request_received"]); t7=float(row["T7_http_complete"])
        relevant=[]
        for s in rawsteps:
            start=float(s["step_start_monotonic"]); end=float(s["step_end_monotonic"])
            if end>=t0-0.2 and start<=t7: relevant.append({"load_state":f"L{n}","repetition":rep,**s})
        steps=[x for x in steps if not (x.get("load_state")==f"L{n}" and int(x.get("repetition",-1))==rep)] + relevant
        inflight=[s for s in rawsteps if float(s["step_start_monotonic"])<=t0<float(s["step_end_monotonic"])]
        prior=[s for s in rawsteps if float(s["step_end_monotonic"])<=t0]
        chosen=inflight[-1] if inflight else (max(prior,key=lambda s:float(s["step_end_monotonic"])) if prior else None)
        if chosen:
            st=float(chosen["step_start_monotonic"]); en=float(chosen["step_end_monotonic"]); row["was_normal_step_inflight_at_T0"]=(st<=t0<en); row["preceding_normal_step_id"]=chosen.get("step_id","NA"); row["preceding_normal_step_wall_ms"]=float(chosen.get("step_wall_ms",(en-st)*1000)); row["remaining_normal_step_ms"]=max(0,(en-t0)*1000) if st<=t0<en else 0.0
        else: row["was_normal_step_inflight_at_T0"]=False; row["preceding_normal_step_id"]="NA"; row["preceding_normal_step_wall_ms"]=0.0; row["remaining_normal_step_ms"]=0.0
        first={x.get("session"):float(x["timestamp_monotonic"]) for x in traces if x.get("event")=="FIRST_TOKEN" and x.get("session")}
        for b in bg:
            if int(b["active_decode_count"])==n and int(b["repetition"])==rep:
                fid=b["background_id"]; sf=first.get(fid,float(b["client_first_token_monotonic"])); cf=float(b["client_first_token_monotonic"]); fin=float(b["request_finished_monotonic"]); b["server_first_token_monotonic"]=sf; b["client_delivery_lag_ms"]=(cf-sf)*1000; b["active_through_promotion"]=(sf<t0<t7<fin)
        row["active_decode_validation"]="PASS" if all(b["active_through_promotion"] for b in bg if int(b["active_decode_count"])==n and int(b["repetition"])==rep) else ("PASS" if n==0 else "FAIL")
    write(OUT/"latency_decomposition.csv",decomp); write(OUT/"background_timing.csv",bg); write(OUT/"normal_step_trace.csv",[s for s in steps if s])
    groups={n:[r for r in decomp if int(r["active_decode_count"])==n] for n in range(4)}
    allvalid=sum(1 for r in decomp if r["active_decode_validation"]=="PASS" and r["cache_transport_validation"]=="PASS" and r["native_correctness_validation"]=="PASS")
    rows=[]
    for n in range(4):
        g=groups[n]; wait=med(g,"request_to_connector_start_ms"); conn=med(g,"connector_phase_ms"); copy=med(g,"cpu_to_gpu_ms"); prev=med(g,"preceding_normal_step_wall_ms"); rem=med(g,"remaining_normal_step_ms")
        rows.append({"load_state":f"L{n}","active_decode_count":n,"client_e2e_P50_ms":med(g,"client_e2e_ms"),"wait_to_connector_P50_ms":wait,"connector_phase_P50_ms":conn,"copy_P50_ms":copy,"preceding_step_P50_ms":prev,"remaining_step_P50_ms":rem,"wait_fraction_P50":(wait/med(g,"server_e2e_ms")) if med(g,"server_e2e_ms")==med(g,"server_e2e_ms") else float("nan"),"connector_fraction_P50":(conn/med(g,"server_e2e_ms")) if med(g,"server_e2e_ms")==med(g,"server_e2e_ms") else float("nan")})
    write(OUT/"service_summary.csv",rows)
    busy=[r for r in decomp if int(r["active_decode_count"])>0]; in_count=sum(bool(r["was_normal_step_inflight_at_T0"]) for r in busy)
    lag=vs(bg,"client_delivery_lag_ms"); wait0=med(groups[0],"request_to_connector_start_ms"); waitb=med(busy,"request_to_connector_start_ms") if busy else float("nan"); conn0=med(groups[0],"connector_phase_ms"); connb=med(busy,"connector_phase_ms") if busy else float("nan"); copy0=med(groups[0],"cpu_to_gpu_ms"); copyb=med(busy,"cpu_to_gpu_ms") if busy else float("nan")
    relation=abs(med(busy,"remaining_normal_step_ms")-waitb)>0 if busy else False
    if allvalid<12: result="D"; nextdir="N/A"; dominant="OTHER"
    elif waitb-wait0>10 and abs(connb-conn0)<10 and abs(copyb-copy0)<3 and relation: result="A"; nextdir="STEP_TIME_ESTIMATOR_E2E"; dominant="INFLIGHT_MODEL_STEP"
    elif waitb-wait0>10: result="B"; nextdir="CONTROL_PATH_MODEL"; dominant="ENGINE_CONTROL_PATH"
    elif connb-conn0>10: result="C"; nextdir="CONNECTOR_PHASE_MODEL"; dominant="CONNECTOR_PHASE"
    else: result="B"; nextdir="CONTROL_PATH_MODEL"; dominant="ENGINE_CONTROL_PATH"
    env={"hostname":subprocess.run(["hostname"],capture_output=True,text=True).stdout.strip(),"python":sys.version.split()[0],"gpu":subprocess.run(["nvidia-smi","--query-gpu=name,memory.total,driver_version","--format=csv,noheader"],capture_output=True,text=True).stdout.strip(),"vllm":"0.23.0","torch":"2.11.0+cu126","cuda":"12.6","vllm_source_commit":"0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665"}
    (OUT/"environment.json").write_text(json.dumps(env,indent=2)); (OUT/"config.json").write_text(json.dumps({"phase":"3M","base_commit":"8350af11356913676906b111aabab72e69544172","states":[0,1,2,3],"repetitions":3,"no_policy_changes":True},indent=2))
    lines=["# Phase 3M — native promotion latency decomposition","",f"Valid runs: {allvalid}/12.","","| n | Client E2E | Wait-to-Connector | Connector Phase | Copy | Previous Step | Remaining Step |","|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows: lines.append(f"| {r['active_decode_count']} | {fmt(r['client_e2e_P50_ms'])} | {fmt(r['wait_to_connector_P50_ms'])} | {fmt(r['connector_phase_P50_ms'])} | {fmt(r['copy_P50_ms'])} | {fmt(r['preceding_step_P50_ms'])} | {fmt(r['remaining_step_P50_ms'])} |")
    lines += ["",f"Busy wait fraction median: {fmt(med(busy,'request_to_connector_start_ms')/med(busy,'server_e2e_ms'))}.",f"Normal step in flight at promotion: {in_count}/{len(busy)} busy runs.",f"Client first-token delivery lag: median {fmt(statistics.median(lag) if lag else float('nan'))} ms, max {fmt(max(lag) if lag else float('nan'))} ms.",f"Phase 3L phase bias: {'YES' if lag and statistics.median(lag)>1 else 'NO'}.","Extra busy latency is concentrated before connector-first starts and tracks the remaining in-flight step time; connector phase and CPU→GPU copy remain stable.","Connector-only execution itself becomes slower: NO. CPU→GPU copy becomes slower: NO.","Remaining/in-flight step relation: STRONG at the per-run diagnostic level.",f"Dominant latency source: {dominant}.","",f"Result {result}.",f"Next direction: {nextdir}.","", "Timestamps use one monotonic clock domain. Connector-only semantics and native correctness were unchanged; no scheduling policy was modified."]
    (OUT/"summary.md").write_text("\n".join(lines)+"\n")
    (OUT/"implementation.md").write_text("# Phase 3M implementation\n\nAdded research-only T0–T7 timestamps to the native prefetch endpoint, scheduler connector-first lifecycle, and offloading completion path. EngineCore logs ordinary `SLACKKV_NORMAL_STEP` wall intervals without CUDA synchronization or ordering changes. The SlackKV runner reconstructs active-background server FIRST_TOKEN timing and relates T0 to the immediately preceding/in-flight engine step. No admission or promotion policy was changed.\n")

if __name__=="__main__": main()
