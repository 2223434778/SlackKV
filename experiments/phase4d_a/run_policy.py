from __future__ import annotations

import csv, json, math, statistics, threading, time, re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from experiments.phase2d.helpers import source_tokens_delta
from experiments.phase3f.native_client import native_prefetch
from experiments.phase3f.scheduler import latest_feasible_start, merge_blackouts
from experiments.phase4d_a import runtime
from experiments.phase4d_a.experiment import (
    OUT, ROOT, POLICIES, build_resume_prompts, first_boundary_jobs, materialize_and_evict,
    source_snapshot, write_csv, write_json, x_prompt,
)
from experiments.phase4d_a.helpers import (ReservationLedger, duplicate_h2d_bytes, frozen_service_budget,
    prefix_memory_time_mib_s, source_conserved)

def wait_until(target_s:float)->None:
    while True:
        rem=target_s-time.monotonic()
        if rem<=0:return
        time.sleep(min(.002,rem))

def parse_trace(path:Path,t0_ns:int)->tuple[list[dict[str,Any]],list[dict[str,Any]]]:
    trace=[]; leases=[]
    if not path.exists(): return trace,leases
    for line in path.read_text(encoding="utf-8",errors="replace").splitlines():
        if "SLACKKV_RESUME_TRACE " in line:
            try:
                e=json.loads(line.split("SLACKKV_RESUME_TRACE ",1)[1])
                if "timestamp_monotonic" in e:
                    raw=float(e["timestamp_monotonic"]); ms=(raw-t0_ns/1e9)*1000
                    trace.append({"event":e.get("event"),"session":e.get("session"),"request_id":e.get("request_id"),
                        "step_id":e.get("step_id"),"timestamp_monotonic_raw":raw,"timestamp_ms":ms,
                        "request_status":e.get("request_status"),"num_running":e.get("num_running"),
                        "num_waiting":e.get("num_waiting"),"num_slackkv_pre_first_token":e.get("num_slackkv_pre_first_token"),
                        "gate_owner_request_id":e.get("gate_owner_request_id")})
            except (ValueError,TypeError): pass
        if "SLACKKV_LEASE_TRACE " in line:
            try:
                e=json.loads(line.split("SLACKKV_LEASE_TRACE ",1)[1])
                if "timestamp_monotonic" in e:
                    e["timestamp_ms"]=(float(e["timestamp_monotonic"])-t0_ns/1e9)*1000
                    leases.append(e)
            except (ValueError,TypeError): pass
    return trace,leases

def capacity_from_server(log:Path,jobs:list[dict[str,Any]])->dict[str,Any]:
    blocks=__import__("experiments.phase4d_a.experiment",fromlist=["gpu_capacity_blocks"]).gpu_capacity_blocks(log,16)
    from experiments.phase4d_a.helpers import reservation_budget,maximum_simultaneous_jobs
    budget=reservation_budget(blocks,jobs)
    return {"total_gpu_kv_blocks":blocks,"reservation_budget_blocks":budget,
        "reservation_budget_MiB":budget*16*28672/(1024**2),
        "max_simultaneous_frozen_jobs":maximum_simultaneous_jobs(jobs,budget)}

def stream_resume(session:str,entry:dict[str,Any],t0_s:float,deadline_ms:float,gate:bool,states:dict[str,Any],lock:threading.Lock)->dict[str,Any]:
    wait_until(t0_s+deadline_ms/1000)
    sent=time.monotonic_ns()
    episode=entry["job"]["episode_id"]
    with lock: states[episode]["resume_started_ns"]=sent
    params={"slackkv_trace_resume":True,"slackkv_session":session,
            "slackkv_first_token_gate":True,"slackkv_first_token_gate_limit":2}
    if states[episode].get("leased"): params["max_offload_tokens"]=0
    def first_seen(ns):
        with lock: states[episode]["first_token_ns"]=ns
    out=runtime.send(entry["resume_prompt"],entry["expected_output_count"],stream=True,params=params,on_first=first_seen)
    return {"episode_id":episode,"request_sent_ns":sent,"first_token_ns":out["first_ns"],
        "finish_ns":out["done_ns"],"generated_token_count":len(out["tokens"]),
        "output_ids":out["tokens"],"output_sha256":__import__("experiments.phase4c.synthetic_tokens",fromlist=["token_hash"]).token_hash(out["tokens"]),
        "expected_output_sha256":entry["expected_output_sha256"],"deadline_ms":deadline_ms}

def start_background(t0_callback=None):
    result={}
    ready=threading.Event()
    def cb(ns):
        result["first_ns"]=ns; ready.set()
        if t0_callback: t0_callback(ns)
    pool=ThreadPoolExecutor(max_workers=1)
    future=pool.submit(runtime.send,x_prompt(),256,stream=True,params={"max_offload_tokens":0},on_first=cb)
    if not ready.wait(120): raise TimeoutError("background X produced no first token")
    return pool,future,result

def run_promotions(policy:str,jobs:list[dict[str,Any]],data:dict[str,dict[str,Any]],states:dict[str,Any],
                   lock:threading.Lock,t0_s:float,p_safe:float,budget:int|None,log:Path)->tuple[dict[str,Any],list[dict[str,Any]]]:
    if policy=="OnDemand-K2": return {},[]
    results={}; decisions=[]
    pending=set(data)
    ledger=ReservationLedger(budget) if policy=="SlackKV-Oracle-ReserveLease-K2" else None
    deadlines={e:float(data[e]["job"]["deadline_ms"]) for e in data}
    blackouts=merge_blackouts(deadlines,110.0)
    pool=ThreadPoolExecutor(max_workers=8)
    while pending:
        now=(time.monotonic()-t0_s)*1000
        if ledger is not None and ledger.active:
            log_text=log.read_text(encoding="utf-8",errors="replace")
            for key in list(ledger.active):
                if states[key].get("first_token_ns") and '"event":"LEASE_RELEASED"' in log_text and f'"lease_key":"{key}"' in log_text:
                    ledger.release(key)
                    decisions.append({"episode_id":key,"decision":"RUNTIME_RELEASE_AFTER_PHYSICAL_LEASE"})
        if policy=="NaiveJIT-Async-K2":
            eid=min(pending,key=lambda e:(deadlines[e],e))
            target=deadlines[eid]-p_safe
        else:
            target=latest_feasible_start(now,[(e,deadlines[e]) for e in pending],blackouts,p_safe)
            eid=min(pending,key=lambda e:(deadlines[e],e))
        if target>now+0.5:
            time.sleep(min(.01,(target-now)/1000)); continue
        with lock:
            stale=states[eid].get("resume_started_ns") is not None
        if stale:
            decisions.append({"episode_id":eid,"decision":"STALE_PROMOTION_CANCELLED","target_start_ms":target})
            pending.remove(eid); continue
        reserved=False
        if ledger is not None:
            reserved=ledger.reserve(eid,int(data[eid]["job"]["required_blocks"]))
            if not reserved:
                decisions.append({"episode_id":eid,"decision":"RESERVATION_SKIP","target_start_ms":target,
                    "reserved_blocks":ledger.reserved_blocks,"budget_blocks":ledger.budget_blocks})
                states[eid]["promoted"]=False; pending.remove(eid); continue
        lease=policy=="SlackKV-Oracle-ReserveLease-K2"
        key=data[eid]["job"]["lease_key"]
        states[eid]["leased"]=lease
        future=pool.submit(native_prefetch,data[eid]["prefix"],timeout_s=120.0,**({"lease_key":key} if lease else {}))
        states[eid]["promoted"]=True
        results[eid]={"future":future,"target_start_ms":target,"lease_key":key if lease else "",
                      "required_blocks":data[eid]["job"]["required_blocks"],"ledger":ledger}
        decisions.append({"episode_id":eid,"decision":"RESERVATION_PROMOTE" if lease else "PROMOTE",
            "target_start_ms":target,"required_blocks":data[eid]["job"]["required_blocks"],
            "reserved_blocks":ledger.reserved_blocks if ledger else "NA","budget_blocks":budget if budget else "NA"})
        pending.remove(eid)
    for eid,item in results.items():
        item["result"]=item["future"].result(timeout=900)
        item["timestamps"]=item["result"].get("phase3m_timestamps") or {}
        item["completion_ns"]=int(item["result"]["completed_monotonic_s"]*1e9)
    pool.shutdown(wait=True)
    return results,decisions



def run_one(policy:str,repetition:int,measured:bool,jobs:list[dict[str,Any]],data:dict[str,dict[str,Any]],
            p_safe:float,out:Path=OUT)->dict[str,Any]:
    label=f"{'measured' if measured else 'integration'}_{policy.lower().replace('-','_')}_{repetition}"
    proc,log=runtime.start_server(label,connector=True,out=out)
    resume_rows=[]; native_rows=[]; decisions=[]; lease_rows=[]; trace=[]
    try:
        _,block=runtime.wait_ready(proc)
        if block!=16: raise RuntimeError("EXPERIMENT_INVALID_ENVIRONMENT:block_size")
        cap=capacity_from_server(log,jobs)
        materialize_and_evict(proc,log,label,jobs,data,out)
        source_before=source_snapshot()
        states={eid:{"resume_started_ns":None,"promoted":False} for eid in data}
        lock=threading.Lock()
        xpool,xfuture,xmeta=start_background()
        t0_ns=int(xmeta["first_ns"]); t0_s=t0_ns/1e9
        pool=ThreadPoolExecutor(max_workers=8)
        futures={}
        for eid,entry in data.items():
            key=entry["job"]["lease_key"]
            futures[eid]=pool.submit(stream_resume,key,entry,t0_s,float(entry["job"]["deadline_ms"]),True,states,lock)
        promo,decisions=run_promotions(policy,jobs,data,states,lock,t0_s,p_safe,
            cap["reservation_budget_blocks"] if policy=="SlackKV-Oracle-ReserveLease-K2" else None,log)
        for eid,future in futures.items():
            result=future.result(timeout=1800)
            result["episode_id"]=eid
            result["resume_osl"]=data[eid]["expected_output_count"]
            result["expected_output_sha256"]=data[eid]["expected_output_sha256"]
            result["output_identity"]=(result["output_sha256"]==result["expected_output_sha256"])
            result["effective_resume_latency_ms"]=max(0,(result["first_token_ns"]/1e9-t0_s)*1000-result["deadline_ms"])
            resume_rows.append(result)
        xresult=xfuture.result(timeout=900); xpool.shutdown(wait=True)
        source_after=source_snapshot()
        local=source_tokens_delta(source_before,source_after,"local_cache_hit")
        external=source_tokens_delta(source_before,source_after,"external_kv_transfer")
        runtime.stop_server(proc,log,out,label)
        trace,lease_rows=parse_trace(log,t0_ns)
        log.unlink(missing_ok=True)
        trace_by={e:{} for e in data}
        for event in trace:
            session=str(event.get("session") or "")
            eid=session.split(":turn:",1)[0]
            if eid in trace_by: trace_by[eid].setdefault(event["event"],event)
        native_out=[]
        for eid,item in promo.items():
            result=item["result"]; ts=item["timestamps"]
            sent_ns=int(result["sent_monotonic_s"]*1e9)
            t0_native=float(ts.get("T0",result["sent_monotonic_s"]))
            t6=float(ts["T6"]) if "T6" in ts else None
            t7=float(ts["T7"]) if "T7" in ts else None
            arrival=trace_by[eid].get("RESUME_ARRIVED",{}).get("timestamp_monotonic_raw")
            remote=trace_by[eid].get("WAITING_FOR_REMOTE_KVS") is not None
            status=("PROMOTION_INTERNAL_READY_MISS" if t6 is None or arrival is None or t6>=arrival
                    else "PROMOTION_REMOTE_WAIT" if remote else "PROMOTION_USEFUL")
            native_out.append({"policy":policy,"repetition":repetition,"episode_id":eid,
                "target_start_ms":item["target_start_ms"],"client_issue_ms":(sent_ns-t0_ns)/1e6,
                "T0_ms":(t0_native-t0_s)*1000,"T6_ms":(t6-t0_s)*1000 if t6 else "NA",
                "T7_ms":(t7-t0_s)*1000 if t7 else "NA",
                "issue_to_T6_ms":(t6-result["sent_monotonic_s"])*1000 if t6 else "NA",
                "arrival_ms":(arrival-t0_s)*1000 if arrival else "NA",
                "arrival_minus_T6_ms":(arrival-t6)*1000 if arrival and t6 else "NA",
                "required_tokens":data[eid]["job"]["required_prefix_tokens"],
                "external_tokens_loaded":result.get("external_tokens_loaded"),
                "transfer_bytes":result.get("transfer_bytes"),"status":result.get("status"),
                "client_e2e_ms":result.get("client_e2e_ms"),
                "cpu_to_gpu_transfer_ms":result.get("cpu_to_gpu_transfer_ms"),
                "connector_only_step_count":result.get("connector_only_step_count"),
                "normal_requests_delayed":result.get("normal_requests_delayed"),
                "model_forward_count":result.get("model_forward_count"),"generated_token_count":result.get("generated_token_count"),
                "mixed_prefetch_model_step_count":result.get("mixed_prefetch_model_step_count"),
                "promotion_classification":status,"lease_key":item["lease_key"]})
        native_rows=native_out
        for row in resume_rows:
            eid=row["episode_id"]; ev=trace_by[eid]
            arrival=ev.get("RESUME_ARRIVED",{}); admit=ev.get("RESUME_ADMITTED",{}); first=ev.get("FIRST_TOKEN",{}); finish=ev.get("REQUEST_FINISHED",{})
            row.update({"scheduler_arrival_ms":arrival.get("timestamp_ms","NA"),
                "scheduler_admission_ms":admit.get("timestamp_ms","NA"),
                "first_token_ms":first.get("timestamp_ms","NA"),"request_finished_ms":finish.get("timestamp_ms","NA"),
                "gate_blocked_count":sum(x.get("event")=="GATE_BLOCKED" and str(x.get("session","")).startswith(eid+":") for x in trace),
                "promotion_completion_ms":next((x["T6_ms"] for x in native_rows if x["episode_id"]==eid),"NA")})
        finishes=[int(r["finish_ns"]) for r in resume_rows]; firsts=[int(r["first_token_ns"]) for r in resume_rows]
        x_first=int(xresult["first_ns"]); x_finish=int(xresult["done_ns"])
        x_active=max(firsts)<x_finish
        max_pre=max([int(e.get("num_slackkv_pre_first_token") or 0) for e in trace]+[0])
        blocked=sum(e.get("event")=="GATE_BLOCKED" for e in trace)
        lease_counts={}
        for e in lease_rows:
            key=str(e.get("lease_key","")); lease_counts.setdefault(key,[]).append(e.get("event"))
        lease_leaks=sum(1 for names in lease_counts.values() if "LEASE_ACQUIRED" in names and "LEASE_RELEASED" not in names)
        active={}; lease_peak=0
        for e in sorted(lease_rows,key=lambda x:float(x.get("timestamp_monotonic",0))):
            key=str(e.get("lease_key",""))
            if e.get("event")=="LEASE_ACQUIRED": active[key]=int(e.get("num_blocks",0))
            elif e.get("event")=="LEASE_RELEASED": active.pop(key,None)
            lease_peak=max(lease_peak,sum(active.values()))
        trace_valid=all(all(event in trace_by[eid] for event in ("RESUME_ARRIVED","RESUME_ADMITTED","FIRST_TOKEN","REQUEST_FINISHED")) for eid in data)
        lease_sequences=all(lease_counts.get(item["lease_key"])==["LEASE_ACQUIRED","LEASE_HELD_AFTER_PREFETCH_FREE","LEASE_RELEASE_READY","LEASE_RELEASED"] for item in promo.values() if item.get("lease_key"))
        native_correct=all(r["status"]=="ok" and int(r["external_tokens_loaded"])==data[r["episode_id"]]["job"]["required_prefix_tokens"]
            and int(r["transfer_bytes"])>0 and int(r["model_forward_count"])==0
            and int(r["generated_token_count"])==0 and int(r["mixed_prefetch_model_step_count"])==0 for r in native_rows)
        token_correct=all(r["generated_token_count"]==r["resume_osl"] for r in resume_rows)
        conservation=source_conserved(int(round(local)),int(round(external)),sum(j["required_prefix_tokens"] for j in jobs))
        if policy=="OnDemand-K2": conservation &= int(round(local))==0 and int(round(external))==168016
        reservation_peak=max([int(d.get("reserved_blocks",0)) for d in decisions if str(d.get("reserved_blocks",0)).isdigit()]+[0])
        return {"policy":policy,"repetition":repetition,"measured":measured,"t0_monotonic_ns":t0_ns,
            "resume_rows":resume_rows,"native_rows":native_rows,"decisions":decisions,
            "trace":trace,"leases":lease_rows,"capacity":cap,"reservation_peak_blocks":reservation_peak,"local_tokens":int(round(local)),
            "external_tokens":int(round(external)),"max_pre_first":max_pre,"gate_blocked":blocked,
            "lease_leaks":lease_leaks,"lease_peak_blocks":lease_peak,"lease_sequences":lease_sequences,
            "trace_valid":trace_valid,"native_correct":native_correct,"token_correct":token_correct,
            "source_conservation":conservation,"X_first_ns":x_first,"X_finish_ns":x_finish,
            "X_decode_duration_ms":(x_finish-x_first)/1e6,"X_TPOT_ms":(x_finish-x_first)/1e6/255,
            "X_active_through_last_first":x_active,"run_label":label}
    finally:
        runtime.stop_server(proc,log,out,label)

def service_calibration(jobs:list[dict[str,Any]],data:dict[str,dict[str,Any]],out:Path=OUT)->tuple[float,list[dict[str,Any]]]:
    selected=sorted(jobs,key=lambda j:j["required_blocks"])
    idx=math.floor((len(selected)-1)/2)
    picks=[selected[0],selected[idx],selected[-1]]
    rows=[]
    for rep in range(3):
        label=f"service_calibration_{rep}"
        proc,log=runtime.start_server(label,connector=True,out=out)
        try:
            _,block=runtime.wait_ready(proc)
            if block!=16: raise RuntimeError("EXPERIMENT_INVALID_ENVIRONMENT")
            subset={j["episode_id"]:data[j["episode_id"]] for j in picks}
            materialize_and_evict(proc,log,label,[j for j in jobs if j["episode_id"] in subset],subset,out)
            _,xf,meta=start_background()
            for j in picks:
                eid=j["episode_id"]
                result=native_prefetch(subset[eid]["prefix"])
                ts=result.get("phase3m_timestamps") or {}
                if result.get("status")!="ok" or "T6" not in ts or "T7" not in ts:
                    raise RuntimeError("SERVICE_CALIBRATION_INVALID:missing endpoint timestamps")
                issue_to=(float(ts["T6"])-float(result["sent_monotonic_s"]))*1000
                row={"repetition":rep,"episode_id":eid,"required_blocks":j["required_blocks"],
                    "prefix_tokens":j["required_prefix_tokens"],"external_tokens_loaded":result["external_tokens_loaded"],
                    "transfer_bytes":result["transfer_bytes"],"T0_monotonic":ts["T0"],"T6_monotonic":ts["T6"],
                    "T7_monotonic":ts["T7"],"issue_to_T6_ms":issue_to,
                    "model_forward_count":result["model_forward_count"],"generated_token_count":result["generated_token_count"],
                    "mixed_prefetch_model_step_count":result["mixed_prefetch_model_step_count"],"status":result["status"]}
                if int(row["external_tokens_loaded"])!=j["required_prefix_tokens"] or int(row["model_forward_count"])!=0 or int(row["generated_token_count"])!=0 or int(row["mixed_prefetch_model_step_count"])!=0 or int(row["transfer_bytes"])<=0:
                    raise RuntimeError("SERVICE_CALIBRATION_INVALID:bad native response")
                rows.append(row)
            xf.result(timeout=900)
        finally: runtime.stop_server(proc,log,out,label)
    if len(rows)!=9: raise RuntimeError("SERVICE_CALIBRATION_INVALID:not 9 measurements")
    p_safe=frozen_service_budget(max(float(r["issue_to_T6_ms"]) for r in rows))
    write_csv(out/"service_calibration.csv",rows)
    write_json(out/"service_budget.json",{"selected_episode_ids":[j["episode_id"] for j in picks],
        "selection_rule":"min, lower median index floor((N-1)/2), max by required_blocks",
        "valid_measurements":len(rows),"max_observed_issue_to_T6_ms":max(float(r["issue_to_T6_ms"]) for r in rows),
        "P_SAFE_MS":p_safe,"safety_factor":1.20})
    return p_safe,rows



def flatten_run(run:dict[str,Any],p_safe:float,x_baseline:float)->tuple[dict[str,Any],list[dict[str,Any]]]:
    policy=run["policy"]; rep=run["repetition"]
    resumes=run["resume_rows"]; natives=run["native_rows"]
    lat=[float(r["effective_resume_latency_ms"]) for r in resumes]
    finishes=[float(r["finish_ns"]) for r in resumes]
    firsts=[float(r["first_token_ns"]) for r in resumes]
    native_bytes=sum(int(r.get("transfer_bytes") or 0) for r in natives)
    duplicate=duplicate_h2d_bytes(native_bytes,run["external_tokens"],[{"required_prefix_tokens":sum(j["required_prefix_tokens"] for j in run["jobs"])}])
    useful=sum(r["promotion_classification"]=="PROMOTION_USEFUL" for r in natives)
    memory=0.0
    for n in natives:
        if n["T6_ms"]=="NA": continue
        rr=next((r for r in resumes if r["episode_id"]==n["episode_id"]),None)
        if rr and rr.get("scheduler_admission_ms")!="NA":
            memory += prefix_memory_time_mib_s(int(n["required_tokens"]),float(rr["scheduler_admission_ms"]),float(n["T6_ms"]))
    runrow={"policy":policy,"repetition":rep,"p_safe_ms":p_safe,
        "resume_P50_ms":statistics.median(lat),"resume_P95_ms":sorted(lat)[math.ceil(.95*len(lat))-1],
        "resume_max_ms":max(lat),"local_source_tokens":run["local_tokens"],"external_source_tokens":run["external_tokens"],
        "local_source_fraction":run["local_tokens"]/168016,"external_source_fraction":run["external_tokens"]/168016,
        "promotion_count":len(natives),"promotion_useful_count":useful,
        "promotion_useful_rate":useful/len(natives) if natives else "NA",
        "promotion_internal_ready_miss_count":sum(r["promotion_classification"]=="PROMOTION_INTERNAL_READY_MISS" for r in natives),
        "promotion_remote_wait_count":sum(r["promotion_classification"]=="PROMOTION_REMOTE_WAIT" for r in natives),
        "stale_promotion_cancelled_count":sum(r["decision"]=="STALE_PROMOTION_CANCELLED" for r in run["decisions"]),
        "native_H2D_bytes":native_bytes,"native_H2D_MiB":native_bytes/(1024**2),
        "resume_external_H2D_bytes":run["external_tokens"]*28672,
        "total_H2D_bytes":native_bytes+run["external_tokens"]*28672,
        "total_H2D_MiB":(native_bytes+run["external_tokens"]*28672)/(1024**2),
        "minimum_H2D_bytes":sum(j["required_prefix_tokens"] for j in run["jobs"])*28672,
        "duplicate_H2D_bytes":duplicate,"duplicate_H2D_MiB":duplicate/(1024**2),
        "promotion_memory_time_MiB_s":memory,"X_decode_duration_ms":run["X_decode_duration_ms"],
        "X_TPOT_ms":run["X_TPOT_ms"],"X_slowdown":run["X_decode_duration_ms"]/x_baseline,
        "X_active_through_last_first_token":run["X_active_through_last_first"],
        "max_pre_first_concurrency":run["max_pre_first"],"gate_blocked_count":run["gate_blocked"],
        "reservation_peak_blocks":max([int(d.get("reserved_blocks",0)) for d in run["decisions"]]+[0]),
        "physical_lease_peak_blocks":run.get("lease_peak_blocks","NA"),
        "lease_leak_count":run["lease_leaks"],"token_count_validation":run["token_correct"],
        "native_validation":run["native_correct"],"source_conservation_validation":run["source_conservation"]}
    return runrow,resumes

def save_run(run:dict[str,Any],p_safe:float,x_baseline:float)->dict[str,Any]:
    run["jobs"]=JOBS
    runrow,resumes=flatten_run(run,p_safe,x_baseline)
    SESSION_ROWS.extend({"policy":run["policy"],"repetition":run["repetition"],**r} for r in resumes)
    RUN_ROWS.append(runrow)
    PREFETCH_ROWS.extend(run["native_rows"])
    DECISION_ROWS.extend({"policy":run["policy"],"repetition":run["repetition"],**r} for r in run["decisions"])
    TRACE_ROWS.extend({"policy":run["policy"],"repetition":run["repetition"],**r} for r in run["trace"])
    LEASE_ROWS.extend({"policy":run["policy"],"repetition":run["repetition"],**r} for r in run["leases"])
    return runrow

def validate_run(run:dict[str,Any])->dict[str,Any]:
    return {"policy":run["policy"],"repetition":run["repetition"],
        "token_count_8_of_8":run["token_correct"],"source_conservation":run["source_conservation"],
        "K2":run["max_pre_first"]<=2,"X_active_through_last_first_token":run["X_active_through_last_first"],
        "native_correct":run["native_correct"],"lease_leaks":run["lease_leaks"],
        "reservation_overcommit":run.get("reservation_peak_blocks",0)>run["capacity"]["reservation_budget_blocks"] or run["lease_peak_blocks"]>run["capacity"]["reservation_budget_blocks"],
        "max_pre_first_concurrency":run["max_pre_first"],
        "promotion_count":len(run["native_rows"]),"valid":run["token_correct"] and run["source_conservation"]
            and run["max_pre_first"]<=2 and run["X_active_through_last_first"] and run["native_correct"] and run["lease_leaks"]==0
            and run["trace_valid"] and run.get("lease_sequences",True)
            and not (run.get("reservation_peak_blocks",0)>run["capacity"]["reservation_budget_blocks"] or run["lease_peak_blocks"]>run["capacity"]["reservation_budget_blocks"])}

JOBS=[]; RUN_ROWS=[]; SESSION_ROWS=[]; PREFETCH_ROWS=[]; TRACE_ROWS=[]; LEASE_ROWS=[]; DECISION_ROWS=[]

def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=("policy","matrix"),default="matrix")
    args=parser.parse_args()
    out=OUT; out.mkdir(parents=True,exist_ok=True); (out/"server_logs").mkdir(exist_ok=True)
    jobs=first_boundary_jobs(ROOT/"results/phase4c/hybrid_workload.csv")
    global JOBS; JOBS=jobs
    data=build_resume_prompts(jobs)
    write_csv(out/"boundary0_jobs.csv",jobs)
    baseline=list(csv.DictReader((out/"x_only_baseline.csv").open(encoding="utf-8")))
    x_baseline=statistics.median(float(r["decode_duration_ms"]) for r in baseline)
    budget_path=out/"service_budget.json"
    if not budget_path.exists():
        p_safe,calibration_rows=service_calibration(jobs,data,out)
    else:
        p_safe=float(json.loads(budget_path.read_text())["P_SAFE_MS"])
    write_json(out/"x_only_baseline_summary.json",{"runs":len(baseline),"baseline_X_duration_ms":x_baseline,
        "median_TPOT_ms":statistics.median(float(r["TPOT_ms"]) for r in baseline)})
    test_runs=[]
    for policy in POLICIES:
        run=run_one(policy,0,False,jobs,data,p_safe,out)
        test_runs.append(run)
    gate=[validate_run(r) for r in test_runs]
    write_json(out/"integration_gate.json",{"runs":gate,"pass":len(gate)==3 and all(r["valid"] for r in gate)})
    if not all(r["valid"] for r in gate): raise SystemExit("Stage A integration gate failed; Stage B not run")
    measured_runs=[]
    if args.stage=="matrix":
        for policy in POLICIES:
            for rep in range(3):
                run=run_one(policy,rep,True,jobs,data,p_safe,out)
                measured_runs.append(run)
                save_run(run,p_safe,x_baseline)
    write_csv(out/"run_metrics.csv",RUN_ROWS)
    write_csv(out/"session_metrics.csv",SESSION_ROWS)
    write_csv(out/"prefetch_metrics.csv",PREFETCH_ROWS)
    write_csv(out/"reservation_decisions.csv",DECISION_ROWS)
    write_csv(out/"reservation_events.csv",[r for r in DECISION_ROWS if r.get("decision") in {"RESERVATION_PROMOTE","RESERVATION_SKIP","RUNTIME_RELEASE_AFTER_PHYSICAL_LEASE"}])
    write_csv(out/"lease_events.csv",LEASE_ROWS)
    write_csv(out/"scheduler_trace.csv",TRACE_ROWS)
    (out/"validation_report.json").write_text(json.dumps({"integration_valid_runs":sum(r["valid"] for r in gate),
        "integration_runs":len(gate),"measured_runs":sum(bool(r.get("token_correct")) for r in measured_runs),
        "policy_names":list(POLICIES),"stage":"full-matrix" if args.stage=="matrix" else "integration-only"},indent=2)+"\n")
    print(json.dumps({"integration_gate":gate,"p_safe_ms":p_safe,"x_baseline_ms":x_baseline},indent=2))

if __name__=="__main__":
    main()
