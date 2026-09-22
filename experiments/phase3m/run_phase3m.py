from __future__ import annotations

import argparse, csv, json, signal, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

from experiments.phase3g_full import run_phase3g_full as base
from experiments.phase3f.native_client import native_prefetch
from experiments.phase3i import run_phase3i as prev
from experiments.phase3g import run_phase3g as transport

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"results"/"phase3m"; LOADS=(0,1,2,3)

def write_csv(path, rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def start(label):
    base.OUT=OUT; return base.start_server(label)

def stop(p):
    base.stop_server(p)

def parse_log(path, t0):
    traces=[]; steps=[]
    for line in path.read_text(encoding="utf-8",errors="ignore").splitlines():
        if "SLACKKV_RESUME_TRACE " in line:
            try:
                x=json.loads(line.split("SLACKKV_RESUME_TRACE ",1)[1]); traces.append(x)
            except Exception: pass
        if "SLACKKV_NORMAL_STEP " in line:
            try:
                text=line.split("SLACKKV_NORMAL_STEP ",1)[1].replace("'",'"')
                x=json.loads(text); steps.append(x)
            except Exception: pass
    return traces,steps

def send_bg(ids, session, first_event):
    payload=transport.completion_payload(ids, stream=True, max_tokens=128, trace=True, session=session, gate=False, normal=True)
    sent=time.monotonic(); req=urllib.request.Request("http://127.0.0.1:8000/v1/completions",data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"}); token_ids=[]; first=None
    with urllib.request.urlopen(req,timeout=600) as response:
        for line in response:
            if not line.startswith(b"data: "): continue
            data=line[6:].strip()
            if data==b"[DONE]": continue
            choice=(json.loads(data).get("choices") or [{}])[0]; tids=[int(x) for x in (choice.get("token_ids") or [])]
            if tids:
                token_ids.extend(tids)
                if first is None: first=time.monotonic(); first_event.set()
            elif first is None and choice.get("text"): first=time.monotonic(); first_event.set()
    done=time.monotonic()
    if first is None: raise RuntimeError(f"{session} produced no first token")
    return {"sent_monotonic_s":sent,"first_monotonic_s":first,"completed_monotonic_s":done,"token_ids":token_ids}

def run_state(n, rep, prompts):
    label=f"l{n}_{rep}"; p=start(label); OUT.joinpath("server_logs").mkdir(parents=True,exist_ok=True)
    try:
        base.runtime.wait_ready(p); base.setup_target if False else None
        base.send(prompts["a"],stream=False,max_tokens=1,normal=True); time.sleep(1)
        for i in range(1,5): base.send(prompts[f"pressure_{i}"],stream=False,max_tokens=1,normal=False)
        executor=ThreadPoolExecutor(max_workers=max(1,n)); events=[Event() for _ in range(n)]
        fs=[executor.submit(send_bg,prompts[f"pressure_{i+1}"],f"X{i+1}",events[i]) for i in range(n)]
        for e in events:
            if not e.wait(180): raise RuntimeError(f"background barrier failed L{n}")
        sent=time.monotonic(); native=native_prefetch(prompts["a"]); completed=time.monotonic()
        bg=[f.result(timeout=900) for f in fs]; executor.shutdown(wait=True)
        base.send(prompts["a"]+prompts["suffix"],stream=True,max_tokens=1,normal=False)
        log=OUT/"server_logs"/f"{label}.log"; traces,steps=parse_log(log,sent)
        by={x.get("session"):x for x in traces if x.get("session")}
        phase=native.get("phase3m_timestamps") or {}
        # endpoint metadata uses raw monotonic values; preserve them.
        t0=float(phase.get("T0",sent)); t1=float(phase.get("T1",t0)); t2=float(phase.get("T2",t1)); t3=float(phase.get("T3",t2)); t5=float(phase.get("T5",float("nan"))); t6=float(phase.get("T6",t3)); t7=float(phase.get("T7",completed))
        preceding=[s for s in steps if float(s.get("step_end_monotonic",0))<=t0]
        prev_step=preceding[-1] if preceding else None
        inflight=bool(prev_step and float(prev_step["step_start_monotonic"])<=t0<float(prev_step["step_end_monotonic"]))
        active_ok=True; bg_rows=[]
        for i,x in enumerate(bg,1):
            sess=f"X{i}"; ev=by.get(sess,{}); server_first=ev.get("timestamp_monotonic") if ev.get("event")=="FIRST_TOKEN" else None
            if server_first is None: server_first=x["first_monotonic_s"]
            active_ok &= float(server_first)<t0<t7<float(x["completed_monotonic_s"])
            bg_rows.append({"load_state":f"L{n}","active_decode_count":n,"repetition":rep,"background_id":sess,"server_first_token_monotonic":server_first,"client_first_token_monotonic":x["first_monotonic_s"],"client_delivery_lag_ms":(x["first_monotonic_s"]-float(server_first))*1000.0,"request_finished_monotonic":x["completed_monotonic_s"],"active_through_promotion":active_ok})
        row={"load_state":f"L{n}","active_decode_count":n,"repetition":rep,"T0_request_received":t0,"T1_engine_enqueue":t1,"T2_scheduler_seen":t2,"T3_connector_start":t3,"T5_transfer_complete":t5,"T6_scheduler_complete":t6,"T7_http_complete":t7,"client_e2e_ms":native["client_e2e_ms"],"server_e2e_ms":(t7-t0)*1000.0,"http_to_engine_enqueue_ms":(t1-t0)*1000.0,"engine_enqueue_to_scheduler_seen_ms":(t2-t1)*1000.0,"scheduler_seen_to_connector_start_ms":(t3-t2)*1000.0,"request_to_connector_start_ms":(t3-t0)*1000.0,"connector_phase_ms":(t6-t3)*1000.0,"scheduler_complete_to_http_ms":(t7-t6)*1000.0,"connector_only_wall_ms":native["connector_only_wall_ms"],"connector_only_step_count":native["connector_only_step_count"],"cpu_to_gpu_ms":native["cpu_to_gpu_transfer_ms"],"was_normal_step_inflight_at_T0":inflight,"preceding_normal_step_id":prev_step.get("step_id") if prev_step else "NA","preceding_normal_step_wall_ms":prev_step.get("step_wall_ms") if prev_step else 0.0,"remaining_normal_step_ms":max(0.0,(float(prev_step["step_end_monotonic"])-t0)*1000.0) if inflight else 0.0,"num_running_requests":prev_step.get("num_running_requests") if prev_step else n,"num_scheduled_requests":prev_step.get("num_scheduled_requests") if prev_step else n,"total_num_scheduled_tokens":prev_step.get("total_num_scheduled_tokens") if prev_step else 0,"normal_requests_delayed":native["normal_requests_delayed"],"active_decode_validation":"PASS" if active_ok else "FAIL","cache_transport_validation":"PASS" if native["status"]=="ok" and native["external_tokens_loaded"]==4096 else "FAIL","native_correctness_validation":"PASS" if native["model_forward_count"]==0 and native["generated_token_count"]==0 and native["mixed_prefetch_model_step_count"]==0 else "FAIL"}
        step_rows=[{"load_state":f"L{n}","repetition":rep,**s} for s in steps if float(s.get("step_end_monotonic",0))>=t0-0.2 and float(s.get("step_start_monotonic",0))<=t7]
        return row,bg_rows,step_rows
    finally:
        stop(p)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--state",type=int,choices=LOADS,required=True); a=ap.parse_args(); prompts=base.make_prompts(); rows=[]; bg=[]; steps=[]
    for rep in range(3):
        print(f"RUN L{a.state} {rep}",flush=True); r,b,s=run_state(a.state,rep,prompts); rows.append(r); bg+=b; steps+=s
    d=OUT/"batches"/f"l{a.state}"; d.mkdir(parents=True,exist_ok=True); write_csv(d/"latency_decomposition.csv",rows); write_csv(d/"background_timing.csv",bg); write_csv(d/"normal_step_trace.csv",steps); print("STATE_DONE",a.state,flush=True)

if __name__=="__main__": main()
