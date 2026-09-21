from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

from experiments.phase3g_full import run_phase3g_full as base
from experiments.phase3f.native_client import native_prefetch
from experiments.phase3i import run_phase3i as prev
from experiments.phase3g import run_phase3g as transport

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"results"/"phase3l"
BYTES=117440512
LOADS=(0,1,2,3)


def write_csv(path, rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n"); w.writeheader(); w.writerows(rows)


def setup_target(prompts,label,snapshots):
    before=base.capture(label+"_ready",snapshots)
    base.send(prompts["a"],stream=False,max_tokens=1,normal=True)
    after=base.capture(label+"_after_first",snapshots)
    offload=base.gpu_to_cpu_offload_bytes(before,after) if hasattr(base,"gpu_to_cpu_offload_bytes") else None
    time.sleep(1.0)
    for i in range(1,5): base.send(prompts[f"pressure_{i}"],stream=False,max_tokens=1,normal=False)
    base.capture(label+"_after_pressure",snapshots)
    return offload


def run_state(n,rep,prompts):
    label=f"l{n}_{rep}"; base.OUT=OUT; proc=base.start_server(label); snapshots=[]
    try:
        base.runtime.wait_ready(proc); setup_target(prompts,label,snapshots)
        executor=ThreadPoolExecutor(max_workers=max(1,n)); events=[Event() for _ in range(n)]
        futures=[executor.submit(prev.send_x,prompts[f"pressure_{i+1}"],events[i]) for i in range(n)]
        for event in events:
            if not event.wait(timeout=180): raise RuntimeError(f"background X not active for L{n}")
        sent=time.monotonic(); native=native_prefetch(prompts["a"]); completed=time.monotonic();
        bg=[f.result(timeout=900) for f in futures]; executor.shutdown(wait=True)
        # A one-token continuation is only a cache-path check after promotion.
        local_check=base.send(prompts["a"]+prompts["suffix"],stream=True,max_tokens=1,normal=False)
        trace_path=OUT/"server_logs"/f"{label}.log"
        trace=[]; marker="SLACKKV_RESUME_TRACE "
        t0=sent
        for line in trace_path.read_text(encoding="utf-8",errors="ignore").splitlines():
            if marker in line:
                raw=json.loads(line.split(marker,1)[1]); trace.append({"load_state":f"L{n}","repetition":rep,"session":raw.get("session"),"event":raw.get("event"),"step_id":raw.get("step_id"),"timestamp_monotonic_raw":raw.get("timestamp_monotonic"),"timestamp_ms_from_promotion":(float(raw.get("timestamp_monotonic"))-t0)*1000.0,"num_slackkv_pre_first_token":raw.get("num_slackkv_pre_first_token")})
        active_ok=all(x["first_monotonic_s"]<sent<x["completed_monotonic_s"] for x in bg)
        bg_rows=[]
        for i,x in enumerate(bg,1):
            bg_rows.append({"load_state":f"L{n}","active_decode_count":n,"repetition":rep,"background_id":f"X{i}","first_token_ms_from_promotion":(x["first_monotonic_s"]-sent)*1000.0,"finish_ms_from_promotion":(x["completed_monotonic_s"]-sent)*1000.0,"decode_duration_ms":(x["completed_monotonic_s"]-x["first_monotonic_s"])*1000.0,"TPOT_ms":(x["completed_monotonic_s"]-x["first_monotonic_s"])*1000.0/127.0,"generated_tokens":len(x["token_ids"]),"sequence_hash":prev.sequence_hash(x["token_ids"]),"token_count_validation":"PASS" if len(x["token_ids"])==128 else "FAIL"})
        row={"load_state":f"L{n}","active_decode_count":n,"repetition":rep,
             "promotion_sent_ms":0.0,"promotion_completed_ms":(completed-sent)*1000.0,
             "client_e2e_ms":native["client_e2e_ms"],"server_e2e_ms":native["elapsed_ms"],
             "cpu_to_gpu_ms":native["cpu_to_gpu_transfer_ms"],
             "client_noncopy_ms":native["client_e2e_ms"]-native["cpu_to_gpu_transfer_ms"],
             "server_noncopy_ms":native["elapsed_ms"]-native["cpu_to_gpu_transfer_ms"],
             "connector_only_step_count":native["connector_only_step_count"],"connector_only_wall_ms":native["connector_only_wall_ms"],
             "normal_requests_delayed":native["normal_requests_delayed"],"external_tokens_loaded":native["external_tokens_loaded"],
             "model_forward_count":native["model_forward_count"],"generated_token_count":native["generated_token_count"],
             "mixed_prefetch_model_step_count":native["mixed_prefetch_model_step_count"],
             "background_ids":json.dumps([f"X{i}" for i in range(1,n+1)]),
             "background_first_token_times_json":json.dumps([x["first_monotonic_s"]-sent for x in bg]),
             "background_finish_times_json":json.dumps([x["completed_monotonic_s"]-sent for x in bg]),
             "active_decode_validation":"PASS" if active_ok else "FAIL",
             "background_token_count_validation":"PASS" if all(len(x["token_ids"])==128 for x in bg) else "FAIL",
             "cache_transport_validation":"PASS" if native["status"]=="ok" and native["external_tokens_loaded"]==4096 and not any(t["session"]=="A" and t["event"]=="WAITING_FOR_REMOTE_KVS" for t in trace) else "FAIL",
             "local_check_token_count":len(local_check["token_ids"]),"status":native["status"]}
        return row,bg_rows,trace
    finally:
        try: executor.shutdown(wait=False)
        except Exception: pass
        base.stop_server(proc)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--state",type=int,choices=LOADS,required=True); args=ap.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    prompts=base.make_prompts(); rows=[]; bg=[]; traces=[]
    for rep in range(5):
        print(f"RUN L{args.state} {rep}",flush=True); r,b,t=run_state(args.state,rep,prompts); rows.append(r); bg+=b; traces+=t
    d=OUT/"batches"/f"l{args.state}"; d.mkdir(parents=True,exist_ok=True)
    write_csv(d/"calibration_metrics.csv",rows); write_csv(d/"background_metrics.csv",bg); write_csv(d/"scheduler_trace.csv",traces)
    print("STATE_DONE",args.state,flush=True)


if __name__=="__main__": main()
