from __future__ import annotations

import csv
import json
import math
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any

from experiments.phase2d.helpers import gpu_to_cpu_offload_bytes, source_tokens_delta
from experiments.phase3f.native_client import native_prefetch
from experiments.phase3f.scheduler import latest_feasible_start, merge_blackouts
from experiments.phase4c.synthetic_tokens import make_next_prompt, token_hash
from experiments.phase4d_a import runtime
from experiments.phase4d_a.helpers import *

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"results/phase4d_a"
WORKLOAD=ROOT/"results/phase4c/hybrid_workload.csv"
REF=ROOT/"results/phase4c/request_metrics.csv"
POLICIES=("OnDemand-K2","NaiveJIT-Async-K2","SlackKV-Oracle-ReserveLease-K2")

def write_csv(path:Path, rows:list[dict[str,Any]])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n"); w.writeheader(); w.writerows(rows)

def write_json(path:Path,value:Any)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n",encoding="utf-8")

def build_resume_prompts(jobs:list[dict[str,Any]])->dict[str,dict[str,Any]]:
    manifests={r["episode_id"]:r for r in json.loads((OUT/"token_manifest.json").read_text(encoding="utf-8"))}
    data={}
    for j in jobs:
        m=manifests[j["episode_id"]]
        t0=runtime.tokenizer_ids(j["config"],j["session_hash"],0,j["turn0_isl"])
        t0out=[int(x) for x in m["turn0_output_ids"]]
        p1=make_next_prompt(t0+t0out,j["turn1_isl"],j["required_prefix_tokens"],
            phase="Phase4C",config=j["config"],session_hash=j["session_hash"],turn_index=1,
            vocab_size=len(runtime.TOKENIZER),special_ids=runtime.TOKENIZER.all_special_ids)
        if token_hash(p1)!=m["turn1_prompt_sha256"]: raise RuntimeError("reconstructed Turn1 hash mismatch")
        data[j["episode_id"]]={"resume_prompt":p1,"prefix":p1[:j["required_prefix_tokens"]],
            "resume_sha256":m["turn1_prompt_sha256"],"expected_output_sha256":None,
            "expected_output_count":j["resume_osl"],"job":j}
    for r in csv.DictReader(REF.open(encoding="utf-8")):
        if int(r["replay_turn_index"])==1 and r["session_hash"] in {x["job"]["session_hash"] for x in data.values()}:
            for value in data.values():
                if value["job"]["session_hash"]==r["session_hash"]: value["expected_output_sha256"]=r["output_sha256"]
    return data

def gpu_capacity_blocks(server_log:Path, block_size:int)->int:
    text=server_log.read_text(encoding="utf-8",errors="replace")
    matches=re.findall(r"GPU KV cache size:\s*([0-9,]+) tokens",text)
    if not matches: raise RuntimeError("GPU KV cache capacity token count absent from server log")
    tokens=int(matches[-1].replace(",",""))
    if tokens%block_size: raise RuntimeError("reported GPU KV capacity not block aligned")
    return tokens//block_size

def source_snapshot()->dict:
    return runtime.source_samples(runtime.metric_text())

def materialize_and_evict(proc,log:Path,label:str,jobs:list[dict[str,Any]],data:dict[str,dict[str,Any]],out:Path=OUT)->dict[str,Any]:
    if runtime.wait_ready(proc)[1]!=16: raise RuntimeError("EXPERIMENT_INVALID_ENVIRONMENT:block_size")
    total_blocks=gpu_capacity_blocks(log,16)
    budget=reservation_budget(total_blocks,jobs)
    max_fit=maximum_simultaneous_jobs(jobs,budget)
    capacity={"total_gpu_kv_blocks":total_blocks,"reservation_budget_blocks":budget,
        "reservation_budget_MiB":budget*16*BYTES_PER_TOKEN/(1024**2),
        "max_simultaneous_frozen_jobs":max_fit,"budget_fraction":0.75}
    write_json(out/"capacity_config.json",capacity)
    if min(j["required_blocks"] for j in jobs)>=budget: raise RuntimeError("mechanical budget cannot admit one job")
    rows=[]; before=source_snapshot()
    for eid in sorted(data):
        j=data[eid]["job"]; prefix=data[eid]["prefix"]
        b=source_snapshot()
        result=runtime.send(prefix,1,stream=False)
        after=source_snapshot()
        deadline=time.monotonic()+20
        while time.monotonic()<deadline:
            observed_now=int(round(gpu_to_cpu_offload_bytes(b,after)))
            if observed_now>=len(prefix)*BYTES_PER_TOKEN-16*BYTES_PER_TOKEN: break
            time.sleep(.1); after=source_snapshot()
        expected=len(prefix)*BYTES_PER_TOKEN
        observed=int(round(gpu_to_cpu_offload_bytes(b,after)))
        rows.append({"run_label":label,"episode_id":eid,"expected_prefix_bytes":expected,"observed_offload_bytes":observed,
            "difference_bytes":observed-expected,"prompt_tokens":result["prompt_tokens"],
            "generated_tokens":len(result["tokens"]),"validation":abs(observed-expected)<=16*BYTES_PER_TOKEN})
    path=out/"external_materialization.csv"
    existing=list(csv.DictReader(path.open(encoding="utf-8"))) if path.exists() else []
    write_csv(path,existing+rows)
    bad=[r for r in rows if not r["validation"]]
    if bad: raise RuntimeError("EXTERNAL_MATERIALIZATION_INVALID")
    pressure_count=pressure_request_count(total_blocks)
    for i in range(pressure_count):
        h=f"pressure-{label}-{i}"
        p=runtime.tokenizer_ids("phase4d-pressure",h,0,X_PROMPT_TOKENS)
        runtime.send(p,1,params={"max_offload_tokens":0})
    pressure={"pressure_requests":pressure_count,"pressure_full_blocks":X_PROMPT_TOKENS//16,
        "unrelated_full_blocks":pressure_count*(X_PROMPT_TOKENS//16),"gpu_capacity_blocks":total_blocks,
        "capacity_exceeded":pressure_count*(X_PROMPT_TOKENS//16)>total_blocks}
    write_json(out/"pressure_validation.json",pressure)
    if not pressure["capacity_exceeded"]: raise RuntimeError("pressure did not exceed GPU capacity")
    return {"capacity":capacity,"pressure":pressure,"materialization":rows}

def x_prompt()->list[int]:
    return runtime.tokenizer_ids("phase4d-background","background-X",0,X_PROMPT_TOKENS)

def run_x_only(repetition:int)->dict[str,Any]:
    proc,log=runtime.start_server(f"x_only_{repetition}",connector=True,out=OUT)
    try:
        runtime.wait_ready(proc)
        p=x_prompt()
        result=runtime.send(p,X_OUTPUT_TOKENS,stream=True,params={"max_offload_tokens":0})
        count=len(result["tokens"]); duration=(result["done_ns"]-result["first_ns"])/1e6
        return {"repetition":repetition,"first_token_monotonic_ns":result["first_ns"],
            "finish_monotonic_ns":result["done_ns"],"generated_tokens":count,
            "decode_duration_ms":duration,"TPOT_ms":duration/255 if count==256 else "NA",
            "status":"PASS" if count==256 else "FAIL"}
    finally: runtime.stop_server(proc,log,OUT,f"x_only_{repetition}")
