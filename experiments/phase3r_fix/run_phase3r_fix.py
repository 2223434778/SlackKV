from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Event
from typing import Any

from experiments.phase2d import run_phase2d as runtime
from experiments.phase2d.helpers import gpu_to_cpu_offload_bytes, parse_prometheus_samples, source_tokens_delta
from experiments.phase2c_fix.helpers import metric_value
from experiments.phase3r_fix.helpers import capacity_decision, SESSION_KV_MIB, CAPACITY_RESERVE_MIB, REQUIRED_FREE_MIB
from experiments.phase3f.native_client import native_prefetch
from experiments.phase3f.scheduler import latest_feasible_start, merge_blackouts
from experiments.phase3g import run_phase3g as transport
from experiments.phase3g_full import run_phase3g_full as base
from experiments.phase3g_full.metrics import decode_throughput_tok_s, interval_concurrency, percentile_linear
from experiments.phase3h import run_phase3h as gate_config
from experiments.phase3n.run_phase3n import send_x_background
from experiments.phase3r_fix.helpers import ASYNC_SERVICE_MS, BLOCKING_SERVICE_MS, DEADLINES_MS, configured_service_budget, effective_latency_ms, gpu_ready_residency_ms, internal_ready_miss

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase3r_fix"
SESSIONS = tuple("ABCDEFGHIJKL")
POLICIES = ("CapacityAwareAsync40-K2",)
CAPACITY_DECISIONS: list[dict[str, Any]] = []
CAPACITY_SANITY: list[dict[str, Any]] = []
TOKENS = 32
BYTES = 117_440_512
GUARD_MS = 110.0


def send_x_background(ids: list[int], first_token: Event) -> dict[str, Any]:
    payload = transport.completion_payload(ids, stream=True, max_tokens=256, trace=True, session="X", gate=False, normal=True)
    sent = time.monotonic(); first = None; token_ids: list[int] = []
    request = urllib.request.Request("http://127.0.0.1:8000/v1/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=900) as response:
        for line in response:
            if not line.startswith(b"data: "): continue
            body=line[6:].strip()
            if body == b"[DONE]": continue
            choice=(json.loads(body).get("choices") or [{}])[0]; values=[int(x) for x in (choice.get("token_ids") or [])]
            if values:
                token_ids.extend(values)
                if first is None: first=time.monotonic(); first_token.set()
    done=time.monotonic()
    if first is None: raise RuntimeError("X did not generate first token")
    return {"sent_monotonic_s":sent,"first_monotonic_s":first,"completed_monotonic_s":done,"token_ids":token_ids}



def make_prompts_12() -> dict[str, list[int]]:
    tokenizer = base.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", local_files_only=True)
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    prompts = {s.lower(): encode((f"Session {s} ordinary coding context marker {s}. ") * 900)[:4096] for s in SESSIONS}
    if any(len(prompts[s.lower()]) != 4096 for s in SESSIONS) or len({tuple(prompts[s.lower()]) for s in SESSIONS}) != 12:
        raise AssertionError("A-L must be 12 distinct 4096-token prefixes")
    for i,text in enumerate(("PRESSURE ONE unrelated file-search output alpha. ","PRESSURE TWO unrelated compiler output beta. ","PRESSURE THREE unrelated repository status gamma. ","PRESSURE FOUR unrelated terminal output delta. "),1): prompts[f"pressure_{i}"]=encode(text*3000)[:9500]
    prompts["suffix"]=encode("\nTool result: operation completed successfully.\nAssistant:")
    return prompts

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def capture(stage: str, snapshots: list[dict[str, Any]]) -> dict[Any, float]:
    raw = runtime.metric_text()
    snapshots.append({"stage": stage, "timestamp_monotonic_s": time.monotonic(), "raw": raw})
    return parse_prometheus_samples(raw)


def parse_trace(path: Path, policy: str, repetition: int, t0: float | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    marker = "SLACKKV_RESUME_TRACE "
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if marker not in line:
            continue
        raw = json.loads(line.split(marker, 1)[1])
        rows.append({"policy": policy, "repetition": repetition, **raw, "timestamp_ms_from_t0": "NA" if t0 is None else (float(raw["timestamp_monotonic"]) - t0) * 1000.0})
    return rows


def event(rows: list[dict[str, Any]], session: str, name: str) -> dict[str, Any] | None:
    return next((row for row in rows if row.get("session") == session and row.get("event") == name), None)


def wait_x_server_first(log_path: Path, timeout_s: float = 30.0) -> float:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        for row in parse_trace(log_path, "bootstrap", -1):
            if row.get("session") == "X" and row.get("event") == "FIRST_TOKEN":
                return float(row["timestamp_monotonic"])
        time.sleep(0.002)
    raise RuntimeError("X server FIRST_TOKEN was not observed")


def wait_until(absolute_s: float) -> None:
    while time.monotonic() < absolute_s:
        time.sleep(min(0.001, absolute_s - time.monotonic()))


def send_resume(session: str, prompts: dict[str, list[int]], t0: float, ready: Event | None, async_policy: bool) -> dict[str, Any]:
    wait_until(t0 + DEADLINES_MS[session] / 1000.0)
    if ready is not None and not async_policy:
        ready.wait(timeout=300)
    # Intentionally no future/result/ready wait in Async40: the real request
    # observes the scheduler-visible T6 cache state itself.
    response = transport.send(prompts[session.lower()] + prompts["suffix"], stream=True, max_tokens=TOKENS, trace=True, session=session, gate=True, normal=False)
    if len(response["token_ids"]) != TOKENS:
        raise RuntimeError(f"{session} generated {len(response['token_ids'])}, expected {TOKENS}")
    return response


def read_capacity_snapshot() -> dict[str, float]:
    usage = metric_value(parse_prometheus_samples(runtime.metric_text()), "vllm:kv_cache_usage_perc")
    if not 0.0 <= usage <= 1.05: raise RuntimeError(f"unexpected vllm:kv_cache_usage_perc={usage}")
    return {"kv_usage_fraction":usage,"used_mib":usage*1024.0,"free_mib":1024.0*(1.0-usage),"required_mib":REQUIRED_FREE_MIB}

def capacity_sanity(stage: str, samples: dict[Any, float]) -> None:
    usage = metric_value(samples, "vllm:kv_cache_usage_perc")
    CAPACITY_SANITY.append({"stage": stage, "kv_usage_fraction": usage,
                            "used_mib": usage * 1024.0,
                            "free_mib": 1024.0 * (1.0 - usage)})

def schedule_promotions(policy, prompts, t0, ready, output, futures) -> None:
    service=ASYNC_SERVICE_MS; pending=set(SESSIONS); blackouts=merge_blackouts(DEADLINES_MS,GUARD_MS)
    with ThreadPoolExecutor(max_workers=3) as pool:
      while pending:
        now=(time.monotonic()-t0)*1000.0; jobs=[(x,DEADLINES_MS[x]) for x in pending]; target=latest_feasible_start(now,jobs,blackouts,service)
        if target-now>0.5: time.sleep((target-now)/1000.0); continue
        session=min(pending,key=lambda x:(DEADLINES_MS[x],x))
        query_start=time.monotonic(); snap=read_capacity_snapshot(); query_end=time.monotonic()
        decision,free=capacity_decision(snap["kv_usage_fraction"]); check=(query_end-t0)*1000.0
        record={"session":session,"cluster":1+(ord(session)-ord("A"))//4,
                "deadline_ms":DEADLINES_MS[session],"target_start_ms":target,
                "capacity_query_start_ms":(query_start-t0)*1000.0,
                "capacity_query_end_ms":check,
                "capacity_query_latency_ms":(query_end-query_start)*1000.0,
                "capacity_check_ms":check,**snap,"session_kv_mib":SESSION_KV_MIB,
                "reserve_mib":CAPACITY_RESERVE_MIB,"decision":decision,
                "reason":"ENOUGH_HEADROOM" if decision=="PROMOTE" else "INSUFFICIENT_KV_CAPACITY",
                "actual_issue_ms":"NA","issue_lateness_ms":"NA"}
        CAPACITY_DECISIONS.append(record)
        if decision=="PROMOTE":
          issue=time.monotonic(); record["actual_issue_ms"]=(issue-t0)*1000.0; record["issue_lateness_ms"]=record["actual_issue_ms"]-target
          future=pool.submit(native_prefetch,prompts[session.lower()]); futures[session]=future; output[session]={"target_start_ms":target,"client_issue_s":issue,"future":future,"decision":decision}
        else: output[session]={"target_start_ms":target,"decision":decision}
        pending.remove(session)
      for session,future in futures.items(): output[session]["result"]=future.result(timeout=300); ready[session].set()


def policy_run(policy: str, repetition: int, prompts: dict[str, list[int]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    label = f"{policy.lower().replace('-', '_')}_{repetition}"
    base.OUT = OUT
    original_payload = transport.completion_payload
    gate_config._active_gate_limit = 2
    transport.completion_payload = gate_config._payload_with_gate_limit
    proc = base.start_server(label)
    snapshots: list[dict[str, Any]] = []
    try:
        runtime.wait_ready(proc)
        previous = capture("ready", snapshots); CAPACITY_SANITY.clear(); capacity_sanity("server_ready", previous)
        offloads: dict[str, float] = {}
        for session in SESSIONS:
            base.send(prompts[session.lower()], stream=False, max_tokens=1, normal=True)
            current = capture(f"after_{session}_first", snapshots)
            offloads[session] = gpu_to_cpu_offload_bytes(previous, current); previous = current
        if any(value != BYTES for value in offloads.values()):
            raise RuntimeError(f"offload mismatch: {offloads}")
        capacity_sanity("after_A_L_setup", previous)
        time.sleep(1.0)
        for index in range(1, 5):
            base.send(prompts[f"pressure_{index}"], stream=False, max_tokens=1, normal=False)
        pressure_samples=capture("after_pressure", snapshots); capacity_sanity("after_pressure", pressure_samples)
        x_first = Event()
        CAPACITY_DECISIONS.clear()
        promotions: dict[str, dict[str, Any]] = {}; futures: dict[str, Future[Any]] = {}; ready = {s: Event() for s in SESSIONS}
        log_path = OUT / "server_logs" / f"{label}.log"
        with ThreadPoolExecutor(max_workers=16) as pool:
            x_future = pool.submit(send_x_background, prompts["pressure_1"], x_first)
            if not x_first.wait(180): raise RuntimeError("X client first-token barrier failed")
            t0 = wait_x_server_first(log_path)
            # X's prefix lookup has completed at its first generated token.  A
            # fresh source-counter baseline therefore attributes the following
            # 12288 local prompt tokens only to A/B/C real resumes.
            before = capture("after_x_first_token", snapshots)
            promotion_future = None
            if policy != "OnDemand-K2":
                promotion_future = pool.submit(schedule_promotions, policy, prompts, t0, ready, promotions, futures)
            # OnDemand owns no native promotion future and must not inherit a
            # never-set controller event.
            resume = {
                s: pool.submit(
                    send_resume,
                    s,
                    prompts,
                    t0,
                    ready[s] if policy == "Blocking70-K2" else None,
                    policy == "CapacityAwareAsync40-K2",
                )
                for s in SESSIONS
            }
            results = {s: item.result(timeout=600) for s, item in resume.items()}
            if promotion_future is not None: promotion_future.result(timeout=600)
            x = x_future.result(timeout=900)
        after = capture("after_workload", snapshots)
        trace = parse_trace(log_path, policy, repetition, t0)
        x_first_event = event(trace, "X", "FIRST_TOKEN"); x_finish_event = event(trace, "X", "REQUEST_FINISHED")
        if not x_first_event or not x_finish_event: raise RuntimeError("missing X scheduler trace")
        x_first_s = float(x_first_event["timestamp_monotonic"]); x_finish_s = float(x_finish_event["timestamp_monotonic"])
        source_local = source_tokens_delta(before, after, "local_cache_hit")
        source_external = source_tokens_delta(before, after, "external_kv_transfer")
        native_rows: list[dict[str, Any]] = []
        session_rows: list[dict[str, Any]] = []
        for session in SESSIONS:
            arrived, admitted, first, finished = (event(trace, session, name) for name in ("RESUME_ARRIVED", "RESUME_ADMITTED", "FIRST_TOKEN", "REQUEST_FINISHED"))
            if not all((arrived, admitted, first, finished)): raise RuntimeError(f"incomplete trace {policy}/{session}")
            remote_wait = event(trace, session, "WAITING_FOR_REMOTE_KVS") is not None
            arrival_ms, admission_ms, first_ms, finish_ms = ((float(row["timestamp_monotonic"]) - t0) * 1000.0 for row in (arrived, admitted, first, finished))
            promotion = promotions.get(session); result = promotion.get("result") if promotion else None
            t6_ms = "NA" if result is None else (float((result.get("phase3m_timestamps") or {})["T6"]) - t0) * 1000.0
            # Source counters are server-global. The run-level exact 12288-token
            # delta plus each session's actual no-remote-wait trace is the only
            # non-invasive attribution available under concurrent K2 resumes.
            decision=promotion.get("decision","CAPACITY_SKIP") if promotion else "CAPACITY_SKIP"
            if decision=="CAPACITY_SKIP": outcome="CAPACITY_SKIPPED_ONDEMAND"
            elif float(t6_ms)>arrival_ms: outcome="INTERNAL_READY_MISS"
            elif remote_wait: outcome="PROMOTED_REMOTE_WAIT"
            else: outcome="PROMOTED_NO_REMOTE_WAIT"
            session_rows.append({"policy":policy,"repetition":repetition,"session":session,"deadline_ms":DEADLINES_MS[session],"client_request_send_ms":(results[session]["sent_monotonic_s"]-t0)*1000.0,"scheduler_arrival_ms":arrival_ms,"scheduler_admission_ms":admission_ms,"first_token_ms":first_ms,"request_finished_ms":finish_ms,"request_TTFT_ms":results[session]["ttft_ms"],"effective_resume_latency_ms":effective_latency_ms(first_ms, DEADLINES_MS[session]),"effective_completion_latency_ms":finish_ms-DEADLINES_MS[session],"WAITING_FOR_REMOTE_KVS_seen":remote_wait,"promotion_decision":decision,"promotion_outcome":outcome,"T6_ms":t6_ms,"arrival_minus_T6_ms":"NA" if t6_ms == "NA" else arrival_ms-float(t6_ms),"gpu_ready_residency_ms":0.0 if t6_ms == "NA" else gpu_ready_residency_ms(admission_ms,float(t6_ms)),"generated_token_count":len(results[session]["token_ids"])})
            if result is not None:
                ts = result.get("phase3m_timestamps") or {}
                target=promotion["target_start_ms"]; t6=(float(ts["T6"])-t0)*1000.0
                native_rows.append({"policy":policy,"repetition":repetition,"session":session,"service_budget_ms":configured_service_budget(policy),"target_start_ms":target,"client_issue_ms":(promotion["client_issue_s"]-t0)*1000.0,"T0_ms":(float(ts["T0"])-t0)*1000.0,"T6_ms":t6,"target_to_T6_ms":t6-target,"T7_ms":(float(ts["T7"])-t0)*1000.0,"client_return_ms":(float(result["completed_monotonic_s"])-t0)*1000.0,"issue_to_T6_ms":(float(ts["T6"])-promotion["client_issue_s"])*1000.0,"T6_to_client_return_ms":(float(result["completed_monotonic_s"])-float(ts["T6"]))*1000.0,"arrival_minus_T6_ms":arrival_ms-t6,"internal_ready_before_resume":not internal_ready_miss(arrival_ms,t6),"cpu_to_gpu_ms":result["cpu_to_gpu_transfer_ms"],"external_tokens_loaded":result["external_tokens_loaded"],"model_forward_count":result["model_forward_count"],"generated_token_count":result["generated_token_count"],"mixed_prefetch_model_step_count":result["mixed_prefetch_model_step_count"],"status":result["status"]})
        intervals = [(float(row["T0_ms"]),float(row["T6_ms"])) for row in native_rows]
        max_internal, _ = interval_concurrency(intervals) if intervals else (0,0.0)
        issue_intervals = [(float(row["client_issue_ms"]),float(row["client_return_ms"])) for row in native_rows]
        max_http, _ = interval_concurrency(issue_intervals) if issue_intervals else (0,0.0)
        abc_first = [float(row["first_token_ms"]) for row in session_rows]; abc_finish = [float(row["request_finished_ms"]) for row in session_rows]
        abc_eff = [float(row["effective_resume_latency_ms"]) for row in session_rows]; abc_complete = [float(row["effective_completion_latency_ms"]) for row in session_rows]
        pre = [(float(row["scheduler_admission_ms"]),float(row["first_token_ms"])) for row in session_rows]; max_pre,_ = interval_concurrency(pre)
        x_duration=(x_finish_s-x_first_s)*1000.0; x_tpot=x_duration/255.0
        x_active = x_first_s <= t0 and all(float(row["request_finished_ms"])/1000.0+t0 <= x_finish_s for row in session_rows)
        native_ok = all(row["status"]=="ok" and row["external_tokens_loaded"]==4096 and row["model_forward_count"]==0 and row["generated_token_count"]==0 and row["mixed_prefetch_model_step_count"]==0 for row in native_rows)
        token_ok=all(row["generated_token_count"]==TOKENS for row in session_rows)
        local_ok = source_local + source_external == len(SESSIONS) * 4096.0
        all_first=[float(row["first_token_ms"]) for row in session_rows]; all_finish=[float(row["request_finished_ms"]) for row in session_rows]; all_eff=[float(row["effective_resume_latency_ms"]) for row in session_rows]; all_complete=[float(row["effective_completion_latency_ms"]) for row in session_rows]; def_rows=[row for row in session_rows if row["session"] in ("D","E","F")]
        run={"policy":policy,"repetition":repetition,"ABCDEF_P50_effective_resume_ms":percentile_linear(all_eff,50),"ABCDEF_P95_effective_resume_ms":percentile_linear(all_eff,95),"ABCDEF_max_effective_resume_ms":max(all_eff),"ABC_P95_effective_resume_ms":percentile_linear(abc_eff,95),"DEF_P95_effective_resume_ms":percentile_linear([float(r["effective_resume_latency_ms"]) for r in def_rows],95),"ABCDEF_P95_completion_ms":percentile_linear(all_complete,95),"ABCDEF_decode_tok_s":decode_throughput_tok_s(372,all_first,all_finish),"X_decode_duration_ms":x_duration,"X_TPOT_ms":x_tpot,"X_slowdown":1.0,"combined_decode_tok_s":627.0/((max(max(all_finish)/1000.0+t0,x_finish_s)-min(min(all_first)/1000.0+t0,x_first_s))),"full_makespan_ms":max(all_finish)-300.0,"ABC_memory_time_mib_s":sum(float(r["gpu_ready_residency_ms"]) for r in session_rows if r["session"] in ("A","B","C"))*112.0/1000.0,"DEF_memory_time_mib_s":sum(float(r["gpu_ready_residency_ms"]) for r in def_rows)*112.0/1000.0,"total_memory_time_mib_s":sum(float(row["gpu_ready_residency_ms"]) for row in session_rows)*112.0/1000.0,"internal_ready_miss_count":sum(not row["internal_ready_before_resume"] for row in native_rows),"ABC_internal_ready_miss_count":sum(not row["internal_ready_before_resume"] for row in native_rows if row["session"] in ("A","B","C")),"DEF_internal_ready_miss_count":sum(not row["internal_ready_before_resume"] for row in native_rows if row["session"] in ("D","E","F")),"max_pre_first_concurrency":max_pre,"total_gate_wait_ms":0.0,"local_hit_validation":"PASS" if local_ok else "FAIL","token_validation":"PASS" if token_ok else "FAIL","background_validation":"PASS" if x_active else "FAIL","gate_validation":"PASS" if max_pre <=2 else "FAIL","native_validation":"PASS" if native_ok else "FAIL","max_outstanding_prefetch_http_calls":max_http,"max_overlapping_internal_prefetch_intervals":max_internal,"source_local_hit_tokens_total":source_local,"source_external_transfer_tokens_total":source_external}
        background={"policy":policy,"repetition":repetition,"X_first_token_ms":(x_first_s-t0)*1000.0,"X_finished_ms":(x_finish_s-t0)*1000.0,"X_decode_duration_ms":x_duration,"X_TPOT_ms":x_tpot,"X_slowdown":1.0,"active_validation":run["background_validation"]}
        return native_rows,session_rows,trace,run,background,snapshots
    finally:
        transport.completion_payload=original_payload; gate_config._active_gate_limit=None; base.stop_server(proc)


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--stage",choices=("integration","full"),required=True); parser.add_argument("--policy", choices=POLICIES); args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True); OUT.joinpath("server_logs").mkdir(exist_ok=True)
    prompts=make_prompts_12()
    policies=("CapacityAwareAsync40-K2",) if args.stage=="integration" else ((args.policy,) if args.policy else POLICIES)
    native=[]; sessions=[]; traces=[]; runs=[]; backgrounds=[]; snapshots=[]
    reps=(99,) if args.stage=="integration" else range(3)
    for policy in policies:
        for rep in reps:
            print(f"RUN {policy} {rep}",flush=True)
            n,s,t,r,b,shots=policy_run(policy,rep,prompts); native+=n;sessions+=s;traces+=t;runs.append(r);backgrounds.append(b);snapshots.extend({"policy":policy,"repetition":rep,**x} for x in shots)
    if args.stage=="integration":
        write_csv(OUT/"capacity_decisions.csv", CAPACITY_DECISIONS); write_csv(OUT/"session_metrics.csv",sessions); write_csv(OUT/"async_prefetch_metrics.csv",native); write_csv(OUT/"scheduler_trace.csv",traces); write_csv(OUT/"run_metrics.csv",runs)
        (OUT/"capacity_metric_sanity.json").write_text(json.dumps(CAPACITY_SANITY,indent=2))
        source_local=runs[0]["source_local_hit_tokens_total"]; source_external=runs[0]["source_external_transfer_tokens_total"]
        waits=[r["session"] for r in sessions if r["WAITING_FOR_REMOTE_KVS_seen"]]
        q=[float(r["capacity_query_latency_ms"]) for r in CAPACITY_DECISIONS]; late=[float(r["issue_lateness_ms"]) for r in CAPACITY_DECISIONS if r["issue_lateness_ms"]!="NA"]
        issue=[float(r["issue_to_T6_ms"]) for r in native]; target=[float(r["target_to_T6_ms"]) for r in native]
        pct=lambda x,p: percentile_linear(x,p)
        all_usage=[r["kv_usage_fraction"] for r in CAPACITY_SANITY]+[r["kv_usage_fraction"] for r in CAPACITY_DECISIONS]
        sane=all(0<=v<=1.05 for v in all_usage) and max(all_usage)-min(all_usage)>1e-6
        experiment_valid=len(CAPACITY_DECISIONS)==12 and sane and runs[0]["token_validation"]=="PASS" and runs[0]["native_validation"]=="PASS" and runs[0]["gate_validation"]=="PASS" and runs[0]["background_validation"]=="PASS" and source_local+source_external==49152
        misses=sum(r["promotion_outcome"]=="INTERNAL_READY_MISS" for r in sessions); remote=sum(r["promotion_outcome"]=="PROMOTED_REMOTE_WAIT" for r in sessions); skips=sum(r["promotion_decision"]=="CAPACITY_SKIP" for r in sessions)
        result="D" if not experiment_valid else ("C" if not sane else ("B" if remote else ("A" if skips else "B")))
        nxt={"A":"CAPACITY_AWARE_FULL_MATRIX","B":"RESIDENCY_RESERVATION","C":"ENGINE_LOCAL_CAPACITY_SIGNAL","D":"FIX_INSTRUMENTATION"}[result]
        report={"experiment_valid":experiment_valid,"capacity_metric_sane":sane,"capacity_decisions_persisted":len(CAPACITY_DECISIONS),"promote_count":12-skips,"capacity_skip_count":skips,"internal_ready_miss_count":misses,"ready_but_remote_wait_count":remote,"ready_but_remote_wait_sessions":waits,"source_local_tokens":source_local,"source_external_tokens":source_external,"source_total_tokens":source_local+source_external,"local_fraction":source_local/49152,"external_fraction":source_external/49152,"source_conservation_pass":source_local+source_external==49152,"capacity_query_latency":{"median":statistics.median(q),"P95":pct(q,95),"max":max(q)},"issue_lateness":{"median":statistics.median(late),"P95":pct(late,95),"max":max(late)},"issue_to_T6":{"median":statistics.median(issue),"P95":pct(issue,95),"max":max(issue)},"target_to_T6":{"median":statistics.median(target),"P95":pct(target,95),"max":max(target)},"K2_valid":runs[0]["gate_validation"],"background_valid":runs[0]["background_validation"],"native_valid":runs[0]["native_validation"],"token_valid":runs[0]["token_validation"],"result":result,"next_direction":nxt}
        (OUT/"validation_report.json").write_text(json.dumps(report,indent=2)); (OUT/"integration_gate.json").write_text(json.dumps({"experiment_valid":experiment_valid,"policy_pass":result=="A","report":report},indent=2)); print("EXPERIMENT_VALID="+str(experiment_valid)+" RESULT="+result); return
    batch = OUT / "batches" / args.policy if args.policy else OUT
    batch.mkdir(parents=True, exist_ok=True)
    write_csv(batch/"async_prefetch_metrics.csv",native);write_csv(batch/"session_metrics.csv",sessions);write_csv(batch/"scheduler_trace.csv",traces);write_csv(batch/"run_metrics.csv",runs);write_csv(batch/"background_metrics.csv",backgrounds)
    (batch/"metric_snapshots.json").write_text(json.dumps(snapshots,indent=2));print("FULL_DONE")


if __name__=="__main__": main()
