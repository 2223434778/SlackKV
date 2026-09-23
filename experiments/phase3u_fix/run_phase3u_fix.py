from __future__ import annotations

import argparse
import csv
import json
import statistics
import threading
import time
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Event
from typing import Any

from experiments.phase2d import run_phase2d as runtime
from experiments.phase2d.helpers import gpu_to_cpu_offload_bytes, parse_prometheus_samples, source_tokens_delta
from experiments.phase3f.native_client import native_prefetch
from experiments.phase3f.scheduler import latest_feasible_start, merge_blackouts
from experiments.phase3g import run_phase3g as transport
from experiments.phase3g_full import run_phase3g_full as base
from experiments.phase3g_full.metrics import decode_throughput_tok_s, interval_concurrency, percentile_linear as _percentile_linear
from experiments.phase3h import run_phase3h as gate_config
from experiments.phase3n.run_phase3n import send_x_background
from experiments.phase3r_fix.helpers import ASYNC_SERVICE_MS, DEADLINES_MS, effective_latency_ms, gpu_ready_residency_ms, internal_ready_miss
from experiments.phase3t.helpers import ReservationLedger

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase3u_fix"
SESSIONS = tuple("ABCDEFGHIJKL")
POLICIES = ("OnDemand-K2","Blocking70-K2","NaiveAsync40-K2","ReservationLedger2-Async40-K2")
RESERVATION_DECISIONS: list[dict[str, Any]] = []
RESERVATION_EVENTS: list[dict[str, Any]] = []
TOKENS = 32
BYTES = 117_440_512
GUARD_MS = 110.0
X_BASELINE_MS: float | None = None
BLOCKING_METRICS: list[dict[str, Any]] = []
BLOCKING_WAIT_CALLS = 0


def percentile_linear(values, percentile):
    return "NA" if not values else _percentile_linear(values, percentile)


def configured_service_budget(policy: str) -> float | None:
    if policy == "Blocking70-K2":
        return 70.0
    if policy in ("NaiveAsync40-K2", "ReservationLedger2-Async40-K2"):
        return 40.0
    return None


def policy_semantics(policy: str) -> dict[str, Any]:
    return {"service_ms": configured_service_budget(policy), "synchronous": policy == "Blocking70-K2", "block_on_prefetch": policy == "Blocking70-K2", "lease": policy == "ReservationLedger2-Async40-K2", "promote": policy != "OnDemand-K2"}


def maybe_wait_for_prefetch(ready_event, block_on_prefetch: bool, timeout_s: float = 300) -> bool:
    if not block_on_prefetch:
        return False
    if ready_event is None:
        raise AssertionError("blocking resume requires a ready event")
    global BLOCKING_WAIT_CALLS
    BLOCKING_WAIT_CALLS += 1
    if not ready_event.wait(timeout=timeout_s):
        raise TimeoutError("timed out waiting for blocking native prefetch")
    return True


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


def parse_lease_trace(path: Path, repetition: int, t0: float) -> list[dict[str, Any]]:
    rows=[]; marker="SLACKKV_LEASE_TRACE "
    for line in path.read_text(encoding="utf-8",errors="ignore").splitlines():
      if marker not in line: continue
      raw=json.loads(line.split(marker,1)[1]); rows.append({"repetition":repetition,**raw,"timestamp_ms":(float(raw["timestamp_monotonic"])-t0)*1000.0})
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


def send_resume(session: str, prompts: dict[str, list[int]], t0: float, *, ready_event: Event | None = None, block_on_prefetch: bool = False) -> dict[str, Any]:
    wait_until(t0 + DEADLINES_MS[session] / 1000.0)
    reached_s = time.monotonic()
    wait_start_s = time.monotonic()
    wait_called = maybe_wait_for_prefetch(ready_event, block_on_prefetch)
    wait_end_s = time.monotonic()
    send_s = time.monotonic()
    if block_on_prefetch:
        BLOCKING_METRICS.append({"session":session,"deadline_reached_ms":(reached_s-t0)*1000.0,"wait_called":wait_called,"wait_start_ms":(wait_start_s-t0)*1000.0,"wait_end_ms":(wait_end_s-t0)*1000.0,"blocking_wait_ms":max(0.0,(wait_end_s-wait_start_s)*1000.0),"resume_client_send_ms":(send_s-t0)*1000.0})
    # Intentionally no future/result/ready wait in Async40: the real request
    # observes the scheduler-visible T6 cache state itself.
    response = transport.send(prompts[session.lower()] + prompts["suffix"], stream=True, max_tokens=TOKENS, trace=True, session=session, gate=True, normal=False)
    if len(response["token_ids"]) != TOKENS:
        raise RuntimeError(f"{session} generated {len(response['token_ids'])}, expected {TOKENS}")
    return response


def schedule_promotions(policy, prompts, t0, ready, output, futures) -> None:
    service=configured_service_budget(policy); assert service is not None; pending=set(SESSIONS); blackouts=merge_blackouts(DEADLINES_MS,GUARD_MS); ledger=ReservationLedger()
    with ThreadPoolExecutor(max_workers=3) as pool:
      while pending:
        now=(time.monotonic()-t0)*1000.0; jobs=[(x,DEADLINES_MS[x]) for x in pending]; target=latest_feasible_start(now,jobs,blackouts,service)
        if target-now>0.5: time.sleep((target-now)/1000.0); continue
        session=min(pending,key=lambda x:(DEADLINES_MS[x],x))
        decision_time=(time.monotonic()-t0)*1000.0
        expired=ledger.expire(decision_time)
        for reservation,before,after in expired:
          RESERVATION_EVENTS.append({"session":reservation.session,"timestamp_ms":decision_time,"event":"RESERVATION_EXPIRED","reserved_blocks_before":before,"reserved_blocks_after":after,"reservation_start_ms":reservation.start_ms,"reservation_end_ms":reservation.end_ms})
        before_blocks=ledger.reserved_blocks; before_count=len(ledger.active)
        decision=ledger.decide() if policy=="ReservationLedger2-Async40-K2" else "PROMOTE"
        record={"session":session,"cluster":1+(ord(session)-ord("A"))//4,
                "deadline_ms":DEADLINES_MS[session],"target_start_ms":target,
                "decision_time_ms":decision_time,"active_reservation_count_before":before_count,"reserved_blocks_before":before_blocks,"required_blocks":256,"budget_blocks":512,"decision":decision,"reservation_start_ms":target if decision=="RESERVATION_PROMOTE" else "NA","reservation_end_ms":DEADLINES_MS[session]+100.0 if decision=="RESERVATION_PROMOTE" else "NA","reason":"RESERVATION_BUDGET_AVAILABLE" if decision=="RESERVATION_PROMOTE" else ("RESERVATION_BUDGET_FULL" if decision=="RESERVATION_SKIP" else "POLICY_PROMOTE"),"actual_issue_ms":"NA","issue_lateness_ms":"NA"}
        RESERVATION_DECISIONS.append(record)
        if decision in ("RESERVATION_PROMOTE","PROMOTE"):
          if decision=="RESERVATION_PROMOTE":
            reservation=ledger.grant(session,target,DEADLINES_MS[session]); RESERVATION_EVENTS.append({"session":session,"timestamp_ms":decision_time,"event":"RESERVATION_GRANTED","reserved_blocks_before":before_blocks,"reserved_blocks_after":ledger.reserved_blocks,"reservation_start_ms":reservation.start_ms,"reservation_end_ms":reservation.end_ms})
          issue=time.monotonic(); record["actual_issue_ms"]=(issue-t0)*1000.0; record["issue_lateness_ms"]=record["actual_issue_ms"]-target
          kwargs={"lease_key":session} if decision=="RESERVATION_PROMOTE" else {}
          if policy=="Blocking70-K2":
            result=native_prefetch(prompts[session.lower()],**kwargs); output[session]={"target_start_ms":target,"client_issue_s":issue,"decision":decision,"result":result}; ready[session].set()
          else:
            future=pool.submit(native_prefetch,prompts[session.lower()],**kwargs); futures[session]=future; output[session]={"target_start_ms":target,"client_issue_s":issue,"future":future,"decision":decision}
        else: output[session]={"target_start_ms":target,"decision":decision}
        pending.remove(session)
      for session,future in futures.items(): output[session]["result"]=future.result(timeout=300); ready[session].set()


def policy_run(policy: str, repetition: int, prompts: dict[str, list[int]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    global BLOCKING_WAIT_CALLS
    BLOCKING_METRICS.clear()
    BLOCKING_WAIT_CALLS = 0
    label = f"{policy.lower().replace('-', '_')}_{repetition}"
    base.OUT = OUT
    original_payload = transport.completion_payload
    gate_config._active_gate_limit = 2
    transport.completion_payload = gate_config._payload_with_gate_limit
    proc = base.start_server(label)
    snapshots: list[dict[str, Any]] = []
    try:
        runtime.wait_ready(proc)
        previous = capture("ready", snapshots)
        offloads: dict[str, float] = {}
        for session in SESSIONS:
            base.send(prompts[session.lower()], stream=False, max_tokens=1, normal=True)
            current = capture(f"after_{session}_first", snapshots)
            offloads[session] = gpu_to_cpu_offload_bytes(previous, current); previous = current
        if any(value != BYTES for value in offloads.values()):
            raise RuntimeError(f"offload mismatch: {offloads}")
        time.sleep(1.0)
        for index in range(1, 5):
            base.send(prompts[f"pressure_{index}"], stream=False, max_tokens=1, normal=False)
        capture("after_pressure", snapshots)
        x_first = Event()
        RESERVATION_DECISIONS.clear(); RESERVATION_EVENTS.clear()
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
            semantics=policy_semantics(policy)
            resume = {
                s: pool.submit(
                    send_resume,
                    s,
                    prompts,
                    t0,
                    ready_event=ready[s] if semantics["block_on_prefetch"] else None,
                    block_on_prefetch=semantics["block_on_prefetch"],
                )
                for s in SESSIONS
            }
            results = {s: item.result(timeout=600) for s, item in resume.items()}
            if promotion_future is not None: promotion_future.result(timeout=600)
            x = x_future.result(timeout=900)
        after = capture("after_workload", snapshots)
        trace = parse_trace(log_path, policy, repetition, t0)
        lease_trace = parse_lease_trace(log_path, repetition, t0)
        for row in RESERVATION_DECISIONS: row.setdefault("repetition",repetition)
        for row in RESERVATION_EVENTS: row.setdefault("repetition",repetition)
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
            decision=promotion.get("decision","ONDEMAND") if promotion else "ONDEMAND"
            events=[row for row in lease_trace if row.get("lease_key")==session]; names=[row["event"] for row in events]
            lease_ok=names==["LEASE_ACQUIRED","LEASE_HELD_AFTER_PREFETCH_FREE","LEASE_RELEASE_READY","LEASE_RELEASED"]
            if decision in ("ONDEMAND","RESERVATION_SKIP"): outcome=decision
            elif decision=="RESERVATION_PROMOTE" and not lease_ok: outcome="LEASE_ERROR"
            elif float(t6_ms)>arrival_ms: outcome="PROMOTION_INTERNAL_READY_MISS"
            elif remote_wait: outcome="PROMOTION_REMOTE_WAIT"
            else: outcome="PROMOTION_USEFUL"
            session_rows.append({"policy":policy,"repetition":repetition,"session":session,"cluster":1+(ord(session)-ord('A'))//4,"deadline_ms":DEADLINES_MS[session],"reservation_decision":decision,"target_start_ms":promotion.get("target_start_ms","NA") if promotion else "NA","client_issue_ms":"NA" if promotion is None or "client_issue_s" not in promotion else (promotion["client_issue_s"]-t0)*1000.0,"scheduler_arrival_ms":arrival_ms,"scheduler_admission_ms":admission_ms,"first_token_ms":first_ms,"finish_ms":finish_ms,"request_finished_ms":finish_ms,"effective_resume_latency_ms":effective_latency_ms(first_ms, DEADLINES_MS[session]),"effective_completion_latency_ms":finish_ms-DEADLINES_MS[session],"remote_wait_seen":remote_wait,"promotion_outcome":outcome,"T6_ms":t6_ms,"arrival_minus_T6_ms":"NA" if t6_ms == "NA" else arrival_ms-float(t6_ms),"gpu_ready_residency_ms":0.0 if t6_ms == "NA" else gpu_ready_residency_ms(admission_ms,float(t6_ms)),"lease_acquired":"LEASE_ACQUIRED" in names,"lease_released":"LEASE_RELEASED" in names,"generated_token_count":len(results[session]["token_ids"])})
            if result is not None:
                ts = result.get("phase3m_timestamps") or {}
                target=promotion["target_start_ms"]; t6=(float(ts["T6"])-t0)*1000.0
                native_rows.append({"policy":policy,"repetition":repetition,"session":session,"service_budget_ms":ASYNC_SERVICE_MS,"target_start_ms":target,"client_issue_ms":(promotion["client_issue_s"]-t0)*1000.0,"target_to_issue_ms":(promotion["client_issue_s"]-t0)*1000.0-target,"T0_ms":(float(ts["T0"])-t0)*1000.0,"T6_ms":t6,"target_to_T6_ms":t6-target,"T7_ms":(float(ts["T7"])-t0)*1000.0,"client_return_ms":(float(result["completed_monotonic_s"])-t0)*1000.0,"issue_to_T6_ms":(float(ts["T6"])-promotion["client_issue_s"])*1000.0,"T6_to_client_return_ms":(float(result["completed_monotonic_s"])-float(ts["T6"]))*1000.0,"arrival_minus_T6_ms":arrival_ms-t6,"internal_ready_before_resume":not internal_ready_miss(arrival_ms,t6),"cpu_to_gpu_ms":result["cpu_to_gpu_transfer_ms"],"external_tokens_loaded":result["external_tokens_loaded"],"transfer_bytes":result["transfer_bytes"],"model_forward_count":result["model_forward_count"],"generated_token_count":result["generated_token_count"],"mixed_prefetch_model_step_count":result["mixed_prefetch_model_step_count"],"status":result["status"]})
        for row in native_rows:
            row["service_budget_ms"] = configured_service_budget(policy)
        native_by_session={row["session"]:row for row in native_rows}
        for row in BLOCKING_METRICS:
            promotion=native_by_session[row["session"]]
            row.update({"policy":policy,"repetition":repetition,"deadline_ms":DEADLINES_MS[row["session"]],"prefetch_target_ms":promotion["target_start_ms"],"prefetch_issue_ms":promotion["client_issue_ms"],"prefetch_T6_ms":promotion["T6_ms"],"prefetch_client_return_ms":promotion["client_return_ms"],"send_after_prefetch_return":float(row["resume_client_send_ms"])+0.5>=float(promotion["client_return_ms"]),"validation":"PASS" if float(row["resume_client_send_ms"])+0.5>=float(promotion["client_return_ms"]) else "FAIL"})
        intervals = [(float(row["T0_ms"]),float(row["T6_ms"])) for row in native_rows]
        max_internal, _ = interval_concurrency(intervals) if intervals else (0,0.0)
        issue_intervals = [(float(row["client_issue_ms"]),float(row["client_return_ms"])) for row in native_rows]
        max_http, _ = interval_concurrency(issue_intervals) if issue_intervals else (0,0.0)
        abc_first = [float(row["first_token_ms"]) for row in session_rows]; abc_finish = [float(row["request_finished_ms"]) for row in session_rows]
        abc_eff = [float(row["effective_resume_latency_ms"]) for row in session_rows]; abc_complete = [float(row["effective_completion_latency_ms"]) for row in session_rows]
        pre = [(float(row["scheduler_admission_ms"]),float(row["first_token_ms"])) for row in session_rows]; max_pre,_ = interval_concurrency(pre)
        x_duration=(x_finish_s-x_first_s)*1000.0; x_tpot=x_duration/255.0
        l_first=event(trace,"L","FIRST_TOKEN")
        x_active = x_first_s <= t0 and l_first is not None and x_finish_s >= float(l_first["timestamp_monotonic"])
        native_ok = all(row["status"]=="ok" and row["external_tokens_loaded"]==4096 and row["model_forward_count"]==0 and row["generated_token_count"]==0 and row["mixed_prefetch_model_step_count"]==0 for row in native_rows)
        token_ok=all(row["generated_token_count"]==TOKENS for row in session_rows)
        local_ok = source_local + source_external == len(SESSIONS) * 4096.0
        all_first=[float(row["first_token_ms"]) for row in session_rows]; all_finish=[float(row["request_finished_ms"]) for row in session_rows]; all_eff=[float(row["effective_resume_latency_ms"]) for row in session_rows]; all_complete=[float(row["effective_completion_latency_ms"]) for row in session_rows]; def_rows=[row for row in session_rows if row["session"] in ("D","E","F")]
        run={"policy":policy,"repetition":repetition,"ABCDEF_P50_effective_resume_ms":percentile_linear(all_eff,50),"ABCDEF_P95_effective_resume_ms":percentile_linear(all_eff,95),"ABCDEF_max_effective_resume_ms":max(all_eff),"ABC_P95_effective_resume_ms":percentile_linear(abc_eff,95),"DEF_P95_effective_resume_ms":percentile_linear([float(r["effective_resume_latency_ms"]) for r in def_rows],95),"ABCDEF_P95_completion_ms":percentile_linear(all_complete,95),"ABCDEF_decode_tok_s":decode_throughput_tok_s(372,all_first,all_finish),"X_decode_duration_ms":x_duration,"X_TPOT_ms":x_tpot,"X_slowdown":1.0,"combined_decode_tok_s":627.0/((max(max(all_finish)/1000.0+t0,x_finish_s)-min(min(all_first)/1000.0+t0,x_first_s))),"full_makespan_ms":max(all_finish)-300.0,"ABC_memory_time_mib_s":sum(float(r["gpu_ready_residency_ms"]) for r in session_rows if r["session"] in ("A","B","C"))*112.0/1000.0,"DEF_memory_time_mib_s":sum(float(r["gpu_ready_residency_ms"]) for r in def_rows)*112.0/1000.0,"total_memory_time_mib_s":sum(float(row["gpu_ready_residency_ms"]) for row in session_rows)*112.0/1000.0,"internal_ready_miss_count":sum(not row["internal_ready_before_resume"] for row in native_rows),"ABC_internal_ready_miss_count":sum(not row["internal_ready_before_resume"] for row in native_rows if row["session"] in ("A","B","C")),"DEF_internal_ready_miss_count":sum(not row["internal_ready_before_resume"] for row in native_rows if row["session"] in ("D","E","F")),"max_pre_first_concurrency":max_pre,"total_gate_wait_ms":0.0,"local_hit_validation":"PASS" if local_ok else "FAIL","token_validation":"PASS" if token_ok else "FAIL","background_validation":"PASS" if x_active else "FAIL","gate_validation":"PASS" if max_pre <=2 else "FAIL","native_validation":"PASS" if native_ok else "FAIL","max_outstanding_prefetch_http_calls":max_http,"max_overlapping_internal_prefetch_intervals":max_internal,"source_local_hit_tokens_total":source_local,"source_external_transfer_tokens_total":source_external}
        if X_BASELINE_MS is None:
            raise RuntimeError("X-only baseline missing")
        run["X_slowdown"] = x_duration / X_BASELINE_MS
        run.update({"blocking_wait_call_count":BLOCKING_WAIT_CALLS,"blocking_wait_total_ms":sum(float(x["blocking_wait_ms"]) for x in BLOCKING_METRICS),"blocking_wait_max_ms":max([float(x["blocking_wait_ms"]) for x in BLOCKING_METRICS],default=0.0),"blocking_semantics_validation":"PASS" if policy!="Blocking70-K2" or (BLOCKING_WAIT_CALLS==12 and all(x["validation"]=="PASS" for x in BLOCKING_METRICS)) else "FAIL","policy_semantics_validation":"PASS"})
        promoted=[row for row in session_rows if row["reservation_decision"] in ("RESERVATION_PROMOTE","PROMOTE")]; skipped=[row for row in session_rows if row["reservation_decision"]=="RESERVATION_SKIP"]; fallback=[row for row in session_rows if row["reservation_decision"] in ("RESERVATION_SKIP","ONDEMAND")]
        useful=sum(row["promotion_outcome"]=="PROMOTION_USEFUL" for row in promoted); remote=sum(row["promotion_outcome"]=="PROMOTION_REMOTE_WAIT" for row in promoted); misses=sum(row["promotion_outcome"]=="PROMOTION_INTERNAL_READY_MISS" for row in promoted)
        acquired=sum(row["event"]=="LEASE_ACQUIRED" for row in lease_trace); released=sum(row["event"]=="LEASE_RELEASED" for row in lease_trace)
        reservation_policy=policy=="ReservationLedger2-Async40-K2"
        run.update({"reservation_promote_count":sum(x["reservation_decision"]=="RESERVATION_PROMOTE" for x in session_rows),"reservation_skip_count":sum(x["reservation_decision"]=="RESERVATION_SKIP" for x in session_rows),"promotion_useful_count":useful,"promotion_remote_wait_count":remote,"promotion_internal_ready_miss_count":misses,"promotion_useful_rate":useful/len(promoted) if promoted else "NA","active_reservation_max_blocks":max([0]+[int(row["reserved_blocks_after"]) for row in RESERVATION_EVENTS]),"lease_acquire_count":acquired,"lease_release_count":released,"lease_leak_count":acquired-released,"local_source_tokens":source_local,"external_source_tokens":source_external,"local_fraction":source_local/49152.0,"external_fraction":source_external/49152.0,"wasted_promotion_mib":remote*112,"native_promotion_mib":len(promoted)*112,"P50_effective_resume_ms":percentile_linear(all_eff,50),"P95_effective_resume_ms":percentile_linear(all_eff,95),"max_effective_resume_ms":max(all_eff),"promoted_P50_ms":percentile_linear([float(x["effective_resume_latency_ms"]) for x in promoted],50) if promoted else "NA","promoted_P95_ms":percentile_linear([float(x["effective_resume_latency_ms"]) for x in promoted],95) if promoted else "NA","skipped_P50_ms":percentile_linear([float(x["effective_resume_latency_ms"]) for x in skipped],50),"skipped_P95_ms":percentile_linear([float(x["effective_resume_latency_ms"]) for x in skipped],95),"cluster1_P95_ms":percentile_linear([float(x["effective_resume_latency_ms"]) for x in session_rows if x["cluster"]==1],95),"cluster2_P95_ms":percentile_linear([float(x["effective_resume_latency_ms"]) for x in session_rows if x["cluster"]==2],95),"cluster3_P95_ms":percentile_linear([float(x["effective_resume_latency_ms"]) for x in session_rows if x["cluster"]==3],95),"promotion_memory_time_mib_s":sum(float(x["gpu_ready_residency_ms"]) for x in promoted)*112/1000.0,"target_to_issue_median_ms":statistics.median(float(x["target_to_issue_ms"]) for x in native_rows) if native_rows else "NA","target_to_issue_max_ms":max((float(x["target_to_issue_ms"]) for x in native_rows),default="NA"),"target_to_T6_median_ms":statistics.median(float(x["target_to_T6_ms"]) for x in native_rows) if native_rows else "NA","target_to_T6_max_ms":max((float(x["target_to_T6_ms"]) for x in native_rows),default="NA"),"lease_validation":"PASS" if (not reservation_policy or acquired==released==len(promoted)) else "FAIL","reservation_validation":"PASS" if (not reservation_policy or (len(promoted)+len(skipped)==12 and max([0]+[int(x["reserved_blocks_after"]) for x in RESERVATION_EVENTS])<=512)) else "FAIL","source_conservation_validation":"PASS" if local_ok else "FAIL"})
        run["fallback_P50_ms"]=percentile_linear([float(x["effective_resume_latency_ms"]) for x in fallback],50) if fallback else "NA"
        run["fallback_P95_ms"]=percentile_linear([float(x["effective_resume_latency_ms"]) for x in fallback],95) if fallback else "NA"
        run["skipped_P50_ms"]=percentile_linear([float(x["effective_resume_latency_ms"]) for x in skipped],50) if skipped else "NA"
        run["skipped_P95_ms"]=percentile_linear([float(x["effective_resume_latency_ms"]) for x in skipped],95) if skipped else "NA"
        background={"policy":policy,"repetition":repetition,"X_first_token_ms":(x_first_s-t0)*1000.0,"X_finished_ms":(x_finish_s-t0)*1000.0,"X_decode_duration_ms":x_duration,"X_TPOT_ms":x_tpot,"X_slowdown":x_duration/X_BASELINE_MS,"active_validation":run["background_validation"]}
        return native_rows,session_rows,trace,lease_trace,run,background,snapshots,list(RESERVATION_DECISIONS),list(RESERVATION_EVENTS),list(BLOCKING_METRICS)
    finally:
        transport.completion_payload=original_payload; gate_config._active_gate_limit=None; base.stop_server(proc)


def run_x_only() -> None:
    rows=[]
    prompts=base.make_prompts()
    base.OUT=OUT
    for repetition in range(3):
        proc=base.start_server(f"x_only_rep{repetition}")
        try:
            runtime.wait_ready(proc)
            first=threading.Event()
            result=send_x_background(prompts["pressure_1"], first)
            token_count=len(result.get("token_ids", []))
            duration=(float(result["completed_monotonic_s"])-float(result["first_monotonic_s"]))*1000.0
            rows.append({"repetition":repetition,"X_decode_duration_ms":duration,"X_TPOT_ms":duration/255.0,"generated_token_count":token_count,"token_validation":"PASS" if token_count==256 else "FAIL"})
        finally:
            base.stop_server(proc)
    write_csv(OUT/"x_only_baseline.csv",rows)
    if not all(row["token_validation"]=="PASS" for row in rows):
        raise RuntimeError("X-only baseline token validation failed")


def main() -> None:
    global X_BASELINE_MS
    parser=argparse.ArgumentParser(); parser.add_argument("--stage",choices=("x-only","stage-a","stage-b"),required=True); args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True); OUT.joinpath("server_logs").mkdir(exist_ok=True)
    if args.stage == "x-only":
        run_x_only()
        return
    baseline_rows=list(csv.DictReader((OUT/"x_only_baseline.csv").open(newline="")))
    if len(baseline_rows)!=3 or any(row["token_validation"]!="PASS" for row in baseline_rows):
        raise RuntimeError("Stage 0 requires three valid X-only runs")
    X_BASELINE_MS=statistics.median(float(row["X_decode_duration_ms"]) for row in baseline_rows)
    if args.stage=="stage-b":
        gate=json.loads((OUT/"integration_gate.json").read_text())
        if not gate.get("pass"):
            raise RuntimeError("Stage B prohibited: Stage A integration gate did not pass")
    prompts=make_prompts_12()
    policies=POLICIES
    native=[]; sessions=[]; traces=[]; leases=[]; runs=[]; backgrounds=[]; snapshots=[]; decisions=[]; reservation_events=[]; blocking=[]
    reps=(900,) if args.stage=="stage-a" else range(3)
    for policy in policies:
        for rep in reps:
            print(f"RUN {policy} {rep}",flush=True)
            n,s,t,l,r,b,shots,d,e,bm=policy_run(policy,rep,prompts); native+=n;sessions+=s;traces+=t;leases+=l;runs.append(r);backgrounds.append(b);decisions+=d;reservation_events+=e;blocking+=bm;snapshots.extend({"policy":policy,"repetition":rep,**x} for x in shots)
    stage_name="stage_a" if args.stage=="stage-a" else "stage_b"
    if args.stage=="stage-a":
        write_csv(OUT/"stage_a_run_metrics.csv",runs)
        write_csv(OUT/"stage_a_blocking_metrics.csv",blocking)
        valid_by_policy={}
        for policy,run in zip(POLICIES,runs):
            native_p=[x for x in native if x["policy"]==policy]
            wait_calls=int(run["blocking_wait_call_count"])
            semantic=(policy!="Blocking70-K2" or (wait_calls==12 and len(native_p)==12 and all(float(x["service_budget_ms"])==70 for x in native_p) and all(x["validation"]=="PASS" for x in blocking) and int(run["max_outstanding_prefetch_http_calls"])<=1 and int(run["max_overlapping_internal_prefetch_intervals"])<=1 and int(run["lease_acquire_count"])==0 and int(run["lease_release_count"])==0))
            if policy=="OnDemand-K2": semantic=semantic and len(native_p)==0 and int(run["lease_acquire_count"])==0 and int(run["lease_release_count"])==0 and int(run["reservation_promote_count"])==0
            if policy=="NaiveAsync40-K2": semantic=semantic and len(native_p)==12 and wait_calls==0 and int(run["lease_acquire_count"])==0 and all(float(x["service_budget_ms"])==40 for x in native_p)
            if policy=="ReservationLedger2-Async40-K2": semantic=semantic and len(native_p)==6 and int(run["reservation_promote_count"])==6 and int(run["reservation_skip_count"])==6 and wait_calls==0 and int(run["lease_acquire_count"])==6 and int(run["lease_release_count"])==6 and int(run["lease_leak_count"])==0 and int(run["active_reservation_max_blocks"])<=512 and int(run["promotion_remote_wait_count"])==0 and int(run["promotion_internal_ready_miss_count"])==0 and all(float(x["service_budget_ms"])==40 for x in native_p)
            policy_sessions=[x for x in sessions if x["policy"]==policy]
            semantic=semantic and all(run[k]=="PASS" for k in ("token_validation","native_validation","gate_validation","background_validation","source_conservation_validation")) and len(policy_sessions)==12 and int(run["max_pre_first_concurrency"])<=2 and int(run["max_outstanding_prefetch_http_calls"]) <= (1 if policy=="Blocking70-K2" else 12)
            run["policy_semantics_validation"]="PASS" if semantic else "FAIL"
            valid_by_policy[policy]=semantic
        stage_a_pass=all(valid_by_policy.values())
        (OUT/"integration_gate.json").write_text(json.dumps({"pass":stage_a_pass,"policy_valid":valid_by_policy,"runs":runs,"blocking_metrics":blocking},indent=2))
        write_csv(OUT/"stage_a_native_prefetch_metrics.csv",native); write_csv(OUT/"stage_a_session_metrics.csv",sessions)
        print("STAGE_A_PASS="+str(stage_a_pass)); return
    write_csv(OUT/"reservation_decisions.csv",decisions); write_csv(OUT/"reservation_events.csv",reservation_events); write_csv(OUT/"lease_events.csv",leases); write_csv(OUT/"session_metrics.csv",sessions); write_csv(OUT/"async_prefetch_metrics.csv",native); write_csv(OUT/"scheduler_trace.csv",traces); write_csv(OUT/"run_metrics.csv",runs); write_csv(OUT/"background_metrics.csv",backgrounds); write_csv(OUT/"blocking_stage_b_metrics.csv",blocking); (OUT/"metric_snapshots.json").write_text(json.dumps(snapshots,indent=2))
    print("STAGE_B_DONE")


if __name__=="__main__": main()
