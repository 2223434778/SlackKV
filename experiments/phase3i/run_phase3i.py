from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from typing import Any

from experiments.phase3g import run_phase3g as transport
from experiments.phase3g_full import run_phase3g_full as base
from experiments.phase3g_full.metrics import (
    decode_throughput_tok_s,
    gate_wait_ms,
    gpu_ready_residency_ms,
    interval_concurrency,
    overlap,
    percentile_linear,
    sequence_hash,
    same_step_pairs,
)
from experiments.phase3f.helpers import write_csv
from experiments.phase3f.native_client import native_prefetch
from experiments.phase3f.scheduler import DEADLINE_MS, RESUME_GUARD_MS, P_SAFE_MS
from experiments.phase2d.helpers import source_tokens_delta, gpu_to_cpu_offload_bytes

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase3i"
SESSIONS = ("A", "B", "C")
ALL_POLICIES = (("ResumeAware-K3", None), ("FirstTokenGate-K1", 1), ("FirstTokenGate-K2", 2))
TOKENS = 32
X_TOKENS = 128
BYTES = 117_440_512
_original_payload = transport.completion_payload
_gate_limit: int | None = None
_all_snapshots: list[dict[str, Any]] = []


def payload_with_limit(*args: Any, **kwargs: Any) -> dict[str, Any]:
    payload = _original_payload(*args, **kwargs)
    if kwargs.get("gate") and _gate_limit is not None:
        payload["kv_transfer_params"]["slackkv_first_token_gate_limit"] = _gate_limit
    return payload


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty output: {path}")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def send_x(ids: list[int], first_event: Event) -> dict[str, Any]:
    payload = transport.completion_payload(ids, stream=True, max_tokens=X_TOKENS,
                                           trace=True, session="X", gate=False, normal=True)
    sent = time.monotonic()
    req = urllib.request.Request("http://127.0.0.1:8000/v1/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    token_ids: list[int] = []
    first = None
    with urllib.request.urlopen(req, timeout=600) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                continue
            choice = (json.loads(data).get("choices") or [{}])[0]
            tids = [int(x) for x in (choice.get("token_ids") or [])]
            if tids:
                token_ids.extend(tids)
                if first is None:
                    first = time.monotonic()
                    first_event.set()
            elif first is None and choice.get("text"):
                first = time.monotonic()
                first_event.set()
    done = time.monotonic()
    if first is None:
        raise RuntimeError("X produced no first token")
    return {"sent_monotonic_s": sent, "first_monotonic_s": first,
            "completed_monotonic_s": done, "token_ids": token_ids,
            "generated_token_count": len(token_ids)}


def first_event(rows: list[dict[str, Any]], session: str, name: str) -> dict[str, Any] | None:
    return next((row for row in rows if row.get("session") == session and row.get("event") == name), None)


def trace_rows(path: Path, policy: str, rep: int, t0: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    marker = "SLACKKV_RESUME_TRACE "
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if marker not in line:
            continue
        raw = json.loads(line.split(marker, 1)[1])
        ts = float(raw["timestamp_monotonic"])
        rows.append({"policy": policy, "repetition": rep, "session": raw.get("session"),
                     "request_id": raw.get("request_id"), "event": raw.get("event"),
                     "step_id": raw.get("step_id"), "timestamp_monotonic_raw": ts,
                     "timestamp_ms_from_t0": (ts - t0) * 1000.0,
                     "request_status": raw.get("request_status"), "num_running": raw.get("num_running"),
                     "num_waiting": raw.get("num_waiting"), "num_skipped_waiting": raw.get("num_skipped_waiting"),
                     "gate_limit": raw.get("gate_limit"),
                     "gate_owner_request_ids": json.dumps(raw.get("gate_owner_request_ids")),
                     "num_slackkv_pre_first_token": raw.get("num_slackkv_pre_first_token"),
                     "num_scheduled_requests": raw.get("num_scheduled_requests"),
                     "total_num_scheduled_tokens": raw.get("total_num_scheduled_tokens"),
                     "num_computed_tokens": raw.get("num_computed_tokens"),
                     "num_output_tokens": raw.get("num_output_tokens")})
    return rows


def native_rows(policy: str, rep: int, promotions: dict[str, dict[str, Any]], sessions: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = []
    blackouts = base.merge_blackouts({s: DEADLINE_MS[s] for s in sessions}, RESUME_GUARD_MS)
    for session, item in promotions.items():
        start = (item["sent_monotonic_s"] - item["t0"]) * 1000.0
        finish = (item["completed_monotonic_s"] - item["t0"]) * 1000.0
        conflict = any(start < end and finish > begin for begin, end in blackouts)
        rows.append({"policy": policy, "repetition": rep, "session": session,
                     "target_start_ms": item["target_start_ms"], "actual_start_ms": start,
                     "completion_ms": finish, "external_tokens_loaded": item["external_tokens_loaded"],
                     "client_e2e_ms": item["client_e2e_ms"], "server_e2e_ms": item["elapsed_ms"],
                     "cpu_to_gpu_ms": item["cpu_to_gpu_transfer_ms"],
                     "deadline_stall_ms": max(0.0, finish - DEADLINE_MS[session]),
                     "blackout_conflict": conflict, "long_prefetch": item["client_e2e_ms"] > 50.0,
                     "connector_only_step_count": item["connector_only_step_count"],
                     "normal_requests_delayed": item["normal_requests_delayed"],
                     "model_forward_count": item["model_forward_count"],
                     "generated_token_count": item["generated_token_count"],
                     "mixed_prefetch_model_step_count": item["mixed_prefetch_model_step_count"],
                     "status": item["status"]})
    return rows


def run_once(policy: str, limit: int | None, rep: int, prompts: dict[str, list[int]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any], float]:
    global _gate_limit
    _gate_limit = limit
    base.OUT = OUT
    old_payload = transport.completion_payload
    transport.completion_payload = payload_with_limit
    label = f"{policy.lower().replace('-', '_')}_{rep}"
    proc = base.start_server(label)
    snapshots: list[dict[str, Any]] = []
    try:
        base.runtime.wait_ready(proc)
        before, offloads = base.setup_external(prompts, label, snapshots, SESSIONS)
        if any(value != BYTES for value in offloads.values()):
            raise RuntimeError(f"offload mismatch: {offloads}")
        x_ready = Event()
        executor = ThreadPoolExecutor(max_workers=5)
        x_future = executor.submit(send_x, prompts["pressure_1"], x_ready)
        if not x_ready.wait(timeout=180):
            raise RuntimeError("X did not produce first token")
        t0 = time.monotonic()
        ready = {s: Event() for s in SESSIONS}
        promotions: dict[str, dict[str, Any]] = {}
        gated = limit is not None
        pf = executor.submit(base.promote_worker, SESSIONS, t0, prompts, ready, promotions)
        futures = {s: executor.submit(base.resume_request, s, t0, prompts, ready[s], gated) for s in SESSIONS}
        resumes = {s: f.result(timeout=900) for s, f in futures.items()}
        pf.result(timeout=900)
        for item in promotions.values():
            item["t0"] = t0
        x_result = x_future.result(timeout=900)
        executor.shutdown(wait=True)
        after = base.capture(label + "_after", snapshots)
        _all_snapshots.extend(snapshots)
        log = OUT / "server_logs" / f"{label}.log"
        trace = trace_rows(log, policy, rep, t0)
        local_delta = source_tokens_delta(before, after, "local_cache_hit")
        cache_ok = local_delta == float(len(SESSIONS) * 4096) and all(
            item["external_tokens_loaded"] == 4096 and item["model_forward_count"] == 0
            and item["generated_token_count"] == 0 and item["mixed_prefetch_model_step_count"] == 0
            and item["status"] == "ok" for item in promotions.values())
        nrows = native_rows(policy, rep, promotions, SESSIONS)
        srows: list[dict[str, Any]] = []
        for s in SESSIONS:
            client = resumes[s]
            arrived = first_event(trace, s, "RESUME_ARRIVED")
            admitted = first_event(trace, s, "RESUME_ADMITTED")
            first_model = first_event(trace, s, "FIRST_MODEL_SCHEDULED")
            first = first_event(trace, s, "FIRST_TOKEN")
            finished = first_event(trace, s, "REQUEST_FINISHED")
            blocked = first_event(trace, s, "GATE_BLOCKED")
            if not all((arrived, admitted, first_model, first, finished)):
                raise RuntimeError(f"incomplete trace for {s}")
            ams = float(admitted["timestamp_ms_from_t0"]); fms = float(first["timestamp_ms_from_t0"]); zms = float(finished["timestamp_ms_from_t0"])
            sent_ms = (client["sent_monotonic_s"] - t0) * 1000.0
            promo = promotions[s]
            pfinish = (promo["completed_monotonic_s"] - t0) * 1000.0
            srows.append({"policy": policy, "repetition": rep, "session": s,
                "original_deadline_ms": DEADLINE_MS[s], "client_request_sent_ms": sent_ms,
                "scheduler_arrival_ms": arrived["timestamp_ms_from_t0"], "scheduler_arrival_step": arrived["step_id"],
                "arrival_transport_delay_ms": float(arrived["timestamp_ms_from_t0"]) - sent_ms,
                "remote_kv_wait_start_ms": "NA", "remote_kv_ready_ms": "NA",
                "gate_first_blocked_ms": "NA" if blocked is None else blocked["timestamp_ms_from_t0"],
                "gate_wait_ms": gate_wait_ms(None if blocked is None else float(blocked["timestamp_ms_from_t0"]), ams),
                "gate_blocked_steps": sum(r["session"] == s and r["event"] == "GATE_BLOCKED" for r in trace),
                "scheduler_admission_ms": ams, "scheduler_admission_step": admitted["step_id"],
                "first_model_scheduled_ms": first_model["timestamp_ms_from_t0"], "first_model_scheduled_step": first_model["step_id"],
                "first_token_time_ms": fms, "first_token_step": first["step_id"],
                "request_finished_ms": zms, "request_finished_step": finished["step_id"],
                "effective_TTFT_ms": fms - DEADLINE_MS[s], "completion_latency_ms": zms - DEADLINE_MS[s],
                "admission_to_first_token_ms": fms - ams, "decode_duration_ms": zms - fms, "TPOT_ms": (zms - fms) / 31.0,
                "promotion_completion_ms": pfinish, "gpu_ready_residency_ms": max(0.0, ams - pfinish),
                "generated_token_count": len(client["token_ids"]), "generated_token_ids_json": json.dumps(client["token_ids"]),
                "sequence_hash": sequence_hash(client["token_ids"]), "token_count_validation": "PASS" if len(client["token_ids"]) == TOKENS else "FAIL",
                "cache_transport_validation": "PASS" if cache_ok else "FAIL"})
        x_first = (x_result["first_monotonic_s"] - t0) * 1000.0
        x_finish = (x_result["completed_monotonic_s"] - t0) * 1000.0
        bg = {"policy": policy, "repetition": rep, "X_first_token_ms": x_first, "X_finish_ms": x_finish,
              "X_decode_duration_ms": x_finish - x_first, "X_TPOT_ms": (x_finish - x_first) / 127.0,
              "X_generated_tokens": len(x_result["token_ids"]), "X_sequence_hash": sequence_hash(x_result["token_ids"]),
              "X_slowdown": None, "background_overlap_validation": "PENDING",
              "status": "PASS" if len(x_result["token_ids"]) == X_TOKENS else "FAIL"}
        run = aggregate(policy, rep, srows, nrows, bg, trace, cache_ok)
        bg["background_overlap_validation"] = "PASS" if x_first < 0 and x_finish > float(max(r["first_token_time_ms"] for r in srows)) else "FAIL"
        run["background_overlap_validation"] = bg["background_overlap_validation"]
        return nrows, srows, trace, run, bg, t0
    finally:
        try:
            executor.shutdown(wait=False)
        except Exception:
            pass
        base.stop_server(proc)
        transport.completion_payload = old_payload
        _gate_limit = None


def aggregate(policy: str, rep: int, rows: list[dict[str, Any]], native: list[dict[str, Any]], bg: dict[str, Any], trace: list[dict[str, Any]], cache_ok: bool) -> dict[str, Any]:
    abc = sorted(rows, key=lambda r: r["session"])
    pre = [(float(r["scheduler_admission_ms"]), float(r["first_token_time_ms"])) for r in abc]
    dec = [(float(r["first_token_time_ms"]), float(r["request_finished_ms"])) for r in abc]
    pre_max, pre_area = interval_concurrency(pre); dec_max, dec_area = interval_concurrency(dec)
    pre_pairs = [overlap(pre[i], pre[j]) for i in range(3) for j in range(i+1,3)]
    dec_pairs = [overlap(dec[i], dec[j]) for i in range(3) for j in range(i+1,3)]
    ttft = [float(r["effective_TTFT_ms"]) for r in abc]; comp = [float(r["completion_latency_ms"]) for r in abc]
    gated = "K1" in policy or "K2" in policy
    owners = [int(r["num_slackkv_pre_first_token"]) for r in trace if r["num_slackkv_pre_first_token"] is not None]
    limit = 1 if "K1" in policy else 2 if "K2" in policy else 3
    structure = True
    if gated:
        amap = {r["session"]: r for r in abc}
        if limit == 1:
            structure = (amap["A"]["scheduler_admission_step"] < amap["B"]["scheduler_admission_step"] < amap["C"]["scheduler_admission_step"]
                         and amap["A"]["first_token_step"] <= amap["B"]["scheduler_admission_step"]
                         and amap["B"]["first_token_step"] <= amap["C"]["scheduler_admission_step"])
        elif limit == 2:
            first_ab = min(amap["A"]["first_token_step"], amap["B"]["first_token_step"])
            structure = first_ab <= amap["C"]["scheduler_admission_step"]
    all_first = [float(r["first_token_time_ms"]) for r in abc] + [float(bg["X_first_token_ms"])]
    all_finish = [float(r["request_finished_ms"]) for r in abc] + [float(bg["X_finish_ms"])]
    return {"policy": policy, "repetition": rep, "gate_limit": limit, "promotion_deadline_miss_count": sum(float(n["deadline_stall_ms"]) > 0 for n in native),
        "blackout_conflict_count": sum(bool(n["blackout_conflict"]) for n in native), "long_prefetch_count": sum(bool(n["long_prefetch"]) for n in native),
        "ABC_pre_first_overlap_pair_count": sum(x > 0 for x in pre_pairs), "ABC_pre_first_pairwise_overlap_ms": sum(pre_pairs),
        "max_concurrent_pre_first_token": pre_max, "area_under_pre_first_token_concurrency_ms": pre_area,
        "ABC_total_decode_overlap_ms": sum(dec_pairs), "max_concurrent_post_first_token_decode": dec_max,
        "area_under_post_first_token_decode_concurrency_ms": dec_area, "ABC_total_gate_wait_ms": sum(float(r["gate_wait_ms"]) for r in abc),
        "ABC_P50_TTFT_ms": percentile_linear(ttft,50), "ABC_P95_TTFT_ms": percentile_linear(ttft,95), "ABC_max_TTFT_ms": max(ttft),
        "overall_P95_TTFT_ms": percentile_linear(ttft,95), "ABC_P95_completion_latency_ms": percentile_linear(comp,95),
        "ABC_completion_makespan_ms": max(float(r["request_finished_ms"]) for r in abc) - 300.0,
        "ABC_decode_throughput_tok_s": decode_throughput_tok_s(93,[float(r["first_token_time_ms"]) for r in abc],[float(r["request_finished_ms"]) for r in abc]),
        "ABC_total_decode_overlap_ms": sum(dec_pairs), "median_TPOT_ms": percentile_linear([float(r["TPOT_ms"]) for r in abc],50),
        "X_first_token_ms": bg["X_first_token_ms"], "X_finish_ms": bg["X_finish_ms"], "X_decode_duration_ms": bg["X_decode_duration_ms"],
        "X_TPOT_ms": bg["X_TPOT_ms"], "X_generated_tokens": bg["X_generated_tokens"], "X_slowdown": bg.get("X_slowdown"),
        "combined_decode_throughput_tok_s": 220.0 / max((max(all_finish)-min(all_first))/1000.0,1e-9),
        "ABC_same_step_admission_pairs": same_step_pairs([int(r["scheduler_admission_step"]) for r in abc]),
        "token_count_validation": "PASS" if all(r["token_count_validation"] == "PASS" for r in abc) and bg["status"] == "PASS" else "FAIL",
        "cache_transport_validation": "PASS" if cache_ok else "FAIL", "timeline_validation": "PASS" if bg["background_overlap_validation"] == "PASS" else "FAIL",
        "trace_validation": "PASS", "gate_invariant_validation": "PASS" if (not gated or (max(owners or [0]) <= limit and structure)) else "FAIL",
        "decode_overlap_validation": "PASS" if dec_max >= 2 and sum(dec_pairs) > 0 else "FAIL", "offload_validation": "PASS"}


def run_baseline(prompts: dict[str, list[int]]) -> None:
    rows=[]
    for rep in range(3):
        label=f"x_baseline_{rep}"; base.OUT=OUT; proc=base.start_server(label)
        try:
            base.runtime.wait_ready(proc); result=base.send(prompts["pressure_1"], stream=True, max_tokens=X_TOKENS, normal=True)
            rows.append({"repetition":rep,"first_token_ms":result["ttft_ms"],"decode_duration_ms":result["e2e_ms"]-result["ttft_ms"],"TPOT_ms":(result["e2e_ms"]-result["ttft_ms"])/127.0,"generated_tokens":len(result["token_ids"]),"sequence_hash":sequence_hash(result["token_ids"]),"status":"PASS" if len(result["token_ids"])==X_TOKENS else "FAIL"})
        finally: base.stop_server(proc)
    write_rows(OUT/"x_baseline_metrics.csv",rows)


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--stage",choices=("baseline","integration","full"),required=True); args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True); prompts=base.make_prompts()
    if args.stage=="baseline": run_baseline(prompts); return
    baseline=list(csv.DictReader((OUT/"x_baseline_metrics.csv").open(encoding="utf-8")))
    median_baseline=statistics.median(float(r["decode_duration_ms"]) for r in baseline)
    if args.stage=="integration":
        checks=[]
        for policy,limit in ALL_POLICIES[1:]:
            n,s,t,r,b,t0=run_once(policy,limit,99,prompts); b["X_slowdown"]=float(b["X_decode_duration_ms"])/median_baseline; checks.append(r)
        (OUT/"integration_gate.json").write_text(json.dumps({"pass":all(r["cache_transport_validation"]=="PASS" and r["token_count_validation"]=="PASS" for r in checks),"checks":checks},indent=2)); return
    alln=[]; alls=[]; allt=[]; allr=[]; allb=[]; context=[]
    for policy,limit in ALL_POLICIES:
        for rep in range(3):
            print(f"RUN {policy} {rep}",flush=True); n,s,t,r,b,t0=run_once(policy,limit,rep,prompts); b["X_slowdown"]=float(b["X_decode_duration_ms"])/median_baseline; r["X_slowdown"]=b["X_slowdown"]; alln+=n; alls+=s; allt+=t; allr.append(r); allb.append(b); context.append({"policy":policy,"repetition":rep,"t0_monotonic_s":t0})
    write_rows(OUT/"native_prefetch_metrics.csv",alln); write_rows(OUT/"session_metrics.csv",alls); write_rows(OUT/"scheduler_trace.csv",allt); write_rows(OUT/"step_metrics.csv",allt); write_rows(OUT/"run_metrics.csv",allr); write_rows(OUT/"background_metrics.csv",allb); write_rows(OUT/"run_context.csv",context)
    (OUT/"metric_snapshots.json").write_text(json.dumps(_all_snapshots, indent=2))
    print("FULL_DONE",flush=True)

if __name__=="__main__": main()
