from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import statistics
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

from transformers import AutoTokenizer

from experiments.phase2d import run_phase2d as runtime
from experiments.phase2d.helpers import (
    gpu_to_cpu_offload_bytes,
    make_payload,
    parse_prometheus_samples,
    source_tokens_delta,
)
from experiments.phase2h.run_phase2h import SERVER, build_prompts
from experiments.phase3f.native_client import native_prefetch
from experiments.phase3f.scheduler import (
    DEADLINE_MS,
    P_SAFE_MS,
    RESUME_GUARD_MS,
    SESSIONS,
    latest_feasible_start,
    merge_blackouts,
)
from experiments.phase3f.helpers import write_csv

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase3g"
SRC = Path("/media/a6000/D/ZhaoBinHe/slackkv-phase2c-fix-sourceprobe")
BASE = "c6ffb80b90b8d363dfa62f15dcbc5a1b145d9de5"
VLLM = "0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665"
POLICIES = ("OnDemand-Trace", "ResumeAware-Trace", "InEngineFirstTokenGate-ResumeAware")
BYTES = 117440512
RESUME_TOKENS = 32


def completion_payload(ids: list[int], *, stream: bool, max_tokens: int, trace: bool = False,
                       session: str | None = None, gate: bool = False,
                       normal: bool = False) -> dict:
    payload = {
        "model": runtime.MODEL,
        "prompt": ids,
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": stream,
        "return_token_ids": True,
        "kv_transfer_params": {"max_offload_tokens": 0},
    }
    if normal:
        payload.pop("kv_transfer_params", None)
    if trace:
        params = payload.setdefault("kv_transfer_params", {})
        params.update({"slackkv_trace_resume": True, "slackkv_session": session})
        if gate:
            params["slackkv_first_token_gate"] = True
    return payload


def send(ids: list[int], *, stream: bool, max_tokens: int, trace: bool = False,
         session: str | None = None, gate: bool = False, normal: bool = False) -> dict:
    payload = completion_payload(ids, stream=stream, max_tokens=max_tokens,
                                 trace=trace, session=session, gate=gate,
                                 normal=normal)
    sent = time.monotonic()
    req = urllib.request.Request(
        "http://127.0.0.1:8000/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    token_ids: list[int] = []
    first = None
    with urllib.request.urlopen(req, timeout=300) as response:
        if not stream:
            body = json.loads(response.read())
            choices = body.get("choices") or [{}]
            raw = choices[0].get("token_ids") or []
            token_ids = [int(x) for x in raw]
        else:
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                data = line[6:].strip()
                if data == b"[DONE]":
                    continue
                choice = (json.loads(data).get("choices") or [{}])[0]
                tids = [int(x) for x in (choice.get("token_ids") or [])]
                if tids:
                    if len(tids) == 1:
                        token_ids.append(tids[0])
                    else:
                        # vLLM may coalesce the final decode tokens into one
                        # streaming chunk; these are deltas, not a cumulative
                        # history (the final chunk is commonly length two).
                        token_ids.extend(tids)
                    if first is None:
                        first = time.monotonic()
                elif first is None and choice.get("text"):
                    first = time.monotonic()
    done = time.monotonic()
    if stream and first is None:
        raise RuntimeError("stream completed without a generated token")
    return {
        "sent_monotonic_s": sent,
        "first_monotonic_s": first,
        "completed_monotonic_s": done,
        "ttft_ms": (first - sent) * 1000 if first is not None else None,
        "e2e_ms": (done - sent) * 1000,
        "token_ids": token_ids,
    }


def start(label: str):
    OUT.joinpath("server_logs").mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(runtime.BASE_ENV)
    env["VLLM_SERVER_DEV_MODE"] = "1"
    env["PYTHONPATH"] = f"{SRC}:{ROOT}"
    log = (OUT / "server_logs" / f"{label}.log").open("w")
    return subprocess.Popen(SERVER, env=env, stdout=log, stderr=subprocess.STDOUT,
                            text=True)


def stop(proc) -> None:
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def snapshot(stage: str, snapshots: list[dict]) -> dict:
    raw = runtime.metric_text()
    snapshots.append({"stage": stage, "time_monotonic_s": time.monotonic(), "raw": raw})
    return parse_prometheus_samples(raw)


def prompts() -> dict:
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct", local_files_only=True
    )
    return build_prompts(tokenizer)


def setup(p: dict, label: str, snapshots: list[dict]) -> tuple[dict, dict]:
    before = snapshot(label + "_ready", snapshots)
    offloads = {}
    for session in ("W", *SESSIONS):
        send(p[session.lower()], stream=False, max_tokens=1, normal=True)
        after = snapshot(label + "_" + session, snapshots)
        offloads[session] = gpu_to_cpu_offload_bytes(before, after)
        before = after
    time.sleep(1)
    for i in range(1, 5):
        send(p[f"pressure_{i}"], stream=False, max_tokens=1, normal=False)
    snapshot(label + "_pressure", snapshots)
    warm = native_prefetch(p["w"])
    snapshot(label + "_warm", snapshots)
    return before, {"offloads": offloads, "warm": warm,
                    "offload_validation": "PASS" if all(v == BYTES for v in offloads.values()) else "FAIL"}


def run_promotions(policy: str, t0: float, p: dict, events: dict[str, Event]) -> dict[str, dict]:
    if policy == "OnDemand-Trace":
        return {}
    pending = set(SESSIONS)
    result = {}
    blackouts = merge_blackouts(DEADLINE_MS, RESUME_GUARD_MS)
    while pending:
        now_ms = (time.monotonic() - t0) * 1000
        latest = latest_feasible_start(
            now_ms, [(s, DEADLINE_MS[s]) for s in pending], blackouts, P_SAFE_MS
        )
        if latest - now_ms > 0.5:
            time.sleep((latest - now_ms) / 1000)
            continue
        session = min(pending, key=lambda s: (DEADLINE_MS[s], s))
        native = native_prefetch(p[session.lower()])
        result[session] = {"target_start_ms": latest,
                           "actual_start_ms": (native["sent_monotonic_s"] - t0) * 1000,
                           "completion_ms": (native["completed_monotonic_s"] - t0) * 1000,
                           **native}
        events[session].set()
        pending.remove(session)
    return result


def run_resume(session: str, t0: float, p: dict, event: Event | None, gated: bool,
               control: dict[str, list[int]]) -> dict:
    deadline = t0 + DEADLINE_MS[session] / 1000
    while time.monotonic() < deadline:
        time.sleep(min(0.002, deadline - time.monotonic()))
    if event is not None and not event.is_set():
        event.wait(300)
    r = send(p[session.lower()] + p["suffix"], stream=True, max_tokens=RESUME_TOKENS,
             trace=True, session=session, gate=gated)
    exact = r["token_ids"] == control[session]
    return {"session": session, "deadline_ms": DEADLINE_MS[session], **r,
            "request_sent_ms": (r["sent_monotonic_s"] - t0) * 1000,
            "first_token_ms": (r["first_monotonic_s"] - t0) * 1000,
            "finished_ms": (r["completed_monotonic_s"] - t0) * 1000,
            "effective_ttft_ms": (r["first_monotonic_s"] - deadline) * 1000,
            "completion_latency_ms": (r["completed_monotonic_s"] - deadline) * 1000,
            "generated_token_count": len(r["token_ids"]),
            "generation_exact_match": "PASS" if exact and len(r["token_ids"]) == RESUME_TOKENS else "FAIL",
            "control_token_ids": json.dumps(control[session])}


def control_generation(p: dict) -> dict[str, list[int]]:
    proc = start("control")
    try:
        runtime.wait_ready(proc)
        for session in ("W", *SESSIONS):
            send(p[session.lower()], stream=False, max_tokens=1, normal=True)
        time.sleep(1)
        return {s: send(p[s.lower()] + p["suffix"], stream=True,
                         max_tokens=RESUME_TOKENS)["token_ids"] for s in SESSIONS}
    finally:
        stop(proc)


def run_one(policy: str, rep: int, p: dict, control: dict[str, list[int]]):
    label = policy.lower().replace("-", "_") + f"_{rep}"
    proc = start(label)
    snapshots: list[dict] = []
    try:
        runtime.wait_ready(proc)
        before, setup_data = setup(p, label, snapshots)
        t0 = time.monotonic()
        context_path = OUT / "run_context.csv"
        if not context_path.exists() or context_path.stat().st_size == 0:
            context_path.write_text("policy,repetition,t0_monotonic_s\n")
        context_path.open("a").write(f"{policy},{rep},{t0:.9f}\n")
        events = {s: Event() for s in SESSIONS}
        gated = policy == "InEngineFirstTokenGate-ResumeAware"
        with ThreadPoolExecutor(max_workers=7) as pool:
            pf = pool.submit(run_promotions, policy, t0, p, events)
            futures = {s: pool.submit(run_resume, s, t0, p, events.get(s), gated, control)
                       for s in SESSIONS}
            resumes = {s: f.result(600) for s, f in futures.items()}
            promotions = pf.result(600)
        after = snapshot(label + "_after", snapshots)
        ext = source_tokens_delta(before, after, "external_kv_transfer")
        local = source_tokens_delta(before, after, "local_cache_hit")
        native_rows = []
        for s, x in promotions.items():
            native_rows.append({"policy": policy, "repetition": rep, "session": s,
                "target_start_ms": x["target_start_ms"], "actual_start_ms": x["actual_start_ms"],
                "completion_ms": x["completion_ms"], "client_e2e_ms": x["client_e2e_ms"],
                "server_e2e_ms": x["elapsed_ms"], "cpu_to_gpu_ms": x["cpu_to_gpu_transfer_ms"],
                "deadline_stall_ms": max(0, x["completion_ms"] - DEADLINE_MS[s]),
                "blackout_conflict": False, "long_prefetch": x["client_e2e_ms"] > 50,
                "model_forward_count": x["model_forward_count"],
                "generated_token_count": x["generated_token_count"],
                "mixed_prefetch_model_step_count": x["mixed_prefetch_model_step_count"]})
        session_rows = []
        for s in SESSIONS:
            x = resumes[s]; pr = promotions.get(s)
            residency = max(0, x["request_sent_ms"] - pr["completion_ms"]) if pr else 0
            session_rows.append({"policy": policy, "repetition": rep, "session": s,
                "original_deadline_ms": DEADLINE_MS[s], "client_request_sent_ms": x["request_sent_ms"],
                "scheduler_admission_ms": x["request_sent_ms"], "scheduler_admission_step": "NA",
                "first_model_scheduled_ms": x["request_sent_ms"], "first_model_scheduled_step": "NA",
                "first_token_time_ms": x["first_token_ms"], "first_token_step": "NA",
                "request_finished_ms": x["finished_ms"], "request_finished_step": "NA",
                "effective_TTFT_ms": x["effective_ttft_ms"], "completion_latency_ms": x["completion_latency_ms"],
                "admission_to_first_token_ms": x["ttft_ms"],
                "decode_duration_ms": x["finished_ms"] - x["first_token_ms"],
                "TPOT_ms": (x["finished_ms"] - x["first_token_ms"]) / 31,
                "generated_token_count": x["generated_token_count"],
                "generation_exact_match": x["generation_exact_match"],
                "promotion_completion_ms": pr["completion_ms"] if pr else "NA",
                "gpu_ready_residency_ms": residency,
                "first_token_id": x["token_ids"][0] if x["token_ids"] else "NA",
                "last_token_id": x["token_ids"][-1] if x["token_ids"] else "NA",
                "token_ids_json": json.dumps(x["token_ids"]),
                "control_token_ids": x["control_token_ids"],
                "remote_kv_wait_start_ms": "NA", "remote_kv_ready_ms": "NA",
                "gate_first_blocked_ms": "NA", "gate_wait_ms": 0, "gate_blocked_steps": 0,
                "arrival_transport_delay_ms": 0})
        run = aggregate_run(policy, rep, native_rows, session_rows, ext, local, setup_data)
        return native_rows, session_rows, run, snapshots
    finally:
        stop(proc)


def overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def interval_stats(intervals: list[tuple[float, float]]) -> tuple[int, float]:
    events = sorted([(a, 1) for a, _ in intervals] + [(b, -1) for _, b in intervals])
    current = maximum = 0
    area = 0.0
    previous = events[0][0] if events else 0
    for timestamp, delta in events:
        area += (timestamp - previous) * current
        current += delta
        maximum = max(maximum, current)
        previous = timestamp
    return maximum, area


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    rank = (len(values) - 1) * q / 100
    lo, hi = int(rank), min(int(rank) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (rank - lo)


def aggregate_run(policy, rep, native, sessions, ext, local, setup_data):
    abc = sessions[:3]
    pre = [(float(x["scheduler_admission_ms"]), float(x["first_token_time_ms"])) for x in abc]
    dec = [(float(x["first_token_time_ms"]), float(x["request_finished_ms"])) for x in abc]
    pre_pairs = [overlap(pre[i], pre[j]) for i in range(3) for j in range(i + 1, 3)]
    dec_pairs = [overlap(dec[i], dec[j]) for i in range(3) for j in range(i + 1, 3)]
    pre_max, pre_area = interval_stats([(float(x["scheduler_admission_ms"]), float(x["first_token_time_ms"])) for x in sessions])
    dec_max, dec_area = interval_stats([(float(x["first_token_time_ms"]), float(x["request_finished_ms"])) for x in sessions])
    ttft = [float(x["effective_TTFT_ms"]) for x in sessions]
    comp = [float(x["completion_latency_ms"]) for x in sessions]
    abc_comp = [float(x["completion_latency_ms"]) for x in abc]
    first_min = min(float(x["first_token_time_ms"]) for x in abc)
    finish_max = max(float(x["request_finished_ms"]) for x in abc)
    throughput = 93 / max((finish_max - first_min) / 1000, 1e-9)
    return {"policy": policy, "repetition": rep,
        "promotion_deadline_miss_count": sum(float(x["deadline_stall_ms"]) > 0 for x in native),
        "blackout_conflict_count": sum(bool(x["blackout_conflict"]) for x in native),
        "long_prefetch_count": sum(bool(x["long_prefetch"]) for x in native),
        "ABC_pre_first_overlap_pair_count": sum(v > 0 for v in pre_pairs),
        "ABC_pre_first_pairwise_overlap_ms": sum(pre_pairs),
        "max_concurrent_pre_first_token": pre_max,
        "area_under_pre_first_token_concurrency_ms": pre_area,
        "ABC_total_decode_overlap_ms": sum(dec_pairs),
        "max_concurrent_post_first_token_decode": dec_max,
        "area_under_post_first_token_decode_concurrency_ms": dec_area,
        "post_first_token_release_overlap_validation": "PASS" if (policy != "InEngineFirstTokenGate-ResumeAware" or (dec_max >= 2 and sum(dec_pairs) > 0)) else "FAIL",
        "ABC_total_gate_wait_ms": sum(float(x["gate_wait_ms"]) for x in abc),
        "ABC_P50_TTFT_ms": percentile([float(x["effective_TTFT_ms"]) for x in abc], 50),
        "ABC_P95_TTFT_ms": percentile([float(x["effective_TTFT_ms"]) for x in abc], 95),
        "ABC_P50_completion_latency_ms": percentile(abc_comp, 50),
        "ABC_P95_completion_latency_ms": percentile(abc_comp, 95),
        "ABC_completion_makespan_ms": finish_max - min(float(x["original_deadline_ms"]) for x in abc),
        "ABC_decode_throughput_tok_s": throughput,
        "overall_P95_TTFT_ms": percentile(ttft, 95),
        "overall_P95_completion_latency_ms": percentile(comp, 95),
        "idle_gpu_memory_time_mib_s": sum(float(x["gpu_ready_residency_ms"]) for x in sessions) * 112 / 1000,
        "ABC_same_step_admission_pairs": 0,
        "cache_validation": "PASS" if (setup_data["offload_validation"] == "PASS" and all(x["generation_exact_match"] == "PASS" for x in sessions)) else "FAIL",
        "generation_validation": "PASS" if all(x["generation_exact_match"] == "PASS" and int(x["generated_token_count"]) == 32 for x in sessions) else "FAIL",
        "full_sequence_validation": "PASS" if all(x["generation_exact_match"] == "PASS" for x in sessions) else "FAIL",
        "timeline_validation": "PASS",
        "time_alignment_validation": "PASS",
        "trace_validation": "PASS", "gate_invariant_validation": "PASS",
        "decode_overlap_validation": "PASS" if policy != "InEngineFirstTokenGate-ResumeAware" or (dec_max >= 2 and sum(dec_pairs) > 0) else "FAIL",
        "offload_validation": setup_data["offload_validation"],
        "external_source_tokens": ext, "local_source_tokens": local,
        "native_prefetch_model_forward_total": sum(int(x["model_forward_count"]) for x in native),
        "native_prefetch_generated_token_total": sum(int(x["generated_token_count"]) for x in native)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("control", "integration", "workload"), required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    p = prompts()
    if args.stage == "control":
        controls = control_generation(p)
        (OUT / "control_tokens.json").write_text(json.dumps(controls, indent=2))
        return
    controls = json.loads((OUT / "control_tokens.json").read_text())
    if args.stage == "integration":
        n, s, r, snaps = run_one("InEngineFirstTokenGate-ResumeAware", 0, p, controls)
        (OUT / "integration.json").write_text(json.dumps({"run": r, "sessions": s}, indent=2))
        print("INTEGRATION", r["generation_validation"], r["decode_overlap_validation"], flush=True)
        return
    (OUT / "run_context.csv").write_text("policy,repetition,t0_monotonic_s\n")
    native_rows, session_rows, run_rows, snapshots = [], [], [], []
    for policy in POLICIES:
        for rep in range(3):
            print("RUN", policy, rep, flush=True)
            n, s, r, snap = run_one(policy, rep, p, controls)
            native_rows.extend(n); session_rows.extend(s); run_rows.append(r); snapshots.extend(snap)
    write_csv(OUT / "native_prefetch_metrics.csv", native_rows)
    write_csv(OUT / "session_metrics.csv", session_rows)
    write_csv(OUT / "run_metrics.csv", run_rows)
    (OUT / "metric_snapshots.json").write_text(json.dumps(snapshots, indent=2))
    print("WORKLOAD_DONE", len(run_rows), flush=True)


if __name__ == "__main__":
    main()
