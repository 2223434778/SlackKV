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

from experiments.phase3g import run_phase3g as transport
from experiments.phase3g_full import run_phase3g_full as base
from experiments.phase3f.native_client import native_prefetch
from experiments.phase3f.scheduler import DEADLINE_MS, RESUME_GUARD_MS, latest_feasible_start, merge_blackouts
from experiments.phase3g_full.metrics import percentile_linear
from experiments.phase3i import run_phase3i as prev

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase3j"
SESSIONS = ("A", "B", "C")
BYTES = 117_440_512
P_IDLE_MS = 30.0
GATE_LIMIT = 2


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError(f"empty output {path}")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader(); w.writerows(rows)


def setup_one(prompts, label, snapshots):
    before = base.capture(label + "_ready", snapshots)
    base.send(prompts["a"], stream=False, max_tokens=1, normal=True)
    after = base.capture(label + "_after_first", snapshots)
    offload = base.gpu_to_cpu_offload_bytes(before, after) if hasattr(base, "gpu_to_cpu_offload_bytes") else None
    time.sleep(1.0)
    for i in range(1, 5):
        base.send(prompts[f"pressure_{i}"], stream=False, max_tokens=1, normal=False)
    base.capture(label + "_after_pressure", snapshots)
    return offload


def calibrate_one(load_state: str, rep: int, prompts: dict) -> dict:
    label = f"cal_{load_state.lower()}_{rep}"
    base.OUT = OUT
    proc = base.start_server(label)
    snapshots = []
    try:
        base.runtime.wait_ready(proc)
        setup_one(prompts, label, snapshots)
        x_future = None
        x_event = Event()
        executor = ThreadPoolExecutor(max_workers=2)
        if load_state == "L1":
            x_future = executor.submit(prev.send_x, prompts["pressure_1"], x_event)
            if not x_event.wait(timeout=180):
                raise RuntimeError("X did not become active")
        native = native_prefetch(prompts["a"])
        if x_future is not None:
            x = x_future.result(timeout=900)
            x_start = x["first_monotonic_s"] < native["sent_monotonic_s"]
            x_end = x["completed_monotonic_s"] > native["completed_monotonic_s"]
        else:
            x_start = x_end = False
        executor.shutdown(wait=True)
        return {"load_state": load_state, "repetition": rep,
                "client_e2e_ms": native["client_e2e_ms"], "server_e2e_ms": native["elapsed_ms"],
                "cpu_to_gpu_ms": native["cpu_to_gpu_transfer_ms"],
                "connector_only_step_count": native["connector_only_step_count"],
                "normal_requests_delayed": native["normal_requests_delayed"],
                "X_active_at_start": x_start, "X_active_at_completion": x_end,
                "external_tokens_loaded": native["external_tokens_loaded"],
                "model_forward_count": native["model_forward_count"],
                "generated_token_count": native["generated_token_count"],
                "mixed_prefetch_model_step_count": native["mixed_prefetch_model_step_count"],
                "status": native["status"]}
    finally:
        base.stop_server(proc)


def promote_with_budget(sessions, t0, prompts, ready, output, budget_ms):
    pending = set(sessions)
    blackouts = merge_blackouts({s: DEADLINE_MS[s] for s in sessions}, RESUME_GUARD_MS)
    while pending:
        now_ms = (time.monotonic() - t0) * 1000.0
        target = latest_feasible_start(now_ms, [(s, DEADLINE_MS[s]) for s in pending], blackouts, budget_ms)
        if target - now_ms > 0.5:
            time.sleep((target - now_ms) / 1000.0)
            continue
        session = min(pending, key=lambda s: (DEADLINE_MS[s], s))
        item = native_prefetch(prompts[session.lower()])
        output[session] = {"target_start_ms": target, **item}
        ready[session].set()
        pending.remove(session)


def run_validation(rep: int, prompts: dict, p_busy_ms: float):
    old_out, old_payload, old_promoter = prev.OUT, transport.completion_payload, base.promote_worker
    prev.OUT = OUT; base.OUT = OUT; prev._gate_limit = GATE_LIMIT
    transport.completion_payload = prev.payload_with_limit
    def promoter(sessions, t0, p, ready, output):
        return promote_with_budget(sessions, t0, p, ready, output, p_busy_ms)
    base.promote_worker = promoter
    try:
        native, sessions, trace, run, bg, t0 = prev.run_once("FirstTokenGate-K2", GATE_LIMIT, rep, prompts)
        for row in native:
            row["service_budget_ms"] = p_busy_ms
            row["budget_overrun"] = float(row["client_e2e_ms"]) > p_busy_ms
        run["service_budget_ms"] = p_busy_ms
        run["budget_overrun_count"] = sum(bool(row["budget_overrun"]) for row in native)
        run["promotion_deadline_miss_count"] = sum(float(row["deadline_stall_ms"]) > 0 for row in native)
        run["blackout_conflict_count"] = sum(str(row["blackout_conflict"]) == "True" for row in native)
        return native, sessions, trace, run, bg, t0
    finally:
        base.promote_worker = old_promoter
        transport.completion_payload = old_payload
        prev.OUT = old_out; base.OUT = old_out; prev._gate_limit = None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", choices=("calibration", "validation"), required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    prompts = base.make_prompts()
    if args.stage == "calibration":
        rows = []
        for state in ("L0", "L1"):
            for rep in range(5):
                print(f"CAL {state} {rep}", flush=True)
                rows.append(calibrate_one(state, rep, prompts))
        write_csv(OUT / "calibration_metrics.csv", rows)
        l1 = [float(r["client_e2e_ms"]) for r in rows if r["load_state"] == "L1"]
        raw = max(l1) + 10.0
        p_busy = math.ceil(raw / 5.0) * 5.0
        (OUT / "frozen_budget.json").write_text(json.dumps({"p_idle_ms": P_IDLE_MS, "p_busy_raw_ms": raw, "p_busy_ms": p_busy}, indent=2) + "\n")
        print(f"P_BUSY_MS={p_busy}", flush=True)
        return
    budget = json.loads((OUT / "frozen_budget.json").read_text())
    p_busy_ms = float(budget["p_busy_ms"])
    all_native=[]; all_sessions=[]; all_trace=[]; all_runs=[]; all_bg=[]; context=[]
    for rep in range(5):
        print(f"VALIDATE {rep}", flush=True)
        n,s,t,r,b,t0 = run_validation(rep, prompts, p_busy_ms)
        all_native += n; all_sessions += s; all_trace += t; all_runs.append(r); all_bg.append(b); context.append({"policy":"FirstTokenGate-K2","repetition":rep,"t0_monotonic_s":t0,"p_busy_ms":p_busy_ms})
    write_csv(OUT/"native_prefetch_metrics.csv", all_native); write_csv(OUT/"session_metrics.csv", all_sessions); write_csv(OUT/"scheduler_trace.csv", all_trace); write_csv(OUT/"run_metrics.csv", all_runs); write_csv(OUT/"background_metrics.csv", all_bg); write_csv(OUT/"run_context.csv", context)
    (OUT/"metric_snapshots.json").write_text("[]")
    print("VALIDATION_DONE", flush=True)


if __name__ == "__main__": main()
