from __future__ import annotations

import json
import argparse
from pathlib import Path

from experiments.phase3g import run_phase3g as transport
from experiments.phase3g_full import run_phase3g_full as base
from experiments.phase3i import run_phase3i as p3i
from experiments.phase3j.run_phase3j import promote_with_budget

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase3k"
POLICIES = (("ResumeAware-K3", None), ("FirstTokenGate-K1", 1), ("FirstTokenGate-K2", 2))
P_BUSY_MS = 70.0


def run_one(policy, limit, rep, prompts):
    old_out, old_payload, old_promoter = p3i.OUT, transport.completion_payload, base.promote_worker
    p3i.OUT = OUT; base.OUT = OUT; p3i._gate_limit = limit
    transport.completion_payload = p3i.payload_with_limit
    def promoter(sessions, t0, p, ready, output):
        return promote_with_budget(sessions, t0, p, ready, output, P_BUSY_MS)
    base.promote_worker = promoter
    try:
        native, sessions, trace, run, bg, t0 = p3i.run_once(policy, limit, rep, prompts)
        for row in native:
            row["service_budget_ms"] = P_BUSY_MS
            row["budget_overrun"] = float(row["client_e2e_ms"]) > P_BUSY_MS
        run["service_budget_ms"] = P_BUSY_MS
        run["budget_overrun_count"] = sum(bool(row["budget_overrun"]) for row in native)
        run["promotion_validation"] = "PASS" if run["promotion_deadline_miss_count"] == 0 and run["blackout_conflict_count"] == 0 and run["budget_overrun_count"] == 0 else "FAIL"
        return native, sessions, trace, run, bg, t0
    finally:
        base.promote_worker = old_promoter
        transport.completion_payload = old_payload
        p3i.OUT = old_out; base.OUT = old_out; p3i._gate_limit = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=("ResumeAware-K3", "FirstTokenGate-K1", "FirstTokenGate-K2"), default=None)
    selected = [item for item in POLICIES if item[0] == parser.parse_args().policy] if parser.parse_args().policy else list(POLICIES)
    OUT.mkdir(parents=True, exist_ok=True)
    prompts = base.make_prompts()
    all_native=[]; all_sessions=[]; all_trace=[]; all_runs=[]; all_bg=[]; context=[]
    for policy, limit in selected:
        for rep in range(5):
            print(f"RUN {policy} {rep}", flush=True)
            n,s,t,r,b,t0 = run_one(policy, limit, rep, prompts)
            all_native += n; all_sessions += s; all_trace += t; all_runs.append(r); all_bg.append(b)
            context.append({"policy":policy,"repetition":rep,"t0_monotonic_s":t0,"p_busy_ms":P_BUSY_MS})
    p3i.write_rows(OUT/"native_prefetch_metrics.csv", all_native)
    p3i.write_rows(OUT/"session_metrics.csv", all_sessions)
    p3i.write_rows(OUT/"scheduler_trace.csv", all_trace)
    p3i.write_rows(OUT/"step_metrics.csv", all_trace)
    p3i.write_rows(OUT/"run_metrics.csv", all_runs)
    p3i.write_rows(OUT/"background_metrics.csv", all_bg)
    p3i.write_rows(OUT/"run_context.csv", context)
    (OUT/"metric_snapshots.json").write_text("[]")
    print("FULL_DONE", flush=True)


if __name__ == "__main__": main()
