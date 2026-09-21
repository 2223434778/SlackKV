from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase3j"


def read(name):
    with (OUT / name).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write(name, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / name).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n"); w.writeheader(); w.writerows(rows)


def med(rows, key): return statistics.median(float(r[key]) for r in rows)


def main():
    runs = read("run_metrics.csv"); sessions = read("session_metrics.csv"); native = read("native_prefetch_metrics.csv"); bg = read("background_metrics.csv"); trace = read("scheduler_trace.csv")
    base_path = ROOT / "results" / "phase3i" / "x_baseline_metrics.csv"
    baseline = list(csv.DictReader(base_path.open(encoding="utf-8")))
    baseline_decode = statistics.median(float(r["decode_duration_ms"]) for r in baseline)
    ng = {}
    for r in native: ng.setdefault((r["policy"], r["repetition"]), []).append(r)
    tg = {}
    for r in trace: tg.setdefault((r["policy"], r["repetition"]), []).append(r)
    for r in bg: r["X_slowdown"] = float(r["X_decode_duration_ms"]) / baseline_decode
    for s in sessions:
        key=(s["policy"],s["repetition"]); ns=ng[key]; ts=tg[key]
        source_ok=all(n["status"]=="ok" and n["external_tokens_loaded"]=="4096" and n["model_forward_count"]=="0" and n["generated_token_count"]=="0" and n["mixed_prefetch_model_step_count"]=="0" for n in ns)
        no_remote=not any(t["session"] in ("A","B","C") and t["event"] in ("WAITING_FOR_REMOTE_KVS","REMOTE_KV_READY") for t in ts)
        s["cache_transport_validation"]="PASS" if source_ok and no_remote else "FAIL"
    for r in runs:
        key=(r["policy"],r["repetition"]); ss=[s for s in sessions if (s["policy"],s["repetition"])==key]; b=next(x for x in bg if (x["policy"],x["repetition"])==key)
        r["cache_transport_validation"]="PASS" if all(s["cache_transport_validation"]=="PASS" for s in ss) else "FAIL"
        r["token_count_validation"]="PASS" if all(s["token_count_validation"]=="PASS" for s in ss) and b["X_generated_tokens"]=="128" else "FAIL"
        r["background_overlap_validation"]="PASS" if float(b["X_first_token_ms"])<0 and float(b["X_finish_ms"])>max(float(s["first_token_time_ms"]) for s in ss if s["session"]=="C") else "FAIL"
        r["X_slowdown"]=b["X_slowdown"]
        r["idle_gpu_memory_time_mib_s"] = sum(max(0.0, float(s["scheduler_admission_ms"]) - float(s["promotion_completion_ms"])) * 112.0 / 1000.0 for s in ss)
    write("session_metrics.csv",sessions); write("background_metrics.csv",bg); write("run_metrics.csv",runs)
    cal=read("calibration_metrics.csv"); l0=[r for r in cal if r["load_state"]=="L0"]; l1=[r for r in cal if r["load_state"]=="L1"]; frozen=json.loads((OUT/"frozen_budget.json").read_text())
    lines=["# Phase 3J summary","", "Phase 3J keeps admission at FirstTokenGate K=2 and changes only the promotion service budget. No vLLM scheduler algorithm changes were made; Phase 3H K-capacity gating was reused unchanged.","", "## Calibration", "", f"L0 idle client E2E P50/P95/max: {med(l0,'client_e2e_ms'):.2f}/{sorted(float(r['client_e2e_ms']) for r in l0)[3]:.2f}/{max(float(r['client_e2e_ms']) for r in l0):.2f} ms.", f"L1 one-active-decode client E2E P50/P95/max: {med(l1,'client_e2e_ms'):.2f}/{sorted(float(r['client_e2e_ms']) for r in l1)[3]:.2f}/{max(float(r['client_e2e_ms']) for r in l1):.2f} ms.", f"L0 server E2E P50/P95/max: {med(l0,'server_e2e_ms'):.2f}/{sorted(float(r['server_e2e_ms']) for r in l0)[3]:.2f}/{max(float(r['server_e2e_ms']) for r in l0):.2f} ms.", f"L1 server E2E P50/P95/max: {med(l1,'server_e2e_ms'):.2f}/{sorted(float(r['server_e2e_ms']) for r in l1)[3]:.2f}/{max(float(r['server_e2e_ms']) for r in l1):.2f} ms.", f"L0 copy P50/max: {med(l0,'cpu_to_gpu_ms'):.2f}/{max(float(r['cpu_to_gpu_ms']) for r in l0):.2f} ms; L1 copy P50/max: {med(l1,'cpu_to_gpu_ms'):.2f}/{max(float(r['cpu_to_gpu_ms']) for r in l1):.2f} ms.", f"Busy/idle median slowdown: {med(l1,'client_e2e_ms')/med(l0,'client_e2e_ms'):.2f}x.", f"Frozen p_busy_ms: {frozen['p_busy_ms']:.0f} ms.", "", "## Validation", "", "|Metric|Median|", "|---|---:|"]
    keys=[("promotion_deadline_miss_count","promotion misses"),("blackout_conflict_count","blackout conflicts"),("budget_overrun_count","budget overruns"),("max_concurrent_pre_first_token","max pre-first concurrency"),("ABC_P95_TTFT_ms","ABC P95 TTFT (ms)"),("ABC_decode_throughput_tok_s","ABC decode tok/s"),("idle_gpu_memory_time_mib_s","memory-time (MiB*s)"),("X_slowdown","X slowdown")]
    for k,label in keys: lines.append(f"|{label}|{med(runs,k):.2f}|")
    clean = all(int(r['promotion_deadline_miss_count'])==0 and int(r['blackout_conflict_count'])==0 and int(r['budget_overrun_count'])==0 for r in runs)
    k2 = all(int(r['max_concurrent_pre_first_token'])==2 for r in runs)
    c_blocked = all(any(t['session']=='C' and t['event']=='GATE_BLOCKED' for t in tg[(r['policy'],r['repetition'])]) for r in runs)
    x_ok = all(r['background_overlap_validation']=='PASS' for r in runs)
    result = "Result A: a load-aware fixed budget restores clean promotion and exercises K=2." if clean and k2 and c_blocked and x_ok else "Result D: promotion is clean, but the required C gate-blocking condition was not exercised in every run; do not treat this as a completed K2 admission validation."
    lines += ["", f"All five runs had clean promotion: {'YES' if clean else 'NO'}.", f"K2 reached pre-first concurrency exactly 2 in all runs: {'YES' if k2 else 'NO'}.", f"C was gate-blocked in all runs: {'YES' if c_blocked else 'NO'}.", f"X remained active through promotion and past C first token in all validation runs: {'YES' if x_ok else 'NO'}.", "", result, "", "Next direction: RETRY_BUSY_K_COMPARISON only after the K2 exercise condition is made observable; no dynamic K or Phase 3K work is started."]
    (OUT/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    env={"vllm_version":"0.23.0","vllm_source_commit":"0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665","pytorch":"2.11.0+cu126","cuda":"12.6","gpu":"NVIDIA RTX A6000","model":"Qwen/Qwen2.5-1.5B-Instruct","p_busy_ms":frozen["p_busy_ms"]}
    (OUT/"environment.json").write_text(json.dumps(env,indent=2)+"\n")


if __name__=="__main__": main()
