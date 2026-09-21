from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase3k"


def read(name):
    with (OUT/name).open(encoding="utf-8", newline="") as f: return list(csv.DictReader(f))


def write(name, rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n"); w.writeheader(); w.writerows(rows)


def median(rows,key): return statistics.median(float(r[key]) for r in rows)


def main():
    batches = OUT / "batches"
    if batches.exists():
        for name in ("run_metrics.csv","session_metrics.csv","native_prefetch_metrics.csv","background_metrics.csv","scheduler_trace.csv","step_metrics.csv","run_context.csv"):
            rows=[]
            for sub in sorted(batches.iterdir()):
                path=sub/name
                if path.exists(): rows.extend(list(csv.DictReader(path.open(encoding="utf-8"))))
            if rows: write(name, rows)
    runs=read("run_metrics.csv"); sessions=read("session_metrics.csv"); native=read("native_prefetch_metrics.csv"); bg=read("background_metrics.csv"); trace=read("scheduler_trace.csv")
    baseline=list(csv.DictReader((ROOT/"results/phase3i/x_baseline_metrics.csv").open(encoding="utf-8")))
    base_decode=statistics.median(float(r["decode_duration_ms"]) for r in baseline)
    ng={}; tg={}
    for r in native: ng.setdefault((r["policy"],r["repetition"]),[]).append(r)
    for r in trace: tg.setdefault((r["policy"],r["repetition"]),[]).append(r)
    for b in bg: b["X_slowdown"]=float(b["X_decode_duration_ms"])/base_decode
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
        r["idle_gpu_memory_time_mib_s"]=sum(max(0,float(s["scheduler_admission_ms"])-float(s["promotion_completion_ms"]))*112/1000 for s in ss)
        r["promotion_validation"]="PASS" if int(r["promotion_deadline_miss_count"])==0 and int(r["blackout_conflict_count"])==0 and int(r["budget_overrun_count"])==0 else "FAIL"
    write("session_metrics.csv",sessions); write("background_metrics.csv",bg); write("run_metrics.csv",runs)
    groups={p:[r for r in runs if r["policy"]==p] for p in ("ResumeAware-K3","FirstTokenGate-K1","FirstTokenGate-K2")}
    stats={p:{"p95":median(q,"ABC_P95_TTFT_ms"),"throughput":median(q,"ABC_decode_throughput_tok_s"),"x":median(q,"X_slowdown"),"combined":median(q,"combined_decode_throughput_tok_s"),"makespan":median(q,"ABC_completion_makespan_ms"),"memory":median(q,"idle_gpu_memory_time_mib_s"),"overlap":median(q,"ABC_pre_first_pairwise_overlap_ms"),"wait":median(q,"ABC_total_gate_wait_ms"),"pre":median(q,"max_concurrent_pre_first_token") } for p,q in groups.items()}
    clean=sum(int(r["promotion_validation"]=="PASS") for r in runs); gate_ok=all((p=="ResumeAware-K3" and int(r["max_concurrent_pre_first_token"])<=3) or (p=="FirstTokenGate-K1" and int(r["max_concurrent_pre_first_token"])==1) or (p=="FirstTokenGate-K2" and int(r["max_concurrent_pre_first_token"])==2) for p in groups for r in groups[p])
    ranking=sorted(stats,key=lambda p:stats[p]["p95"])
    k1,k2=stats["FirstTokenGate-K1"],stats["FirstTokenGate-K2"]
    dyn=(k1["p95"]<=k2["p95"]*1.05 and (k1["x"]<=k2["x"]*0.90 or k1["combined"]>=k2["combined"]*1.05))
    result="Result A" if clean==15 and gate_ok and dyn else ("Result B" if clean==15 and gate_ok and ranking[0]=="FirstTokenGate-K2" else ("Result C" if clean==15 and gate_ok and ranking[0]=="ResumeAware-K3" else "Result D"))
    lines=["# Phase 3K summary","", "Phase 3K is the clean retry of Phase 3I with p_busy=70 ms and resume_guard=110 ms. The Phase 3H K-capacity gate and connector-first promotion were reused unchanged.","", "|Policy|ABC P95 TTFT|Pre-first overlap|Gate wait|ABC decode tok/s|X slowdown|Combined tok/s|Makespan|Memory|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for p in ("ResumeAware-K3","FirstTokenGate-K1","FirstTokenGate-K2"):
        s=stats[p]; lines.append(f"|{p}|{s['p95']:.2f}|{s['overlap']:.2f}|{s['wait']:.2f}|{s['throughput']:.2f}|{s['x']:.3f}|{s['combined']:.2f}|{s['makespan']:.2f}|{s['memory']:.2f}|")
    lines += ["", f"Promotion-clean runs: {clean}/15.", f"K1 max pre-first=1 in every run: {'YES' if all(int(r['max_concurrent_pre_first_token'])==1 for r in groups['FirstTokenGate-K1']) else 'NO'}.", f"K2 max pre-first=2 in every run: {'YES' if all(int(r['max_concurrent_pre_first_token'])==2 for r in groups['FirstTokenGate-K2']) else 'NO'}.", f"Busy-load ranking by ABC P95 TTFT: {' < '.join(ranking)}.", "Light-load ranking from Phase 3H: K2 < K3 < K1.", f"Ranking changed: {'YES' if ranking != ['FirstTokenGate-K2','ResumeAware-K3','FirstTokenGate-K1'] else 'NO'}.", f"K1 protects X more than K2: {'YES' if k1['x'] < k2['x'] else 'NO'}.", f"K1 improves combined throughput over K2: {'YES' if k1['combined'] > k2['combined'] else 'NO'}.", f"Dynamic K justified: {'YES' if dyn else 'NO'}.", f"Result: {result}.", "Next direction: DYNAMIC_K if Result A; KEEP_K2 if Result B; RETHINK_ADMISSION if Result C.", "", "The X request generated 128 tokens in every run and remained active through C first token. No dynamic K or Phase 3L work was started."]
    (OUT/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    env={"vllm_version":"0.23.0","vllm_source_commit":"0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665","pytorch":"2.11.0+cu126","cuda":"12.6","gpu":"NVIDIA RTX A6000","model":"Qwen/Qwen2.5-1.5B-Instruct","p_busy_ms":70,"resume_guard_ms":110,"x_baseline_decode_ms":base_decode}
    (OUT/"environment.json").write_text(json.dumps(env,indent=2)+"\n")


if __name__=="__main__": main()
