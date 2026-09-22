import csv,json,statistics as S
from pathlib import Path
O=Path(__file__).resolve().parents[2]/"results"/"phase3p"; P=["OnDemand-K2","Blocking70-K2","Async40-K2"]
def R(n):
 with open(O/n,newline="",encoding="utf8") as f:return list(csv.DictReader(f))
def F(r,k):return float(r[k])
def M(xs):return S.median(xs)
def Q(xs):return S.quantiles(xs,n=20,method="inclusive")[18]
run,ses,nat,bg,x=R("run_metrics.csv"),R("session_metrics.csv"),R("async_prefetch_metrics.csv"),R("background_metrics.csv"),R("x_only_baseline.csv")
assert (len(run),len(ses),len(nat),len(bg),len(x))==(9,54,36,9,3)
for p in P:assert len([r for r in run if r["policy"]==p])==3 and len([r for r in ses if r["policy"]==p])==18
for p in ["Async40-K2","Blocking70-K2"]: assert len([r for r in nat if r["policy"]==p])==18 
keys=["ABCDEF_P50_effective_resume_ms","ABCDEF_P95_effective_resume_ms","ABCDEF_max_effective_resume_ms","ABC_P95_effective_resume_ms","DEF_P95_effective_resume_ms","ABCDEF_P95_completion_ms","ABCDEF_decode_tok_s","combined_decode_tok_s","full_makespan_ms","ABC_memory_time_mib_s","DEF_memory_time_mib_s","total_memory_time_mib_s"]
base=M([F(r,"decode_duration_ms") for r in x]); rows=[]
for p in P:
 q=[r for r in run if r["policy"]==p]; z={"policy":p};z.update({k:M([F(r,k) for r in q]) for k in keys});z["X_baseline_decode_ms"]=base;z["X_slowdown"]=M([F(r,"X_decode_duration_ms") for r in q])/base;rows.append(z)
A=next(z for z in rows if z["policy"]=="Async40-K2");B=next(z for z in rows if z["policy"]=="Blocking70-K2")
asr=[r for r in ses if r["policy"]=="Async40-K2"];an=[r for r in nat if r["policy"]=="Async40-K2"]; it=[F(r,"issue_to_T6_ms") for r in an];abc=[F(r,"issue_to_T6_ms") for r in an if r["session"] in "ABC"];de=[F(r,"issue_to_T6_ms") for r in an if r["session"] in "DEF"]
ready=sum(F(r,"scheduler_arrival_ms")>=F(r,"T6_ms") for r in asr);local=sum(F(r,"local_cache_hit_tokens")==4096 and F(r,"external_kv_transfer_tokens")==0 for r in asr);remote=sum(r["WAITING_FOR_REMOTE_KVS_seen"]=="True" for r in asr);native=sum(F(r,"external_tokens_loaded")==4096 and F(r,"model_forward_count")==0 and F(r,"generated_token_count")==0 and F(r,"mixed_prefetch_model_step_count")==0 and r["status"]=="ok" for r in an);tokens=sum(int(float(r["generated_token_count"]))==32 for r in ses);bgok=sum(r["background_validation"]=="PASS" for r in run);k2=sum(int(float(r["max_pre_first_concurrency"]))==2 and r["gate_validation"]=="PASS" for r in run);enough=max(it)<40
result="A" if ready==18 and local==18 and remote==0 and native==18 and tokens==54 and bgok==9 and k2==9 and enough and A["ABCDEF_P95_effective_resume_ms"]<=1.1*B["ABCDEF_P95_effective_resume_ms"] and A["total_memory_time_mib_s"]<=.8*B["total_memory_time_mib_s"] and A["combined_decode_tok_s"]>=.95*B["combined_decode_tok_s"] else "B"
report={"raw_validation":{"run_rows":len(run),"session_rows":len(ses),"native_rows":len(nat),"background_rows":len(bg),"x_only_rows":len(x)},"async":{"internal_ready":ready,"local_hit":local,"remote_wait":remote,"native_correct":native,"token_correct":tokens,"background_valid":bgok,"k2_valid":k2,"issue_to_T6":{"ABC_median":M(abc),"ABC_max":max(abc),"DEF_median":M(de),"DEF_max":max(de),"overall_median":M(it),"overall_p95":Q(it),"overall_max":max(it)},"p_async_40_sufficient":enough},"x_only":{"tokens":192,"denominator":191,"median_decode_ms":base},"policies":rows,"result":result}
with open(O/"policy_summary.csv","w",newline="",encoding="utf8") as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
(O/"validation_report.json").write_text(json.dumps(report,indent=2)+"\n")
t=["# Phase 3P-FIX","","Frozen Phase 3P raw CSVs were postprocessed only. No GPU, vLLM, Qwen, or native prefetch workload was rerun.","","Raw validation: 9 runs; 54 sessions; 36 native records; 9 background records; 3 X-only records.","","Async internal-ready/local-hit/remote-wait: %d/18, %d/18, %d. Native correctness %d/18; token correctness %d/54; background %d/9; K2 %d/9."%(ready,local,remote,native,tokens,bgok,k2),"Async issue-to-T6 ms: ABC median/max %.2f/%.2f; DEF %.2f/%.2f; all median/P95/max %.2f/%.2f/%.2f; p_async=40 sufficient %s."%(M(abc),max(abc),M(de),max(de),M(it),Q(it),max(it),"YES" if enough else "NO"),"","|Policy|A-F P95|ABC P95|DEF P95|Completion P95|Decode tok/s|X slowdown|Combined tok/s|Makespan|Memory|","|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
for z in rows:t.append("|%s|%.2f|%.2f|%.2f|%.2f|%.2f|%.3f|%.2f|%.2f|%.2f|"%(z["policy"],z["ABCDEF_P95_effective_resume_ms"],z["ABC_P95_effective_resume_ms"],z["DEF_P95_effective_resume_ms"],z["ABCDEF_P95_completion_ms"],z["ABCDEF_decode_tok_s"],z["X_slowdown"],z["combined_decode_tok_s"],z["full_makespan_ms"],z["total_memory_time_mib_s"]))
t+=["","X-only baseline: 192 tokens, 191 post-first tokens, median %.2f ms."%base,"Async vs Blocking: P95 %.2f%%; DEF %.2f%%; memory %.2f%%; combined throughput %.2f%%."%((A["ABCDEF_P95_effective_resume_ms"]/B["ABCDEF_P95_effective_resume_ms"]-1)*100,(A["DEF_P95_effective_resume_ms"]/B["DEF_P95_effective_resume_ms"]-1)*100,(A["total_memory_time_mib_s"]/B["total_memory_time_mib_s"]-1)*100,(A["combined_decode_tok_s"]/B["combined_decode_tok_s"]-1)*100),"Result %s; next direction: Phase 3Q stable async workload."%result]
(O/"summary.md").write_text("\n".join(t)+"\n")
(O/"config.json").write_text(json.dumps({"phase":"3P-FIX","raw_replay":False,"p_async_ms":40,"p_blocking_ms":70,"x_tokens":192,"post_first_denominator":191,"gate_limit":2},indent=2)+"\n")
(O/"environment.json").write_text(json.dumps({"source":"frozen Phase 3P CSV artifacts","gpu_workload_rerun":False},indent=2)+"\n")
(O/"implementation.md").write_text("# Phase 3P-FIX\n\nReads immutable raw CSV artifacts only. Validates full schema; recomputes policy medians with real schema names; checks T6-before-arrival, local-hit/no-external recovery, native zero-forward fields, 192-token counts, K2 and background validation; derives X slowdown from the 192-token/191-post-first baseline.\n")
(O/"commands.sh").write_text("python3 experiments/phase3p/postprocess.py\npython3 -m pytest -q tests/test_phase3p.py\n")

