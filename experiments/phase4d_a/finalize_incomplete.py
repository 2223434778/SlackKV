from __future__ import annotations
import csv, hashlib, json, platform, re, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"results/phase4d_a"
def j(name): return json.loads((OUT/name).read_text(encoding="utf-8"))
def w(name,value): (OUT/name).write_text(json.dumps(value,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def cmd(args):
    try: return subprocess.run(args,check=True,text=True,capture_output=True).stdout.strip()
    except Exception as e: return f"UNAVAILABLE: {type(e).__name__}: {e}"
def main():
    sha=hashlib.sha256((ROOT/"results/phase4c/hybrid_workload.csv").read_bytes()).hexdigest()
    od=j("integration_ondemand_retry.json"); rv=j("integration_reservation_retry.json"); nv=j("integration_naive_retry.json")
    log=(OUT/"server_logs/integration_naivejit_async_k2_2.log").read_text(encoding="utf-8",errors="replace")
    events={e:len(re.findall(r'"event":"' + e + r'"',log)) for e in ("RESUME_ARRIVED","WAITING_FOR_REMOTE_KVS","REMOTE_KV_READY","RESUME_ADMITTED","FIRST_TOKEN","REQUEST_FINISHED")}
    diag={"policy":"NaiveJIT-Async-K2","status":"INTEGRATION_FAILED_NONPROGRESS",
      "native_http_200":len(re.findall(r"POST /slackkv/prefetch.*200 OK",log)),
      "native_http_500_on_shutdown":len(re.findall(r"POST /slackkv/prefetch.*500",log)),
      "connector_only_step_lines":log.count("SLACKKV_CONNECTOR_ONLY_STEP"),"scheduler_event_counts":events,
      "client_error":nv.get("error"),"read_timeout_seconds":120,
      "oom_or_server_crash_observed":bool(re.search(r"CUDA out of memory|OutOfMemoryError|EngineCore.*exited",log))}
    w("naive_failure_diagnostic.json",diag)
    gate={"pass":False,"valid_policy_runs":2,"required_policy_runs":3,"stage_b_started":False,
      "runs":[{"policy":"OnDemand-K2",**od},{"policy":"NaiveJIT-Async-K2",**diag,"valid":False},
              {"policy":"SlackKV-Oracle-ReserveLease-K2",**rv}],
      "stop_reason":"NaiveJIT integration failed to complete all eight resumes; frozen gate requires 8/8 before Stage B."}
    w("integration_gate.json",gate)
    cap=j("capacity_config.json"); svc=j("service_budget.json")
    base=list(csv.DictReader((OUT/"x_only_baseline.csv").open(encoding="utf-8")))
    stage=j("stage_a0.json")
    w("config.json",{"phase":"4D-A","start_commit":"2fea2d810305ef94d65068c812a236b08a19e73c",
      "workload_sha256":sha,"policies":["OnDemand-K2","NaiveJIT-Async-K2","SlackKV-Oracle-ReserveLease-K2"],
      "episodes":8,"boundary":0,"gpu_kv_bytes":2147483648,"cpu_kv_bytes":8589934592,"block_size":16,
      "total_gpu_kv_blocks":cap["total_gpu_kv_blocks"],"reservation_budget_blocks":cap["reservation_budget_blocks"],
      "reservation_fraction":0.75,"P_SAFE_MS":svc["P_SAFE_MS"],"resume_guard_ms":110,"K2":2,
      "stage_A0":stage,"measured_matrix_started":False})
    mem={k:v.strip() for line in Path("/proc/meminfo").read_text().splitlines()
      if line.startswith(("MemTotal:","MemAvailable:")) for k,v in [line.split(":",1)]}
    w("environment.json",{"python":platform.python_version(),"torch_cuda":cmd([sys.executable,"-c","import torch; print(torch.__version__, torch.version.cuda)"]),
      "transformers":cmd([sys.executable,"-c","import transformers; print(transformers.__version__)"]),
      "vllm":cmd([sys.executable,"-c","from importlib.metadata import version; print(version('vllm'))"]),
      "gpu_driver":cmd(["nvidia-smi","--query-gpu=name,driver_version","--format=csv,noheader"]),
      "gpu_inventory_system":cmd(["nvidia-smi","--query-gpu=name","--format=csv,noheader"]),
      "gpu_visible_to_process":cmd([sys.executable,"-c","import torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"]),
      "topology":cmd(["nvidia-smi","topo","-m"]),
      "meminfo":mem,"cuda_visible_devices":"0","model_snapshot":"989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
      "vllm_source":"/media/a6000/D/ZhaoBinHe/slackkv-phase2c-fix-sourceprobe","vllm_commit":"0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665",
      "max_model_len":32768,"gpu_kv_bytes":2147483648,"cpu_kv_bytes":8589934592,"actual_block_size":16,
      "connector":"OffloadingConnector","prefix_caching":True})
    w("validation_report.json",{"phase":"4D-A","status":"INCOMPLETE_INTEGRATION_GATE","workload_sha256":sha,
      "calibration_valid_measurements":svc["valid_measurements"],"P_SAFE_MS":svc["P_SAFE_MS"],
      "integration_valid":"2/3","measured_runs":"0/9","StageB_started":False,"NaiveJIT_diagnostic":diag,
      "official_result":None,"result_note":"No A/B/C result assigned; comparative matrix was not run."})
    (OUT/"implementation.md").write_text("Phase 3S residency-lease vLLM patch reused unchanged; no new vLLM source modifications. Stage 0A and 0D passed; X-only baseline completed 3/3; OnDemand and Reservation integrations passed. NaiveJIT repeatedly stalled with three pending connector-only promotions and did not complete its eight resumes. Stage B was not started.\n",encoding="utf-8")
    (OUT/"vllm_phase4d_a_delta.patch").write_text("Phase 3S residency-lease vLLM patch reused unchanged.\nNo new vLLM source modification.\n",encoding="utf-8")
    median=sorted(float(r["decode_duration_ms"]) for r in base)[1]
    medt=sorted(float(r["TPOT_ms"]) for r in base)[1]
    (OUT/"summary.md").write_text(
      "# Phase 4D-A — integration gate stopped\n\n"
      "**INCOMPLETE. No Stage B runs; no RESULT_A/B/C assigned.** The required integration gate did not pass because NaiveJIT-Async-K2 failed to finish all eight requests.\n\n"
      "## Frozen setup\n\n"
      f"- Phase 4C workload SHA256: {sha}.\n- Boundary 0, 8 episodes; cached tokens 168016, blocks 10501, minimum H2D 4594.1875 MiB.\n"
      f"- Runtime capacity: {cap['total_gpu_kv_blocks']} blocks; frozen 75% budget {cap['reservation_budget_blocks']} blocks ({cap['reservation_budget_MiB']} MiB); at most {cap['max_simultaneous_frozen_jobs']} frozen jobs.\n"
      f"- Real-size calibration: {svc['valid_measurements']}/9 valid; max issue-to-T6 {svc['max_observed_issue_to_T6_ms']:.3f} ms; frozen P_SAFE {svc['P_SAFE_MS']} ms. Stage A0 prefix calibration passed.\n"
      f"- X-only baseline: {len(base)}/3; median decode {median:.3f} ms; median TPOT {medt:.3f} ms.\n\n"
      "## Integration only\n\n"
      f"- OnDemand: valid, 8/8 tokens, local/external {od['local_source_tokens']}/{od['external_source_tokens']}, K2 and X checks pass.\n"
      f"- Reservation: valid, 8/8 tokens, local/external {rv['local_source_tokens']}/{rv['external_source_tokens']}, promotions {rv['promotion_count']}, peak logical/physical {rv['reservation_peak_blocks']}/{rv['physical_lease_peak_blocks']} blocks, lease leaks {rv['lease_leaks']}.\n"
      f"- Naive: latest run had {diag['native_http_200']}/8 completed native calls and {diag['native_http_500_on_shutdown']} pending calls errored on shutdown; scheduler observed {events['RESUME_ARRIVED']} arrivals, {events['FIRST_TOKEN']} first tokens, {events['REQUEST_FINISHED']} finishes and {diag['connector_only_step_lines']} connector-only steps before 120-second client timeout. This is consistent with unprotected residency non-progress, not a latency measurement.\n\n"
      "## Stop decision\n\n"
      "The frozen protocol requires eight completed resumes for every integration policy before Stage B. Naive did not progress to request completion; therefore no measured matrix or comparative performance claim is valid. This is not classified as a SlackKV lease failure or an invalid runtime environment.\n",encoding="utf-8")
    headers={"run_metrics.csv":"policy,repetition,status","session_metrics.csv":"policy,repetition,episode_id,status",
      "prefetch_metrics.csv":"policy,repetition,episode_id,status","scheduler_trace.csv":"policy,repetition,session,event,step_id,timestamp_ms",
      "reservation_decisions.csv":"policy,repetition,episode_id,decision","reservation_events.csv":"policy,repetition,episode_id,event",
      "lease_events.csv":"policy,repetition,lease_key,event","h2d_accounting.csv":"policy,repetition,native_h2d_bytes,external_h2d_bytes,total_h2d_bytes",
      "background_metrics.csv":"policy,repetition,X_first_token_ms,X_finished_ms,X_decode_duration_ms,X_TPOT_ms,X_slowdown,active_through_last_first_token"}
    for name,header in headers.items():
      if not (OUT/name).exists(): (OUT/name).write_text(header+"\n",encoding="utf-8")
    (OUT/"commands.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\ncd /media/a6000/D/ZhaoBinHe/SlackKV\nPYTHONPATH=. python -m pytest -q tests/test_phase4d_a.py\nPYTHONPATH=. python -m experiments.phase4d_a.run_policy --stage policy\n# Stage B intentionally not run: NaiveJIT integration did not complete.\n",encoding="utf-8")
if __name__=="__main__": main()
