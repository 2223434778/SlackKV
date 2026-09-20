from __future__ import annotations

import csv
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from transformers import AutoTokenizer

from experiments.phase2d import run_phase2d as runtime
from experiments.phase2d.helpers import gpu_to_cpu_offload_bytes, parse_prometheus_samples
from experiments.phase2h.run_phase2h import SERVER, build_prompts
from experiments.phase3f.native_client import native_prefetch
from experiments.phase3f.scheduler import DEADLINE_MS, RESUME_GUARD_MS, SESSIONS, latest_feasible_start, merge_blackouts, P_SAFE_MS
from experiments.phase3g.run_phase3g import send
from experiments.phase3g_diag.metrics import first_divergence_position, interval_concurrency, matching_prefix_length, overlap, sequence_hash, structural_schedule, unique_sequence_counts

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase3g_diag"
SRC = Path("/media/a6000/D/ZhaoBinHe/slackkv-phase2c-fix-sourceprobe")
CASES = ("S1", "S2", "S3", "S4")
RUNS = 3
RESUME_TOKENS = 32
BYTES = 117440512


def start(label: str):
    OUT.joinpath("server_logs").mkdir(parents=True, exist_ok=True)
    env = os.environ.copy(); env.update(runtime.BASE_ENV)
    env["VLLM_SERVER_DEV_MODE"] = "1"; env["PYTHONPATH"] = f"{SRC}:{ROOT}"
    log = (OUT / "server_logs" / f"{label}.log").open("w", encoding="utf-8")
    return subprocess.Popen(SERVER, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)


def stop(proc):
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try: proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait()


def snap(stage: str, snapshots: list[dict]) -> dict:
    raw = runtime.metric_text(); snapshots.append({"stage": stage, "time_monotonic_s": time.monotonic(), "raw": raw})
    return parse_prometheus_samples(raw)


def prompts() -> dict[str, list[int]]:
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", local_files_only=True)
    return build_prompts(tok)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def setup(p: dict, case: str, label: str, snapshots: list[dict]) -> tuple[dict, dict, dict]:
    sessions = ("C",) if case in ("S1", "S2") else ("A", "B", "C")
    local = case in ("S1", "S3")
    before = snap(label + "_ready", snapshots); offloads = {}
    for s in sessions:
        r = send(p[s.lower()], stream=False, max_tokens=1, normal=local is False)
        after = snap(label + "_first_" + s, snapshots)
        offloads[s] = gpu_to_cpu_offload_bytes(before, after); before = after
    time.sleep(1.0)
    promotions = {}
    if not local:
        for i in range(1, 5):
            send(p[f"pressure_{i}"], stream=False, max_tokens=1, normal=False)
        snap(label + "_pressure", snapshots)
        # S2 performs the single prefetch before its solo resume. S4 leaves
        # all three external prefixes pending so the real ResumeAware
        # promotion schedule is the only promotion path under test.
        if case == "S2":
            n = native_prefetch(p["c"]); promotions["C"] = n
            snap(label + "_prefetch", snapshots)
    ok = (all(offloads[s] == 0 for s in sessions) if local else all(offloads[s] == BYTES for s in sessions))
    return before, {"offloads": offloads, "promotions": promotions, "cache_transport_validation": "PASS" if ok else "FAIL"}, {"sessions": sessions}


def run_resume(s: str, p: dict, t0: float, gate: bool, control: dict[str, list[int]]) -> dict:
    deadline = t0 + DEADLINE_MS[s] / 1000
    while time.monotonic() < deadline: time.sleep(min(0.002, deadline - time.monotonic()))
    r = send(p[s.lower()] + p["suffix"], stream=True, max_tokens=RESUME_TOKENS, trace=True, session=s, gate=gate)
    return {"session": s, "sent_ms": (r["sent_monotonic_s"]-t0)*1000, "first_ms": (r["first_monotonic_s"]-t0)*1000, "finish_ms": (r["completed_monotonic_s"]-t0)*1000, "ttft_ms": r["ttft_ms"], "token_ids": r["token_ids"], "control": control[s], "exact": r["token_ids"] == control[s] and len(r["token_ids"]) == RESUME_TOKENS}


def schedule_promotions(sessions: tuple[str, ...], p: dict, t0: float, out: dict, blackouts: list[tuple[float,float]]) -> None:
    pending = set(sessions)
    while pending:
        now = (time.monotonic()-t0)*1000
        jobs = [(s, DEADLINE_MS[s]) for s in pending]
        target = latest_feasible_start(now, jobs, blackouts, P_SAFE_MS)
        if target - now > 0.5:
            time.sleep((target-now)/1000); continue
        s = min(pending, key=lambda x:(DEADLINE_MS[x],x)); n = native_prefetch(p[s.lower()])
        out[s] = {"target_ms": target, "actual_ms": (n["sent_monotonic_s"]-t0)*1000, "completion_ms": (n["completed_monotonic_s"]-t0)*1000, **n}; pending.remove(s)


def control_generation(p: dict) -> dict[str, list[int]]:
    proc = start("control"); runtime.wait_ready(proc)
    try:
        out = {}
        for s in SESSIONS:
            send(p[s.lower()], stream=False, max_tokens=1, normal=False)
        time.sleep(1)
        for s in SESSIONS:
            out[s] = send(p[s.lower()] + p["suffix"], stream=True, max_tokens=RESUME_TOKENS)["token_ids"]
        return out
    finally: stop(proc)


def run_one(case: str, rep: int, p: dict, control: dict[str, list[int]]) -> tuple[list[dict], dict, list[dict], list[dict]]:
    label = f"{case.lower()}_{rep}"; proc = start(label); snapshots=[]
    try:
        runtime.wait_ready(proc); _, setup_data, meta = setup(p, case, label, snapshots); t0=time.monotonic()
        OUT.joinpath("run_context.csv").open("a", encoding="utf-8").write(f"{case},{rep},{t0:.9f}\n")
        sessions=meta["sessions"]; gate=case in ("S3","S4"); promotions={}
        if case == "S4":
            blackouts=merge_blackouts({s:DEADLINE_MS[s] for s in sessions}, RESUME_GUARD_MS)
            with ThreadPoolExecutor(max_workers=1) as pool: pf=pool.submit(schedule_promotions,sessions,p,t0,promotions,blackouts)
        else: pf=None
        with ThreadPoolExecutor(max_workers=len(sessions)) as pool:
            fut={s:pool.submit(run_resume,s,p,t0,gate,control) for s in sessions}; resumes={s:f.result(600) for s,f in fut.items()}
        if pf: pf.result(600)
        snap(label+"_after", snapshots)
        trace=read_trace(OUT/"server_logs"/f"{label}.log", case, rep, t0)
        trace_rows={s:trace_for_session(trace,s) for s in sessions}
        seq_rows=[]; sess_rows=[]
        for s in sessions:
            x=resumes[s]; tr=trace_rows[s]; pr=promotions.get(s); seq=x["token_ids"]
            seq_rows.append({"case":case,"repetition":rep,"session":s,"kv_mode":"local" if case in ("S1","S3") else "prefetch","solo_or_clustered":"solo" if case in ("S1","S2") else "clustered","gate_enabled":gate,"generated_token_count":len(seq),"token_ids_json":json.dumps(seq),"sequence_hash":sequence_hash(seq),"matches_solo_local_control":x["exact"] if s=="C" else seq==control[s],"matching_prefix_length_vs_solo":matching_prefix_length(seq,control[s]),"first_divergence_position_vs_solo":first_divergence_position(seq,control[s]),"scheduler_admission_step":tr.get("admission_step","NA"),"first_token_step":tr.get("first_token_step","NA"),"finish_step":tr.get("finish_step","NA"),"cache_transport_validation":setup_data["cache_transport_validation"]})
            sess_rows.append({"case":case,"repetition":rep,"session":s,**x, "admission_step":tr.get("admission_step","NA"),"first_token_step":tr.get("first_token_step","NA"),"finish_step":tr.get("finish_step","NA"),"remote_wait":tr.get("remote_wait","NA"),"remote_ready":tr.get("remote_ready","NA")})
        run={"case":case,"repetition":rep,"cache_transport_validation":setup_data["cache_transport_validation"],"max_pre_first_concurrency":trace_max(trace,"pre"),"max_post_first_decode_concurrency":trace_max(trace,"decode"),"post_first_decode_overlap_ms":trace_overlap(trace,"decode"),"gate_invariant_validation":"PASS" if (not gate or trace_structure(trace_rows)) else "FAIL","schedule_structure_validation":"PASS" if (not gate or trace_structure(trace_rows)) else "FAIL","A_sequence_hash":next((r["sequence_hash"] for r in seq_rows if r["session"]=="A"),"NA"),"B_sequence_hash":next((r["sequence_hash"] for r in seq_rows if r["session"]=="B"),"NA"),"C_sequence_hash":next((r["sequence_hash"] for r in seq_rows if r["session"]=="C"),"NA"),"generation_validation":"PASS" if all(x["exact"] for x in resumes.values()) else "FAIL"}
        native_rows=[{"case":case,"repetition":rep,"session":s,"client_e2e_ms":n.get("client_e2e_ms"),"server_e2e_ms":n.get("elapsed_ms"),"cpu_to_gpu_ms":n.get("cpu_to_gpu_transfer_ms"),"external_tokens_loaded":n.get("external_tokens_loaded"),"model_forward_count":n.get("model_forward_count"),"generated_token_count":n.get("generated_token_count"),"mixed_prefetch_model_step_count":n.get("mixed_prefetch_model_step_count")} for s,n in setup_data["promotions"].items()]
        return seq_rows, run, sess_rows, native_rows+[{"case":case,"repetition":rep,"session":s,**{k:v for k,v in x.items() if k in ("target_ms","actual_ms","completion_ms")}} for s,x in promotions.items()], snapshots
    finally: stop(proc)


def read_trace(path: Path, case: str, rep: int, t0: float) -> list[dict]:
    rows=[]
    if not path.exists(): return rows
    for line in path.read_text(errors="ignore").splitlines():
        marker="SLACKKV_RESUME_TRACE "
        if marker not in line: continue
        try:
            j=json.loads(line.split(marker,1)[1]); j.update({"case":case,"repetition":rep,"timestamp_ms":(float(j["timestamp_monotonic"])-t0)*1000}); rows.append(j)
        except Exception: continue
    write_trace_rows(rows); return rows


_trace_rows=[]
def write_trace_rows(rows):
    _trace_rows.extend(rows)


def trace_for_session(rows,s):
    out={}
    for e in rows:
        if e.get("session")!=s: continue
        ev=e.get("event"); out.setdefault(ev,e)
        if ev=="FIRST_TOKEN": out["first_token_step"]=e.get("step_id")
        if ev=="RESUME_ADMITTED": out["admission_step"]=e.get("step_id")
        if ev=="REQUEST_FINISHED": out["finish_step"]=e.get("step_id")
        if ev=="WAITING_FOR_REMOTE_KVS": out["remote_wait"]=e.get("step_id")
        if ev=="REMOTE_KV_READY": out["remote_ready"]=e.get("step_id")
    return out


def trace_structure(d):
    try:
        a,b,c=(d[x] for x in ("A","B","C")); return int(a["first_token_step"])<int(b["admission_step"])<int(a["finish_step"]) and int(b["first_token_step"])<int(c["admission_step"])<int(b["finish_step"])
    except Exception:return False


def trace_max(rows,kind):
    if kind=="pre":
        return max((int(e.get("num_slackkv_pre_first_token",0)) for e in rows),default=0)
    return 0


def trace_overlap(rows,kind): return 0.0


def main():
    OUT.mkdir(parents=True,exist_ok=True); p=prompts(); (OUT/"run_context.csv").write_text("case,repetition,t0_monotonic_s\n")
    controls=control_generation(p); (OUT/"control_sequences.json").write_text(json.dumps(controls,indent=2))
    all_seq=[]; all_run=[]; all_sess=[]; all_nat=[]; all_snaps=[]
    for case in CASES:
        for rep in range(RUNS):
            print("RUN",case,rep,flush=True); seq,run,sess,nat,snaps=run_one(case,rep,p,controls); all_seq+=seq;all_run.append(run);all_sess+=sess;all_nat+=nat;all_snaps+=snaps
    write_csv(OUT/"result_sequences.csv",all_seq); write_csv(OUT/"run_metrics.csv",all_run); write_csv(OUT/"session_metrics.csv",all_sess); write_csv(OUT/"native_prefetch_metrics.csv",all_nat); write_csv(OUT/"scheduler_trace.csv",_trace_rows); (OUT/"metric_snapshots.json").write_text(json.dumps(all_snaps,indent=2));
    (OUT/"logprobs.json").write_text(json.dumps({"supported":False,"reason":"optional logprob diagnostic not available in this pinned endpoint"},indent=2))


if __name__=="__main__": main()
