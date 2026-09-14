#!/usr/bin/env python3
"""Phase 1B-Fix: closed-loop, context-capped TraceLab QueueJIT replay."""
import csv
import heapq
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCAL_DUCKDB = ROOT / "data/tracelab/python"
sys.path[:0] = [str(ROOT), str(LOCAL_DUCKDB)]
import duckdb

from slackkv.simulator.jobs import PromotionJob
from slackkv.simulator.policies import Policy, eligibility_time, queue_safe_idle_s
from slackkv.simulator.simulator import EPS, JobOutcome, SimulationResult, aggregate, percentile, simulate
from slackkv.simulator.transfer_model import TransferModel

KIB = 1024
FOOTPRINTS = (56 * KIB, 128 * KIB)
MAX_CONTEXT = 131072
PAUSE_BUCKETS = ((0,.05,"<50ms"),(.05,.1,"50-100ms"),(.1,.25,"100-250ms"),(.25,.5,"250-500ms"),(.5,1,"500ms-1s"),(1,5,"1-5s"),(5,math.inf,">=5s"))
KV_BUCKETS = ((0,1,"<1GB"),(1,4,"1-4GB"),(4,8,"4-8GB"),(8,math.inf,">=8GB"))
FIELDS = ("workload","footprint_bytes_per_token","concurrency","rho","seed","policy")

def write_csv(path, fields, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader()
        for row in rows:
            writer.writerow({key: format(value, ".9g") if isinstance(value, float) else value for key, value in row.items()})

def quantiles(values):
    return {f"p{int(q*100)}": percentile(values, q) for q in (.5,.9,.95,.99)}

def extract_events(db_path):
    con = duckdb.connect(str(db_path), read_only=True)
    query = """SELECT r.round_pk,r.provider,r.session_id,r.round_index,r.input_tokens_total,r.output_tokens,
      r.reasoning_output_tokens,string_agg(t.tool_name,'|' ORDER BY t.tool_index) tools,
      max(t.emitted_at) release_at,max(t.result_at) deadline_at,count(*) n,count(t.emitted_at) emits,count(t.result_at) results
      FROM rounds r JOIN tool_calls t USING(round_pk) GROUP BY ALL ORDER BY r.session_id,r.round_pk"""
    valid = []; skipped = Counter(); token_audit = []
    for row in con.sql(query).fetchall():
        pk, provider, session, round_index, input_tokens, output_tokens, reasoning, tools, release, deadline, n, emits, results = row
        if emits != n or results != n: skipped["missing_tool_timing"] += 1; continue
        if input_tokens is None or output_tokens is None: skipped["missing_token_count"] += 1; continue
        pause = (deadline-release).total_seconds()
        if pause <= 0: skipped["nonpositive_pause"] += 1; continue
        visible_output = max(0, output_tokens - (reasoning or 0))
        reusable = input_tokens + visible_output
        if reusable <= 0: skipped["nonpositive_reusable_tokens"] += 1; continue
        item = {"event_id": str(pk), "provider": provider, "session_id": session, "round_index": round_index,
                "tool_category": tools.split("|")[0].lower(), "release_epoch_s": release.timestamp(), "pause_s": pause,
                "input_tokens": input_tokens, "output_tokens": output_tokens, "reasoning_tokens": reasoning or 0,
                "visible_output_tokens": visible_output, "reusable_tokens": reusable}
        valid.append(item)
        token_audit.append({"provider": provider, "input_tokens": input_tokens, "output_tokens": output_tokens,
                            "reasoning_tokens": reasoning or 0, "visible_output_tokens": visible_output,
                            "reusable_tokens": reusable, "legacy_tokens": input_tokens + output_tokens})
    con.close()
    return valid, skipped, token_audit

def audit_semantics(db_path, output):
    con = duckdb.connect(str(db_path), read_only=True)
    aggregate = con.sql("""SELECT count(DISTINCT c.continuation_of_tool_call_id),count(*),
      sum(CASE WHEN cr.round_index > pr.round_index + 1 THEN 1 ELSE 0 END)
      FROM tool_calls c JOIN rounds cr USING(round_pk) JOIN tool_calls p ON c.continuation_of_tool_call_id=p.tool_call_id
      JOIN rounds pr ON p.round_pk=pr.round_pk WHERE c.continuation_of_tool_call_id IS NOT NULL""").fetchone()
    examples = con.sql("""WITH ranked AS (
      SELECT cr.session_id,pr.round_index parent_round,cr.round_index continuation_round,
        cr.round_index-pr.round_index round_gap,p.emitted_at parent_emitted,p.result_at parent_result,
        c.emitted_at continuation_emitted,c.result_at continuation_result,
        row_number() OVER (PARTITION BY c.continuation_of_tool_call_id ORDER BY cr.round_index,c.tool_index) ordinal
      FROM tool_calls c JOIN rounds cr USING(round_pk) JOIN tool_calls p ON c.continuation_of_tool_call_id=p.tool_call_id
      JOIN rounds pr ON p.round_pk=pr.round_pk WHERE c.continuation_of_tool_call_id IS NOT NULL)
      SELECT * FROM ranked WHERE ordinal=1 ORDER BY session_id,parent_round LIMIT 30""").fetchall()
    con.close()
    chains, polls, intermediate = aggregate
    lines = ["# TraceLab continuation and deadline semantics audit", "", "- Inspected 30 distinct Codex `exec_command` roots with their first `write_stdin` continuation.",
             f"- The released database has {chains:,} continuation roots and {polls:,} continuation calls; {intermediate:,} calls occur more than one round after their root.",
             "- A continuation is emitted in a later TraceLab round.  Thus a new LLM round exists before a later poll; merging the whole command chain would erase intervening LLM turns.",
             "- Decision: retain one promotion pause per valid round, from that round's final tool emission to final tool result.  No chain latency is summed.",
             "- `timing_events` has output-side reasoning/text/tool events but no direct next-GPU-inference-start event.  `result_at` remains the conservative earliest-availability deadline; no unobserved post-result slack is invented.", "",
             "|sample|parent round|continuation round|round gap|parent emitted|parent result|continuation emitted|continuation result|", "|---:|---:|---:|---:|---|---|---|---|"]
    for i, row in enumerate(examples, 1):
        _, parent, cont, gap, pe, pr, ce, cr, _ = row
        lines.append(f"|{i}|{parent}|{cont}|{gap}|{pe}|{pr}|{ce}|{cr}|")
    output.write_text("\n".join(lines) + "\n")

def closed_loop(groups, offsets, footprint, policy, model):
    """Propagate each completed promotion stall into its session's later releases."""
    future_sessions = []
    states = {}
    for index, (session_id, events) in enumerate(sorted(groups.items())):
        origin = events[0]["release_epoch_s"]
        states[session_id] = {"events": events, "origin": origin, "offset": offsets[session_id], "shift": 0.0, "next": 0}
        events[0]["actual_release_s"] = offsets[session_id]
        heapq.heappush(future_sessions, (offsets[session_id], index, session_id))
    future_promotions, pending, outcomes = [], [], []
    serial = 0; now = future_sessions[0][0] if future_sessions else 0.0; busy = 0.0
    def release_event(session_id):
        nonlocal serial
        state = states[session_id]; event = state["events"][state["next"]]
        release = event["actual_release_s"]; size = event["reusable_tokens"] * footprint
        service = model.piecewise_service_time_s(size)
        job = PromotionJob(f"{session_id}:{state['next']}", release, release + event["pause_s"], size, service)
        heapq.heappush(future_promotions, (eligibility_time(job, policy), serial, session_id, job)); serial += 1
    def schedule_next(session_id, completion):
        state = states[session_id]; current = state["events"][state["next"]]
        stall = max(0.0, completion - (current["actual_release_s"] + current["pause_s"]))
        state["shift"] += stall; state["next"] += 1
        if state["next"] >= len(state["events"]): return
        nxt = state["events"][state["next"]]
        nominal = state["offset"] + (nxt["release_epoch_s"] - state["origin"]) + state["shift"]
        nxt["actual_release_s"] = max(nominal, completion)
        heapq.heappush(future_sessions, (nxt["actual_release_s"], serial, session_id))
    while future_sessions or future_promotions or pending:
        while future_sessions and future_sessions[0][0] <= now + EPS:
            _, _, session_id = heapq.heappop(future_sessions); release_event(session_id)
        while future_promotions and future_promotions[0][0] <= now + EPS:
            _, _, session_id, job = heapq.heappop(future_promotions); pending.append((session_id, job))
        if not pending:
            candidates = [item[0] for item in (future_sessions[:1] + future_promotions[:1])]
            if not candidates: break
            now = min(candidates); continue
        if policy is Policy.QUEUE_JIT:
            safe = queue_safe_idle_s(now, [job for _, job in pending])
            next_time = min([item[0] for item in (future_sessions[:1] + future_promotions[:1])] or [math.inf])
            if safe > EPS and next_time < now + safe - EPS:
                now = next_time; continue
        session_id, job = min(pending, key=lambda item: (item[1].deadline_s, item[1].release_time_s, item[1].job_id))
        pending.remove((session_id, job)); start = now; completion = start + job.service_time_s
        outcomes.append(JobOutcome(job, start, completion)); busy += job.service_time_s; now = completion
        schedule_next(session_id, completion)
    return SimulationResult(policy, tuple(outcomes), busy)

def static_jobs(events, footprint, model, prefix):
    jobs=[]
    for i, event in enumerate(events):
        size=event["reusable_tokens"]*footprint; release=event["release_time_s"]
        jobs.append(PromotionJob(f"{prefix}-{i}",release,release+event["pause_s"],size,model.piecewise_service_time_s(size)))
    return jobs

def session_workload(events, m, seed):
    grouped=defaultdict(list)
    for event in events: grouped[event["session_id"]].append(dict(event))
    candidates=[(sid, rows) for sid, rows in grouped.items() if 50 <= len(rows) <= 100]
    rng=random.Random(seed); chosen=rng.sample(candidates,m); offsets={sid:rng.uniform(0,60) for sid,_ in chosen}
    return dict(chosen), offsets

def poisson_workload(events, rho, seed, footprint, model):
    rng=random.Random(seed); chosen=[dict(rng.choice(events)) for _ in range(5000)]
    mean_service=statistics.mean(model.piecewise_service_time_s(e["reusable_tokens"]*footprint) for e in chosen)
    release=0.0
    for event in chosen: release += rng.expovariate(rho/mean_service); event["release_time_s"] = release
    return chosen

def characterize(events, footprint, model):
    token=[event["reusable_tokens"] for event in events]; pause=[event["pause_s"] for event in events]
    sizes=[value*footprint/1e9 for value in token]; roundtrips=[model.piecewise_service_time_s(value*footprint)*2 for value in token]
    rows=[]
    for name, values in (("reusable_context_tokens",token),("kv_size_gb",sizes),("pause_ms",[x*1000 for x in pause]),("roundtrip_ms",[x*1000 for x in roundtrips])):
        rows += [{"measure":name,"statistic":key,"value":value} for key,value in quantiles(values).items()]
    ratios=[p/r for p,r in zip(pause,roundtrips)]
    buckets=((0,1,"<1"),(1,1.25,"1-1.25"),(1.25,1.5,"1.25-1.5"),(1.5,2,"1.5-2"),(2,3,"2-3"),(3,math.inf,">=3"))
    rows += [{"measure":"slack_ratio_fraction","statistic":label,"value":sum(low<=x<high for x in ratios)/len(ratios)} for low,high,label in buckets]
    return rows

def main():
    out=ROOT/'results/phase1b_fix'; out.mkdir(parents=True,exist_ok=True)
    db=ROOT/'data/tracelab/syfi_coding_trace.duckdb'; audit_semantics(db,out/'trace_semantics_audit.md')
    all_events, skipped, token_audit = extract_events(db)
    events=[event for event in all_events if event['reusable_tokens'] <= MAX_CONTEXT]
    model=TransferModel.from_phase0(ROOT,'cpu',all_payload_sizes=True)
    write_csv(ROOT/'data/tracelab/phase1b_fix_opportunities.csv', tuple(events[0]), events)
    rows=[]; workload=[]
    for footprint in FOOTPRINTS:
      for m in (4,8,16,32,64):
       for seed in range(10):
        groups,offsets=session_workload(events,m,seed)
        for policy in Policy:
            result=closed_loop(groups,offsets,footprint,policy,model)
            rows.append({"workload":"closed_loop_session","footprint_bytes_per_token":footprint,"concurrency":m,"rho":"","seed":seed,"policy":policy.value,**aggregate(result)})
        open_events=[]
        for session_id, session_events in groups.items():
            origin=session_events[0]["release_epoch_s"]
            for event in session_events:
                open_events.append({**event,"release_time_s":offsets[session_id] + event["release_epoch_s"] - origin})
        for policy in Policy:
            result=simulate(static_jobs(open_events,footprint,model,f"open-{m}-{seed}"),policy)
            rows.append({"workload":"open_loop_session_audit","footprint_bytes_per_token":footprint,"concurrency":m,"rho":"","seed":seed,"policy":policy.value,**aggregate(result)})
        workload.append({"workload":"closed_loop_session","footprint_bytes_per_token":footprint,"concurrency":m,"rho":"","seed":seed,"job_count":sum(len(v) for v in groups.values()),"session_count":m})
        workload.append({"workload":"open_loop_session_audit","footprint_bytes_per_token":footprint,"concurrency":m,"rho":"","seed":seed,"job_count":len(open_events),"session_count":m})
      for rho in (.25,.5,.75,.9):
       for seed in range(10):
        replay=poisson_workload(events,rho,1000+seed,footprint,model); jobs=static_jobs(replay,footprint,model,f"poisson-{rho}-{seed}")
        for policy in Policy:
            rows.append({"workload":"poisson_tuple","footprint_bytes_per_token":footprint,"concurrency":"","rho":rho,"seed":1000+seed,"policy":policy.value,**aggregate(simulate(jobs,policy))})
        workload.append({"workload":"poisson_tuple","footprint_bytes_per_token":footprint,"concurrency":"","rho":rho,"seed":1000+seed,"job_count":len(jobs),"session_count":0})
    metric_fields=FIELDS+tuple(aggregate(SimulationResult(Policy.QUEUE_JIT,(),0)).keys())
    # Obtain stable fields from a representative completed result.
    metric_fields=FIELDS+tuple(key for key in rows[0] if key not in FIELDS)
    write_csv(out/'aggregate_metrics.csv',metric_fields,rows); write_csv(out/'workload_summary.csv',("workload","footprint_bytes_per_token","concurrency","rho","seed","job_count","session_count"),workload)
    char=[]
    for footprint in FOOTPRINTS: char += [{"footprint_bytes_per_token":footprint,**row} for row in characterize(events,footprint,model)]
    char += [{"footprint_bytes_per_token":"all","measure":"coverage","statistic":"valid_events", "value":len(all_events)}, {"footprint_bytes_per_token":"all","measure":"coverage","statistic":"retained_events", "value":len(events)}, {"footprint_bytes_per_token":"all","measure":"coverage","statistic":"retained_fraction", "value":len(events)/len(all_events)}]
    write_csv(out/'characterization.csv',("footprint_bytes_per_token","measure","statistic","value"),char)
    audit_rows=[]
    for key in ("input_tokens","output_tokens","reasoning_tokens","visible_output_tokens","reusable_tokens","legacy_tokens"):
        audit_rows += [{"measure":key,"statistic":stat,"value":value} for stat,value in quantiles([x[key] for x in token_audit]).items()]
    write_csv(out/'token_accounting_audit.csv',("measure","statistic","value"),audit_rows)
    config={"phase":"1B-Fix","phase0_calibration_commit":model.phase0_commit,"context_cap":MAX_CONTEXT,"footprints":list(FOOTPRINTS),"token_mapping":"input_tokens_total + max(0, output_tokens - reasoning_output_tokens)","deadline":"max(result_at), conservative earliest result availability; timing_events lacks next GPU inference start","session_replay":"closed loop: each promotion stall shifts later events in its own session; offsets U[0,60], M=4/8/16/32/64, 10 seeds","poisson":"tuple-preserving 5000 jobs, rho=.25/.5/.75/.9, 10 seeds","skips":dict(skipped),"valid_before_cap":len(all_events),"retained_after_cap":len(events)}
    (out/'config.json').write_text(json.dumps(config,indent=2)+"\n")
    (out/'commands.sh').write_text("#!/usr/bin/env bash\nset -euo pipefail\n/home/a6000/anaconda3/bin/python experiments/phase1b_fix/run_phase1b_fix.py\n")
    print('valid',len(all_events),'retained',len(events),'metrics',len(rows))
if __name__=='__main__': main()
