#!/usr/bin/env python3
"""TraceLab v0.0.2 extraction and Phase 1B oracle-deadline replay simulator."""
import argparse, csv, datetime as dt, hashlib, json, math, random, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCAL_DUCKDB = ROOT / "data/tracelab/python"
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
if str(LOCAL_DUCKDB) not in sys.path: sys.path.insert(0, str(LOCAL_DUCKDB))
import duckdb
from slackkv.simulator.jobs import PromotionJob
from slackkv.simulator.policies import Policy
from slackkv.simulator.simulator import aggregate, percentile, simulate
from slackkv.simulator.transfer_model import TransferModel

KIB = 1024
FOOTPRINTS = (64 * KIB, 128 * KIB, 256 * KIB)
PAUSE_BUCKETS = ((0,.05,"<50ms"),(.05,.1,"50-100ms"),(.1,.25,"100-250ms"),(.25,.5,"250-500ms"),(.5,1,"500ms-1s"),(1,5,"1-5s"),(5,float("inf"),">=5s"))
KV_BUCKETS = ((0,1e9,"<1GiB"),(1e9,4e9,"1-4GiB"),(4e9,8e9,"4-8GiB"),(8e9,float("inf"),">8GiB"))

RAW_FIELDS = ("event_id","session_id","provider","round_id","tool_category","tool_names","release_epoch_s","pause_duration_s","context_tokens")
AGG_PREFIX = ("workload","source","footprint_bytes_per_token","concurrency","rho","seed","policy")
WORKLOAD_FIELDS = ("workload","source","footprint_bytes_per_token","concurrency","rho","seed","job_count","session_count","pause_ms_p50","pause_ms_p95","context_tokens_p50","kv_gb_p50")

def csv_write(path, fields, rows):
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader()
        for row in rows: w.writerow({k:format(v,".9g") if isinstance(v,float) else v for k,v in row.items()})

def quantiles(values): return {f"p{int(q*100)}":percentile(values,q) for q in (.5,.9,.95,.99)}

def extract(db_path, raw_path):
    con=duckdb.connect(str(db_path),read_only=True)
    total_rounds=con.sql("SELECT count(DISTINCT round_pk) FROM tool_calls").fetchone()[0]
    q="""SELECT r.round_pk,r.provider,r.session_id,r.round_id,r.input_tokens_total,r.output_tokens,
      string_agg(t.tool_name,'|' ORDER BY t.tool_index) tools,max(t.emitted_at) release_at,max(t.result_at) deadline_at,count(*) tool_count,
      count(t.emitted_at) emitted_count,count(t.result_at) result_count
      FROM rounds r JOIN tool_calls t USING(round_pk) GROUP BY ALL
      ORDER BY r.session_id, r.round_pk"""
    rows=[]; skips=Counter()
    for item in con.sql(q).fetchall():
        pk,provider,session,round_id,input_tokens,output_tokens,tools,release,deadline,tool_count,emitted,result=item
        if emitted != tool_count or result != tool_count: skips["missing_tool_timing"]+=1; continue
        if input_tokens is None or output_tokens is None: skips["missing_token_count"]+=1; continue
        pause=(deadline-release).total_seconds()
        if pause<=0: skips["nonpositive_pause"]+=1; continue
        tokens=input_tokens+output_tokens
        if tokens<=0: skips["nonpositive_context_tokens"]+=1; continue
        names=tools.split("|"); category=names[0].lower()
        rows.append({"event_id":str(pk),"session_id":session,"provider":provider,"round_id":round_id,"tool_category":category,"tool_names":tools,"release_epoch_s":release.timestamp(),"pause_duration_s":pause,"context_tokens":tokens})
    csv_write(raw_path,RAW_FIELDS,rows)
    sha=hashlib.sha256(db_path.read_bytes()).hexdigest()
    metadata={"release_tag":"v0.0.2","asset_filename":db_path.name,"sha256":sha,"download_date":dt.date.today().isoformat(),"round_count":con.sql("SELECT count(*) FROM rounds").fetchone()[0],"tool_count":con.sql("SELECT count(*) FROM tool_calls").fetchone()[0],"session_count":con.sql("SELECT count(DISTINCT session_id) FROM rounds").fetchone()[0],"tool_round_count":total_rounds,"valid_promotion_opportunities":len(rows),"skipped_rounds":dict(skips),"schema":{"rounds":[x[0] for x in con.sql('DESCRIBE rounds').fetchall()],"tool_calls":[x[0] for x in con.sql('DESCRIBE tool_calls').fetchall()],"timing_events":[x[0] for x in con.sql('DESCRIBE timing_events').fetchall()]}}
    con.close(); return rows,metadata

def load_events(path):
    with path.open(newline="") as f:
        return [{**r,"release_epoch_s":float(r["release_epoch_s"]),"pause_duration_s":float(r["pause_duration_s"]),"context_tokens":int(r["context_tokens"])} for r in csv.DictReader(f)]

def jobs_from_events(events, footprint, prefix):
    jobs=[]; meta={}
    for i,e in enumerate(events):
        size=e["context_tokens"]*footprint; service=MODEL.piecewise_service_time_s(size)
        job=PromotionJob(f"{prefix}-{i}",e["release_time_s"],e["release_time_s"]+e["pause_duration_s"],size,service)
        jobs.append(job);meta[job.job_id]=e
    return jobs,meta

def session_replay(grouped, concurrency, seed, footprint):
    rng=random.Random(seed); candidates=[events for events in grouped.values() if 50<=len(events)<=100]
    if len(candidates)<concurrency: raise RuntimeError("insufficient 50-100-event sessions")
    selected=rng.sample(candidates,concurrency); events=[]
    for group in selected:
        origin=min(e["release_epoch_s"] for e in group); offset=rng.uniform(0,60)
        for e in group:
            events.append({**e,"release_time_s":e["release_epoch_s"]-origin+offset})
    return jobs_from_events(events,footprint,f"session-{concurrency}-{seed}"),len(selected)

def poisson_replay(events, rho, seed, footprint, count=5000):
    rng=random.Random(seed); chosen=[rng.choice(events) for _ in range(count)]
    mean_service=statistics.mean(MODEL.piecewise_service_time_s(e["context_tokens"]*footprint) for e in chosen)
    release=0.; replay=[]
    for e in chosen:
        release+=rng.expovariate(rho/mean_service); replay.append({**e,"release_time_s":release})
    return jobs_from_events(replay,footprint,f"poisson-{rho}-{seed}"),0

def bucket(value,bounds):
    for low,high,label in bounds:
        if low<=value<high:return label
    raise AssertionError

def simple_subset_metrics(outcomes):
    stalls=[o.stall_s*1000 for o in outcomes]
    return {"job_count":len(outcomes),"deadline_miss_rate":sum(x>0 for x in stalls)/len(stalls),"stall_ms_p95":percentile(stalls,.95),"stall_ms_p99":percentile(stalls,.99)}

def classify_misses(result, meta):
    outcomes=list(result.outcomes); counts=Counter(); big=4e9
    for i,o in enumerate(outcomes):
        if o.stall_s<=0: continue
        j=o.job
        if j.service_time_s>j.deadline_s-j.release_time_s: reason="late_arrival"
        elif any(x.job.kv_size_bytes>=big and x.start_time_s<o.start_time_s and x.completion_time_s>j.deadline_s-j.service_time_s for x in outcomes[:i]): reason="large_kv_domination"
        else:
            relevant=[x.job for x in outcomes if x.job.release_time_s<=j.deadline_s and x.job.deadline_s<=j.deadline_s]
            earliest=min(x.release_time_s for x in relevant)
            reason="inherently_infeasible" if sum(x.service_time_s for x in relevant)>j.deadline_s-earliest else "queue_collision"
        counts[reason]+=1
    return counts

def add_run(context,jobs,meta,session_count,aggregate_rows,workload_rows,strata,miss_rows):
    pauses=[j.deadline_s-j.release_time_s for j in jobs]; sizes=[j.kv_size_gb for j in jobs]
    workload_rows.append({**context,"job_count":len(jobs),"session_count":session_count,"pause_ms_p50":percentile([x*1000 for x in pauses],.5),"pause_ms_p95":percentile([x*1000 for x in pauses],.95),"context_tokens_p50":percentile([meta[j.job_id]["context_tokens"] for j in jobs],.5),"kv_gb_p50":percentile(sizes,.5)})
    for policy in Policy:
        result=simulate(jobs,policy); aggregate_rows.append({**context,"policy":policy.value,**aggregate(result)})
        if policy is Policy.QUEUE_JIT:
            for reason,count in classify_misses(result,meta).items(): miss_rows.append({**context,"policy":policy.value,"reason":reason,"miss_count":count})
        if context["workload"]=="session" and context["concurrency"]==64 and context["footprint_bytes_per_token"]==128*KIB:
            by=defaultdict(list)
            for outcome in result.outcomes:
                event=meta[outcome.job.job_id]; by[("pause",bucket(event["pause_duration_s"],PAUSE_BUCKETS))].append(outcome);by[("kv",bucket(outcome.job.kv_size_bytes,KV_BUCKETS))].append(outcome);by[("provider",event["provider"])].append(outcome);by[("tool",event["tool_category"])].append(outcome)
            for (kind,label),items in by.items():
                if len(items)>=30:strata.append({**context,"policy":policy.value,"stratification":kind,"bucket":label,**simple_subset_metrics(items)})

def characterize(events,footprint):
    pauses=[e["pause_duration_s"] for e in events];tokens=[e["context_tokens"] for e in events];sizes=[x*footprint/1e9 for x in tokens]
    mean_p,mean_s=statistics.mean(pauses),statistics.mean(sizes); corr=sum((x-mean_p)*(y-mean_s) for x,y in zip(pauses,sizes))/math.sqrt(sum((x-mean_p)**2 for x in pauses)*sum((y-mean_s)**2 for y in sizes))
    rows=[]
    for label,values in (("pause_duration_ms",[x*1000 for x in pauses]),("context_tokens",tokens),("kv_size_gb_128k",sizes)):
        rows += [{"measure":label,"statistic":k,"value":v} for k,v in quantiles(values).items()]
    rows += [{"measure":"pause_fraction","statistic":f"<{x}s","value":sum(v<x for v in pauses)/len(pauses)} for x in (.05,.1,.25,.5,1,5)]
    rows.append({"measure":"pause_kv_pearson_correlation","statistic":"value","value":corr});return rows

def compact_trace_plot_data(events, footprint):
    rows=[]
    for label,values in (("pause_duration_s",sorted(e["pause_duration_s"] for e in events)),("kv_size_gb",sorted(e["context_tokens"]*footprint/1e9 for e in events))):
        for index in range(1001):
            position=round(index*(len(values)-1)/1000)
            rows.append({"measure":label,"value":values[position],"cdf":position/(len(values)-1)})
    sample=random.Random(0).sample(events,min(2000,len(events)))
    scatter=[{"pause_duration_s":e["pause_duration_s"],"kv_size_gb":e["context_tokens"]*footprint/1e9,"provider":e["provider"]} for e in sample]
    return rows,scatter

def main():
    global MODEL
    ap=argparse.ArgumentParser();ap.add_argument('--db',default='data/tracelab/syfi_coding_trace.duckdb');ap.add_argument('--output-dir',default='results/phase1b');ap.add_argument('--raw-opportunities',default='data/tracelab/phase1b_opportunities.csv');ap.add_argument('--extract-only',action='store_true');args=ap.parse_args()
    db=(ROOT/args.db).resolve();out=(ROOT/args.output_dir).resolve();out.mkdir(parents=True,exist_ok=True);raw=(ROOT/args.raw_opportunities).resolve();raw.parent.mkdir(parents=True,exist_ok=True)
    events,metadata=extract(db,raw); (out/'dataset.json').write_text(json.dumps(metadata,indent=2)+"\n")
    if args.extract_only:
        print('valid events',len(events),'raw local',raw); return
    MODEL=TransferModel.from_phase0(ROOT,'cpu',all_payload_sizes=True)
    grouped=defaultdict(list)
    for e in events: grouped[e['session_id']].append(e)
    for group in grouped.values():group.sort(key=lambda e:e['release_epoch_s'])
    agg=[];work=[];strata=[];miss=[]
    for m in (4,8,16,32,64):
        for seed in range(10):
            (jobs,meta),sessions=session_replay(grouped,m,seed,128*KIB);add_run({"workload":"session","source":"tracelab_session","footprint_bytes_per_token":128*KIB,"concurrency":m,"rho":"","seed":seed},jobs,meta,sessions,agg,work,strata,miss)
    for rho in (.25,.5,.75,.9):
        for seed in range(10):
            (jobs,meta),sessions=poisson_replay(events,rho,1000+seed,128*KIB);add_run({"workload":"poisson","source":"tracelab_tuple","footprint_bytes_per_token":128*KIB,"concurrency":"","rho":rho,"seed":1000+seed},jobs,meta,sessions,agg,work,strata,miss)
    for footprint in FOOTPRINTS:
        for seed in range(3):
            (jobs,meta),sessions=session_replay(grouped,32,2000+seed,footprint);add_run({"workload":"kv_footprint_sensitivity","source":"tracelab_session","footprint_bytes_per_token":footprint,"concurrency":32,"rho":"","seed":2000+seed},jobs,meta,sessions,agg,work,strata,miss)
    fields=AGG_PREFIX+tuple(aggregate(simulate([PromotionJob('x',0,1,1,.1)],Policy.ON_DEMAND)).keys())
    cdf,scatter=compact_trace_plot_data(events,128*KIB)
    csv_write(out/'aggregate_metrics.csv',fields,agg);csv_write(out/'workload_summary.csv',WORKLOAD_FIELDS,work);csv_write(out/'trace_characterization.csv',('measure','statistic','value'),characterize(events,128*KIB));csv_write(out/'trace_cdf.csv',('measure','value','cdf'),cdf);csv_write(out/'trace_scatter_sample.csv',('pause_duration_s','kv_size_gb','provider'),scatter);csv_write(out/'stratified_metrics.csv',AGG_PREFIX+('stratification','bucket','job_count','deadline_miss_rate','stall_ms_p95','stall_ms_p99'),strata);csv_write(out/'miss_analysis.csv',AGG_PREFIX+('reason','miss_count'),miss)
    config={"phase0_calibration_commit":MODEL.phase0_commit,"transfer_direction":MODEL.direction,"transfer_model":"P50 CPU->GPU0 all Phase 0 payloads; linear interpolation between knots; >8GiB equals repeated 8GiB chunks plus interpolated remainder","kv_bytes_per_token_primary":128*KIB,"kv_footprint_sensitivity":list(FOOTPRINTS),"extraction_rule":"per round: release=max(emitted_at), deadline=max(result_at); all tool calls in the round require timestamps; pause must be positive. write_stdin continuations remain their own round calls and are not latency-summed.","session_replay":{"M":[4,8,16,32,64],"seeds":10,"eligible_session_event_range":[50,100],"selection_rationale":"complete moderately sized sessions only; no truncation or policy-dependent filtering","initial_offset_window_s":[0,60]},"poisson_replay":{"rho":[.25,.5,.75,.9],"jobs":5000,"seeds":10,"tuple_sampling":"sample complete TraceLab event tuple jointly"}}
    (out/'config.json').write_text(json.dumps(config,indent=2)+"\n");(out/'commands.sh').write_text("#!/usr/bin/env bash\nset -euo pipefail\nPYTHON=/home/a6000/anaconda3/bin/python\n$PYTHON -m pip install --target data/tracelab/python duckdb==1.1.3\n$PYTHON experiments/phase1b/run_phase1b.py\n$PYTHON scripts/plot_phase1b.py\n");(out/'commands.sh').chmod(0o755)
    print('valid events',len(events),'aggregate rows',len(agg),'raw local',raw)
if __name__=='__main__':main()
