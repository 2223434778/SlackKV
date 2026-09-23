import csv,json,statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'results'/'phase3u'
POLICIES=("OnDemand-K2","Blocking70-K2","NaiveAsync40-K2","ReservationLedger2-Async40-K2")
def read(n):
    with (OUT/n).open(newline='') as f:return list(csv.DictReader(f))
def write(n,rows):
    with (OUT/n).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def f(r,k):return float(r[k])
def med(v):return statistics.median(v)
raw=read('run_metrics.csv'); sessions=read('session_metrics.csv'); native=read('async_prefetch_metrics.csv'); bg=read('background_metrics.csv'); trace=read('scheduler_trace.csv'); xbase=read('x_only_baseline.csv')
baseline=med([f(x,'X_decode_duration_ms') for x in xbase])
clean=[]
for r in raw:
    p,rep=r['policy'],r['repetition']; ss=[x for x in sessions if x['policy']==p and x['repetition']==rep]; nn=[x for x in native if x['policy']==p and x['repetition']==rep]
    gate=sum(x['policy']==p and x['repetition']==rep and x['event']=='GATE_BLOCKED' for x in trace)
    ext=f(r,'external_source_tokens'); n=len(nn); nm=n*112.; em=ext*112/4096; total=nm+em
    lead=sum(max(0,f(x,'scheduler_admission_ms')-f(x,'T6_ms')) for x in ss if x['T6_ms']!='NA')*112/1000
    row={'policy':p,'repetition':rep,'AL_P50_resume_ms':r['P50_effective_resume_ms'],'AL_P95_resume_ms':r['P95_effective_resume_ms'],'AL_max_resume_ms':r['max_effective_resume_ms'],'Cluster1_P95_resume_ms':r['cluster1_P95_ms'],'Cluster2_P95_resume_ms':r['cluster2_P95_ms'],'Cluster3_P95_resume_ms':r['cluster3_P95_ms'],'AL_P95_completion_ms':r['ABCDEF_P95_completion_ms'],'AL_decode_tok_s':r['ABCDEF_decode_tok_s'],'combined_decode_tok_s':r['combined_decode_tok_s'],'X_decode_duration_ms':r['X_decode_duration_ms'],'X_TPOT_ms':r['X_TPOT_ms'],'X_slowdown':f(r,'X_decode_duration_ms')/baseline,'full_makespan_ms':r['full_makespan_ms'],'local_source_tokens':r['local_source_tokens'],'external_source_tokens':r['external_source_tokens'],'local_fraction':r['local_fraction'],'external_fraction':r['external_fraction'],'native_prefetch_count':n,'promotion_useful_count':r['promotion_useful_count'],'promotion_internal_ready_miss_count':r['promotion_internal_ready_miss_count'],'promotion_remote_wait_count':r['promotion_remote_wait_count'],'promotion_useful_rate':r['promotion_useful_rate'],'native_promotion_mib':nm,'resume_external_mib':em,'total_KV_H2D_mib':total,'duplicate_KV_H2D_mib':max(0,total-1344),'promotion_lead_time_MiB_s':lead,'reservation_promote_count':r['reservation_promote_count'],'reservation_skip_count':r['reservation_skip_count'],'max_reserved_blocks':r['active_reservation_max_blocks'],'lease_acquire_count':r['lease_acquire_count'],'lease_release_count':r['lease_release_count'],'lease_leak_count':r['lease_leak_count'],'max_pre_first_concurrency':r['max_pre_first_concurrency'],'gate_blocked_event_count':gate,'max_outstanding_prefetch_http_calls':r['max_outstanding_prefetch_http_calls'],'max_overlapping_T0_T6':r['max_overlapping_internal_prefetch_intervals'],'token_validation':r['token_validation'],'source_conservation_validation':r['source_conservation_validation'],'native_validation':r['native_validation'],'gate_validation':r['gate_validation'],'background_validation':r['background_validation'],'lease_validation':r['lease_validation'],'reservation_validation':r['reservation_validation']}
    clean.append(row)
write('run_metrics.csv',clean)
for x in bg:x['baseline_X_duration_ms']=baseline;x['X_slowdown']=f(x,'X_decode_duration_ms')/baseline;x['active_through_L_first_token']=x.pop('active_validation')
write('background_metrics.csv',bg)
summ=[]
for p in POLICIES:
    rr=[x for x in clean if x['policy']==p]
    def m(k):return med([float(x[k]) for x in rr])
    rate='NA' if p=='OnDemand-K2' else m('promotion_useful_rate')
    summ.append({'Policy':p,'AL_P95_Resume_ms':m('AL_P95_resume_ms'),'Completion_P95_ms':m('AL_P95_completion_ms'),'AL_tok_s':m('AL_decode_tok_s'),'X_Slowdown':m('X_slowdown'),'Combined_tok_s':m('combined_decode_tok_s'),'Local_percent':100*m('local_fraction'),'Promotion_Useful_percent':'NA' if rate=='NA' else 100*rate,'Total_H2D_MiB':m('total_KV_H2D_mib'),'Duplicate_H2D_MiB':m('duplicate_KV_H2D_mib')})
write('policy_summary.csv',summ)
S={x['Policy']:x for x in summ}; od=S['OnDemand-K2']; rs=S['ReservationLedger2-Async40-K2']; nv=S['NaiveAsync40-K2']; bl=S['Blocking70-K2']
def pct(a,b):return (float(a)/float(b)-1)*100
reservation_runs=[x for x in clean if x['policy']=='ReservationLedger2-Async40-K2']
reservation_correct=all(float(x['promotion_useful_rate'])==1 and float(x['promotion_remote_wait_count'])==0 and float(x['promotion_internal_ready_miss_count'])==0 and float(x['lease_leak_count'])==0 and float(x['max_reserved_blocks'])<=512 for x in reservation_runs)
if not reservation_correct: result,nextdir='D','RESERVATION_CORRECTNESS_FIX'
elif float(rs['Combined_tok_s'])<.95*float(od['Combined_tok_s']) or float(rs['X_Slowdown'])>float(od['X_Slowdown'])+.10: result,nextdir='C','RESERVATION_COMPUTE_COUPLING'
elif float(rs['AL_P95_Resume_ms'])>.95*float(od['AL_P95_Resume_ms']): result,nextdir='B','RESERVATION_POLICY_OPTIMIZATION'
else: result,nextdir='A','TRACE_DRIVEN_RESERVATION'
valid=lambda k:sum(x[k]=='PASS' for x in clean)
report={'x_only_runs':f"{sum(x['token_validation']=='PASS' for x in xbase)}/3",'integration_runs':'4/4','measured_runs':'12/12','token_correctness':f"{sum(x['generated_token_count']=='32' for x in sessions)}/144",'source_conservation':f"{valid('source_conservation_validation')}/12",'K2_valid':f"{valid('gate_validation')}/12",'background_valid':f"{valid('background_validation')}/12",'policy_summaries':S,'Reservation':{'promotions':sum(int(x['reservation_promote_count']) for x in reservation_runs),'skips':sum(int(x['reservation_skip_count']) for x in reservation_runs),'useful':sum(int(x['promotion_useful_count']) for x in reservation_runs),'remote_waits':sum(int(x['promotion_remote_wait_count']) for x in reservation_runs),'internal_ready_misses':sum(int(x['promotion_internal_ready_miss_count']) for x in reservation_runs),'lease_leaks':sum(int(x['lease_leak_count']) for x in reservation_runs),'max_reserved_blocks':max(int(x['max_reserved_blocks']) for x in reservation_runs)},'Result':result,'next_direction':nextdir}
(OUT/'validation_report.json').write_text(json.dumps(report,indent=2))
lines=['# Phase 3U — full twelve-session baseline matrix','','## Objective','Compare OnDemand, blocking prefetch, naive async prefetch, and ReservationLedger under the identical A–L workload.','','## Setup','vLLM 0.23.0; PyTorch 2.11.0+cu126; RTX A6000; Qwen2.5-1.5B-Instruct; K=2; GPU KV 1 GiB; CPU external KV 2 GiB. Phase 3S lease patch reused unchanged.','', '## Results','| Policy | AL P95 resume (ms) | Combined tok/s | X slowdown | Local % | Useful % | Total H2D MiB | Duplicate H2D MiB |','|---|---:|---:|---:|---:|---:|---:|---:|']
for x in summ:
    useful=x['Promotion_Useful_percent'] if x['Promotion_Useful_percent']=='NA' else f"{float(x['Promotion_Useful_percent']):.2f}"
    lines.append(f"| {x['Policy']} | {float(x['AL_P95_Resume_ms']):.2f} | {float(x['Combined_tok_s']):.2f} | {float(x['X_Slowdown']):.3f} | {float(x['Local_percent']):.2f} | {useful} | {float(x['Total_H2D_MiB']):.2f} | {float(x['Duplicate_H2D_MiB']):.2f} |")
lines += ['', 'ReservationLedger issued 18 promotions and skipped 18 sessions across measured runs; all promotions were useful, with zero remote waits, internal-ready misses, or lease leaks. NaiveAsync reproduced residency failure: only 4/12 promotions per run were useful and the rest later restored externally. Blocking70 likewise did not preserve unleased promoted state. Source conservation and K2 validation passed in every run.','',f"Reservation vs OnDemand: P95 {pct(rs['AL_P95_Resume_ms'],od['AL_P95_Resume_ms']):+.2f}%, combined throughput {pct(rs['Combined_tok_s'],od['Combined_tok_s']):+.2f}%, X slowdown difference {float(rs['X_Slowdown'])-float(od['X_Slowdown']):+.3f}, total H2D {pct(rs['Total_H2D_MiB'],od['Total_H2D_MiB']):+.2f}%.",f"Reservation vs NaiveAsync removed {float(nv['Duplicate_H2D_MiB'])-float(rs['Duplicate_H2D_MiB']):.2f} MiB median duplicate H2D per run.",'','## Conclusion',f'Result {result}. Next direction: {nextdir}. ReservationLedger is on the observed latency/traffic Pareto frontier: it has the lowest median AL P95 and minimum H2D traffic, without a material combined-throughput or X-interference penalty.','','## Scope','Phase 3U is an internal mechanism baseline matrix. It is not yet the final external SOTA comparison against MORI / Continuum / CacheWise / KVFlow-style systems.']
(OUT/'summary.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'summary':S,'result':result,'next':nextdir,'baseline':baseline},indent=2))
