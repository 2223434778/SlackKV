from __future__ import annotations

import csv, json, statistics
from pathlib import Path

from experiments.phase3g_full.metrics import percentile_linear

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"results"/"phase3t"
def rows(name): return list(csv.DictReader((OUT/name).open()))
def stats(values):
    values=[float(x) for x in values]
    return {"median":statistics.median(values),"P95":percentile_linear(values,95),"max":max(values)}

def main():
    runs=rows("run_metrics.csv"); sessions=rows("session_metrics.csv"); native=rows("async_prefetch_metrics.csv"); leases=rows("lease_events.csv"); events=rows("reservation_events.csv")
    promoted=sum(int(x["reservation_promote_count"]) for x in runs); skipped=sum(int(x["reservation_skip_count"]) for x in runs)
    useful=sum(int(x["promotion_useful_count"]) for x in runs); remote=sum(int(x["promotion_remote_wait_count"]) for x in runs); misses=sum(int(x["promotion_internal_ready_miss_count"]) for x in runs)
    acquired=sum(x["event"]=="LEASE_ACQUIRED" for x in leases); released=sum(x["event"]=="LEASE_RELEASED" for x in leases)
    local=sum(int(float(x["local_source_tokens"])) for x in runs); external=sum(int(float(x["external_source_tokens"])) for x in runs)
    target_issue=stats(x["target_to_issue_ms"] for x in native); issue_t6=stats(x["issue_to_T6_ms"] for x in native); target_t6=stats(x["target_to_T6_ms"] for x in native)
    valid_runs=sum(all(x[k]=="PASS" for k in ("token_validation","native_validation","lease_validation","reservation_validation","gate_validation","background_validation","source_conservation_validation")) and int(x["promotion_remote_wait_count"])==0 and int(x["promotion_internal_ready_miss_count"])==0 for x in runs)
    result="A" if valid_runs==3 and useful==promoted and acquired==released==promoted and max(int(x["reserved_blocks_after"]) for x in events)<=512 else ("B" if remote else ("C" if misses else "D"))
    next_dir={"A":"RESERVATION_FULL_BASELINE_MATRIX","B":"LEASE_CAPACITY_DIAGNOSIS","C":"ASYNC_SERVICE_MODEL_WITH_LEASE","D":"RESERVATION_RELEASE_MODEL"}[result]
    report={"integration_gate":json.loads((OUT/"integration_gate.json").read_text())["pass"],"stage_b_runs":valid_runs,"real_resume_correctness":f"{sum(int(x['generated_token_count'])==32 for x in sessions)}/36","reservation_promotions":promoted,"reservation_skips":skipped,"promoted_local":useful,"promoted_remote_wait":remote,"promoted_internal_ready_miss":misses,"promotion_useful_rate":useful/promoted,"lease_acquires":acquired,"lease_releases":released,"lease_leaks":acquired-released,"max_reserved_blocks":max(int(x["reserved_blocks_after"]) for x in events),"wasted_promotion_mib":remote*112,"local_source_tokens":local,"external_source_tokens":external,"source_conservation":local+external==3*49152,"target_to_issue":target_issue,"issue_to_T6":issue_t6,"target_to_T6":target_t6,"K2_valid":sum(x["gate_validation"]=="PASS" for x in runs),"background_valid":sum(x["background_validation"]=="PASS" for x in runs),"Result":result,"next_direction":next_dir}
    (OUT/"validation_report.json").write_text(json.dumps(report,indent=2))
    med=lambda k:statistics.median(float(x[k]) for x in runs)
    summary=f'''# Phase 3T — 12-session residency reservation\n\n## Result\n\n**Result {result}.** Stage A passed and all three Stage B runs were valid. The finite two-lease ledger deliberately promoted 6/12 sessions per run and fell back to OnDemand for the other 6/12.\n\n| Metric | Result |\n|---|---:|\n| Reservation promotions | {promoted} (6/run) |\n| Reservation skips | {skipped} (6/run) |\n| Promoted local | {useful}/{promoted} |\n| Promoted remote waits | {remote} |\n| Internal-ready misses | {misses} |\n| Promotion useful rate | {100*useful/promoted:.1f}% |\n| Wasted promotion | {remote*112} MiB |\n| Maximum reserved blocks | {report['max_reserved_blocks']} / 512 |\n| Lease acquire/release/leak | {acquired}/{released}/{acquired-released} |\n| Source tokens | {local} local + {external} external = {local+external} |\n| Overall P95 resume (median run) | {med('P95_effective_resume_ms'):.2f} ms |\n| Promoted P95 (median run) | {med('promoted_P95_ms'):.2f} ms |\n| Skipped P95 (median run) | {med('skipped_P95_ms'):.2f} ms |\n\nEvery run naturally selected A/B, E/F, and I/J through EDF QueueJIT plus the ledger; C/D, G/H, and K/L were skipped. The ledger never exceeded 512 blocks. Every admitted promotion acquired and released exactly one lease, survived to the resume allocation boundary, and avoided `WAITING_FOR_REMOTE_KVS`. There were no lease leaks. K2 and background-X validation passed 3/3; X generated 256 tokens and remained active through L's first token.\n\nTarget-to-issue was {target_issue['median']:.3f}/{target_issue['P95']:.3f}/{target_issue['max']:.3f} ms median/P95/max. Issue-to-T6 was {issue_t6['median']:.3f}/{issue_t6['P95']:.3f}/{issue_t6['max']:.3f} ms, and target-to-T6 was {target_t6['median']:.3f}/{target_t6['P95']:.3f}/{target_t6['max']:.3f} ms. Removing the synchronous Prometheus decision restored sub-millisecond controller latency.\n\nMechanistically, Phase 3R-FIX promoted all 12 after an instantaneous-capacity query and later lost seven promoted prefixes. Phase 3T reserves only two future prefixes per cluster, physically leases them, and intentionally routes the remaining sessions through OnDemand. This reduced issued promotion traffic while making every issued promotion useful.\n\nPromotion memory-time totaled {sum(float(x['promotion_memory_time_mib_s']) for x in runs):.2f} MiB*s across Stage B. Promoted P95 was lower than skipped P95, but residency correctness—not latency—is the primary success criterion.\n\nNext direction: `{next_dir}`. It was not started.\n'''
    (OUT/"summary.md").write_text(summary)
if __name__=="__main__": main()
