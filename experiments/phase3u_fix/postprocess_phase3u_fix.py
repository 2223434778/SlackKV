import csv
import json
import math
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path('/media/a6000/D/ZhaoBinHe/SlackKV')
OUT = ROOT / 'results/phase3u_fix'
POLICIES = ('OnDemand-K2', 'Blocking70-K2', 'NaiveAsync40-K2', 'ReservationLedger2-Async40-K2')


def read_csv(name):
    with (OUT / name).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(name, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / name).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def median(rows, key):
    values = [float(r[key]) for r in rows if r.get(key) not in ('', 'NA', None)]
    return statistics.median(values) if values else 'NA'


def pct_change(new, old):
    return (new / old - 1) * 100 if old else 'NA'


def live_lease_block_peaks(rows):
    grouped = {}
    for row in rows:
        rep = int(row['repetition'])
        live, peak = grouped.get(rep, (0, 0))
        if row['event'] == 'LEASE_ACQUIRED':
            live += int(row['num_blocks'])
            peak = max(peak, live)
        elif row['event'] == 'LEASE_RELEASED':
            live -= int(row['num_blocks'])
        grouped[rep] = (live, peak)
    return {rep: {'live_blocks_at_end': live, 'peak_blocks': peak} for rep, (live, peak) in grouped.items()}


def integration_lease_peak(log_path):
    active = peak = 0
    marker = 'SLACKKV_LEASE_TRACE '
    for line in log_path.read_text(encoding='utf-8', errors='ignore').splitlines():
        if marker not in line:
            continue
        row = json.loads(line.split(marker, 1)[1])
        if row['event'] == 'LEASE_ACQUIRED':
            active += int(row['num_blocks'])
            peak = max(peak, active)
        elif row['event'] == 'LEASE_RELEASED':
            active -= int(row['num_blocks'])
    return active, peak


def main():
    runs = read_csv('run_metrics.csv')
    sessions = read_csv('session_metrics.csv')
    native = read_csv('async_prefetch_metrics.csv')
    stage_a_native = read_csv('stage_a_native_prefetch_metrics.csv')
    blocking_a = read_csv('stage_a_blocking_metrics.csv')
    blocking_b = read_csv('blocking_stage_b_metrics.csv')
    leases = read_csv('lease_events.csv')
    lease_peaks = live_lease_block_peaks(leases)
    blocking = ([dict(stage='integration', **r) for r in blocking_a]
                + [dict(stage='measured', **r) for r in blocking_b])
    write_csv('blocking_metrics.csv', blocking)

    for r in runs:
        policy = r['policy']; rep = r['repetition']
        matches = [s for s in sessions if s['policy'] == policy and s['repetition'] == rep]
        nrows = [n for n in native if n['policy'] == policy and n['repetition'] == rep]
        r['AL_P50_resume_ms'] = r['P50_effective_resume_ms']
        r['AL_P95_resume_ms'] = r['P95_effective_resume_ms']
        r['AL_max_resume_ms'] = r['max_effective_resume_ms']
        r['AL_P95_completion_ms'] = r.get('AL_P95_completion_ms', r.get('ABCDEF_P95_completion_ms', 'NA'))
        r['AL_decode_tok_s'] = r.get('AL_decode_tok_s', r.get('ABCDEF_decode_tok_s', 'NA'))
        r['Cluster1_P95_resume_ms'] = r['cluster1_P95_ms']
        r['Cluster2_P95_resume_ms'] = r['cluster2_P95_ms']
        r['Cluster3_P95_resume_ms'] = r['cluster3_P95_ms']
        r['native_prefetch_count'] = len(nrows)
        r['gate_blocked_event_count'] = sum(1 for t in read_csv('scheduler_trace.csv')
            if t['policy'] == policy and t['repetition'] == rep and t.get('event') == 'GATE_BLOCKED')
        r['native_promotion_mib'] = len(nrows) * 112.0
        r['resume_external_mib'] = float(r['source_external_transfer_tokens_total']) * 112.0 / 4096.0
        r['total_KV_H2D_mib'] = float(r['native_promotion_mib']) + float(r['resume_external_mib'])
        r['duplicate_KV_H2D_mib'] = max(0.0, float(r['total_KV_H2D_mib']) - 1344.0)
        r['promotion_lead_time_MiB_s'] = sum(float(s['gpu_ready_residency_ms']) for s in matches
            if s['reservation_decision'] in ('PROMOTE', 'RESERVATION_PROMOTE')) * 112.0 / 1000.0
        r['max_outstanding_prefetch_http_calls'] = r.get('max_outstanding_prefetch_http_calls', 'NA')
        r['max_overlapping_T0_T6'] = r.get('max_overlapping_internal_prefetch_intervals', 'NA')
        r['blocking_wait_call_count'] = r.get('blocking_wait_call_count', 0)
        r['blocking_wait_total_ms'] = r.get('blocking_wait_total_ms', 0.0)
        r['blocking_wait_max_ms'] = r.get('blocking_wait_max_ms', 0.0)
        r['max_active_leased_blocks'] = lease_peaks.get(int(rep), {}).get('peak_blocks', 0)
        for field in ('target_to_issue_ms', 'issue_to_T6_ms', 'target_to_T6_ms'):
            values = [float(n[field]) for n in nrows if n.get(field) not in ('', 'NA', None)]
            r[field + '_P50'] = statistics.median(values) if values else 'NA'
            r[field + '_P95'] = (sorted(values)[math.ceil(.95 * len(values)) - 1] if values else 'NA')
            r[field + '_max'] = max(values) if values else 'NA'
        r['source_validation'] = r['source_conservation_validation']
        r['native_prefetch_count_validation'] = 'PASS' if len(nrows) == (0 if policy == POLICIES[0] else 12 if policy in POLICIES[1:3] else 6) else 'FAIL'
        r['service_budget_validation'] = 'PASS' if all(float(n['service_budget_ms']) == (70 if policy == POLICIES[1] else 40) for n in nrows) else ('PASS' if not nrows and policy == POLICIES[0] else 'FAIL')
        r['blocking_semantics_validation'] = r.get('blocking_semantics_validation', 'FAIL')
        r['policy_semantics_validation'] = r.get('policy_semantics_validation', 'FAIL')
        for key in ('ABCDEF_P50_effective_resume_ms', 'ABCDEF_P95_effective_resume_ms', 'ABCDEF_max_effective_resume_ms',
                    'ABCDEF_P95_completion_ms', 'ABCDEF_decode_tok_s'):
            r.pop(key, None)
        for key in ('ABC_P95_effective_resume_ms', 'DEF_P95_effective_resume_ms'):
            r.pop(key, None)
    write_csv('run_metrics.csv', runs)

    summaries = []
    for p in POLICIES:
        subset = [r for r in runs if r['policy'] == p]
        summaries.append({
            'Policy': p,
            'AL_P50_resume_ms': median(subset, 'AL_P50_resume_ms'),
            'AL_P95_resume_ms': median(subset, 'AL_P95_resume_ms'),
            'AL_max_resume_ms': median(subset, 'AL_max_resume_ms'),
            'AL_P95_completion_ms': median(subset, 'AL_P95_completion_ms'),
            'AL_decode_tok_s': median(subset, 'AL_decode_tok_s'),
            'combined_decode_tok_s': median(subset, 'combined_decode_tok_s'),
            'X_slowdown': median(subset, 'X_slowdown'),
            'local_fraction': median(subset, 'local_fraction'),
            'external_fraction': median(subset, 'external_fraction'),
            'promotion_useful_rate': median(subset, 'promotion_useful_rate'),
            'native_promotion_mib': median(subset, 'native_promotion_mib'),
            'resume_external_mib': median(subset, 'resume_external_mib'),
            'total_KV_H2D_mib': median(subset, 'total_KV_H2D_mib'),
            'duplicate_KV_H2D_mib': median(subset, 'duplicate_KV_H2D_mib'),
            'promotion_count': median(subset, 'native_prefetch_count'),
            'promotion_remote_wait_count': median(subset, 'promotion_remote_wait_count'),
            'promotion_internal_ready_miss_count': median(subset, 'promotion_internal_ready_miss_count'),
            'reservation_skip_count': median(subset, 'reservation_skip_count'),
            'lease_leak_count': median(subset, 'lease_leak_count'),
            'max_reserved_blocks': median(subset, 'active_reservation_max_blocks'),
            'max_active_leased_blocks': median(subset, 'max_active_leased_blocks'),
            'blocking_wait_call_count': median(subset, 'blocking_wait_call_count'),
            'blocking_wait_total_ms': median(subset, 'blocking_wait_total_ms'),
            'blocking_wait_max_ms': median(subset, 'blocking_wait_max_ms'),
            'target_to_issue_P50_ms': median(subset, 'target_to_issue_ms_P50'),
            'target_to_issue_P95_ms': median(subset, 'target_to_issue_ms_P95'),
            'target_to_issue_max_ms': median(subset, 'target_to_issue_ms_max'),
            'issue_to_T6_P50_ms': median(subset, 'issue_to_T6_ms_P50'),
            'issue_to_T6_P95_ms': median(subset, 'issue_to_T6_ms_P95'),
            'issue_to_T6_max_ms': median(subset, 'issue_to_T6_ms_max'),
            'target_to_T6_P50_ms': median(subset, 'target_to_T6_ms_P50'),
            'target_to_T6_P95_ms': median(subset, 'target_to_T6_ms_P95'),
            'target_to_T6_max_ms': median(subset, 'target_to_T6_ms_max'),
        })
    write_csv('policy_summary.csv', summaries)

    xrows = read_csv('x_only_baseline.csv')
    integration = json.loads((OUT / 'integration_gate.json').read_text())
    measured_block_waits = [float(row['blocking_wait_ms']) for row in blocking_b]
    blocking_violations = sum(1 for row in blocking if row.get('validation') != 'PASS')
    measured_valid = []
    for r in runs:
        p = r['policy']; rep = r['repetition']
        b = [x for x in blocking_b if x['policy'] == p and x['repetition'] == rep]
        policy_native = [n for n in native if n['policy'] == p and n['repetition'] == rep]
        valid = (r['token_validation'] == r['source_conservation_validation'] == r['gate_validation'] == r['background_validation'] == r['native_validation'] == r['policy_semantics_validation'] == 'PASS'
                 and float(r['source_local_hit_tokens_total']) + float(r['source_external_transfer_tokens_total']) == 49152)
        if p == POLICIES[1]:
            valid = valid and len(b) == 12 and int(r['blocking_wait_call_count']) == 12 and all(x['validation'] == 'PASS' for x in b) and int(r['max_outstanding_prefetch_http_calls']) <= 1 and int(r['max_overlapping_internal_prefetch_intervals']) <= 1
        if p == POLICIES[3]:
            valid = valid and int(r['reservation_promote_count']) == 6 and int(r['reservation_skip_count']) == 6 and int(r['lease_leak_count']) == 0 and int(r['active_reservation_max_blocks']) <= 512 and float(r['promotion_useful_rate']) == 1.0 and lease_peaks.get(int(rep), {}).get('peak_blocks', 0) <= 512 and lease_peaks.get(int(rep), {}).get('live_blocks_at_end', 0) == 0
        measured_valid.append(valid)
    junit = ET.parse(OUT / 'tests.xml').getroot()
    test_suites = [junit] if junit.tag == 'testsuite' else list(junit.findall('testsuite'))
    tests_total = sum(int(s.attrib.get('tests', 0)) for s in test_suites)
    tests_failures = sum(int(s.attrib.get('failures', 0)) + int(s.attrib.get('errors', 0)) for s in test_suites)
    tests_status = {'passed': tests_total - tests_failures, 'total': tests_total}
    reservation = [r for r in runs if r['policy'] == POLICIES[3]]
    integration_live, integration_peak = integration_lease_peak(OUT / 'server_logs/reservationledger2_async40_k2_900.log')
    native_all = native + stage_a_native
    native_bytes_valid = all(int(float(n['transfer_bytes'])) == 117440512 for n in native_all)
    native_mechanism_valid = all(n['status'] == 'ok' and int(n['external_tokens_loaded']) == 4096 and int(n['model_forward_count']) == 0 and int(n['generated_token_count']) == 0 and int(n['mixed_prefetch_model_step_count']) == 0 for n in native_all)
    report = {
        'previous_phase3u_status': 'INVALID', 'invalid_reason': 'BLOCKING_SEMANTICS_BUG',
        'tests': tests_status, 'x_only': f"{sum(x['token_validation']=='PASS' and int(x['generated_token_count'])==256 for x in xrows)}/3",
        'stage_a': f"{sum(integration.get('policy_valid', {}).values())}/4", 'stage_b': f"{sum(measured_valid)}/12",
        'real_resume_correctness': f"{sum(int(r['token_validation']=='PASS') * 12 for r in runs)}/144",
        'source_conservation': f"{sum(r['source_conservation_validation']=='PASS' for r in runs)}/12",
        'K2': f"{sum(r['gate_validation']=='PASS' for r in runs)}/12",
        'background': f"{sum(r['background_validation']=='PASS' for r in runs)}/12",
        'blocking': {'wait_helper_calls': sum(int(r['blocking_wait_call_count']) for r in runs if r['policy']==POLICIES[1]), 'expected_wait_helper_calls': 36,
            'send_before_prefetch_return_violations': blocking_violations, 'service_budget_validation': all(n['service_budget_ms']=='70.0' for n in native if n['policy']==POLICIES[1]),
            'measured_wait_median_ms': statistics.median(measured_block_waits), 'measured_wait_p95_ms': sorted(measured_block_waits)[math.ceil(.95*len(measured_block_waits))-1], 'measured_wait_max_ms': max(measured_block_waits)},
        'naive': {'wait_helper_calls': sum(int(r['blocking_wait_call_count']) for r in runs if r['policy']==POLICIES[2]), 'service_budget_validation': all(n['service_budget_ms']=='40.0' for n in native if n['policy']==POLICIES[2])},
        'reservation': {'promotions':sum(int(r['reservation_promote_count']) for r in reservation), 'skips':sum(int(r['reservation_skip_count']) for r in reservation),
            'useful':sum(int(r['promotion_useful_count']) for r in reservation), 'remote_waits':sum(int(r['promotion_remote_wait_count']) for r in reservation),
            'internal_ready_misses':sum(int(r['promotion_internal_ready_miss_count']) for r in reservation), 'lease_leaks':sum(int(r['lease_leak_count']) for r in reservation),
            'max_reserved_blocks':max(int(r['active_reservation_max_blocks']) for r in reservation),
            'max_active_leased_blocks': max(v['peak_blocks'] for v in lease_peaks.values()),
            'integration_max_active_leased_blocks': integration_peak, 'integration_live_blocks_at_end': integration_live},
        'native_transfer_bytes_validation': native_bytes_valid,
        'native_prefetch_invariants_validation': native_mechanism_valid,
        'policy_summaries': {r['Policy']:r for r in summaries},
        'result': 'VALID_BASELINE_MATRIX' if all(measured_valid) and integration.get('pass') and len(xrows)==3 and tests_total>0 and tests_failures==0 and native_bytes_valid and native_mechanism_valid else 'EXPERIMENT_INVALID',
        'next_direction': 'TRACE_DRIVEN_RESERVATION' if all(measured_valid) and integration.get('pass') and len(xrows)==3 and tests_total>0 and tests_failures==0 and native_bytes_valid and native_mechanism_valid else 'FIX_IMPLEMENTATION'
    }
    (OUT / 'validation_report.json').write_text(json.dumps(report, indent=2))

    byp = {r['Policy']:r for r in summaries}
    rows = ['# Phase 3U-FIX summary', '', 'Previous Phase 3U status: INVALID. Its Blocking70 resumes were not coupled to native-prefetch completion; old raw CSVs are preserved and not used here.',
            '', '## Validation', '', f"X-only controls: {report['x_only']} (median decode {statistics.median(float(x['X_decode_duration_ms']) for x in xrows):.2f} ms; median TPOT {statistics.median(float(x['X_TPOT_ms']) for x in xrows):.2f} ms); Stage A: {report['stage_a']}; Stage B: {report['stage_b']}; real resumes: {report['real_resume_correctness']}; source conservation: {report['source_conservation']}; K2: {report['K2']}; background: {report['background']}.",
            '', 'Blocking70 wait helper: 12 calls/run; measured waits may be zero when prefetch completed before the deadline. Every measured resume was checked against its own client-return timestamp with 0.5-ms tolerance.',
            '', '## Median across three fresh measured runs', '', '| Policy | AL P95 resume (ms) | Combined tok/s | X slowdown | Local % | Total H2D MiB | Duplicate H2D MiB | Promotion useful |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for p in POLICIES:
        r=byp[p]; rows.append(f"| {p} | {float(r['AL_P95_resume_ms']):.2f} | {float(r['combined_decode_tok_s']):.2f} | {float(r['X_slowdown']):.3f} | {100*float(r['local_fraction']):.1f}% | {float(r['total_KV_H2D_mib']):.1f} | {float(r['duplicate_KV_H2D_mib']):.1f} | {r['promotion_useful_rate']} |")
    od=byp[POLICIES[0]]; naive=byp[POLICIES[2]]; res=byp[POLICIES[3]]; blocking_sum=byp[POLICIES[1]]
    rows += ['', f"Blocking70 actual waiting: measured wait median {report['blocking']['measured_wait_median_ms']:.3f} ms, P95 {report['blocking']['measured_wait_p95_ms']:.3f} ms, max {report['blocking']['measured_wait_max_ms']:.3f} ms, sum {sum(measured_block_waits):.3f} ms. A zero wait is valid when that session's native prefetch returned before its deadline.",
              f"Physical lease high-water mark: Stage A {integration_peak} blocks (end-live {integration_live}); measured max {report['reservation']['max_active_leased_blocks']} blocks.",
              '', '## Comparisons', '', f"ReservationLedger vs OnDemand: P95 resume change {pct_change(float(res['AL_P95_resume_ms']),float(od['AL_P95_resume_ms'])):.1f}%; combined throughput change {pct_change(float(res['combined_decode_tok_s']),float(od['combined_decode_tok_s'])):.1f}%; X slowdown difference {float(res['X_slowdown'])-float(od['X_slowdown']):+.3f}; total H2D change {pct_change(float(res['total_KV_H2D_mib']),float(od['total_KV_H2D_mib'])):.1f}%.",
              f"ReservationLedger vs NaiveAsync40: P95 resume change {pct_change(float(res['AL_P95_resume_ms']),float(naive['AL_P95_resume_ms'])):.1f}%; duplicate H2D removed {float(naive['duplicate_KV_H2D_mib'])-float(res['duplicate_KV_H2D_mib']):.1f} MiB; combined throughput change {pct_change(float(res['combined_decode_tok_s']),float(naive['combined_decode_tok_s'])):.1f}%; X slowdown difference {float(res['X_slowdown'])-float(naive['X_slowdown']):+.3f}.",
              f"ReservationLedger vs Blocking70: P95 resume change {pct_change(float(res['AL_P95_resume_ms']),float(blocking_sum['AL_P95_resume_ms'])):.1f}%; duplicate H2D change {float(res['duplicate_KV_H2D_mib'])-float(blocking_sum['duplicate_KV_H2D_mib']):+.1f} MiB; combined throughput change {pct_change(float(res['combined_decode_tok_s']),float(blocking_sum['combined_decode_tok_s'])):.1f}%.",
              f"Native timing medians (target→issue / issue→T6 / target→T6, ms): Blocking70 {blocking_sum['target_to_issue_P50_ms']} / {blocking_sum['issue_to_T6_P50_ms']} / {blocking_sum['target_to_T6_P50_ms']}; NaiveAsync40 {byp[POLICIES[2]]['target_to_issue_P50_ms']} / {byp[POLICIES[2]]['issue_to_T6_P50_ms']} / {byp[POLICIES[2]]['target_to_T6_P50_ms']}; Reservation {res['target_to_issue_P50_ms']} / {res['issue_to_T6_P50_ms']} / {res['target_to_T6_P50_ms']}.",
              '', f"Reservation usefulness: {report['reservation']['useful']}/{report['reservation']['promotions']}; remote waits {report['reservation']['remote_waits']}; internal-ready misses {report['reservation']['internal_ready_misses']}; lease leaks {report['reservation']['lease_leaks']}; maximum reserved blocks {report['reservation']['max_reserved_blocks']}.",
              '', 'X slowdown uses the median of three fresh X-only 256-token decode durations. No Phase 3U numbers were reused. Three repeats are descriptive mechanism evidence, not a population-level inference.',
              '', f"Result: {report['result']}. Next direction: {report['next_direction']}."]
    (OUT / 'summary.md').write_text('\n'.join(rows)+'\n')


if __name__ == '__main__':
    main()
