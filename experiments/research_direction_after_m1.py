"""Evidence-only CPU audit. No serving policy, GPU initialization or network IO."""
import argparse
import gzip
import hashlib
import json
import lzma
from pathlib import Path
import statistics

from experiments.large_model.verify_archive import verify
from experiments.host_prefix.analyze_gpu import delta
from experiments.host_prefix.followup_analyze import transfer
from experiments.opportunity_decision.analyze import distribution, flatten


def archived(base, name):
    manifest = json.loads((base/'raw_manifest.json').read_text())
    record = manifest.get(name)
    if record is None:raise ValueError('unarchived input: '+name)
    path = base/record['stored']
    raw = path.read_bytes()
    if path.suffix == '.gz':raw = gzip.decompress(raw)
    elif path.suffix == '.xz':raw = lzma.decompress(raw)
    if len(raw) != record['bytes'] or hashlib.sha256(raw).hexdigest() != record['sha256']:
        raise ValueError('input identity mismatch: '+name)
    return raw


def persistent_reuse_bound(capacity_tokens, initial_lengths, later_lcps, block_tokens=16):
    """Future-aware optimal fixed private prefixes; not a rolling-cache bound.

    Unit blocks have decreasing marginal future reference counts along a prefix.
    Select the best exposed next block, respecting prefix closure. No task timing,
    physical refs, GPU state, migration cost or fairness guarantee is modelled.
    """
    import heapq
    if capacity_tokens < 0 or block_tokens <= 0:raise ValueError('invalid capacity/block size')
    if initial_lengths.keys() != later_lcps.keys():raise ValueError('session mismatch')
    heap = []; shares = {s:0 for s in initial_lengths}
    def marginal(s, depth):
        if depth > initial_lengths[s]:return 0
        return sum(lcp >= depth for lcp in later_lcps[s])
    for s in sorted(shares):
        score = marginal(s,block_tokens)
        if score:heapq.heappush(heap,(-score,s))
    reward = 0
    for _ in range(capacity_tokens//block_tokens):
        if not heap:break
        neg_score,s = heapq.heappop(heap)
        reward += -neg_score*block_tokens
        shares[s] += block_tokens
        score = marginal(s,shares[s]+block_tokens)
        if score:heapq.heappush(heap,(-score,s))
    return dict(saved_tokens=reward,shares_tokens=shares,
                scope='Offline future-aware fixed-prefix allocation, private paths, initial write only; token-count objective. Not online, not a global KV policy/E2E upper bound.')


def large_model(base):
    validation = verify(base)
    payloads = {}; reports = {}; identities = None
    for policy in ('nohost','full','fair'):
        name = 'screen_'+policy+'1'
        rows = json.loads(archived(base,name+'/requests.json'))
        inputs = json.loads(archived(base,name+'/inputs.json'))
        prompts = {(x['role'],x['session'],x['turn']):x['prompt'] for x in inputs}
        hashes = {}
        for r in rows:
            k = (r['role'],r['session'],r['turn']);prompt = prompts[k]
            assert r['length']==len(prompt) and len(r['tokens'])==128
            hashes[k] = hashlib.sha256(json.dumps(prompt).encode()).hexdigest()
            assert hashes[k]==r['prompt_sha256']
        assert len(rows)==len(prompts)==66
        if identities is None:identities = hashes
        assert hashes==identities
        analysis = json.loads((base/name/'analysis.json').read_text())
        before=archived(base,name+'/before.prom').decode()
        after=archived(base,name+'/after.prom').decode()
        sources={s:delta(before,after,'vllm:prompt_tokens_by_source_total',f'source="{s}"')
                 for s in ('local_compute','local_cache_hit','external_kv_transfer')}
        if any(v is None for v in sources.values()):raise ValueError('unknown prompt source counter')
        assert sources==analysis['sources'] and sum(sources.values())==sum(r['length'] for r in rows)
        execution=json.loads((base/name/'execution.json').read_text())
        assert execution['status']=='COMPLETE' and execution['gpu_release_verified']
        if policy=='nohost':
            assert '--kv-transfer-config' not in execution['server_command']
            h2d=d2h=0
        else:
            h2d=transfer(before,after,'CPU_to_GPU',sources['external_kv_transfer'])
            d2h=transfer(before,after,'GPU_to_CPU',sources['external_kv_transfer'])
        assert h2d==analysis['h2d_bytes'] and d2h==analysis['d2h_bytes']
        e2es=[];resumes=[]
        for s in range(21):
            own=sorted([r for r in rows if r['role']=='agent' and r['session']==s],key=lambda r:r['turn'])
            assert [r['turn'] for r in own]==[0,1,2]
            e2es.append((own[-1]['done_ns']-own[0]['issue_ns'])/1e6)
            resumes.extend((r['first_ns']-r['issue_ns'])/1e6 for r in own[1:])
        median=statistics.median(e2es);p95=sorted(resumes)[39]
        assert median==analysis['session_e2e_median_ms'] and p95==analysis['resume_p95_ms']
        backgrounds=[dict(turn=r['turn'],e2e_ms=(r['done_ns']-r['issue_ns'])/1e6,
            ttft_ms=(r['first_ns']-r['issue_ns'])/1e6,
            first_token_to_response_end_mean_ms=(r['done_ns']-r['first_ns'])/127e6)
            for r in sorted(rows,key=lambda r:r['turn']) if r['role']=='bg']
        assert backgrounds==analysis['background_requests']
        reports[policy] = dict(session_e2e_median_ms=median,
            resume_p95_ms=p95,sources=sources,h2d_bytes=h2d,d2h_bytes=d2h,
            background_requests=backgrounds,
            by_turn={str(t):dict(requests=len(part),
                median_ttft_ms=statistics.median((r['first_ns']-r['issue_ns'])/1e6 for r in part),
                median_first_to_end_mean_ms=statistics.median((r['done_ns']-r['first_ns'])/127e6 for r in part))
                for t in range(3) for part in [[r for r in rows if r['role']=='agent' and r['turn']==t]]},
            total_input_tokens=sum(r['length'] for r in rows),
            total_output_tokens=sum(len(r['tokens']) for r in rows))
        if policy=='fair':payloads = prompts
    lengths={};later={};unbounded_reuse=0
    assert len({tuple(payloads[('agent',s,0)][:16]) for s in range(21)})==21
    for s in range(21):
        initial = payloads[('agent',s,0)]; lengths[s] = len(initial); later[s] = []
        for t in (1,2):
            current=payloads[('agent',s,t)];previous=payloads[('agent',s,t-1)]
            assert current[:len(previous)]==previous
            later[s].append(len(initial))
            unbounded_reuse += len(previous)//16*16
    host_capacity_tokens=64*2**30//262144
    oracle = persistent_reuse_bound(host_capacity_tokens,lengths,later)
    observed = reports['fair']['sources']['external_kv_transfer']
    assert oracle['saved_tokens']==observed==524288
    total=reports['fair']['total_input_tokens']
    missing=unbounded_reuse-observed
    assert total-unbounded_reuse==648192 and missing==690688
    return dict(archive_validation=validation,matched_input_identities=len(identities),reports=reports,
        fixed_prefix_reference=dict(**oracle,host_capacity_tokens=host_capacity_tokens,
            observed_fair_restored_tokens=observed,extra_saved_tokens=oracle['saved_tokens']-observed,
            qualification='All21 roots28672 tokens and two resumptions. Fair fills the fixed-prefix token reuse bound; alternative allocation can change fairness/nonlinear compute cost. New writes/recycling/tier placement fall outside this bound.'),
        unbounded_storage_reference=dict(reusable_prior_prompt_tokens=unbounded_reuse,
            compulsory_input_tokens_including_background=total-unbounded_reuse,
            fair_remaining_reusable_compute_tokens=missing,
            qualification='Unlimited usable prior-input KV, no load/setup cost. Token reference only; infeasible at the frozen capacity in general, not a latency prediction.'),
        uniform_storage_target=dict(initial_all_roots_gib=sum(lengths.values())*262144/2**30,
            gpu_pool_gib=70000*262144/2**30,host_budget_gib=64,
            qualification='Persistent all-root retention target, not actual simultaneous residency or a production/minimum-capacity claim.'),
        unknown=['scheduler admission/compute/remote-load critical-path decomposition',
                 'per-prefix saved GPU time curve and queueing sensitivity',
                 'physical GPU/host overlap or reclaimable memory-time',
                 'same-budget rolling or cost-weighted prefix opportunity',
                 'independent-run repeatability and stronger related-work baselines'])


def proxy_metadata(path):
    raw=path.read_bytes()
    expected='d078503145b09533298928438fe186bbbc40c726901f920872510df04cd47994'
    if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('proxy dataset version mismatch')
    containers=[];input_lengths=[];output_lengths=[];fit_containers=[]
    for index,line in enumerate(raw.splitlines()):
        trace=json.loads(line);requests,_=flatten(trace);groups={}
        for r in requests:
            groups.setdefault((r['container'],r['model']),[]).append(r)
            input_lengths.append(r['in']);output_lengths.append(r['out'])
        for (container,model),rows in groups.items():
            record=dict(trace_index=index,container=container,model=model,requests=len(rows),
                input_min=min(r['in'] for r in rows),input_max=max(r['in'] for r in rows),
                all_input_plus_output_within_32768=all(r['in']+r['out']<=32768 for r in rows))
            containers.append(record)
            if record['all_input_plus_output_within_32768']:fit_containers.append(record)
    return dict(source_sha256=expected,request_count=len(input_lengths),container_model_groups=len(containers),
        calls_per_container_model=distribution([c['requests'] for c in containers]),
        proxy_input_tokens=distribution(input_lengths),output_tokens=distribution(output_lengths),
        entire_container_model_fitting_old_max_model_len=len(fit_containers),
        multi_call_entire_container_model_fitting_old_max_model_len=sum(c['requests']>=3 for c in fit_containers),
        requests_over_32768_proxy_input=sum(n>32768 for n in input_lengths),containers=containers,
        qualification='Existing hashed-prefix public proxy metadata, original Claude models/endpoints. Containers are not proven linear sessions; timestamps and proxy tokens are not Qwen runtime. Fitting fragments must not be selected as representative or truncated to obtain a favorable test.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--large-model',type=Path,required=True)
    parser.add_argument('--proxy',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    result=dict(gpu_hours_added=0,large_model=large_model(args.large_model),
                proxy=proxy_metadata(args.proxy),analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(gpu_hours_added=0,archive_validation=result['large_model']['archive_validation'],
        fixed_prefix_extra_tokens=result['large_model']['fixed_prefix_reference']['extra_saved_tokens'],
        proxy_container_calls=result['proxy']['calls_per_container_model'],
        proxy_fitting_multi_call_groups=result['proxy']['multi_call_entire_container_model_fitting_old_max_model_len']),indent=2))


if __name__=='__main__':main()
