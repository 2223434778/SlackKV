"""Matched raw-payload checks and frozen end-to-end screen gate."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from experiments.host_prefix.analyze_gpu import delta
from experiments.host_prefix.followup_analyze import read, transfer
from experiments.large_model.screen import quotas


def percentile(values, fraction):
    return sorted(values)[math.ceil(len(values)*fraction)-1]


def analyze(path):
    execution = json.loads((path/'execution.json').read_text())
    if execution['status']=='RUNNING':
        raise ValueError('unresolved run')
    if execution['status']!='COMPLETE':
        return dict(run=path.name,status=execution['status'],gpu_hours=execution['gpu_hours'])
    cfg=json.loads((path/'workload.json').read_text())
    assert not execution['batch_invariant']
    assert cfg['policy']==execution['mode']
    assert (cfg['sessions'],cfg['turns'],cfg['output'],cfg['seed'])==(21,3,128,914477)
    assert cfg['cpu_bytes_budget']==64*2**30 and cfg['gpu_bytes_per_rank']==70000*131072
    assert cfg['tool_gap_s']==5 and cfg['first_release_spacing_ms']==20
    assert cfg['quotas_tokens']==quotas(cfg['cpu_bytes_budget'])
    rows=json.loads(read(path/'requests.json'))
    inputs=json.loads(read(path/'inputs.json'))
    payloads={(x['role'],x['session'],x['turn']):x['prompt'] for x in inputs}
    assert len(payloads)==len(inputs)==len(rows)==66
    identities={}
    for row in rows:
        key=(row['role'],row['session'],row['turn'])
        assert key not in identities
        prompt=payloads[key]
        assert row['role'] in ('agent','bg')
        if row['role']=='agent':
            assert 0<=row['session']<21 and 0<=row['turn']<3
            assert len(prompt)==28672+row['turn']*512
            expected_limit=len(prompt) if cfg['policy']=='full' else cfg['quotas_tokens'][row['session']] if cfg['policy']=='fair' else 0
        else:
            assert row['session']==-1 and 0<=row['turn']<3 and len(prompt)==8192
            expected_limit=0
        assert row['max_offload_tokens']==expected_limit
        digest=hashlib.sha256(json.dumps(prompt).encode()).hexdigest()
        assert digest==row['prompt_sha256'] and len(prompt)==row['length']
        assert len(row['tokens'])==128
        assert row['issue_ns']<=row['first_ns']<=row['done_ns']
        identities[':'.join(map(str,key))]=digest
    sessions=[]
    for session in range(21):
        own=sorted((r for r in rows if r['role']=='agent' and r['session']==session),key=lambda r:r['turn'])
        assert [r['turn'] for r in own]==[0,1,2]
        for a,b in zip(own,own[1:]):
            assert b['issue_ns']>=a['done_ns']+4.9e9
            assert payloads[('agent',session,b['turn'])][:a['length']]==payloads[('agent',session,a['turn'])]
        resumes=[(r['first_ns']-r['issue_ns'])/1e6 for r in own[1:]]
        sessions.append(dict(session=session,e2e_ms=(own[-1]['done_ns']-own[0]['issue_ns'])/1e6,
                             resume_ttft_ms=resumes))
    bg=sorted((r for r in rows if r['role']=='bg'),key=lambda r:r['turn'])
    assert len(bg)==3 and [r['turn'] for r in bg]==[0,1,2]
    assert all(b['issue_ns']>=a['done_ns'] for a,b in zip(bg,bg[1:]))
    assert all(r['issue_ns']>=bg[0]['first_ns'] for r in rows if r['role']=='agent')
    before=read(path/'before.prom').decode();after=read(path/'after.prom').decode()
    sources={s:delta(before,after,'vllm:prompt_tokens_by_source_total',f'source="{s}"')
             for s in ['local_compute','local_cache_hit','external_kv_transfer']}
    total_input=sum(r['length'] for r in rows)
    assert total_input==1863168
    if all(v is not None for v in sources.values()):
        assert sum(sources.values())==total_input
    external=sources['external_kv_transfer']
    if cfg['policy']=='nohost':
        assert '--kv-transfer-config' not in execution['server_command']
        assert external in (None,0)
        h2d=d2h=0
        transfer_scope='zero by absent connector configuration'
    else:
        h2d=transfer(before,after,'CPU_to_GPU',external)
        d2h=transfer(before,after,'GPU_to_CPU',external)
        transfer_scope='native all-TP-worker aggregated operation bytes'
        if external is not None and h2d is not None:
            assert h2d==external*262144
    assert execution['actual_gpu_pool_tokens']==[70000]
    assert execution['gpu_release_verified'] and not execution['owned_live_pids_after_cleanup']
    span=(max(r['done_ns'] for r in rows)-min(r['issue_ns'] for r in rows))/1e9
    argv=execution['server_command'].copy()
    if '--kv-transfer-config' in argv:
        position=argv.index('--kv-transfer-config')
        connector=json.loads(argv[position+1])
        assert connector==dict(kv_connector='OffloadingConnector',kv_role='kv_both',
                               kv_connector_extra_config=dict(block_size=16,cpu_bytes_to_use=64*2**30))
        del argv[position:position+2]
    stack_signature=dict(command=argv,scheduler_sha256=execution['scheduler_sha256'],batch_invariant=False)
    return dict(run=path.name,status='VALID',config=cfg,session_metrics=sessions,
                stack_signature=stack_signature,
                session_e2e_median_ms=statistics.median(s['e2e_ms'] for s in sessions),
                resume_p95_ms=percentile([v for s in sessions for v in s['resume_ttft_ms']],.95),
                background_makespan_ms=(bg[-1]['done_ns']-bg[0]['issue_ns'])/1e6,
                background_requests=[dict(turn=r['turn'],e2e_ms=(r['done_ns']-r['issue_ns'])/1e6,
                                          ttft_ms=(r['first_ns']-r['issue_ns'])/1e6,
                                          first_token_to_response_end_mean_ms=(r['done_ns']-r['first_ns'])/127e6)
                                     for r in bg],
                output_tokens_per_second=sum(len(r['tokens']) for r in rows)/span,
                total_input_tokens=total_input,total_output_tokens=sum(len(r['tokens']) for r in rows),
                sources=sources,h2d_bytes=h2d,d2h_bytes=d2h,transfer_scope=transfer_scope,
                input_identity=identities,gpu_hours=execution['gpu_hours'])


def compare(fair,baseline):
    assert fair['status']==baseline['status']=='VALID'
    assert fair['input_identity']==baseline['input_identity']
    assert fair['stack_signature']==baseline['stack_signature']
    for key in ['sessions','turns','output','seed','cpu_bytes_budget','tool_gap_s','gpu_bytes_per_rank']:
        assert fair['config'][key]==baseline['config'][key]
    change=lambda a,b:100*(a/b-1)
    per=[change(a['e2e_ms'],b['e2e_ms']) for a,b in zip(fair['session_metrics'],baseline['session_metrics'])]
    e2e=change(fair['session_e2e_median_ms'],baseline['session_e2e_median_ms'])
    bg=change(fair['background_makespan_ms'],baseline['background_makespan_ms'])
    return dict(baseline=baseline['run'],session_e2e_median_change_pct=e2e,
                resume_p95_change_pct=change(fair['resume_p95_ms'],baseline['resume_p95_ms']),
                background_makespan_change_pct=bg,per_session_e2e_change_pct=per,
                primary_gate_pass=e2e<=-10 and bg<=5 and max(per)<=5)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path)
    parser.add_argument('--compare',type=Path,nargs=3,metavar=('NOHOST','FULL','FAIR'))
    args=parser.parse_args()
    if bool(args.directory)==bool(args.compare):
        parser.error('choose either one directory or three matched comparison directories')
    if args.compare:
        nohost,full,fair=[analyze(path) for path in args.compare]
        assert [r['config']['policy'] for r in (nohost,full,fair)]==['nohost','full','fair']
        comparisons=[compare(fair,baseline) for baseline in (nohost,full)]
        result=dict(comparisons=comparisons,primary_gate_pass=all(r['primary_gate_pass'] for r in comparisons),
                    caveat='one frozen matched triple; no statistical significance claim')
        (args.compare[-1].parent/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))
        return
    result=analyze(args.directory)
    (args.directory/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['input_identity','session_metrics']},indent=2))


if __name__=='__main__':
    main()
