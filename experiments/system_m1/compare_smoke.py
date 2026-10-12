"""Optional single-pair screen analysis; refuse content/performance mixing."""
import argparse
import json
from pathlib import Path
import re
import statistics


def load(directory, policy):
    execution=json.loads((directory/'execution.json').read_text())
    if execution['status']!='COMPLETE' or not execution['cleanup_verified'] or execution['mode']!='mechanism' or execution['policy']!=policy:
        raise ValueError('matched completed mechanism-mode run required')
    pools = re.findall(r'GPU KV cache size: ([\d,]+) tokens', (directory/'server.log').read_text(errors='replace'))
    if not pools:raise ValueError('actual GPU KV pool missing')
    execution['actual_gpu_pool_tokens'] = sorted({int(p.replace(',','')) for p in pools})
    rows=json.loads((directory/'requests.json').read_text())
    if len(rows)!=12 or any(len(r['tokens'])!=execution['config']['output_tokens'] for r in rows):
        raise ValueError('prescribed real request/output count differs')
    labels={r['label']:r for r in rows}
    if len(labels)!=len(rows):raise ValueError('duplicate request identity')
    sessions={key:(labels[end]['done_ns']-labels[start]['issue_ns'])/1e6 for key,start,end in
              [('long','long0','long_end'),('short','short0','short_end')]}
    return execution,labels,sessions


def compare(m1,fair):
    a,ar,ae=load(m1,'m1');b,br,be=load(fair,'fair')
    if a['config']!=b['config'] or ar.keys()!=br.keys():raise ValueError('configuration/input schedule differs')
    if a['actual_gpu_pool_tokens'] != b['actual_gpu_pool_tokens']:raise ValueError('actual GPU KV capacity differs')
    for key in ar:
        if any(ar[key][field]!=br[key][field] for field in ('key','length','generation','end','cold','prompt_sha256')):
            raise ValueError('actual inputs differ')
    result=dict(kind='single synthetic matched pair, not stable/significant benefit',
                actual_gpu_pool_tokens=a['actual_gpu_pool_tokens'],
                per_request={key:dict(m1_ttft_ms=ar[key]['ttft_ms'],fair_ttft_ms=br[key]['ttft_ms'],
                                      m1_source=ar[key]['source_trace'],fair_source=br[key]['source_trace'],
                                      m1_transfer_delta=ar[key]['transfer_bytes_delta'],fair_transfer_delta=br[key]['transfer_bytes_delta']) for key in ar},
                session_e2e_ms={'m1':ae,'fair':be},
                median_session_e2e_change_percent=100*(statistics.median(ae.values())/statistics.median(be.values())-1),
                dynamic_benefit_observed='UNKNOWN',not_run_baselines=['Full/NoHost not evaluated by this comparison'],
                causal_note='Compare long3 external depth/compute and store traffic; timing alone does not prove elasticity benefit.')
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--m1',type=Path,required=True);parser.add_argument('--fair',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.write_text(json.dumps(compare(args.m1,args.fair),indent=2)+'\n')


if __name__=='__main__':main()
