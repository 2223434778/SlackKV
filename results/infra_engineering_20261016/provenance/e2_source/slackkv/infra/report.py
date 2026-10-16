"""Global aggregate metrics; absent data is UNKNOWN, never per-request attribution."""
import csv
import json
from pathlib import Path
import re
import statistics

from slackkv.infra.runtime import save


def counter(raw,name,label=None):
    values=[]
    for line in raw.splitlines():
        if not line.startswith(name+'{'):continue
        if label is not None and label not in line:continue
        values.append(float(line.rsplit(' ',1)[1]))
    return sum(values) if values else None


def delta(before,after,name,label):
    a=counter(before,name,label);b=counter(after,name,label)
    if a is None or b is None:return None
    if b<a:raise ValueError('metric counter reset/decreased')
    return b-a


def create(out):
    out=Path(out);rows=json.loads((out/'requests.json').read_text(encoding='utf-8'));sessions=json.loads((out/'sessions.json').read_text(encoding='utf-8'))
    before=(out/'metrics_before.prom').read_text(encoding='utf-8');after=(out/'metrics_after.prom').read_text(encoding='utf-8')
    metrics={s:delta(before,after,'vllm:prompt_tokens_by_source_total',f'source="{s}"')
             for s in ('local_compute','local_cache_hit','external_kv_transfer')}
    traffic={d:delta(before,after,'vllm:kv_offload_total_bytes_total',f'transfer_type="{d}"')
             for d in ('GPU_to_CPU','CPU_to_GPU')}
    table=[]
    for row in rows:
        if not row['issue_ns']<=row['first_ns']<=row['done_ns']:raise ValueError('invalid request timestamp')
        usage=row.get('usage') or {};n=usage.get('completion_tokens')
        table.append(dict(session_id=row['session_id'],generation=row['generation'],request_id=row['request_id'],
            ttft_first_content_ms=(row['first_ns']-row['issue_ns'])/1e6,
            response_ms=(row['done_ns']-row['issue_ns'])/1e6,
            first_content_to_end_mean_ms=None if n is None or n<=1 else (row['done_ns']-row['first_ns'])/((n-1)*1e6),
            prompt_tokens=usage.get('prompt_tokens'),completion_tokens=n))
    overlap=any(a['session_id']!=b['session_id'] and max(a['issue_ns'],b['issue_ns'])<min(a['done_ns'],b['done_ns']) for a in rows for b in rows)
    result=dict(status='COMPLETE',requests=len(rows),sessions=len(sessions),cross_session_request_overlap=overlap,
        session_e2e_ms={s['session_id']:(s['done_ns']-s['started_ns'])/1e6 for s in sessions},
        prompt_sources=metrics,transfer_bytes=traffic,requests_table=table,
        missing='null means UNKNOWN; absent transfer labels are not filled with zero',
        attribution='global counters over this demo; no per-request attribution during overlap',
        timing='client first nonempty SSE content; interval proxy is not a TPOT distribution',
        physical_host_occupancy='UNKNOWN; configured capacity is not actual occupancy',
        comparison='free generation histories can differ by policy; not a strict throughput ranking')
    save(out/'report.json',result)
    with (out/'requests.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    (out/'summary.md').write_text('# SlackKV real-chat demo\n\n'+
        f"- Requests: {len(rows)}; sessions: {len(sessions)}; overlapping requests: {overlap}.\n"+
        '- Client-driven local JSON tool; actual assistant history retained.\n'+
        '- Missing values: UNKNOWN. Global counters do not attribute concurrent requests.\n'+
        '- Free generation demo is not a fair throughput benchmark.\n\n'+
        '```json\n'+json.dumps({k:v for k,v in result.items() if k!='requests_table'},ensure_ascii=False,indent=2)+'\n```\n',encoding='utf-8')
    return result
