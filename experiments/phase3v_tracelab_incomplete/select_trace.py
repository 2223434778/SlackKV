from __future__ import annotations
import csv,hashlib,json,statistics
from collections import defaultdict
from pathlib import Path
import duckdb

ROOT=Path(__file__).resolve().parents[2]; DB=ROOT/'external/tracelab/syfi_coding_trace.duckdb'; OUT=ROOT/'results/phase3v'
EXPECTED='97715265367cc72376475f5d444c8e1900b88cab1482aa7b9a742894d9f15619'
def pct(xs,p):
 xs=sorted(xs); q=(len(xs)-1)*p/100; lo=int(q); hi=min(lo+1,len(xs)-1); return xs[lo]+(xs[hi]-xs[lo])*(q-lo)
def klass(m): return 'small' if m<3072 else ('medium' if m<5632 else 'large')
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 assert hashlib.sha256(DB.read_bytes()).hexdigest()==EXPECTED
 c=duckdb.connect(str(DB),read_only=True)
 rows=c.execute('''
 WITH tools AS (
  SELECT round_pk, min(emitted_at) emit, max(result_at) result,
         datediff('millisecond',min(emitted_at),max(result_at)) pause_ms,
         min(tool_name) tool_name, count(*) tool_count
  FROM tool_calls WHERE emitted_at IS NOT NULL AND result_at IS NOT NULL
  GROUP BY round_pk
 )
 SELECT r.session_id,r.provider,r.model,r.round_index,r.round_pk,
        r.input_tokens_total kv_tokens,n.input_tokens_total next_input_tokens,
        greatest(0,n.input_tokens_total-r.input_tokens_total) append_tokens,
        tools.pause_ms,tools.tool_name,tools.tool_count
 FROM rounds r JOIN rounds n ON n.session_id=r.session_id AND n.round_index=r.round_index+1
 JOIN tools ON tools.round_pk=r.round_pk
 WHERE r.input_tokens_total BETWEEN 1024 AND 8192
   AND n.input_tokens_total<=9000 AND tools.pause_ms BETWEEN 100 AND 3000
 ORDER BY r.session_id,r.round_index
 ''').fetchall()
 by=defaultdict(list)
 for row in rows:
  keys=('session_id','provider','model','round_index','round_pk','kv_tokens','next_input_tokens','append_tokens','tool_pause_ms','tool_name_category','tool_count')
  by[row[0]].append(dict(zip(keys,row)))
 episodes=[]
 for sid,rr in by.items():
  rr=sorted(rr,key=lambda x:x['round_index'])
  for i in range(len(rr)-3):
   w=rr[i:i+4]
   if [x['round_index'] for x in w] != list(range(w[0]['round_index'],w[0]['round_index']+4)): continue
   lengths=[w[0]['kv_tokens']]+[x['next_input_tokens'] for x in w]
   if any(b<a for a,b in zip(lengths,lengths[1:])): continue
   m=statistics.median(x['kv_tokens'] for x in w)
   episodes.append((klass(m),hashlib.sha256((sid+'|SlackKV-Phase3V').encode()).hexdigest(),sid,w,m));break
 selected=[]
 for k in ('small','medium','large'):
  candidates=sorted((x for x in episodes if x[0]==k),key=lambda x:x[1])[:2]
  assert len(candidates)==2,k;selected+=candidates
 out=[]
 for ei,(k,h,sid,w,m) in enumerate(selected):
  eid=f'E{ei:02d}'
  for ti,x in enumerate(w,1):
   out.append({'episode_id':eid,'logical_session_hash':hashlib.sha256(sid.encode()).hexdigest(),'provider':x['provider'],'turn_index':ti,'kv_tokens':x['kv_tokens'],'next_input_tokens':x['next_input_tokens'],'append_tokens':x['append_tokens'],'tool_pause_ms':x['tool_pause_ms'],'tool_name_category':x['tool_name_category'],'trace_round_index':x['round_index'],'size_class':k,'lease_key':f'{eid}:turn:{ti}'})
 with (OUT/'trace_selection.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
 (OUT/'trace_selection.json').write_text(json.dumps(out,indent=2))
 kv=[x['kv_tokens'] for x in out];pa=[x['tool_pause_ms'] for x in out];ap=[x['append_tokens'] for x in out]
 dist={'kv_tokens':{'min':min(kv),'P25':pct(kv,25),'median':pct(kv,50),'P75':pct(kv,75),'max':max(kv)},'tool_pause_ms':{'min':min(pa),'P25':pct(pa,25),'median':pct(pa,50),'P75':pct(pa,75),'max':max(pa)},'append_tokens':{'min':min(ap),'median':pct(ap,50),'max':max(ap)},'episode_classes':{k:sum(x[0]==k for x in selected) for k in ('small','medium','large')}}
 (OUT/'trace_distribution.json').write_text(json.dumps(dist,indent=2))
 print(json.dumps(dist,indent=2));print([(x[0],x[1][:8],x[4]) for x in selected])
if __name__=='__main__':main()
