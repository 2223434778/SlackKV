import gzip
import hashlib
import json
import math
from pathlib import Path

from experiments.system_m1.analyze_contention import load
from experiments.system_m1.contention import correct_private_marker

root=Path.cwd();base=root/'results/system_m1_smoke';out=root/'results/system_m1_cost'
server=load(out/'analysis.json');local=load(out/'analysis_verified.json')
diff=[]
def compare(a,b,path=''):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:compare(a[k],b[k],path+'/'+k)
    elif isinstance(a,list):
        assert len(a)==len(b)
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'/'+str(i))
    elif a!=b:
        budget=path in ('/cohort/additional_gpu_hours','/cohort/continuous_gpu_hours')
        sums=path.startswith('/existing_event_diagnosis/original_screen_cost_decomposition/') and path.rsplit('/',1)[-1] in ('request_e2e_sum_ms','pressure_request_e2e_sum_ms','between_requests_ms')
        # Python 3.11 and 3.13 float sum implementations differ in their last bits.
        assert (budget or sums) and math.isclose(a,b,abs_tol=1e-15 if budget else 1e-8,rel_tol=0)
        diff.append(dict(path=path,server=a,local=b))
compare(server,local)
checks={}
for name in ('m1_cost_ref1','m1_cost_opt1','m1_cost_fair1'):
    p=base/name;e=load(p/'execution.json');rows=load(p/'requests.json')
    assert e['status']=='COMPLETE' and e['cleanup_verified']
    assert len(rows)==19 and sum(r['length'] for r in rows)==153088
    assert sum(len(r['tokens']) for r in rows)==304
    sources=dict(local_cache_hit=0,external_kv_transfer=0,local_compute=0)
    for r in rows:
        assert len(r['source_trace'])==1
        t=r['source_trace'][0]
        sources['local_cache_hit']+=t['local_cached_tokens']
        sources['external_kv_transfer']+=t['external_cached_tokens']
        sources['local_compute']+=r['length']-t['local_cached_tokens']-t['external_cached_tokens']
    assert sources==server['cohort']['runs'][name]['source_tokens']
    assert hashlib.sha256((out/'executed_sources/contention.py').read_bytes()).hexdigest()==e['plan']['additional_runner_sources']['contention.py']
    assert hashlib.sha256((out/'executed_sources/smoke.py').read_bytes()).hexdigest()==e['runner_sha256']
    native=out/('reference' if name.endswith('ref1') else 'optimized')/'native.py'
    assert hashlib.sha256(native.read_bytes()).hexdigest()==e['sources']['native.py']
    assert load(p/'state_events.json')==[]
    checkpoints=load(p/'checkpoints.json')
    assert len(checkpoints)==2 and checkpoints[0]['free_blocks']==512
    assert all(c['pending_store_blocks']==c['load_protected_blocks']==0 for c in checkpoints)
    checks[name]=dict(sources_from_actual_admissions=sources,compiled=e['command'].count('--enforce-eager')==0,
                     checkpoints=checkpoints,source_hashes_verified=True)
p=base/'m1_cost_content1';rows={r['label']:r for r in load(p/'requests.json')}
cold=load(p/'cold.json');comparisons=[]
for c,label,marker in zip(cold,('a1','b1','b_after_a_end'),('CEDAR17','MAPLE42','MAPLE42')):
    assert correct_private_marker(c['text'],marker)
    assert len(c['tokens'])==16 and rows[label]['tokens']==c['tokens']
    assert len(c['source_trace'])==1
    assert c['source_trace'][0]['local_cached_tokens']==c['source_trace'][0]['external_cached_tokens']==0
    assert rows[label]['source_trace'][0]['external_cached_tokens']>512
    comparisons.append(dict(label=label,expected=marker,full_16_ids_match=True,strict_current_grammar_pass=True,
                            cold_text=c['text'],source=rows[label]['source_trace'][0]))
states=load(p/'state_events.json')
assert len(states)==70
assert all(s['physical_ready_blocks']+s['physical_pending_store_blocks']+s['physical_free_blocks']==512 for s in states)
assert states[-1]['physical_free_blocks']==512 and states[-1]['members']==[]
assert max(s['physical_shared_blocks'] for s in states)==32
assert load(out/'native_audit.json')['status']=='PASS'
assert len(load(out/'native_audit.json')['checks'])==29
new_cost=sum(load(base/n/'execution.json')['gpu_hours'] for n in
             ('m1_cost_ref1','m1_cost_content1','m1_cost_opt1','m1_cost_fair1'))
extra=[]
for file in sorted((out/'executed_sources').glob('*.py')):
    raw=file.read_bytes()
    extra.append(dict(path=str(file.relative_to(root)).replace('\\','/'),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
(out/'spotcheck.json').write_text(json.dumps(dict(performance=checks,content=comparisons,
    content_state_count=70,native_audit_groups=29,new_gpu_hours=new_cost,
    additional_gpu_hours=server['cohort']['additional_gpu_hours'],
    local_server_float_differences=diff,executed_source_supplement=extra,
    qualification='Current strict oracle offline verified; remote runner bytes pinned by preparation hash. No reinterpretation of failures or new GPU run.'),indent=2)+'\n',encoding='utf-8')
print('independent sources/content/checkpoints/provenance verification PASS',new_cost,diff)
