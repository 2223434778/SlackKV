import pytest

from slackkv.m1.controller import PrefixController


def path(name,n):return tuple(f'{name}-{i}'.encode() for i in range(n))


def test_new_terminal_does_not_trim_or_rotate_existing_retention():
    c=PrefixController(512)
    c.observe('a0','a',0,path('a',512));c.finish('a0')
    before=(c.members[('a',0)].target,c.cursor,c.version)
    terminal=c.observe('t','t',0,path('t',512),terminal=True)
    assert c.allowed(terminal)==0 and c.members[('t',0)].retention_blocks==0
    assert (c.members[('a',0)].target,c.cursor,c.version)==before
    c.end('t',0);c.finish('t');c.collect(('t',0))
    assert (c.members[('a',0)].target,c.cursor,c.version)==before
    normal=c.observe('normal','normal',0,path('normal',512))
    assert c.allowed(normal)==c.members[('a',0)].target==256


def test_existing_terminal_growth_keeps_old_demand_and_odd_remainder():
    c=PrefixController(9)
    c.observe('a0','a',0,path('a',8));c.finish('a0')
    c.observe('b0','b',0,path('b',8));c.finish('b0')
    before=({t:m.target for t,m in c.members.items()},c.cursor,c.version)
    p=c.observe('a-end','a',0,path('a',12),terminal=True)
    assert p.path==path('a',12) and c.members[('a',0)].retention_blocks==8
    assert c.allowed(p)==0
    assert ({t:m.target for t,m in c.members.items()},c.cursor,c.version)==before


def test_shared_root_is_charged_once_and_last_logical_reference_survives():
    c=PrefixController(9,shared_prefix_blocks=3);root=path('root',3)
    a=c.observe('a0','a',0,root+path('a',6));c.finish('a0')
    b=c.observe('b0','b',0,root+path('b',6))
    assert a.reason is None and b.reason is None
    assert [m.target for m in c.members.values()]==[6,6]
    assert c.snapshot()['target_distinct_blocks']==9
    assert all(c.key_refs[k]=={('a',0),('b',0)} for k in root)
    c.end('a',0);c.collect(('a',0))
    assert all(c.key_refs[k]=={('b',0)} for k in root)
    assert c.members[('b',0)].target==9 and c.allowed(b)==6
    c.finish('b0');assert c.observe('b1','b',0,root+path('b',6)).ceiling==9


@pytest.mark.parametrize('capacity',range(10))
def test_shared_root_larger_than_capacity_never_overreserves(capacity):
    c=PrefixController(capacity,shared_prefix_blocks=6)
    for s in ['a','b','c']:
        p=c.observe(s,s,0,path('root',6)+path(s,6))
        assert p.reason is None and c.snapshot()['target_distinct_blocks']<=capacity
    if capacity<=6:assert all(m.target==capacity for m in c.members.values())


def test_private_merge_or_shared_beyond_declared_root_is_rejected():
    c=PrefixController(10,shared_prefix_blocks=2)
    root=path('root',2);c.observe('a','a',0,root+path('a',3))
    p=c.observe('merge','b',0,root+path('b',1)+path('a',2))
    assert p.reason=='unsupported_shared_shape' and c.allowed(p)==0
    q=c.observe('copy','c',0,root+path('a',3))
    assert q.reason=='unsupported_shared_shape' and c.allowed(q)==0


def test_retired_private_identity_wait_is_transient_in_shared_mode():
    c=PrefixController(8,shared_prefix_blocks=2);p=path('root',2)+path('a',4)
    c.observe('old','a',0,p);c.finish('old')
    q=c.observe('new','a',1,p)
    assert q.reason=='retired_identity_pending'
    c.collect(('a',0));c.finish('new')
    assert c.observe('retry','a',1,p).reason is None


def test_shared_runner_keeps_detailed_logs_in_content_only_and_common_full_gate():
    import json
    from pathlib import Path
    from experiments.system_m1.smoke import command
    from experiments.system_m1.contention import SHARED_SEQUENCE
    c=json.loads(Path('experiments/system_m1/shared_config.json').read_text())
    extra=json.loads(command(c,'m1','mechanism',234881024)[-1])['kv_connector_extra_config']
    assert extra['slackkv_m1_trace'] is False and extra['slackkv_shared_prefix_blocks']==32
    assert json.loads(command(c,'m1','content',234881024)[-1])['kv_connector_extra_config']['slackkv_m1_trace'] is True
    assert json.loads(command(c,'full','mechanism',234881024)[-1])['kv_connector_extra_config']['spec_name']=='TerminalCPUOffloadingSpec'
    labels=[s[0] for s in SHARED_SEQUENCE]
    assert len(labels)==19 and labels.index('a_end')<labels.index('pressure3')<labels.index('b_after_a_end')<labels.index('b_end')
    from experiments.system_m1.contention import store_cap
    cap=store_cap(c,'fair','a',8192)
    assert cap==2432 and 4*cap-3*512==c['host_tokens']


@pytest.mark.parametrize('text,valid',[
    ('MAPLE42<|im_end|>',True),
    ('The secret password is **MAPLE42**.<|im_end|>',True),
    ('CEDAR17<|im_end|>',False),
    ('CEDAR17 or MAPLE42',False),
    ('123456',False),
    ('Unknown<|im_end|>MAPLE42',False),
    ('MAPLE42 MAPLE42',False)])
def test_marker_oracle_rejects_cross_identity_ambiguous_and_post_eos_answers(text,valid):
    from experiments.system_m1.contention import correct_private_marker
    assert correct_private_marker(text,'MAPLE42') is valid


def test_marker_oracle_does_not_accept_a_negated_correct_marker():
    from experiments.system_m1.contention import correct_private_marker
    assert not correct_private_marker('The password is not MAPLE42.','MAPLE42')


def test_analysis_rejects_duplicate_policy_and_suffix_keeps_all_budget(tmp_path):
    import json
    from pathlib import Path
    import shutil
    from experiments.system_m1.analyze_contention import analyze
    # CPU-only copied-artifact fixture, not another GPU execution/measurement.
    source=Path('results/system_m1_smoke')
    for name in ['m1_shared_m11','m1_shared_fair1','m1_shared_full1']:
        shutil.copytree(source/name,tmp_path/name)
    shutil.copytree(source/'m1_shared_m11',tmp_path/'duplicate')
    with pytest.raises(ValueError,match='multiple runs per policy'):
        analyze(tmp_path,'shared_v1')
    selected=analyze(tmp_path,'shared_v1','1')
    assert len(selected['runs'])==3 and 'duplicate' not in selected['runs']
    assert len(selected['executions'])==4
    spent=sum(json.loads(p.read_text())['gpu_hours'] for p in tmp_path.glob('*/execution.json'))
    assert selected['additional_gpu_hours']==pytest.approx(spent)
