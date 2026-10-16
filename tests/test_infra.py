import json
from pathlib import Path
import threading

import pytest

from slackkv.infra.config import Config,layout
from slackkv.infra.http import IncompleteResponse,post
from slackkv.infra.report import delta
from slackkv.infra.report import create as create_report
from slackkv.infra.service import budget,command,init_budget
from slackkv.infra.sessions import SessionError,SessionPool
from slackkv.infra.workloads import chat_tokens


def config(**extra):return Config('local-model','runtime',**extra)


def result(text='actual assistant'):
    return dict(text=text,issue_ns=1,first_ns=2,done_ns=3,request_id='native-id',usage={'completion_tokens':2})


def test_real_history_terminal_generation_and_fixed_slots():
    sent=[]
    def transport(url,body,*args):sent.append(body);return result()
    pool=SessionPool(config(),[16,32],transport)
    a=pool.open('a');pool.open('b')
    with pytest.raises(SessionError):pool.open('c')
    pool.send('a','first')
    pool.send('a','second',terminal=True)
    assert sent[-1]['messages'][-2]=={'role':'assistant','content':'actual assistant'}
    assert sent[0]['kv_transfer_params']['max_offload_tokens']==16
    assert sent[1]['kv_transfer_params']['slackkv_terminal']['end_after_request'] is True
    with pytest.raises(SessionError):pool.send('a','late')
    newer=pool.open('a');assert newer.generation==1 and newer.salt!=a.salt
    assert len(newer.messages)==1
    assert pool.close('a')==pool.close('a')


def test_failure_is_atomic_and_requires_new_identity():
    def failure(*args):raise IncompleteResponse('disconnected')
    pool=SessionPool(config(),[16,16],failure);s=pool.open('a');before=list(s.messages)
    with pytest.raises(IncompleteResponse):pool.send('a','not committed')
    assert s.state=='FAILED' and s.messages==before and not s.pending
    with pytest.raises(SessionError):pool.send('a','retry')
    pool.close('a');new=pool.open('a');assert new.salt!=s.salt


def test_same_session_rejects_concurrent_send_and_close_other_session_progresses():
    entered=threading.Event();release=threading.Event()
    def slow(url,body,*args):
        if body['messages'][-1]['content']=='slow':entered.set();assert release.wait(2)
        return result()
    pool=SessionPool(config(),[16,16],slow);pool.open('a');pool.open('b')
    thread=threading.Thread(target=lambda:pool.send('a','slow'));thread.start();assert entered.wait(1)
    try:
        with pytest.raises(SessionError):pool.send('a','duplicate')
        with pytest.raises(SessionError):pool.close('a')
        assert pool.send('b','fast')['text']=='actual assistant'
    finally:release.set();thread.join(2)
    assert not thread.is_alive()


def test_context_rejection_does_not_poison_or_commit_history():
    pool=SessionPool(config(max_model_len=256),[16,16],token_count=lambda messages:200)
    s=pool.open('a')
    with pytest.raises(SessionError):pool.send('a','long')
    assert s.state=='OPEN' and len(s.messages)==1 and not s.pending


@pytest.mark.parametrize('values',[{'policy':'m1'},{'gpu':True},{'port':0},{'session_slots':65},
    {'host':'0.0.0.0'},{'max_run_seconds':1801},{'batch_invariant':'false'}])
def test_invalid_configuration(values):
    with pytest.raises(ValueError):config(**values)


def test_nohost_is_real_absent_connector_full_and_fair_same_pool():
    assert '--kv-transfer-config' not in command(config(policy='nohost'),4096)
    assert command(config(policy='full'),4096)==command(config(policy='fair'),4096)


def test_quota_rounds_down_and_unused_slots_never_borrow(tmp_path):
    shape=dict(model_type='qwen2',max_position_embeddings=32768,num_hidden_layers=28,
        num_key_value_heads=2,hidden_size=1536,num_attention_heads=12,use_sliding_window=False,sliding_window=32768)
    (tmp_path/'config.json').write_text(json.dumps(shape))
    c=Config(str(tmp_path),'runtime',host_kv_bytes=17*16*28672+3,session_slots=3)
    l=layout(c);assert l['quota_tokens']==[96,96,80] and l['unused_host_bytes']==3


def test_metric_absence_stays_unknown_and_reset_fails():
    assert delta('','m{x="y"} 1','m','x="y"') is None
    with pytest.raises(ValueError):delta('m{x="y"} 2','m{x="y"} 1','m','x="y"')


def test_continuous_engineering_budget_cannot_reuse_unresolved_run(tmp_path):
    init_budget(tmp_path)
    assert budget(tmp_path)[1]==0
    out=tmp_path/'runs/a';out.mkdir(parents=True)
    (out/'execution.json').write_text(json.dumps(dict(status='RUNNING',cleanup_verified=False,gpu_hours=0)))
    with pytest.raises(RuntimeError):budget(tmp_path)
    (out/'execution.json').write_text(json.dumps(dict(status='FAILED',cleanup_verified=True,gpu_hours=.1)))
    assert budget(tmp_path)[1]==.1
    with pytest.raises(FileExistsError):init_budget(tmp_path)


def test_partial_sse_and_cancel_do_not_become_complete(monkeypatch):
    class Response:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def __iter__(self):return iter([b'data: {"choices":[{"delta":{"content":"half"}}]}\n'])
    monkeypatch.setattr('urllib.request.urlopen',lambda *a,**k:Response())
    with pytest.raises(IncompleteResponse):post('http://localhost',{'stream':True})
    cancel=threading.Event();cancel.set()
    with pytest.raises(IncompleteResponse):post('http://localhost',{'stream':True},cancel=cancel)


def test_chat_token_count_never_uses_transformers_mapping_length():
    class Tokenizer:
        def apply_chat_template(self,messages,**kwargs):
            return [1]*200 if kwargs.get('return_dict') is False else {'input_ids':[[1]*200],'attention_mask':[[1]*200]}
    assert len(chat_tokens(Tokenizer(),[]))==200


def test_unknown_chat_tokenizer_layout_rejected():
    class Tokenizer:
        def apply_chat_template(self,*args,**kwargs):return {'input_ids':[[1,2]]}
    with pytest.raises(ValueError):chat_tokens(Tokenizer(),[])


def test_utf8_report_reads_real_non_ascii_history(tmp_path):
    row=dict(session_id='项目甲',generation=0,request_id='r',issue_ns=1,first_ns=2,done_ns=3,
             usage=dict(prompt_tokens=10,completion_tokens=2),text='你好')
    (tmp_path/'requests.json').write_text(json.dumps([row],ensure_ascii=False),encoding='utf-8')
    (tmp_path/'sessions.json').write_text(json.dumps([dict(session_id='项目甲',started_ns=1,done_ns=3)],ensure_ascii=False),encoding='utf-8')
    for name in ('metrics_before.prom','metrics_after.prom'):(tmp_path/name).write_text('',encoding='utf-8')
    result=create_report(tmp_path)
    assert result['session_e2e_ms']['项目甲']==2e-6
    assert all(v is None for v in result['transfer_bytes'].values())


def test_terminal_slot_cannot_be_reused_before_request_settlement():
    pool=SessionPool(config(session_slots=1),[16]);s=pool.open('a')
    # Reachable between terminal commit and the send() finally block.
    s.state='CLOSED';s.pending=True
    with pytest.raises(SessionError):pool.open('a')
    with pytest.raises(SessionError):pool.open('b')
    assert pool.generations['a']==0
    s.pending=False
    assert pool.open('a').generation==1


def test_archived_metrics_report_writes_separate_output(tmp_path):
    import gzip
    test_utf8_report_reads_real_non_ascii_history(tmp_path)
    original=(tmp_path/'report.json').read_bytes()
    for name in ('metrics_before.prom','metrics_after.prom'):
        path=tmp_path/name
        with gzip.open(str(path)+'.gz','wb') as stream:stream.write(path.read_bytes())
        path.unlink()
    destination=tmp_path/'regenerated'
    assert create_report(tmp_path,destination)['requests']==1
    assert (destination/'requests.csv').is_file()
    assert (tmp_path/'report.json').read_bytes()==original


def test_source_digest_ignores_only_generated_top_level_version(tmp_path):
    from slackkv.infra.runtime import source_tree_digest
    (tmp_path/'module.py').write_bytes(b'x=1\n')
    before=source_tree_digest(tmp_path)
    (tmp_path/'_version.py').write_text('generated')
    assert source_tree_digest(tmp_path)==before
    (tmp_path/'module.py').write_bytes(b'x=1\r\n')
    assert source_tree_digest(tmp_path)==before
    nested=tmp_path/'nested';nested.mkdir();(nested/'_version.py').write_text('upstream')
    assert source_tree_digest(tmp_path)!=before
