import json

import pytest

from slackkv.infra.benchmark import compare,prepare
from slackkv.infra.config import Config


def test_frozen_benchmark_records_exact_shapes_and_rejects_wrong_regime(tmp_path,monkeypatch):
    model=tmp_path/'model';model.mkdir()
    (model/'config.json').write_text(json.dumps(dict(model_type='qwen2',max_position_embeddings=32768,
        num_hidden_layers=28,num_key_value_heads=2,hidden_size=1536,num_attention_heads=12,use_sliding_window=False)))
    class Tokenizer:
        def encode(self,*args,**kwargs):return [100]*70000
    monkeypatch.setattr('slackkv.infra.benchmark.tokenizer',lambda cfg:Tokenizer())
    config=Config(str(model),'runtime')
    plan=prepare(config,'host-limited',tmp_path/'limited')
    assert [len(plan['inputs']['cedar'+str(i)]) for i in range(3)]==[8192,8256,8320]
    assert len(plan['inputs']['pressure'])==16384
    assert 'policy' not in plan['config_signature'] and plan['output_tokens']==16
    with pytest.raises(ValueError):prepare(config,'host-enough',tmp_path/'wrong')
    with pytest.raises(FileExistsError):prepare(config,'host-limited',tmp_path/'limited')


def test_comparison_never_silently_chooses_unmatched_or_repeated_policy(tmp_path):
    dirs=[]
    for name in ('fair','full','nohost'):
        path=tmp_path/name;path.mkdir();dirs.append(path)
        (path/'benchmark.json').write_text(json.dumps(dict(policy=name,status='PASS',prompt_identity={'a':'same'},input_tokens=10,output_tokens=16,actual_gpu_tokens=100)))
        (path/'benchmark_plan.json').write_text(json.dumps({'same':'config'}))
        (path/'execution.json').write_text(json.dumps(dict(status='COMPLETE',cleanup_verified=True)))
    assert len(compare(dirs,tmp_path/'good.json'))==3
    (dirs[-1]/'benchmark_plan.json').write_text(json.dumps({'different':'budget'}))
    with pytest.raises(ValueError):compare(dirs,tmp_path/'bad.json')
    assert not (tmp_path/'bad.json').exists()


def test_compare_rejects_unsettled_execution(tmp_path):
    p=tmp_path/'one';p.mkdir()
    for name,value in [('benchmark',{}),('benchmark_plan',{}),('execution',dict(status='RUNNING',cleanup_verified=False))]:
        (p/(name+'.json')).write_text(json.dumps(value))
    with pytest.raises(ValueError):compare([p,p,p],tmp_path/'result.json')
