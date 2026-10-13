import json
from pathlib import Path

import pytest

from experiments.system_m1.smoke import budget, command, execute, resources, server_environment, transfer_delta, wall_limit


def config():
    return json.loads(Path('experiments/system_m1/smoke_config.json').read_text())


def test_all_policies_preserve_compiled_resources_and_true_nohost():
    c = config()
    for policy in ('m1', 'fair', 'full', 'nohost'):
        argv = command(c, policy, 'mechanism', 234881024)
        assert '--enforce-eager' not in argv
        assert argv[argv.index('--tensor-parallel-size')+1] == '1'
        assert argv[argv.index('--kv-cache-memory-bytes')+1] == str(c['gpu_kv_bytes'])
        if policy == 'nohost':
            assert '--kv-transfer-config' not in argv
        else:
            extra = json.loads(argv[argv.index('--kv-transfer-config')+1])['kv_connector_extra_config']
            assert extra['cpu_bytes_to_use'] == 234881024
            if policy == 'm1':assert extra['slackkv_m1_trace'] is False
    extra = json.loads(command(c,'m1','content',234881024)[-1])['kv_connector_extra_config']
    assert extra['slackkv_m1_trace'] is True


def test_smoke_ledger_counts_failed_and_other_policy_runs_and_rejects_unknown_cleanup(tmp_path):
    for name, cost in [('m1-failed', .1), ('fair-renamed', .2)]:
        directory = tmp_path/name;directory.mkdir()
        (directory/'execution.json').write_text(json.dumps(dict(status='FAILED',cleanup_verified=True,gpu_hours=cost)))
    assert budget(tmp_path) == pytest.approx(.3)
    with pytest.raises(RuntimeError,match='insufficient'):
        wall_limit(config(), .5, .49)
    with pytest.raises(ValueError,match='allowance'):
        wall_limit(config(), float('nan'), .3)
    (tmp_path/'fair-renamed'/'execution.json').write_text(json.dumps(dict(status='COMPLETE',cleanup_verified=False,gpu_hours=.2)))
    with pytest.raises(RuntimeError,match='unresolved'):budget(tmp_path)


def test_fp16_host_size_uses_full_real_model_layout_not_parameter_count(tmp_path):
    (tmp_path/'config.json').write_text(json.dumps(dict(num_hidden_layers=28,num_key_value_heads=2,
                                                     hidden_size=1536,num_attention_heads=12,use_sliding_window=False)))
    assert resources(config(),tmp_path) == dict(kv_bytes_per_token=28672,host_bytes=234881024,host_blocks=512)
    changed = config();changed['expected_kv_bytes_per_token'] *= 2
    with pytest.raises(ValueError,match='layout'):resources(changed,tmp_path)
    changed = config();changed['max_model_len'] = 32768
    with pytest.raises(ValueError,match='physical GPU'):resources(changed,tmp_path)
    changed = config();changed['max_model_len'] = 16384
    with pytest.raises(ValueError,match='prompt/output'):resources(changed,tmp_path)


def test_missing_transfer_counter_does_not_become_successful_zero():
    raw='vllm:kv_offload_total_bytes_total{transfer_type="CPU_to_GPU"} 100\n'
    assert transfer_delta('',raw)['CPU_to_GPU'] is None
    assert transfer_delta(raw,raw)['CPU_to_GPU'] == 0
    with pytest.raises(ValueError,match='decreased'):transfer_delta(raw,raw.replace('100','50'))


def test_changing_ledger_root_cannot_reset_gpu_budget(tmp_path):
    with pytest.raises(ValueError,match='ledger root'):
        execute(config(), tmp_path, tmp_path, 'm1', 'content', 0, 'renamed-run', .5)


def test_comparison_refuses_content_mode_as_performance_evidence(tmp_path):
    from experiments.system_m1.compare_smoke import load
    (tmp_path/'execution.json').write_text(json.dumps(dict(status='COMPLETE',cleanup_verified=True,mode='content',policy='m1')))
    with pytest.raises(ValueError,match='mechanism'):load(tmp_path,'m1')


def test_server_inherits_conda_build_tools_and_isolates_modes(monkeypatch):
    monkeypatch.setenv('PATH', '/usr/bin')
    monkeypatch.setenv('SLACKKV_RETENTION_CONTROL', '1')
    c=config()
    env=server_environment(c,'content',1)
    assert env['PATH']==str(Path(c['env_root']))+'/bin:/usr/bin'
    assert env['CUDA_VISIBLE_DEVICES']=='1' and env['SLACKKV_RETENTION_CONTROL']=='0'
    assert env['VLLM_BATCH_INVARIANT']=='1'
    assert server_environment(c,'mechanism',1)['VLLM_BATCH_INVARIANT']=='0'
