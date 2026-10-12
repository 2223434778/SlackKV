import hashlib
from experiments.large_model.acquire import verify


def test_weight_hash_and_length_are_both_required(tmp_path):
    path = tmp_path / 'weight'
    path.write_bytes(b'abcd')
    record = dict(bytes=4, sha256=hashlib.sha256(b'abcd').hexdigest(), git_blob_sha1=None)
    assert verify(path, record)
    path.write_bytes(b'abce')
    assert not verify(path, record)
    path.write_bytes(b'abcdx')
    assert not verify(path, record)


def test_compiled_tp2_command_preserves_actual_resource_configuration():
    import json
    from pathlib import Path
    from experiments.large_model.run import command
    argv = command(Path('/model'), 32 * 2**30)
    assert '--enforce-eager' not in argv
    assert '--kv-cache-memory-bytes' not in argv
    assert argv[argv.index('--tensor-parallel-size') + 1] == '2'
    assert argv[argv.index('--gpu-memory-utilization') + 1] == '0.85'
    config = json.loads(argv[argv.index('--kv-transfer-config') + 1])
    assert config['kv_connector_extra_config']['cpu_bytes_to_use'] == 32 * 2**30


def test_small_file_uses_git_blob_identity(tmp_path):
    raw = b'{"config":1}\n'
    path = tmp_path / 'config.json'
    path.write_bytes(raw)
    record = dict(bytes=len(raw), sha256=None,
                  git_blob_sha1=hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest())
    assert verify(path, record)


def test_large_model_analysis_rejects_unresolved_execution(tmp_path):
    import json
    import pytest
    from experiments.large_model.analyze import analyze
    (tmp_path / 'execution.json').write_text(json.dumps(dict(status='RUNNING')))
    with pytest.raises(ValueError, match='unresolved'):
        analyze(tmp_path)


def test_large_model_budget_counts_both_gpu_lifecycles(tmp_path):
    import json
    import pytest
    from experiments.large_model.run import budget
    for family in ('autonomous_fork', 'autonomous_round4', 'host_prefix_cost_20261010', 'host_prefix_followup_20261010'):
        path = tmp_path/'results'/family/'run'/'execution.json'
        path.parent.mkdir(parents=True)
        key = 'gpu_hours_conservative' if family in ('autonomous_fork', 'autonomous_round4') else 'gpu_hours'
        path.write_text(json.dumps(dict(status='COMPLETE', **{key: .2})))
    path = tmp_path/'results/large_model_20261012/tp2/execution.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(status='COMPLETE', gpu_hours=.4)))
    assert sum(budget(tmp_path).values()) == pytest.approx(1.2)
    path.write_text(json.dumps(dict(status='RUNNING')))
    with pytest.raises(RuntimeError, match='unresolved'):
        budget(tmp_path)


def test_strong_nohost_has_no_connector_and_matches_measured_pool():
    from pathlib import Path
    from experiments.large_model.run import command
    argv = command(Path('/model'), 64*2**30, 70000*131072, connector=False)
    assert '--kv-transfer-config' not in argv
    assert argv[argv.index('--kv-cache-memory-bytes')+1] == str(70000*131072)


def test_tp2_prefix_quotas_conserve_real_host_blocks():
    from experiments.large_model.screen import quotas
    result = quotas(64*2**30)
    assert len(result)==21
    assert result[:-1]==[12480]*20 and result[-1]==12544
    assert sum(result)*262144==64*2**30


def test_oracle_rejects_performance_execution(tmp_path):
    import json
    import pytest
    from experiments.large_model.analyze_oracle import analyze
    (tmp_path/'execution.json').write_text(json.dumps(dict(status='COMPLETE', batch_invariant=False)))
    with pytest.raises(ValueError, match='batch-invariant'):
        analyze(tmp_path)


def test_screen_rejects_unfinished_execution(tmp_path):
    import json
    import pytest
    from experiments.large_model.analyze_screen import analyze
    (tmp_path/'execution.json').write_text(json.dumps(dict(status='RUNNING')))
    with pytest.raises(ValueError, match='unresolved'):
        analyze(tmp_path)


def test_primary_gate_rejects_background_or_individual_session_harm():
    from copy import deepcopy
    from experiments.large_model.analyze_screen import compare
    config=dict(sessions=21,turns=3,output=128,seed=914477,cpu_bytes_budget=64*2**30,
                tool_gap_s=5,gpu_bytes_per_rank=70000*131072)
    baseline=dict(status='VALID',run='baseline',input_identity={'a':'same'},stack_signature={'scheduler':'same'},
                  config=config,session_metrics=[dict(e2e_ms=100) for _ in range(21)],
                  session_e2e_median_ms=100,resume_p95_ms=100,background_makespan_ms=100)
    fair=deepcopy(baseline)
    fair['session_e2e_median_ms']=85
    assert compare(fair,baseline)['primary_gate_pass']
    fair['background_makespan_ms']=106
    assert not compare(fair,baseline)['primary_gate_pass']
    fair['background_makespan_ms']=100
    fair['session_metrics'][0]['e2e_ms']=106
    assert not compare(fair,baseline)['primary_gate_pass']


def test_watchdog_closes_only_owned_group_before_raising(monkeypatch):
    import signal
    from types import SimpleNamespace
    import pytest
    from experiments.large_model.run import expire_owned_group
    calls=[]
    monkeypatch.setattr('experiments.large_model.run.os.killpg',lambda pid,sig:calls.append((pid,sig)),raising=False)
    with pytest.raises(TimeoutError,match='watchdog'):
        expire_owned_group(SimpleNamespace(pid=12345))
    assert calls==[(12345,signal.SIGTERM)]


def test_compressed_evidence_verification_detects_modified_content(tmp_path):
    import json
    import lzma
    import pytest
    from experiments.large_model.verify_archive import verify
    raw=b'[1,2,3]'
    path=tmp_path/'inputs.json.xz'
    path.write_bytes(lzma.compress(raw))
    (tmp_path/'raw_manifest.json').write_text(json.dumps({'inputs.json':dict(bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),stored='inputs.json.xz')}))
    (tmp_path/'inputs.json').write_bytes(b'[9,8,7]')
    assert verify(tmp_path)['validated_files']==1
    path.write_bytes(lzma.compress(b'[1,2,4]'))
    with pytest.raises(ValueError,match='identity mismatch'):
        verify(tmp_path)
