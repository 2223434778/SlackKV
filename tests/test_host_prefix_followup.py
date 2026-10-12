import pytest
from experiments.host_prefix.followup_screen import BLOCK, TOKEN_BYTES, cap, shares


@pytest.mark.parametrize('sessions', [1, 2, 4, 7])
def test_physical_shares_are_aligned_and_conserve_capacity(sessions):
    quotas = shares(256*2**20, sessions)
    assert len(quotas) == sessions
    assert all(q > 0 and q % BLOCK == 0 for q in quotas)
    assert sum(quotas) == 256*2**20//(BLOCK*TOKEN_BYTES)*BLOCK
    assert max(quotas)-min(quotas) <= (sessions-1)*BLOCK


def test_nohost_only_disables_native_store():
    assert cap('nohost', 28672, 9360) == 0
    assert cap('fair', 28672, 9360) == 9360
    assert cap('fair', 1024, 9360) == 1024
    assert cap('full', 28672, 9360) == 28672
    with pytest.raises(ValueError):
        cap('fake', 28672, 9360)


def test_insufficient_capacity_is_rejected():
    with pytest.raises(ValueError):
        shares(BLOCK*TOKEN_BYTES, 2)


def test_absent_transfer_is_zero_only_in_known_empty_connector():
    from experiments.host_prefix.followup_analyze import transfer
    header = '# HELP vllm:kv_offload_total_bytes_total Native transfer bytes\n'
    assert transfer('', '', 'CPU_to_GPU', 0) is None
    assert transfer(header, header, 'CPU_to_GPU', 0) == 0
    assert transfer(header, header, 'CPU_to_GPU', 16) is None
    line = 'vllm:kv_offload_total_bytes_total{transfer_type="CPU_to_GPU"} '
    with pytest.raises(ValueError):
        transfer(line+'20', line+'10', 'CPU_to_GPU', 0)


def test_unfinished_execution_cannot_be_analyzed(tmp_path):
    from experiments.host_prefix.followup_analyze import analyze
    (tmp_path/'execution.json').write_text('{"status":"RUNNING"}')
    with pytest.raises(ValueError, match='unresolved'):
        analyze(tmp_path)


def test_compact_payload_read_is_lossless(tmp_path):
    import lzma
    from experiments.host_prefix.followup_analyze import read
    path = tmp_path/'inputs.json'
    raw = b'[{"prompt":[1,2,3]}]\n'
    path.with_suffix('.json.xz').write_bytes(lzma.compress(raw))
    assert read(path) == raw


def test_budget_carries_history_and_refuses_live_run(tmp_path):
    import json
    from experiments.host_prefix.followup_screen import FAMILY, budget
    for family in ('autonomous_fork', 'autonomous_round4', 'host_prefix_cost_20261010', FAMILY,
                   'deployment_opportunity_20261011', 'large_model_20261012'):
        directory = tmp_path/'results'/family/'run'
        directory.mkdir(parents=True)
        field = 'gpu_hours_conservative' if family in ('autonomous_fork', 'autonomous_round4') else 'gpu_hours'
        (directory/'execution.json').write_text(json.dumps(dict(status='COMPLETE', **{field: .2})))
    assert sum(budget(tmp_path).values()) == pytest.approx(1.2)
    (tmp_path/'results/large_model_20261012/run/execution.json').write_text('{"status":"RUNNING"}')
    with pytest.raises(RuntimeError, match='unresolved'):
        budget(tmp_path)


def test_empty_directory_cannot_report_zero_gpu_success(tmp_path, monkeypatch):
    from experiments.host_prefix.followup_analyze import main
    monkeypatch.setattr('sys.argv', ['followup_analyze', '--directory', str(tmp_path)])
    with pytest.raises(ValueError, match='empty directory is UNKNOWN'):
        main()
    assert not (tmp_path/'analysis.json').exists()
