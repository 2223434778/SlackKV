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
