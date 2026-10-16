"""Optional real native metadata; ordinary CPU CI has no vLLM/torch dependency."""
import pytest


def test_native_adapter_contract_without_cuda():
    pytest.importorskip('torch',reason='native integration requires the pinned torch build')
    pytest.importorskip('vllm',reason='native integration requires pinned Linux vLLM; not replaced by mocks')
    from slackkv.infra.audit import run
    assert run()['status']=='PASS'
