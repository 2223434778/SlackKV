import pytest

from experiments.phase2d.helpers import (
    effective_resume_latency_ms,
    make_payload,
    prefetch_timing,
    token_ids_match,
)


def test_prefetch_stall_formula():
    assert prefetch_timing(20) == (20, 0)


def test_early_residency_formula():
    assert prefetch_timing(-40) == (0, 40)


def test_effective_resume_latency():
    assert effective_resume_latency_ms(20, 95) == 115
    with pytest.raises(ValueError):
        effective_resume_latency_ms(-1, 0)


def test_prefetch_payload_keeps_request_out_of_external_tier():
    payload = make_payload("model", [1, 2, 3], stream=False)
    assert payload["kv_transfer_params"] == {"max_offload_tokens": 0}
    assert payload["max_tokens"] == 1
    assert payload["temperature"] == 0


def test_greedy_resume_token_id_equality():
    assert token_ids_match([123], [123])
    assert not token_ids_match([123], [124])
    assert not token_ids_match(None, [123])
