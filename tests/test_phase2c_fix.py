import pytest

from experiments.phase2c_fix.helpers import counter_delta, make_completion_payload, parse_prometheus_samples, ttft_ms


def test_keep_payload_contains_zero_offload_cap():
    payload = make_completion_payload("model", [1, 2], keep=True)
    assert payload["kv_transfer_params"] == {"max_offload_tokens": 0}


def test_prometheus_labeled_counter_delta():
    before = parse_prometheus_samples('vllm:prompt_tokens_by_source_total{source="external_kv_transfer"} 1\n')
    after = parse_prometheus_samples('vllm:prompt_tokens_by_source_total{source="external_kv_transfer"} 4097\n')
    assert counter_delta(before, after, "vllm:prompt_tokens_by_source_total", {"source": "external_kv_transfer"}) == 4096


def test_streaming_ttft_formula():
    assert ttft_ms(10.0, 10.025) == pytest.approx(25.0)
