import pytest
from experiments.phase2c.metrics import (
    counter_delta,
    exact_prefix_tokens,
    parse_prometheus_counters,
    pressure_prompt_is_distinct,
    ttft_ms,
)


def test_prefix_is_exactly_4096_tokens():
    tokens = exact_prefix_tokens(lambda text: list(range(len(text))), "ordinary ", 4096)
    assert len(tokens) == 4096


def test_pressure_prompt_does_not_share_session_prefixes():
    assert pressure_prompt_is_distinct("Pressure unique context.", "Session A text.", "Session B text.")


def test_prometheus_counter_delta_parsing():
    before = parse_prometheus_counters("vllm:prefix_cache_hits 2\n")
    after = parse_prometheus_counters("vllm:prefix_cache_hits 5\n")
    assert counter_delta(before, after, "vllm:prefix_cache_hits") == 3


def test_ttft_from_streaming_timestamps():
    assert ttft_ms(10.0, 10.025) == pytest.approx(25.0)
