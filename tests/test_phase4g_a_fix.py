from __future__ import annotations

import hashlib
import time

import pytest

from experiments.phase4g_a_fix import background
from experiments.phase4g_a_fix.run_phase4g_a_fix import (
    BG_CAL_ORDER,
    OUT,
    POLICY_NAMES,
    RUN_ORDER,
    TEMP_LOGS,
    VLLM_EXPECTED_DIFF_SHA,
    _temporary_server_start,
    configure_legacy_lifecycle,
    vllm_diff_sha,
)
from experiments.phase4g_a import run_phase4g_a as legacy
from experiments.phase4g_a.workload import SOURCE_SHA, build_manifest, load_source


class FakeRuntime:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.issue_times = []

    def metric_text(self) -> str:
        return ""

    def source_samples(self, _raw: str) -> dict:
        return {}

    def send(self, prompt, max_tokens, *, stream, params, on_first=None):
        assert stream is True
        assert params == {"max_offload_tokens": 0}
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        issue = time.monotonic_ns()
        self.issue_times.append(issue)
        first = issue + 1_000_000
        done = issue + 2_000_000
        if on_first:
            on_first(first)
        time.sleep(.001)
        self.active -= 1
        return {"issue_ns": issue, "first_ns": first, "done_ns": done,
                "tokens": [int(prompt[0])] * max_tokens}


def make_chunk(chunk_id: int, first_ns: int, done_ns: int, gap_ms: float = 0.0) -> dict:
    return {
        "chunk_id": chunk_id,
        "prompt_tokens": background.BG_PROMPT_TOKENS,
        "generated_tokens": background.BG_CHUNK_OUTPUT_TOKENS,
        "issue_ns": first_ns,
        "first_ns": first_ns,
        "done_ns": done_ns,
        "decode_ms": (done_ns - first_ns) / 1e6,
        "TPOT_ms": (done_ns - first_ns) / 1e6 / (background.BG_CHUNK_OUTPUT_TOKENS - 1),
        "previous_chunk_done_ns": "NA" if chunk_id == 0 else done_ns - round(gap_ms * 1e6),
        "inter_chunk_idle_gap_ms": gap_ms,
        "metric_window_start_ns": first_ns,
        "metric_window_end_ns": done_ns,
        "source_local_delta_raw_tokens": 0,
        "prompt_local_hit_tokens": 0,
        "prompt_sha256": "same-prompt",
    }


def test_fixed_bg_protocol_constants_and_post_first_tokens() -> None:
    assert background.BG_PROMPT_TOKENS == 9500
    assert background.BG_CHUNK_OUTPUT_TOKENS == 2048
    assert background.BG_CHUNKS == 4
    assert background.BG_TOTAL_OUTPUT_TOKENS == 8192
    assert background.BG_POST_FIRST_TOKENS == 8188


def test_weighted_tpot_uses_8188_post_first_tokens() -> None:
    chunks = [make_chunk(i, i * 10_000_000, i * 10_000_000 + 2_047_000_000) for i in range(4)]
    summary = background.summarize_chunks(chunks)
    assert summary["weighted_TPOT_ms"] == pytest.approx(
        sum(row["decode_ms"] for row in chunks) / 8188
    )
    assert 4 * (2048 - 1) == background.BG_POST_FIRST_TOKENS
    assert background.BG_POST_FIRST_TOKENS != 8191


def test_same_prompt_and_sequential_non_overlapping_requests() -> None:
    fake = FakeRuntime()
    prompt = [17] * background.BG_PROMPT_TOKENS
    result = background.run_chunks(fake, prompt)
    assert len(result["chunks"]) == 4
    assert result["tokens"] == [17] * background.BG_TOTAL_OUTPUT_TOKENS
    assert len({row["prompt_sha256"] for row in result["chunks"]}) == 1
    expected = hashlib.sha256(b"".join((17).to_bytes(8, "little") for _ in prompt)).hexdigest()
    assert result["prompt_sha256"] == expected
    assert fake.max_active == 1


def test_run_chunks_rejects_wrong_prompt_length() -> None:
    with pytest.raises(ValueError, match="prompt length"):
        background.run_chunks(FakeRuntime(), [1])


def test_chunk_output_count_is_hard_gate(monkeypatch) -> None:
    fake = FakeRuntime()
    original = fake.send

    def short_send(*args, **kwargs):
        value = original(*args, **kwargs)
        value["tokens"] = value["tokens"][:-1]
        return value

    monkeypatch.setattr(fake, "send", short_send)
    with pytest.raises(RuntimeError, match="generated"):
        background.run_chunks(fake, [1] * background.BG_PROMPT_TOKENS)


def test_inter_chunk_idle_gap_formula_and_50ms_gate() -> None:
    chunks = [make_chunk(i, i * 3_000_000_000, i * 3_000_000_000 + 2_000_000_000,
                         10.0 if i else 0.0) for i in range(4)]
    assert max(row["inter_chunk_idle_gap_ms"] for row in chunks[1:]) == 10
    assert background.summarize_chunks(chunks)["max_inter_chunk_idle_gap_ms"] == 10
    chunks[2]["inter_chunk_idle_gap_ms"] = 50.01
    assert background.summarize_chunks(chunks)["max_inter_chunk_idle_gap_ms"] == 50.01
    valid, _tail = background.background_valid(chunks, chunks[-1]["done_ns"] - 1)
    assert not valid


def test_chunked_request_issue_gap_is_measured_from_previous_done() -> None:
    fake = FakeRuntime()
    result = background.run_chunks(fake, [4] * background.BG_PROMPT_TOKENS)
    for prior, current in zip(result["chunks"], result["chunks"][1:]):
        assert current["inter_chunk_idle_gap_ms"] == pytest.approx(
            (current["issue_ns"] - prior["done_ns"]) / 1e6
        )


def test_background_wall_and_first_token_tail_are_separate() -> None:
    chunks = [make_chunk(i, i * 2_000_000_000, i * 2_000_000_000 + 1_000_000_000) for i in range(4)]
    first_agent = 3_000_000_000
    finish_agent = 4_000_000_000
    first_tail, finish_tail = background.background_tails(chunks[-1]["done_ns"], first_agent, finish_agent)
    assert first_tail == 4_000
    assert finish_tail == 3_000


def test_validity_uses_last_request_finished_and_calibration_margin() -> None:
    chunks = [make_chunk(i, i * 10_000_000_000, i * 10_000_000_000 + 5_000_000_000) for i in range(4)]
    final_done = chunks[-1]["done_ns"]
    ok, tail = background.background_valid(chunks, final_done - 30_000_000_000, calibration=True)
    assert ok and tail == 30_000
    ok, tail = background.background_valid(chunks, final_done - 29_999_000_000, calibration=True)
    assert not ok and tail == 29_999
    ok, tail = background.background_valid(chunks, final_done, calibration=False)
    assert not ok and tail == 0


def test_measured_source_hits_subtract_concurrent_resumed_agent_tokens() -> None:
    chunk = make_chunk(0, 1_000_000_000, 2_000_000_000)
    chunk["metric_window_start_ns"] = 1_000_000_000
    chunk["metric_window_end_ns"] = 2_000_000_000
    chunk["source_local_delta_raw_tokens"] = 140
    event = {"timestamp_monotonic": 1.5, "local_cached_tokens": 96}
    result = background.adjust_local_hit_attribution([chunk], [event])[0]
    assert result["agent_local_tokens_during_metric_window"] == 96
    assert result["prompt_local_hit_tokens"] == 44


def test_source_counter_unavailable_is_na_not_fabricated() -> None:
    chunk = make_chunk(0, 1_000_000_000, 2_000_000_000)
    chunk["source_local_delta_raw_tokens"] = "NA"
    row = background.adjust_local_hit_attribution([chunk], [])[0]
    assert row["prompt_local_hit_tokens"] == "NA"


def test_frozen_multi_turn_workload_and_order_are_unchanged() -> None:
    manifest = build_manifest(load_source())
    assert len(load_source()) == 40
    assert len(manifest) == 32
    assert SOURCE_SHA == "9b381336a70e73b0efbddbb79d2c88608c19663247e819c1e7a364efd2971105"
    assert BG_CAL_ORDER == (("A", 1), ("D", 2), ("D", 3), ("A", 4))
    assert RUN_ORDER == (("A", "D"), ("D", "A")) * 4
    assert POLICY_NAMES == {"A": "OnDemand-K2", "D": "FullSlackKV+PSA1-K2"}


def test_phase4g_a_fix_has_separate_output_and_frozen_vllm_fingerprint() -> None:
    assert OUT.name == "phase4g_a_fix"
    assert OUT != OUT.parent / "phase4g_a"
    assert vllm_diff_sha() == VLLM_EXPECTED_DIFF_SHA


def test_legacy_lifecycle_output_gate_uses_four_chunk_total() -> None:
    configure_legacy_lifecycle()
    assert legacy.X_OUTPUT_TOKENS == background.BG_TOTAL_OUTPUT_TOKENS


def test_temporary_server_proxy_calls_original_starter_once(monkeypatch) -> None:
    calls = []

    def original(label, *, connector, gpu_bytes, out):
        calls.append((label, connector, gpu_bytes, out))
        return "proc", "log"

    from experiments.phase4g_a_fix import run_phase4g_a_fix as runner

    monkeypatch.setattr(runner, "ORIGINAL_START_SERVER", original)
    assert _temporary_server_start("proxy-test", out=OUT) == ("proc", "log")
    assert calls == [("proxy-test", True, 2147483648, TEMP_LOGS)]
