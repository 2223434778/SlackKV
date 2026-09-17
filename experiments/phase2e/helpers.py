"""Prompt and request helpers for the Phase 2E vLLM harness."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from experiments.phase2c_fix.helpers import exact_token_prefix, make_completion_payload
from experiments.phase2d.helpers import token_ids_match

PREFIX_TOKENS = 4096
PRESSURE_TOKENS = 9500


def build_prompts(encode: Callable[[str], Sequence[int]]) -> dict[str, list[int]]:
    a = exact_token_prefix(encode, "Session A ordinary coding context marker A. " * 900, PREFIX_TOKENS)
    b = exact_token_prefix(encode, "Session B ordinary coding context marker B. " * 900, PREFIX_TOKENS)
    pressure_1 = exact_token_prefix(
        encode, "PRESSURE UNIQUE CONTEXT unrelated file-search output item. " * 3000, PRESSURE_TOKENS
    )
    pressure_2 = exact_token_prefix(
        encode, "SECOND PRESSURE UNIQUE CONTEXT unrelated compiler output item. " * 3000, PRESSURE_TOKENS
    )
    if a == b or pressure_1 == pressure_2 or a == pressure_1[:PREFIX_TOKENS] or b == pressure_2[:PREFIX_TOKENS]:
        raise AssertionError("session prefixes and pressure prompts must be distinct")
    return {
        "a": a,
        "b": b,
        "pressure_1": pressure_1,
        "pressure_2": pressure_2,
    }


def request_payload(model: str, prompt: Sequence[int], *, stream: bool, keep_local: bool = False) -> dict[str, Any]:
    return make_completion_payload(model, prompt, stream=stream, keep=keep_local)


def resume_token_matches(token_id: int | None, control_token_id: int | None) -> bool:
    observed = [token_id] if token_id is not None else None
    control = [control_token_id] if control_token_id is not None else None
    return token_ids_match(observed, control)
