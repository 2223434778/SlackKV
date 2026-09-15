"""Pure helpers for the Phase 2C serving harness and capability probe."""
from __future__ import annotations

import re
from typing import Callable, Dict, Sequence

PROMETHEUS_SAMPLE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([-+]?\d+(?:\.\d+)?)$"
)


def exact_prefix_tokens(
    encode: Callable[[str], Sequence[int]], paragraph: str, target_tokens: int
) -> list[int]:
    """Repeat ordinary text and truncate the deterministic token sequence exactly."""
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    tokens = list(encode(paragraph * (target_tokens + 1)))
    if len(tokens) < target_tokens:
        raise ValueError("paragraph encoder did not provide enough tokens")
    return tokens[:target_tokens]


def pressure_prompt_is_distinct(
    pressure_prompt: str, session_a_prefix: str, session_b_prefix: str
) -> bool:
    return not (
        pressure_prompt.startswith(session_a_prefix)
        or pressure_prompt.startswith(session_b_prefix)
        or session_a_prefix.startswith(pressure_prompt)
        or session_b_prefix.startswith(pressure_prompt)
    )


def parse_prometheus_counters(payload: str) -> Dict[str, float]:
    """Sum labeled and unlabeled samples by metric name."""
    counters: Dict[str, float] = {}
    for line in payload.splitlines():
        if not line or line.startswith("#"):
            continue
        match = PROMETHEUS_SAMPLE.match(line)
        if match:
            name, value = match.groups()
            counters[name] = counters.get(name, 0.0) + float(value)
    return counters


def counter_delta(before: Dict[str, float], after: Dict[str, float], name: str) -> float:
    return after.get(name, 0.0) - before.get(name, 0.0)


def ttft_ms(request_sent_s: float, first_token_received_s: float) -> float:
    if first_token_received_s < request_sent_s:
        raise ValueError("first token timestamp precedes request timestamp")
    return (first_token_received_s - request_sent_s) * 1000
