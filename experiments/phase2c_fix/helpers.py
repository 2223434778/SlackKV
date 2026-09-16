"""Small pure helpers used by the Phase 2C-Fix serving harness."""
from __future__ import annotations

import re
from typing import Any, Callable, Mapping, Sequence

_SAMPLE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)$"
)


def exact_token_prefix(encode: Callable[[str], Sequence[int]], text: str, target_tokens: int) -> list[int]:
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    tokens = list(encode(text))
    if len(tokens) < target_tokens:
        raise ValueError("text did not encode to enough tokens")
    return tokens[:target_tokens]


def make_completion_payload(model: str, prompt: Sequence[int], *, stream: bool = False, keep: bool = False) -> dict[str, Any]:
    payload: dict[str, Any] = {"model": model, "prompt": list(prompt), "max_tokens": 1, "temperature": 0, "stream": stream}
    if keep:
        payload["kv_transfer_params"] = {"max_offload_tokens": 0}
    return payload


def parse_prometheus_samples(payload: str) -> dict[tuple[str, tuple[tuple[str, str], ...]], float]:
    samples: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
    for line in payload.splitlines():
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE.match(line)
        if not match:
            continue
        labels: list[tuple[str, str]] = []
        raw_labels = match.group("labels") or ""
        for item in re.finditer(r'(\w+)="((?:\\.|[^"\\])*)"', raw_labels):
            labels.append((item.group(1), item.group(2).replace('\\"', '"')))
        samples[(match.group("name"), tuple(sorted(labels)))] = float(match.group("value"))
    return samples


def metric_value(samples: Mapping[tuple[str, tuple[tuple[str, str], ...]], float], name: str, labels: Mapping[str, str] | None = None) -> float:
    wanted = dict(labels or {})
    total = 0.0
    for (sample_name, sample_labels), value in samples.items():
        if sample_name != name:
            continue
        actual = dict(sample_labels)
        if all(actual.get(key) == val for key, val in wanted.items()):
            total += float(value)
    return total


def counter_delta(before: Mapping[tuple[str, tuple[tuple[str, str], ...]], float], after: Mapping[tuple[str, tuple[tuple[str, str], ...]], float], name: str, labels: Mapping[str, str] | None = None) -> float:
    return metric_value(after, name, labels) - metric_value(before, name, labels)


def ttft_ms(request_sent_s: float, first_token_received_s: float) -> float:
    if first_token_received_s < request_sent_s:
        raise ValueError("first token timestamp precedes request timestamp")
    return (first_token_received_s - request_sent_s) * 1000.0
