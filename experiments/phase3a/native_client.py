"""Client for the research-only vLLM native prefetch endpoint."""
from __future__ import annotations

import time
from typing import Any

import requests


def parse_native_prefetch_response(payload: dict[str, Any]) -> dict[str, Any]:
    required = {
        "status",
        "external_tokens_loaded",
        "transfer_bytes",
        "cpu_to_gpu_transfer_ms",
        "model_forward_count",
        "generated_token_count",
        "elapsed_ms",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"native prefetch response missing {sorted(missing)}")
    if payload["status"] != "ok":
        raise ValueError(f"native prefetch failed: {payload!r}")
    return payload


def native_prefetch(
    prompt_token_ids: list[int],
    *,
    base_url: str = "http://127.0.0.1:8000",
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    sent = time.monotonic()
    response = requests.post(
        f"{base_url}/slackkv/prefetch",
        json={"prompt_token_ids": prompt_token_ids},
        timeout=timeout_s,
    )
    completed = time.monotonic()
    response.raise_for_status()
    parsed = parse_native_prefetch_response(response.json())
    return {
        **parsed,
        "sent_monotonic_s": sent,
        "completed_monotonic_s": completed,
        "client_e2e_ms": (completed - sent) * 1000.0,
    }
