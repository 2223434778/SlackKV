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
        "connector_only_step_count",
        "connector_only_wall_ms",
        "mixed_prefetch_model_step_count",
        "normal_requests_delayed",
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
    lease_key: str | None = None,
    base_url: str = "http://127.0.0.1:8000",
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    sent = time.monotonic()
    request_payload: dict[str, Any] = {"prompt_token_ids": prompt_token_ids}
    if lease_key is not None:
        request_payload["lease_key"] = lease_key
    response = requests.post(
        f"{base_url}/slackkv/prefetch",
        json=request_payload,
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
