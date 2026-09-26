from __future__ import annotations

import json
import time
import urllib.request
from typing import Any

from experiments.phase4d_b.capacity import parse_prefetch_response

PREFETCH_URL = "http://127.0.0.1:8000/slackkv/prefetch"


def build_payload(
    prompt_token_ids: list[int],
    *,
    capacity_guard: bool = False,
    required_blocks: int | None = None,
    lease_key: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"prompt_token_ids": [int(x) for x in prompt_token_ids]}
    if lease_key is not None:
        payload["lease_key"] = lease_key
    if capacity_guard:
        if required_blocks is None:
            raise ValueError("guarded requests require required_blocks")
        payload["capacity_guard"] = True
        payload["required_blocks"] = int(required_blocks)
    return payload


def capacity_prefetch(
    prompt_token_ids: list[int],
    *,
    required_blocks: int,
    lease_key: str | None = None,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    payload = build_payload(
        prompt_token_ids,
        capacity_guard=True,
        required_blocks=required_blocks,
        lease_key=lease_key,
    )
    issue_ns = time.monotonic_ns()
    request = urllib.request.Request(
        PREFETCH_URL,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        status_code = response.status
        data = json.loads(response.read().decode("utf-8"))
    parse_prefetch_response(data)
    returned_ns = time.monotonic_ns()
    return {
        **data,
        "http_status": status_code,
        "client_issue_ns": issue_ns,
        "client_return_ns": returned_ns,
        "client_response_ms": (returned_ns - issue_ns) / 1e6,
    }
