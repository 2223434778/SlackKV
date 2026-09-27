from __future__ import annotations

import json
import time
import urllib.request
from typing import Any

from experiments.phase4d_b.capacity_client import PREFETCH_URL
from experiments.phase4f_a.logic import rejected_response_valid


def progress_prefetch(
    prompt_token_ids: list[int], *, required_blocks: int,
    lease_key: str, protected_lease_key: str, protected_progress_blocks: int,
    progress_guard: bool = True, timeout_s: float = 5.0,
) -> dict[str, Any]:
    payload = {
        "prompt_token_ids": [int(token) for token in prompt_token_ids],
        "capacity_guard": True,
        "required_blocks": required_blocks,
        "lease_key": lease_key,
    }
    if progress_guard:
        payload.update({
            "progress_guard": True,
            "protected_lease_key": protected_lease_key,
            "protected_progress_blocks": protected_progress_blocks,
        })
    issue_ns = time.monotonic_ns()
    request = urllib.request.Request(
        PREFETCH_URL,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        status_code = response.status
        result = json.loads(response.read().decode("utf-8"))
    returned_ns = time.monotonic_ns()
    status = result.get("status")
    if status not in ({"ok", "capacity_rejected", "progress_rejected"}
                      if progress_guard else {"ok", "capacity_rejected"}):
        raise ValueError(f"unexpected prefetch status: {status}")
    if status != "ok" and not rejected_response_valid(result, str(status)):
        raise ValueError(f"invalid rejection lifecycle: {status}")
    return {**result, "http_status": status_code,
            "client_issue_ns": issue_ns, "client_return_ns": returned_ns,
            "client_response_ms": (returned_ns - issue_ns) / 1e6}
