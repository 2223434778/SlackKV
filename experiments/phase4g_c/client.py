"""Phase 4G-C progress-guarded prefetch client with source-RCA metadata."""
from __future__ import annotations

import json
import time
import urllib.request
from typing import Any

from experiments.phase4d_b.capacity_client import PREFETCH_URL
from experiments.phase4f_a.logic import rejected_response_valid


def progress_prefetch_rca(
    prompt_token_ids: list[int], *, required_blocks: int,
    lease_key: str, protected_lease_key: str,
    protected_progress_blocks: int, rca_snapshot: dict[str, Any],
    timeout_s: float = 5.0,
) -> dict[str, Any]:
    payload = {
        "prompt_token_ids": [int(token) for token in prompt_token_ids],
        "capacity_guard": True,
        "required_blocks": int(required_blocks),
        "lease_key": lease_key,
        "progress_guard": True,
        "protected_lease_key": protected_lease_key,
        "protected_progress_blocks": int(protected_progress_blocks),
        "rca_snapshot": rca_snapshot,
    }
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
    if status not in {"ok", "capacity_rejected", "progress_rejected"}:
        raise ValueError(f"unexpected prefetch status: {status}")
    if status != "ok" and not rejected_response_valid(result, str(status)):
        raise ValueError(f"invalid rejection lifecycle: {status}")
    return {
        **result, "http_status": status_code,
        "client_issue_ns": issue_ns, "client_return_ns": returned_ns,
        "client_response_ms": (returned_ns - issue_ns) / 1e6,
    }


def read_only_selection_snapshot(
    *, rca_snapshot: dict[str, Any], timeout_s: float = 3.0,
) -> dict[str, Any]:
    """Synchronously request a scheduler-side, allocation-free exact snapshot."""
    payload = {
        "prompt_token_ids": [1],
        "snapshot_only": True,
        "rca_snapshot": rca_snapshot,
    }
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
    if status_code != 200 or result.get("status") != "snapshot":
        raise ValueError("read-only scheduler snapshot did not complete")
    snapshot = result.get("snapshot")
    if not isinstance(snapshot, dict) or snapshot.get("event") != "SELECTION_SNAPSHOT":
        snapshot_keys = sorted(snapshot) if isinstance(snapshot, dict) else None
        snapshot_event = snapshot.get("event") if isinstance(snapshot, dict) else None
        raise ValueError(
            "invalid read-only scheduler snapshot payload: "
            f"status={result.get('status')!r}, snapshot_type={type(snapshot).__name__}, "
            f"snapshot_event={snapshot_event!r}, snapshot_keys={snapshot_keys!r}"
        )
    return {
        **result,
        "client_issue_ns": issue_ns,
        "client_return_ns": returned_ns,
        "client_roundtrip_ms": (returned_ns - issue_ns) / 1e6,
    }
