# SPDX-License-Identifier: Apache-2.0
# Research-only SlackKV native prefetch endpoint.

import time
import uuid
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from vllm.engine.protocol import EngineClient
from vllm.sampling_params import SamplingParams

router = APIRouter()


class SlackKVPrefetchRequest(BaseModel):
    prompt_token_ids: list[int] = Field(min_length=1)
    lease_key: str | None = None
    capacity_guard: bool = False
    required_blocks: int | None = Field(default=None, ge=1, strict=True)
    progress_guard: bool = False
    protected_lease_key: str | None = None
    protected_progress_blocks: int | None = Field(default=None, ge=1, strict=True)
    rca_snapshot: dict[str, Any] | None = None
    snapshot_only: bool = False

def validate_capacity_guard_request(
    prompt_token_ids: list[int],
    capacity_guard: bool,
    required_blocks: int | None,
) -> None:
    if capacity_guard and (
        required_blocks is None
        or len(prompt_token_ids) % 16 != 0
        or required_blocks != len(prompt_token_ids) // 16
    ):
        raise HTTPException(status_code=422, detail="INVALID_CAPACITY_GUARD_REQUEST")

def validate_prefetch_status(
    status: object, capacity_guard: bool, progress_guard: bool = False,
    snapshot_only: bool = False,
) -> None:
    if status != "ok" and not (
        (capacity_guard and status == "capacity_rejected")
        or (progress_guard and status == "progress_rejected")
        or (snapshot_only and status == "snapshot")
    ):
        raise RuntimeError(f"SlackKV prefetch failed: {status!r}")


@router.post("/slackkv/prefetch")
async def slackkv_prefetch(
    request: SlackKVPrefetchRequest, raw_request: Request
) -> dict[str, object]:
    engine: EngineClient = raw_request.app.state.engine_client
    validate_capacity_guard_request(
        request.prompt_token_ids,
        request.capacity_guard,
        request.required_blocks,
    )
    if request.snapshot_only and request.rca_snapshot is None:
        raise HTTPException(status_code=422, detail="INVALID_RCA_SNAPSHOT_REQUEST")
    if request.progress_guard and (
        not request.capacity_guard
        or not request.lease_key
        or not request.protected_lease_key
        or request.protected_progress_blocks is None
    ):
        raise HTTPException(status_code=422, detail="INVALID_PROGRESS_GUARD_REQUEST")
    request_id = f"slackkv-prefetch-{uuid.uuid4().hex}"
    t0 = time.monotonic()
    params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        extra_args={
            "kv_transfer_params": {
                "slackkv_prefetch_only": True,
                "max_offload_tokens": 0,
                "slackkv_rca_snapshot_only": request.snapshot_only,
                "slackkv_phase3m": {"T0": t0},
                "slackkv_allocator_trace": request.rca_snapshot is not None,
                "slackkv_allocator_trace_once": request.rca_snapshot is not None,
                **({"slackkv_rca": request.rca_snapshot} if request.rca_snapshot else {}),
                **(
                    {"capacity_guard": True, "required_blocks": request.required_blocks}
                    if request.capacity_guard
                    else {}
                ),
                **(
                    {
                        "progress_guard": True,
                        "protected_lease_key": request.protected_lease_key,
                        "protected_progress_blocks": request.protected_progress_blocks,
                    }
                    if request.progress_guard
                    else {}
                ),
            }
        },
    )
    if request.lease_key is not None:
        params.extra_args["kv_transfer_params"][
            "slackkv_residency_lease_key"
        ] = request.lease_key
    t1 = time.monotonic()
    params.extra_args["kv_transfer_params"]["slackkv_phase3m"]["T1"] = t1
    final_output = None
    async for output in engine.generate(
        {"prompt_token_ids": request.prompt_token_ids}, params, request_id
    ):
        final_output = output
    t7 = time.monotonic()
    elapsed_ms = (t7 - t0) * 1000.0
    if final_output is None:
        raise RuntimeError("SlackKV prefetch finished without structured output")
    metadata = (final_output.kv_transfer_params or {}).get(
        "slackkv_prefetch", {}
    )
    validate_prefetch_status(
        metadata.get("status"), request.capacity_guard, request.progress_guard,
        request.snapshot_only,
    )
    return {
        **metadata,
        "phase3m_timestamps": {**(metadata.get("phase3m_timestamps") or {}), "T0": t0, "T1": t1, "T7": t7},
        "elapsed_ms": elapsed_ms,
        "generated_tokens": 0,
        "model_forward_count": 0,
    }


def attach_router(app: FastAPI):
    app.include_router(router)
