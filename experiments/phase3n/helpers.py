"""Pure timing and classification helpers for Phase 3N."""
from __future__ import annotations

from typing import Literal

Boundary = Literal["PRE_READY", "INTERNAL_READY_PRE_CLIENT_RETURN", "POST_CLIENT_RETURN"]


def classify_handoff(arrival_s: float, ready_s: float, client_return_s: float) -> Boundary:
    if arrival_s < ready_s:
        return "PRE_READY"
    if arrival_s < client_return_s:
        return "INTERNAL_READY_PRE_CLIENT_RETURN"
    return "POST_CLIENT_RETURN"


def active_x_valid(first_token_s: float, prefetch_send_s: float, endpoint_return_s: float, finish_s: float) -> bool:
    return first_token_s < prefetch_send_s < endpoint_return_s < finish_s


def local_resume_valid(local_hit_tokens: float, external_hit_tokens: float, saw_remote_wait: bool) -> bool:
    return local_hit_tokens == 4096.0 and external_hit_tokens == 0.0 and not saw_remote_wait


def internal_pre_return(arrival_s: float, ready_s: float, engine_done_s: float, client_return_s: float) -> tuple[bool, bool]:
    return ready_s <= arrival_s < engine_done_s, engine_done_s <= arrival_s < client_return_s
