"""Pure metric and request helpers for Phase 2D."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from experiments.phase2c_fix.helpers import counter_delta, parse_prometheus_samples


def make_payload(model: str, prompt: Sequence[int], *, stream: bool, return_token_ids: bool = False) -> dict[str, Any]:
    return {
        "model": model,
        "prompt": list(prompt),
        "max_tokens": 1,
        "temperature": 0,
        "stream": stream,
        "return_token_ids": return_token_ids,
        "kv_transfer_params": {"max_offload_tokens": 0},
    }


def prefetch_timing(completion_relative_to_deadline_ms: float) -> tuple[float, float]:
    """Return (stall_ms, early_residency_ms) from completion minus deadline."""
    return max(0.0, completion_relative_to_deadline_ms), max(0.0, -completion_relative_to_deadline_ms)


def effective_resume_latency_ms(prefetch_stall_ms: float, resume_ttft_ms: float) -> float:
    if prefetch_stall_ms < 0 or resume_ttft_ms < 0:
        raise ValueError("latencies must be non-negative")
    return prefetch_stall_ms + resume_ttft_ms


def token_ids_match(observed: Sequence[int] | None, control: Sequence[int] | None) -> bool:
    return observed is not None and control is not None and list(observed) == list(control)


def source_tokens_delta(before: Mapping[Any, float], after: Mapping[Any, float], source: str) -> float:
    return counter_delta(
        before,
        after,
        "vllm:prompt_tokens_by_source_total",
        {"source": source},
    )


def external_cache_hit_delta(before: Mapping[Any, float], after: Mapping[Any, float]) -> float:
    return counter_delta(before, after, "vllm:external_prefix_cache_hits_total")


def gpu_to_cpu_offload_bytes(before: Mapping[Any, float], after: Mapping[Any, float]) -> float:
    return counter_delta(before, after, "vllm:kv_offload_total_bytes_total", {"transfer_type": "GPU_to_CPU"})


def gpu_to_cpu_offload_ms(before: Mapping[Any, float], after: Mapping[Any, float]) -> float:
    # vLLM's total-time counter is reported in seconds.
    return counter_delta(before, after, "vllm:kv_offload_total_time_total", {"transfer_type": "GPU_to_CPU"}) * 1000.0


__all__ = [
    "external_cache_hit_delta",
    "effective_resume_latency_ms",
    "gpu_to_cpu_offload_bytes",
    "gpu_to_cpu_offload_ms",
    "make_payload",
    "parse_prometheus_samples",
    "prefetch_timing",
    "source_tokens_delta",
    "token_ids_match",
]
