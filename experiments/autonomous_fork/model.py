"""Exact FP16 full-attention footprint, without assumed simultaneous residency."""
from __future__ import annotations

import math


def bytes_per_token(config: dict, dtype_bytes: int = 2) -> int:
    if config.get("use_sliding_window", False):
        raise ValueError("this model covers full attention only")
    heads = config["num_attention_heads"]
    hidden = config["hidden_size"]
    if hidden % heads:
        raise ValueError("nonintegral attention head size")
    return 2 * config["num_hidden_layers"] * config["num_key_value_heads"] * (hidden // heads) * dtype_bytes


def fork_footprint(config: dict, shared: int, suffix: int, output: int, branches: int) -> dict:
    if min(shared, suffix, output, branches) <= 0:
        raise ValueError("positive dimensions required")
    b = bytes_per_token(config) * 16
    root_blocks = math.ceil(shared / 16)
    private_blocks = math.ceil((suffix + output) / 16)
    return {"bytes_per_token": b // 16, "shared_blocks": root_blocks,
            "unique_bytes": (1 + root_blocks + branches * private_blocks) * b,
            "duplicate_upper_bytes": (1 + branches * (root_blocks + private_blocks)) * b,
            "avoided_copy_upper_bytes": (branches - 1) * root_blocks * b,
            "qualification": "capacity model, not actual residency or measured speedup"}


def streaming_cost(config: dict, shared: int, attention_steps: int,
                   bandwidth_gbps: float) -> dict:
    if min(shared, attention_steps, bandwidth_gbps) <= 0:
        raise ValueError("positive dimensions and measured bandwidth required")
    traffic = bytes_per_token(config) * shared * attention_steps
    return {"root_only_bytes": traffic,
            "dma_seconds_at_reference_bandwidth": traffic / (bandwidth_gbps * 10**9),
            "qualification": "conditional optimistic model: one shared fetch per attention iteration; "
                             "no private KV, staging, compute or interference; historical bandwidth "
                             "is a reference, not a universal hardware bound"}
