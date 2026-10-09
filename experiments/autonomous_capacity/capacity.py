"""Exact pool expectation for the pinned FP16 Qwen2.5-1.5B layout."""
BYTES_PER_TOKEN = 28672
BLOCK_SIZE = 16


def validate_pool(kv_bytes: int, block_size: int, actual_blocks: int) -> dict[str, int]:
    expected = kv_bytes // (BYTES_PER_TOKEN * BLOCK_SIZE)
    if block_size != BLOCK_SIZE or actual_blocks != expected:
        raise ValueError(f"KV layout/pool mismatch: expected {expected}x16, "
                         f"actual {actual_blocks}x{block_size}")
    return {"total_gpu_kv_blocks": actual_blocks,
            "reservation_budget_blocks": actual_blocks * 3 // 4}
