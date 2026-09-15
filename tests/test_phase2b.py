import torch

from experiments.phase2b.run_phase2b import (
    kv_byte_count,
    legacy_shapes,
    map_legacy_cache,
    token_sequences_equal,
)


def test_kv_byte_count_is_correct():
    legacy = ((torch.zeros(2, 3, dtype=torch.float16), torch.zeros(4, dtype=torch.float32)),)
    assert kv_byte_count(legacy) == 2 * 3 * 2 + 4 * 4


def test_legacy_structure_reconstruction_preserves_shapes():
    legacy = (
        (torch.zeros(1, 2, 3), torch.zeros(1, 2, 3)),
        (torch.zeros(1, 4, 5), torch.zeros(1, 4, 5)),
    )
    copied = map_legacy_cache(legacy, lambda tensor: tensor.clone())
    assert legacy_shapes(copied) == legacy_shapes(legacy)


def test_continuation_comparison_detects_equal_and_unequal_sequences():
    assert token_sequences_equal([1, 2, 3], [1, 2, 3])
    assert not token_sequences_equal([1, 2, 3], [1, 2, 4])
