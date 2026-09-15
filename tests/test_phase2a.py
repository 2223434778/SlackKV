import pytest
import torch

from experiments.phase2a.run_phase2a import (
    numel_for_bytes,
    resume_metrics,
    sampled_equal,
)


def test_byte_size_calculation_is_exact():
    assert numel_for_bytes(256 * 2**20, 2) * 2 == 256 * 2**20


def test_sampled_correctness_checker_detects_equality():
    tensor = torch.tensor([0.0, 1.0, 2.0, 3.0])
    indices = torch.tensor([1, 3])
    assert sampled_equal(tensor, indices, torch.tensor([1.0, 3.0]))
    assert not sampled_equal(tensor, indices, torch.tensor([1.0, 2.0]))


def test_resume_stall_and_early_residency_formulas():
    assert resume_metrics(2.5, 3.0) == (0.0, 0.5)
    assert resume_metrics(3.2, 3.0) == pytest.approx((0.2, 0.0))
