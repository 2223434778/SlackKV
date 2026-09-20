from experiments.phase3g_full.metrics import (
    decode_throughput_tok_s,
    gate_wait_ms,
    gpu_ready_residency_ms,
    interval_concurrency,
    overlap,
    same_step_pairs,
)
from pathlib import Path
import csv


def test_gate_wait_ends_at_admission() -> None:
    assert gate_wait_ms(340.0, 360.0) == 20.0


def test_gpu_residency_ends_at_scheduler_admission() -> None:
    assert gpu_ready_residency_ms(270.0, 320.0) == 50.0


def test_pre_and_post_overlap_metrics() -> None:
    assert overlap((100.0, 150.0), (130.0, 180.0)) == 20.0
    maximum, area = interval_concurrency([(100.0, 200.0), (130.0, 230.0)])
    assert maximum == 2
    assert area == 200.0


def test_tpot_and_cluster_throughput() -> None:
    assert (310.0 / 31.0) == 10.0
    assert decode_throughput_tok_s(93, [100.0, 130.0, 160.0], [200.0, 230.0, 260.0]) == 93.0 / 0.16


def test_same_step_pair_count() -> None:
    assert same_step_pairs([1, 1, 2]) == 1


def test_final_artifacts_use_canonical_metrics_and_runtime_native_records() -> None:
    root = Path("results/phase3g_full")
    with (root / "run_metrics.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        assert reader.fieldnames.count("max_concurrent_pre_first_token") == 1
        assert reader.fieldnames.count("max_concurrent_post_first_token_decode") == 1
        assert "max_pre_first_concurrency" not in reader.fieldnames
        assert "max_post_first_decode_concurrency" not in reader.fieldnames
    with (root / "native_prefetch_metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert all(row["external_tokens_loaded"] == "4096" for row in rows)
    assert all(row["model_forward_count"] == "0" for row in rows)
    assert all(row["generated_token_count"] == "0" for row in rows)
