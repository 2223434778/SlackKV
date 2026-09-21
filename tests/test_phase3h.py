from experiments.phase3g_full.metrics import (
    decode_throughput_tok_s,
    gate_wait_ms,
    gpu_ready_residency_ms,
    interval_concurrency,
    overlap,
)
import csv
from pathlib import Path


def test_gate_wait_and_residency_end_at_admission() -> None:
    assert gate_wait_ms(340.0, 360.0) == 20.0
    assert gpu_ready_residency_ms(270.0, 320.0) == 50.0


def test_k2_capacity_metric_is_two_not_three() -> None:
    maximum, _ = interval_concurrency([(300.0, 350.0), (320.0, 360.0), (351.0, 390.0)])
    assert maximum == 2


def test_pre_first_and_decode_overlap_metrics() -> None:
    assert overlap((300.0, 350.0), (320.0, 360.0)) == 30.0
    assert overlap((350.0, 500.0), (360.0, 510.0)) == 140.0


def test_decode_throughput_formula() -> None:
    assert decode_throughput_tok_s(93, [100.0, 120.0, 140.0], [200.0, 220.0, 240.0]) == 93.0 / 0.14


def test_final_k2_artifacts_exercise_two_slot_gate() -> None:
    root = Path("results/phase3h")
    rows = list(csv.DictReader((root / "run_metrics.csv").open()))
    k2 = [row for row in rows if row["policy"] == "FirstTokenGate-K2"]
    assert len(k2) == 3
    assert all(float(row["max_concurrent_pre_first_token"]) == 2 for row in k2)
    assert all(row["gate_invariant_validation"] == "PASS" for row in k2)
    assert all(row["token_count_validation"] == "PASS" for row in k2)
    native = list(csv.DictReader((root / "native_prefetch_metrics.csv").open()))
    assert all(row["external_tokens_loaded"] == "4096" for row in native)
