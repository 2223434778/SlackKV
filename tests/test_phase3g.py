import csv
import json
from pathlib import Path
import pytest

from experiments.phase3g.metrics import (
    aligned_event_ms,
    decode_overlap_ms,
    gate_wait_ms,
    gpu_ready_residency_ms,
    interval_max_concurrency,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "phase3g"


def test_common_monotonic_clock():
    assert aligned_event_ms(10.320, 10.000) == pytest.approx(320.0)


def test_corrected_gpu_residency():
    assert gpu_ready_residency_ms(320.0, 270.0) == 50.0


def test_gate_wait_ends_at_admission():
    assert gate_wait_ms(340.0, 360.0) == 20.0


def test_post_first_decode_overlap():
    assert decode_overlap_ms(100.0, 200.0, 130.0, 230.0) == 70.0


def test_real_gate_has_one_pre_first_owner_and_releases_in_order():
    log = OUT / "server_logs" / "inenginefirsttokengate_resumeaware_0.log"
    assert log.exists()
    events = []
    for line in log.read_text(errors="replace").splitlines():
        if "SLACKKV_RESUME_TRACE " in line:
            events.append(json.loads(line.split("SLACKKV_RESUME_TRACE ", 1)[1]))
    assert max(int(e["num_slackkv_pre_first_token"]) for e in events) == 1
    admitted = {e["session"]: e["step_id"] for e in events if e["event"] == "RESUME_ADMITTED" and e.get("session") in "ABC"}
    first = {e["session"]: e["step_id"] for e in events if e["event"] == "FIRST_TOKEN" and e.get("session") in "ABC"}
    finished = {e["session"]: e["step_id"] for e in events if e["event"] == "REQUEST_FINISHED" and e.get("session") in "ABC"}
    assert admitted["A"] < admitted["B"] < admitted["C"]
    assert first["A"] < admitted["B"] and first["B"] < admitted["C"]
    assert int(admitted["B"]) < int(finished["A"])


def test_integration_detects_full_sequence_blocker():
    artifact = json.loads((OUT / "integration.json").read_text())
    sessions = artifact["sessions"]
    assert all(int(row["generated_token_count"]) == 32 for row in sessions)
    # This is a deliberate diagnostic assertion: the integration gate must
    # fail rather than silently accepting a corrupted continuation.
    assert any(row["generation_exact_match"] == "FAIL" for row in sessions)


def test_interval_concurrency():
    assert interval_max_concurrency([(0.0, 10.0), (5.0, 15.0)]) == 2
