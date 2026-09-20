import csv
import json
from pathlib import Path

from experiments.phase3f_fix.scheduler import FirstTokenGate, memory_time_mib_s


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "phase3f_fix"


def test_gate_has_one_owner_and_fifo_trace():
    gate = FirstTokenGate()
    gate.admit("A")
    assert gate.owner == "A"
    assert not gate.can_admit("B")
    gate.first_token("A")
    assert gate.owner is None
    gate.admit("B")
    assert gate.owner == "B"


def test_real_trace_enforces_gate_order_and_single_owner():
    rows = list(csv.DictReader((OUT / "scheduler_trace.csv").open()))
    gated = [r for r in rows if r["policy"] == "InEngineFirstTokenGate-ResumeAware"]
    assert gated
    assert max(int(r["num_slackkv_pre_first_token"]) for r in gated) <= 1
    for rep in {r["repetition"] for r in gated}:
        admissions = []
        for session in "ABC":
            events = [r for r in gated if r["repetition"] == rep and r["session"] == session and r["event"] == "RESUME_ADMITTED"]
            assert events
            admissions.append((int(events[0]["step_id"]), session))
        assert [s for _, s in sorted(admissions)] == ["A", "B", "C"]


def test_ondemand_remote_kv_trace_is_real():
    rows = list(csv.DictReader((OUT / "scheduler_trace.csv").open()))
    od = [r for r in rows if r["policy"] == "OnDemand-Trace"]
    assert any(r["event"] == "WAITING_FOR_REMOTE_KVS" for r in od)
    assert any(r["event"] == "REMOTE_KV_READY" for r in od)


def test_native_prefetch_zero_forward_and_generation():
    rows = list(csv.DictReader((OUT / "native_prefetch_metrics.csv").open()))
    assert len(rows) == 36
    assert all(float(r["model_forward_count"]) == 0 for r in rows)
    assert all(float(r["generated_token_count"]) == 0 for r in rows)
    assert all(float(r["mixed_prefetch_model_step_count"]) == 0 for r in rows)


def test_artifact_validations_and_memory_formula():
    rows = list(csv.DictReader((OUT / "run_metrics.csv").open()))
    assert len(rows) == 9
    assert all(r["cache_validation"] == "PASS" for r in rows)
    assert all(r["generation_validation"] == "PASS" for r in rows)
    assert memory_time_mib_s([0.0, 100.0]) == 11.2


def test_config_disables_client_gate():
    config = json.loads((OUT / "config.json").read_text())
    assert config["client_event_gate"] is False
    assert config["policies"] == [
        "OnDemand-Trace",
        "ResumeAware-Trace",
        "InEngineFirstTokenGate-ResumeAware",
    ]
