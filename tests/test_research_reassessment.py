import math
import json
from pathlib import Path

import pytest

from scripts.research_reassessment import deletion_improvement, nearest_rank, number, remote_intervals


def trace(extra):
    return [{"event": name, "timestamp_monotonic": time} for name, time in
            [("RESUME_ARRIVED", 0), *extra, ("RESUME_ADMITTED", 8),
             ("FIRST_TOKEN", 10), ("REQUEST_FINISHED", 20)]]


def test_remote_wait_and_post_first_preemption():
    assert remote_intervals(trace([("WAITING_FOR_REMOTE_KVS", 2), ("REMOTE_KV_READY", 3),
                                   ("WAITING_FOR_REMOTE_KVS", 4), ("REMOTE_KV_READY", 6),
                                   ("WAITING_FOR_REMOTE_KVS", 12), ("REMOTE_KV_READY", 14)])) == [(2, 3), (4, 6)]


@pytest.mark.parametrize("extra", [[("WAITING_FOR_REMOTE_KVS", 2)], [("REMOTE_KV_READY", 2)],
                                   [("WAITING_FOR_REMOTE_KVS", 2), ("WAITING_FOR_REMOTE_KVS", 3)]])
def test_missing_or_ambiguous_remote_event_is_unknown(extra):
    assert remote_intervals(trace(extra)) is None


def test_missing_lifecycle_not_assumed_local_hit():
    assert remote_intervals([]) is None
    assert remote_intervals(trace([])) == []


@pytest.mark.parametrize("value", ["", "NA", "UNKNOWN", "nan", "inf", None])
def test_missing_and_nonfinite_are_unknown(value):
    assert number(value) is None


def test_recompute_quantile_after_deletion():
    rows = [{"deadline_latency_ms": value, "delete": 100 if value == 100 else 0}
            for value in range(1, 101)]
    assert deletion_improvement(rows, "delete") == pytest.approx(100 / 95)
    rows[0]["delete"] = None
    assert deletion_improvement(rows, "delete") is None


def test_nearest_rank_not_sum_of_component_quantiles():
    assert nearest_rank(list(range(32))) == 30
    with pytest.raises(ValueError):
        nearest_rank([math.nan])


def test_committed_matrix_reanalysis(tmp_path):
    from scripts.research_reassessment import analyze
    analyze(Path(__file__).resolve().parents[1], tmp_path)
    audit = json.loads((tmp_path / "audit.json").read_text())
    assert audit["boundary_count"] == 512
    assert audit["phase4g_d_eligible_allocator"] == {"True": 11, "False": 35}
    assert audit["phase4g_d_frontier"]["exact_F0_blocked"] == 7
    assert audit["phase4g_c_corrected_allocator"]["exact_missing"] == 69
    assert audit["phase4g_c_corrected_allocator"]["known_mismatch"] == 0
    assert audit["phase4g_paired_P95_change_percent"] == pytest.approx(1.38, abs=0.01)
    result = audit["policies"]["OnDemand-K2"]
    assert result["remote_unknown"]["max"] == 0
    assert result["fixed_path_remote_deletion_P95_improvement_percent"]["median"] == pytest.approx(0.90470948)
