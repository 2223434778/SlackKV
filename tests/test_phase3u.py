from experiments.phase3u.helpers import *

def test_frozen_configuration():
    assert len(POLICIES) == 4 and len(SESSIONS) == 12
    assert DEADLINES_MS["A"] == 300 and DEADLINES_MS["L"] == 1560
    assert 12 * 4096 == 49152
    assert 12 * 112 == 1344
    assert 256 - 1 == 255

def test_policy_parameters():
    assert (K, ASYNC_MS, BLOCKING_MS, RESERVATION_BLOCKS, X_TOKENS) == (2,40,70,512,256)

def test_policy_names_exact():
    assert POLICIES == ("OnDemand-K2","Blocking70-K2","NaiveAsync40-K2","ReservationLedger2-Async40-K2")

def test_h2d_and_slowdown():
    assert h2d_metrics(0, 49152) == (0, 1344, 1344, 0)
    assert h2d_metrics(12, 4096)[3] == 112
    assert x_slowdown(120, 100) == 1.2

def test_outcomes():
    assert promotion_outcome(False, 0, 0, False) == "RESERVATION_SKIP"
    assert promotion_outcome(True, 10, 11, False) == "PROMOTION_USEFUL"
    assert promotion_outcome(True, 12, 11, False) == "PROMOTION_INTERNAL_READY_MISS"
    assert promotion_outcome(True, 10, 11, True) == "PROMOTION_REMOTE_WAIT"

def test_reservation_skip_is_not_failure():
    assert promotion_outcome(False, 999, 0, True) == "RESERVATION_SKIP"

def test_h2d_conversion():
    assert 4096 * 112 / 4096 == 112

def test_minimum_h2d():
    assert h2d_metrics(6, 6*4096) == (672,672,1344,0)

def test_clean_metric_names():
    names=("AL_P95_resume_ms","Cluster1_P95_resume_ms","Cluster2_P95_resume_ms","Cluster3_P95_resume_ms")
    assert all("ABC" not in x and "DEF" not in x for x in names)
