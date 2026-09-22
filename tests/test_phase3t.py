from experiments.phase3t.helpers import *


def test_frozen_constants():
    assert SESSION_LEASE_BLOCKS == 256 and LEASE_BUDGET_BLOCKS == 512
    assert HANDOFF_GUARD_MS == 100 and GATE_LIMIT == 2


def test_two_grants_then_skip():
    ledger=ReservationLedger(); assert ledger.decide()=="RESERVATION_PROMOTE"
    a=ledger.grant("A",210,300); assert a.end_ms==400 and ledger.reserved_blocks==256
    ledger.grant("B",250,320); assert ledger.reserved_blocks==512
    assert ledger.decide()=="RESERVATION_SKIP"


def test_expiry_releases_budget():
    ledger=ReservationLedger(); ledger.grant("A",210,300); ledger.grant("B",250,320)
    expired=ledger.expire(400); assert [x[0].session for x in expired]==["A"]
    assert ledger.reserved_blocks==256 and ledger.decide()=="RESERVATION_PROMOTE"


def test_outcome_classification():
    assert classify_promoted_outcome("RESERVATION_SKIP","NA",300,True,False)=="SKIPPED_ONDEMAND"
    assert classify_promoted_outcome("RESERVATION_PROMOTE",301,300,False,True)=="PROMOTED_INTERNAL_READY_MISS"
    assert classify_promoted_outcome("RESERVATION_PROMOTE",299,300,True,True)=="PROMOTED_REMOTE_WAIT"
    assert classify_promoted_outcome("RESERVATION_PROMOTE",299,300,False,True)=="PROMOTED_LOCAL"
    assert classify_promoted_outcome("RESERVATION_PROMOTE",299,300,False,False)=="LEASE_ERROR"


def test_metrics():
    assert promotion_useful_rate(6,6)==1.0
    assert wasted_promotion_mib(0)==0 and wasted_promotion_mib(2)==224
    assert source_conservation(24576,24576)
