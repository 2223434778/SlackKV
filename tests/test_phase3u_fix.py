import threading
from experiments.phase3u_fix.helpers import POLICIES,h2d,source_conserved,x_slowdown,category_percentiles
from experiments.phase3u_fix.run_phase3u_fix import SESSIONS,TOKENS,maybe_wait_for_prefetch,policy_semantics,configured_service_budget
from experiments.phase3t.helpers import GATE_LIMIT,LEASE_BUDGET_BLOCKS,SESSION_LEASE_BLOCKS

class FakeEvent:
    def __init__(self): self.calls=[]
    def wait(self,timeout=None): self.calls.append(timeout); return True

def test_policy_mapping():
    assert policy_semantics(POLICIES[0])=={"service_ms":None,"synchronous":False,"block_on_prefetch":False,"lease":False,"promote":False}
    assert policy_semantics(POLICIES[1])=={"service_ms":70.0,"synchronous":True,"block_on_prefetch":True,"lease":False,"promote":True}
    assert policy_semantics(POLICIES[2])=={"service_ms":40.0,"synchronous":False,"block_on_prefetch":False,"lease":False,"promote":True}
    assert policy_semantics(POLICIES[3])=={"service_ms":40.0,"synchronous":False,"block_on_prefetch":False,"lease":True,"promote":True}

def test_wait_helper():
    event=FakeEvent(); assert maybe_wait_for_prefetch(event,True); assert len(event.calls)==1
    for policy in POLICIES:
        if policy=="Blocking70-K2":continue
        event=FakeEvent(); assert not maybe_wait_for_prefetch(event,False); assert event.calls==[]
    assert not maybe_wait_for_prefetch(None,False)

def test_service_budgets():
    assert [configured_service_budget(p) for p in POLICIES]==[None,70.0,40.0,40.0]

def test_empty_category_is_na():
    assert category_percentiles([],50)=="NA" and category_percentiles([],95)=="NA"

def test_x_and_h2d():
    assert x_slowdown(120,100)==1.2
    assert h2d(0,49152)==(0,1344,1344,0)
    assert h2d(12,24576)==(1344,672,2016,672)

def test_source_and_k2_reservation():
    assert source_conserved(24576,24576) and not source_conserved(24576,24575)
    assert len(SESSIONS)==12 and TOKENS==32
    assert GATE_LIMIT==2 and SESSION_LEASE_BLOCKS==256 and LEASE_BUDGET_BLOCKS==512
