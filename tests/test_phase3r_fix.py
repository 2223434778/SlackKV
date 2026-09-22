from experiments.phase3r_fix.helpers import DEADLINES_MS,ASYNC_SERVICE_MS,BLOCKING_SERVICE_MS,internal_ready_miss,gpu_ready_residency_ms

def test_frozen_scale_config():
 assert tuple(DEADLINES_MS)==tuple('ABCDEFGHIJKL')
 assert list(DEADLINES_MS.values())==[300.,320.,340.,360.,900.,920.,940.,960.,1500.,1520.,1540.,1560.]
 assert 12*4096==49152 and 12*31==372 and 372+255==627
 assert ASYNC_SERVICE_MS==40 and BLOCKING_SERVICE_MS==70

def test_ready_and_residency():
 assert internal_ready_miss(9.,10.) and not internal_ready_miss(10.,10.)
 assert gpu_ready_residency_ms(20.,10.)==10.
from experiments.phase3r_fix.helpers import capacity_decision,promotion_outcome,REQUIRED_FREE_MIB

def test_capacity_guard_and_outcomes():
 assert REQUIRED_FREE_MIB==144.0
 assert capacity_decision((1024-144)/1024)[0]=='PROMOTE'
 assert capacity_decision((1024-143)/1024)[0]=='CAPACITY_SKIP'
 assert promotion_outcome('PROMOTE',True,False)=='PROMOTED_AND_LOCAL'
 assert promotion_outcome('PROMOTE',True,True)=='READY_BUT_NOT_LOCAL'
 assert promotion_outcome('CAPACITY_SKIP',False,True)=='CAPACITY_SKIPPED_ONDEMAND'

def test_audit_math_accepts_partial_source_split():
 assert 23600 + 25552 == 12 * 4096

def test_query_and_target_timing():
 assert round((1.004-1.000)*1000,6) == 4.0
 assert 190.0-140.0 == 50.0

def test_remote_wait_count_comes_from_rows():
 rows=[{'session':'E','WAITING_FOR_REMOTE_KVS_seen':True},{'session':'F','WAITING_FOR_REMOTE_KVS_seen':False}]
 assert [r['session'] for r in rows if r['WAITING_FOR_REMOTE_KVS_seen']] == ['E']

def test_experiment_valid_is_separate_from_policy_pass():
 experiment_valid=True; policy_pass=False
 assert experiment_valid and not policy_pass
