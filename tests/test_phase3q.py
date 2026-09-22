from experiments.phase3q.helpers import DEADLINES_MS,ASYNC_SERVICE_MS,BLOCKING_SERVICE_MS,internal_ready_miss,gpu_ready_residency_ms

def test_frozen_scale_config():
 assert tuple(DEADLINES_MS)==tuple('ABCDEFGHIJKL')
 assert list(DEADLINES_MS.values())==[300.,320.,340.,360.,900.,920.,940.,960.,1500.,1520.,1540.,1560.]
 assert 12*4096==49152 and 12*31==372 and 372+255==627
 assert ASYNC_SERVICE_MS==40 and BLOCKING_SERVICE_MS==70

def test_ready_and_residency():
 assert internal_ready_miss(9.,10.) and not internal_ready_miss(10.,10.)
 assert gpu_ready_residency_ms(20.,10.)==10.
