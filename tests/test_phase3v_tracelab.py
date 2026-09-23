from experiments.phase3v_tracelab_incomplete.selection import size_bucket, pause_bucket, service_bucket, ceil5, required_blocks, h2d

def test_buckets():
    assert [size_bucket(x) for x in (1024,3072,5632)] == ['small','medium','large']
    assert [pause_bucket(x) for x in (100,500,1500)] == ['short','medium','long']
def test_service_bucket(): assert service_bucket(3500)==4096 and service_bucket(7000)==8192
def test_ceil5(): assert ceil5(17.1)==20
def test_blocks(): assert required_blocks(4097,16)==257
def test_budget(): assert required_blocks(8192,16)==512
def test_h2d(): assert h2d(117440512,117440512)==(117440512,0)
def test_k2(): assert 2 == 2
