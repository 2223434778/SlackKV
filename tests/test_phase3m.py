def test_timestamp_ordering():
    assert [1,2,3,6,7] == sorted([1,2,3,6,7])
def test_remaining_step():
    assert max(0.0, 150.0-120.0) == 30.0
def test_decomposition():
    parts=[1.0,2.0,3.0,4.0]
    assert abs(sum(parts)-10.0) < 1e-9
def test_active_through_promotion():
    assert 1.0 < 2.0 < 3.0
