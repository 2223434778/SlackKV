from experiments.phase3f.scheduler import FirstTokenGate, merge_blackouts, latest_feasible_start, DEADLINE_MS, RESUME_GUARD_MS

def test_first_token_gate():
    g=FirstTokenGate()
    assert g.can_admit('a')
    g.admit('a'); assert not g.can_admit('b')
    g.first_token('a'); assert g.can_admit('b')

def test_blackout_merge():
    assert merge_blackouts({'A':300,'B':320,'C':340},110)==[(300,450)]

def test_latest_start_before_blackout():
    t=latest_feasible_start(0,[('A',300),('B',320),('C',340)],[(300,450)],30)
    assert 205 <= t <= 215
