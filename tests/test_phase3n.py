from experiments.phase3n.helpers import active_x_valid, classify_handoff, internal_pre_return, local_resume_valid


def test_handoff_categories_use_scheduler_arrival() -> None:
    assert classify_handoff(1.0, 2.0, 3.0) == "PRE_READY"
    assert classify_handoff(2.0, 2.0, 3.0) == "INTERNAL_READY_PRE_CLIENT_RETURN"
    assert classify_handoff(3.0, 2.0, 3.0) == "POST_CLIENT_RETURN"


def test_internal_subwindows() -> None:
    assert internal_pre_return(2.2, 2.0, 2.6, 3.0) == (True, False)
    assert internal_pre_return(2.8, 2.0, 2.6, 3.0) == (False, True)


def test_active_x_and_real_cache_path() -> None:
    assert active_x_valid(1.0, 1.2, 1.3, 2.0)
    assert not active_x_valid(1.0, 1.2, 2.1, 2.0)
    assert local_resume_valid(4096.0, 0.0, False)
    assert not local_resume_valid(4096.0, 1.0, False)
    assert not local_resume_valid(4096.0, 0.0, True)
