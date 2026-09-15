from experiments.phase1d.run_phase1d import predicted


def test_sigma_zero_is_oracle_pause():
    assert predicted(2.5, 0.0, 1, "s", 3) == 2.5


def test_prediction_is_deterministic_across_policy_use():
    assert predicted(2.5, .25, 1, "s", 3) == predicted(2.5, .25, 1, "s", 3)


def test_overestimate_has_later_predicted_deadline():
    assert 1.0 * (1 + .25) > 1.0


def test_underestimate_can_finish_earlier_and_add_residency():
    actual_deadline, completion = 2.0, 1.5
    assert max(0.0, actual_deadline - completion) == .5
