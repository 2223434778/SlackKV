from experiments.phase3g_diag.metrics import first_divergence_position, matching_prefix_length, sequence_hash, unique_sequence_counts, structural_schedule

def test_prefix_and_divergence():
    assert matching_prefix_length([1,2,3],[1,2,4]) == 2
    assert first_divergence_position([1,2,3],[1,2,4]) == 3

def test_hash_and_unique_counts():
    assert sequence_hash([1,2]) == sequence_hash([1,2])
    assert unique_sequence_counts([[1],[1],[2]]) == {sequence_hash([1]):2, sequence_hash([2]):1}

def test_structural_schedule():
    rows={"A":{"first_token_step":2,"admission_step":1,"finish_step":5},"B":{"first_token_step":3,"admission_step":3,"finish_step":6},"C":{"first_token_step":4,"admission_step":4,"finish_step":7}}
    assert structural_schedule(rows)
