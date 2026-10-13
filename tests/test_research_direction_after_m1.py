import itertools
from pathlib import Path

import pytest

from experiments.research_direction_after_m1 import large_model, persistent_reuse_bound


def test_uniform_fixed_prefix_quota_has_no_extra_token_opportunity():
    result=persistent_reuse_bound(12,{'a':16,'b':16,'c':16},
                                 {'a':[16,16],'b':[16,16],'c':[16,16]},4)
    assert result['saved_tokens']==24
    # Maximizing token hits is not a fairness proof: tie order can concentrate.
    assert result['shares_tokens']=={'a':12,'b':0,'c':0}


def test_heterogeneous_future_reuse_changes_conditional_oracle():
    result=persistent_reuse_bound(8,{'a':8,'b':8},{'a':[8]*5,'b':[8]},4)
    assert result['saved_tokens']==40
    assert result['shares_tokens']=={'a':8,'b':0}
    # Equal4/4 would save24; this oracle uses FUTURE references and sacrifices B.


@pytest.mark.parametrize('lcps',[[[8,8],[8]],[[4,8],[0,4]],[[0],[4,4]],[[8,4],[8,0]]])
def test_greedy_prefix_reference_matches_small_exhaustive_state_space(lcps):
    lengths={0:8,1:8};later=dict(enumerate(lcps))
    for capacity in range(0,20,4):
        expected=max(sum(min(q,lcp) for s,q in enumerate(quotas) for lcp in lcps[s])
                     for quotas in itertools.product(range(0,12,4),repeat=2) if sum(quotas)<=capacity)
        assert persistent_reuse_bound(capacity,lengths,later,4)['saved_tokens']==expected


def test_reference_rejects_session_mismatch():
    with pytest.raises(ValueError):persistent_reuse_bound(16,{'a':16},{'b':[16]})


def test_archived_32b_hit_work_and_conditional_bound_conserve():
    result=large_model(Path(__file__).resolve().parents[1]/'results/large_model_20261012')
    assert result['archive_validation']['validated_files']==82
    assert result['matched_input_identities']==66
    assert result['fixed_prefix_reference']['extra_saved_tokens']==0
    assert result['unbounded_storage_reference']['fair_remaining_reusable_compute_tokens']==690688
    assert all(r['total_input_tokens']==1863168 and r['total_output_tokens']==8448 for r in result['reports'].values())
