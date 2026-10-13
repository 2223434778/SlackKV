import random
from pathlib import Path

import pytest

from experiments.system_m1.cost_diagnosis import closure
from experiments.system_m1.analyze_cost import cohort, existing_events


@pytest.mark.parametrize('seed', range(30))
def test_protection_early_stop_preserves_union_and_ancestor_closure(seed):
    rng = random.Random(seed)
    # Tree generalization also covers the supported shared-root/private-tail shape.
    parents = {0:None}
    for key in range(1,100):parents[key] = rng.randrange(key)
    needed, _ = closure(parents, set(), rng.sample(range(100), 10), False)
    protected = rng.sample(range(100), 30)
    old, old_visits = closure(parents, needed, protected, False)
    new, new_visits = closure(parents, needed, reversed(protected), True)
    assert new == old
    assert new_visits <= old_visits
    assert all(p is None or p in new for k,p in parents.items() if k in new)


def test_needed_prefix_is_already_closed_no_repeated_walk():
    parents = {k:k-1 if k else None for k in range(512)}
    old, old_visits = closure(parents, set(parents), parents, False)
    new, new_visits = closure(parents, set(parents), parents, True)
    assert new == old == set(parents)
    assert old_visits == 512*513//2 and new_visits == 0


def test_orphan_protected_descendant_keeps_missing_materialized_ancestors():
    parents = {0:None,1:0,2:1,3:2,4:1}
    assert closure(parents,{0},[3,4],True)[0] == set(parents)


def test_existing_event_accounting_against_archived_native_counters():
    base = Path(__file__).resolve().parents[1]/'results/system_m1_smoke'
    result = existing_events(base)
    assert result['admitted_blocks'] == result['reclaimed_blocks'] == 1228
    assert result['written_mib'] == 537.25
    assert result['prefix_depth_inferred_rewritten_mib'] == 70
    writes = [r for r in result['requests'] if r['admitted_blocks']]
    assert [r['label'] for r in writes] == ['a0','b0','s0','d0','a3','b_after_a_end']
    assert writes[-1]['admitted_blocks'] == 204
    assert writes[-1]['prefix_depth_inferred_rewritten_blocks'] == 80
    costs = result['original_screen_cost_decomposition']
    assert all(v['between_requests_ms'] >= v['configured_gap_ms'] for v in costs.values())


def test_equivalent_fix_cohort_preserves_actual_work_and_dma():
    base = Path(__file__).resolve().parents[1]/'results/system_m1_smoke'
    result = cohort(base)
    old = result['runs']['m1_cost_ref1']
    new = result['runs']['m1_cost_opt1']
    assert old['source_tokens'] == new['source_tokens']
    assert old['transfer_bytes'] == new['transfer_bytes']
    assert all(r['prompt_tokens']==153088 and r['output_tokens']==304 for r in result['runs'].values())
    assert all(e['status']!='RUNNING' and e['cleanup_verified'] for e in result['executions'].values())
    assert result['additional_gpu_hours'] <= .5
