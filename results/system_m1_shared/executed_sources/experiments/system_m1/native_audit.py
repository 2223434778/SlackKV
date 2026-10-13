"""Real native metadata contracts, not a GPU, tensor-content or speed oracle."""
import argparse
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace
import time
import traceback


def run():
    import torch
    def forbid_cuda(*args, **kwargs):
        raise AssertionError('CPU audit attempted CUDA initialization')
    torch.cuda._lazy_init = forbid_cuda
    from vllm.v1.kv_offload.base import ReqContext, make_offload_key
    from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
    import vllm.v1.kv_offload.cpu.manager as native_module
    from slackkv.m1.native import M1CPUManager

    checks, snapshots = [], {}
    def keys(label, n):
        return tuple(make_offload_key(hashlib.sha256(f'{label}-{i}'.encode()).digest(), 0) for i in range(n))
    def start(manager, request, session, n, generation=0, path=None, end=False):
        path = path or keys(session, n)
        ctx = ReqContext(request, {'slackkv_m1': dict(session_id=session, generation=generation, end_after_request=end)})
        manager.on_new_request(ctx)
        state = SimpleNamespace(req_context=ctx, req=SimpleNamespace(num_prompt_tokens=n*16),
                                config=SimpleNamespace(kv_group_configs=[SimpleNamespace(sliding_window_size_in_blocks=None)]),
                                group_states=[SimpleNamespace(offload_keys=path)], max_offload_tokens=None)
        manager.slackkv_m1_observe(state)
        return ctx, path, state
    def store(manager, ctx, path, success=True):
        output = manager.prepare_store(path, ctx)
        assert output is not None
        if output.keys_to_store:
            manager.complete_store(output.keys_to_store, ctx, success=success)
        return output
    def snapshot(manager):
        result = manager.snapshot_metrics()
        assert result['physical_ready_blocks']+result['physical_pending_store_blocks']+result['physical_free_blocks']==result['capacity_blocks']
        assert result['physical_load_protected_blocks']<=result['physical_ready_blocks']
        return result

    # The gate does not retire identity before load, nor discard old DMA tickets.
    m=M1CPUManager(4,16,1,terminal_store_gate=True)
    initial,ik,_=start(m,'ordinary','terminal',4)
    pending=m.prepare_store(ik[:2],initial)
    m.complete_store(pending.keys_to_store,initial)
    m.on_request_finished(initial)
    terminal,tk,_=start(m,'terminal-read','terminal',4,end=True)
    assert all(m.lookup(k,terminal) for k in tk[:2])
    m.prepare_load(tk[:2],terminal)
    assert not m.prepare_store(tk[2:],terminal).keys_to_store
    assert 'terminal' not in m.controller.ended
    m.on_request_finished(terminal)
    assert snapshot(m)['physical_load_protected_blocks']==2
    m.complete_load(tk[:2],terminal)
    assert snapshot(m)['physical_free_blocks']==4
    higher,hk,_=start(m,'new-terminal','terminal',4,generation=1,end=True)
    assert not m.prepare_store(hk,higher).keys_to_store
    m.on_request_finished(higher)
    late,lk,_=start(m,'late-terminal','terminal',4,generation=1)
    assert not m.prepare_store(lk,late).keys_to_store
    m.on_request_finished(late)
    assert snapshot(m)['physical_free_blocks']==4
    assert m.metrics['terminal_skipped_store_batches']==2
    checks.append('terminal_gate_preserves_load_until_settlement_new_generation_no_store_tombstone')

    m=M1CPUManager(4,16,1,terminal_store_gate=True)
    initial,ik,_=start(m,'inflight','inflight',4)
    pending=m.prepare_store(ik,initial)
    initial.kv_transfer_params['slackkv_m1']['end_after_request']=True
    assert not m.prepare_store(ik,initial).keys_to_store
    m.on_request_finished(initial)
    assert snapshot(m)['physical_pending_store_blocks']==4
    m.complete_store(pending.keys_to_store,initial)
    assert snapshot(m)['physical_free_blocks']==4
    checks.append('terminal_gate_old_pending_batch_completes_without_leak')

    from slackkv.m1.native import TerminalCPUManager
    full=TerminalCPUManager(4,cache_policy='lru')
    ctx=ReqContext('full-ordinary',{})
    full.on_new_request(ctx)
    fk=keys('full-terminal',4)
    pending=full.prepare_store(fk[:2],ctx)
    full.complete_store(pending.keys_to_store,ctx)
    terminal=ReqContext('full-terminal',{'slackkv_terminal':dict(independent=True,end_after_request=True)})
    full.on_new_request(terminal)
    assert all(full.lookup(k,terminal) for k in fk[:2])
    full.prepare_load(fk[:2],terminal)
    assert not full.prepare_store(fk[2:],terminal).keys_to_store
    full.complete_load(fk[:2],terminal)
    assert len(full.prepare_store(fk[2:],ctx).keys_to_store)==2
    checks.append('common_full_terminal_gate_preserves_native_reads_and_ordinary_store')

    m=M1CPUManager(512,16,1,terminal_store_gate=True)
    a,ak,_=start(m,'terminal-demand-A','a',512);store(m,a,ak);m.on_request_finished(a)
    before=(m.controller.cursor,m.controller.version,m.metrics['reclaimed_blocks'])
    t,tk,state=start(m,'terminal-demand-T','t',512,end=True)
    assert state.max_offload_tokens==0 and m.controller.members[('a',0)].target==512
    assert snapshot(m)['physical_ready_blocks']==512
    assert not store(m,t,tk).keys_to_store
    m.on_request_finished(t)
    assert (m.controller.cursor,m.controller.version,m.metrics['reclaimed_blocks'])==before
    n,nk,_=start(m,'normal-demand-N','n',512)
    assert m.controller.members[('a',0)].target==256
    assert snapshot(m)['physical_ready_blocks']==256
    checks.append('terminal_new_member_no_retention_no_trim_normal_join_still_trims')

    m=M1CPUManager(9,16,1,terminal_store_gate=True)
    a,ak,_=start(m,'grow-A0','a',8);store(m,a,ak);m.on_request_finished(a)
    b,bk,_=start(m,'grow-B0','b',8);store(m,b,bk);m.on_request_finished(b)
    before=({t:x.target for t,x in m.controller.members.items()},m.controller.cursor,m.metrics['reclaimed_blocks'])
    terminal,tk,state=start(m,'grow-A-end','a',12,path=ak+keys('grow-extra',4),end=True)
    assert state.max_offload_tokens==0
    assert ({t:x.target for t,x in m.controller.members.items()},m.controller.cursor,m.metrics['reclaimed_blocks'])==before
    retained=ak[:before[0][('a',0)]]
    assert all(m.lookup(k,terminal) for k in retained)
    m.prepare_load(retained,terminal);m.on_request_finished(terminal)
    assert all(m.lookup(k,b) for k in bk[:before[0][('b',0)]])
    m.complete_load(retained,terminal)
    checks.append('terminal_growth_preserves_old_retention_remainder_and_native_read')

    root=keys('shared-root',3);ak=root+keys('private-A',6);bk=root+keys('private-B',6)
    m=M1CPUManager(9,16,1,shared_prefix_blocks=3,terminal_store_gate=True)
    a,_,_=start(m,'shared-supported-A','a',9,path=ak)
    assert len(store(m,a,ak).keys_to_store)==9;m.on_request_finished(a)
    b,_,_=start(m,'shared-supported-B','b',9,path=bk)
    assert len(store(m,b,bk).keys_to_store)==3
    assert snapshot(m)['physical_ready_blocks']==9 and snapshot(m)['physical_shared_blocks']==3
    assert m.controller.snapshot()['target_distinct_blocks']==9
    assert all(m.lookup(k,b) for k in bk[:6])
    m.on_request_finished(b);m.end_session('a',0)
    assert snapshot(m)['physical_ready_blocks']==6 and ('a',0) not in m.controller.members
    b,_,_=start(m,'shared-supported-B-next','b',9,path=bk)
    assert all(m.lookup(k,b) for k in bk[:6])
    assert len(store(m,b,bk).keys_to_store)==3
    m.prepare_load(bk,b);m.end_session('b',0);m.on_request_finished(b)
    assert snapshot(m)['physical_load_protected_blocks']==9
    m.complete_load(bk,b)
    assert snapshot(m)['physical_free_blocks']==9 and not m.controller.key_refs
    snapshots['shared_dedup_last_release']=snapshot(m)
    checks.append('shared_public_stored_once_private_tails_A_end_B_restore_last_load_releases_all')

    m=M1CPUManager(9,16,1,shared_prefix_blocks=3)
    a,_,_=start(m,'ancestor-A','a',9,path=ak);store(m,a,ak);m.on_request_finished(a)
    b,_,_=start(m,'ancestor-B','b',9,path=bk);store(m,b,bk)
    m.prepare_load(bk[5:6],b);m.end_session('b',0);m.on_request_finished(b)
    assert all(m.lookup(k,a) for k in bk[:6])  # Protect zero-ref ancestors too.
    m.complete_load(bk[5:6],b)
    assert all(m.lookup(k,a) for k in ak[:6]) and all(not m.lookup(k,a) for k in bk[3:6])
    checks.append('shared_private_deepest_load_protects_public_and_private_ancestors')

    root=keys('failure-root',2);ak=root+keys('failure-A',3);bk=root+keys('failure-B',3)
    m=M1CPUManager(8,16,1,shared_prefix_blocks=2)
    a,_,_=start(m,'failure-shared-A','a',5,path=ak)
    pending=m.prepare_store(root,a)
    b,_,_=start(m,'failure-shared-B','b',5,path=bk)
    private=m.prepare_store(bk,b)
    assert private.keys_to_store==list(bk[2:])
    m.complete_store(pending.keys_to_store,a,success=False)
    assert snapshot(m)['physical_pending_store_blocks']==3
    m.complete_store(private.keys_to_store,b)
    assert snapshot(m)['physical_free_blocks']==8  # Orphan descendants reclaimed.
    m.on_request_finished(a);m.on_request_finished(b)
    b,_,_=start(m,'failure-shared-B-retry','b',5,path=bk);store(m,b,bk)
    m.complete_store(pending.keys_to_store,a,success=False)  # Old ticket cannot delete rebuilt root.
    assert all(m.lookup(k,b) for k in bk)
    m.end_session('a',0);m.end_session('b',0);m.on_request_finished(b)
    assert snapshot(m)['physical_free_blocks']==8 and not m.controller.key_refs
    checks.append('shared_pending_dedup_root_failure_orphan_cleanup_retry_duplicate_completion')

    m=M1CPUManager(4,16,1,shared_prefix_blocks=6)
    root=keys('big-root',6)
    a,ak,_=start(m,'big-root-A','a',10,path=root+keys('big-A',4));store(m,a,ak)
    b,bk,_=start(m,'big-root-B','b',10,path=root+keys('big-B',4))
    assert not store(m,b,bk).keys_to_store
    assert snapshot(m)['physical_ready_blocks']==4 and m.controller.snapshot()['target_distinct_blocks']==4
    checks.append('shared_root_itself_capacity_bounded_not_duplicated')

    root=keys('generation-root',2)
    old_path=root+keys('generation-old',4);new_path=root+keys('generation-new',4);bp=root+keys('generation-B',4)
    m=M1CPUManager(8,16,1,shared_prefix_blocks=2)
    old,_,_=start(m,'generation-old','a',6,path=old_path)
    pending=m.prepare_store(old_path,old)
    b,_,_=start(m,'generation-B','b',6,path=bp);store(m,b,bp);m.on_request_finished(b)
    new,_,_=start(m,'generation-new','a',6,generation=1,path=new_path)
    assert not store(m,new,new_path).keys_to_store
    m.on_request_finished(old);m.complete_store(pending.keys_to_store,old)
    assert ('a',0) not in m.controller.members and all(m.lookup(k,b) for k in root)
    assert len(store(m,new,new_path).keys_to_store)==3
    m.complete_store(pending.keys_to_store,old,success=False)
    assert all(m.lookup(k,new) for k in new_path[:5])
    m.end_session('a',0);assert m.controller.members[('a',1)].alive
    m.end_session('a',1);m.on_request_finished(new);m.end_session('b',0)
    assert snapshot(m)['physical_free_blocks']==8 and not m.controller.key_refs
    checks.append('shared_old_generation_pending_settles_public_survives_stale_completion_and_end')

    root=keys('reset-root',2)
    m=M1CPUManager(6,16,1,shared_prefix_blocks=2)
    a,ak,_=start(m,'shared-reset-A','a',4,path=root+keys('reset-A',2))
    pending=m.prepare_store(ak,a)
    m.reset_cache()  # Same scheduler flush/stale-job boundary as native reset.
    m.on_request_finished(a)
    b,bk,_=start(m,'shared-reset-B','b',4,path=root+keys('reset-B',2));store(m,b,bk)
    m.complete_store(pending.keys_to_store,a)
    assert all(m.lookup(k,b) for k in bk)
    m.end_session('a',0);m.end_session('b',0);m.on_request_finished(b)
    assert snapshot(m)['physical_free_blocks']==6 and not m.controller.key_refs
    checks.append('shared_native_reset_boundary_old_ticket_cannot_modify_reused_public_slots')

    # External-module loggers may inherit WARNING and silently drop INFO traces.
    import io
    import logging
    import slackkv.m1.native as adapter
    stream=io.StringIO();handler=logging.StreamHandler(stream)
    adapter.logger.addHandler(handler)
    try:
        assert adapter.logger.isEnabledFor(logging.INFO)
        traced=M1CPUManager(1,16,1,trace=True)
        ctx,path,_=start(traced,'traced','trace',1,end=True)
        ctx.kv_transfer_params['slackkv_checkpoint_tag']='warmup_complete'
        store(traced,ctx,path);traced.on_request_finished(ctx)
        assert stream.getvalue().count('SLACKKV_M1_STATE')==3
        line=next(x for x in stream.getvalue().splitlines() if 'SLACKKV_POOL_CHECKPOINT ' in x)
        checkpoint=json.loads(line.split('SLACKKV_POOL_CHECKPOINT ',1)[1])
        assert checkpoint['free_blocks']==1 and checkpoint['retained_members']==0
    finally:
        adapter.logger.removeHandler(handler);handler.close()
    checks.append('trace_true_visible_at_native_vllm_info_level')
    checks.append('common_lightweight_checkpoint_reports_final_physical_cleanup')

    # Required 10-block counterexample: old ceiling persists, actual stores do not.
    m = M1CPUManager(10, 16, 4*2**20, enable_events=True)
    a, ak, astate = start(m, 'a0', 'a', 10)
    store(m, a, ak[:2])
    b, bk, _ = start(m, 'b0', 'b', 10)
    store(m, b, bk)
    output = store(m, a, ak[2:])
    assert astate.max_offload_tokens == 160 and len(output.keys_to_store)==3
    assert all(m.lookup(k, b) for k in bk[:5])
    assert all(m.lookup(k, a) for k in ak[:5]) and not m.lookup(ak[5], a)
    snapshots['old_plan_new_member'] = snapshot(m)
    checks.append('old10_ceiling_actual5_preserves_B5')

    # A deepest load protects its whole tail ordering; no ancestor hole.
    m = M1CPUManager(8, 16, 1, enable_events=True)
    a, ak, _ = start(m, 'tailA', 'a', 8)
    store(m, a, ak)
    m.prepare_load(ak[-1:], a)
    b, bk, _ = start(m, 'tailB', 'b', 8)
    assert snapshot(m)['physical_free_blocks']==0
    assert all(m.lookup(k, a) for k in ak)
    assert not store(m, b, bk).keys_to_store
    m.complete_load(ak[-1:], a)
    assert all(m.lookup(k, a) for k in ak[:4]) and not m.lookup(ak[4], a)
    store(m, b, bk)
    assert snapshot(m)['physical_ready_blocks']==8
    removed = [k for event in m.take_events() if event.removed for k in event.keys]
    assert removed == list(reversed(ak[4:]))
    checks.append('deepest_load_blocks_trim_then_tail_only_events')

    # Pending store -1, old generation and late/duplicate callbacks.
    m = M1CPUManager(4, 16, 1)
    old, ok, _ = start(m, 'old', 'reuse', 4, end=True)
    pending = m.prepare_store(ok, old)
    assert all(m.lookup(k, old) is None for k in ok)
    m.on_request_finished(old)
    new, nk, _ = start(m, 'new', 'reuse', 4, generation=1, path=keys('new', 4))
    assert snapshot(m)['physical_pending_store_blocks']==4
    assert not store(m, new, nk).keys_to_store
    m.complete_store(pending.keys_to_store, old)
    assert snapshot(m)['physical_free_blocks']==4 and snapshot(m)['tracked_request_contexts']==1
    store(m, new, nk)
    m.complete_store(pending.keys_to_store, old)  # Cannot change reused slots.
    assert all(m.lookup(k, new) for k in nk) and snapshot(m)['physical_ready_blocks']==4
    checks.append('pending_minus1_retired_generation_duplicate_physical_settlement')

    m=M1CPUManager(4,16,1)
    old,ok,_=start(m,'repeat-id','same',4,end=True)
    pending=m.prepare_store(ok,old);m.on_request_finished(old)
    new,_,state=start(m,'repeat-id','same',4,generation=1,path=ok)
    assert state.max_offload_tokens==0
    m.complete_store(pending.keys_to_store,old)
    m.on_request_finished(new)
    newer,_,state=start(m,'repeat-id','same',4,generation=1,path=ok)
    assert state.max_offload_tokens==64
    store(m,newer,ok)
    m.complete_store(pending.keys_to_store,old)
    assert all(m.lookup(k,newer) for k in ok)
    checks.append('same_identity_new_generation_next_request_rebuild_reused_request_id_safe')

    # Failed ancestor write leaves pending descendants, later safely removed.
    m = M1CPUManager(4, 16, 1)
    a, ak, _ = start(m, 'fail', 'a', 4)
    head=m.prepare_store(ak[:2], a);tail=m.prepare_store(ak[2:], a)
    m.complete_store(head.keys_to_store, a, success=False)
    assert snapshot(m)['physical_pending_store_blocks']==2
    assert snapshot(m)['members_physical'][0]['deferred_reclaim_blocks']==2
    m.complete_store(tail.keys_to_store, a)
    assert snapshot(m)['physical_free_blocks']==4
    checks.append('failed_ancestor_orphan_tail_eventual_reclaim')

    # Shared and unknown clients retain native reads, never hidden Full stores.
    m = M1CPUManager(6, 16, 1)
    a, ak, _ = start(m, 'sharedA', 'a', 3)
    store(m, a, ak)
    b, _, state = start(m, 'sharedB', 'b', 3, path=ak)
    assert state.max_offload_tokens==0 and not store(m, b, ak+keys('extra', 3)).keys_to_store
    m.prepare_load(ak, b);m.complete_load(ak, b)
    unknown=ReqContext('unknown')
    assert not m.prepare_store(keys('u', 3), unknown).keys_to_store
    assert snapshot(m)['physical_ready_blocks']==3
    checks.append('shared_unknown_no_store_native_read_identity_preserved')

    # A common native system prefix remains shared, not hidden by session salts.
    m = M1CPUManager(8, 16, 1)
    common = keys('system-prefix', 2)
    a, ak, _ = start(m, 'suffixA', 'a', 4, path=common+keys('suffixA', 2))
    store(m, a, ak)
    b, bk, state = start(m, 'suffixB', 'b', 4, path=common+keys('suffixB', 2))
    assert state.max_offload_tokens == 0 and not store(m, b, bk).keys_to_store
    m.prepare_load(common, b);m.complete_load(common, b)
    assert snapshot(m)['metrics']['fallback_shared_prefix'] == 1
    snapshots['common_system_prefix_distinct_suffix'] = snapshot(m)
    checks.append('common_system_prefix_distinct_suffix_one_shared_fallback_native_read')

    # End survives Member collection and cache reset; ordinary finish does not.
    m = M1CPUManager(4, 16, 1)
    a, ak, _ = start(m, 'ended-first', 'ended', 4)
    store(m, a, ak);m.on_request_finished(a)
    a1, _, state = start(m, 'ordinary-resume', 'ended', 4, path=ak)
    assert state.max_offload_tokens == 64
    m.end_session('ended', 0);m.on_request_finished(a1)
    assert not m.controller.members and snapshot(m)['physical_free_blocks'] == 4
    for reset in (False, True):
        if reset:m.reset_cache()
        late, _, state = start(m, f'late-{reset}', 'ended', 4, path=ak)
        assert state.max_offload_tokens == 0 and not store(m, late, ak).keys_to_store
        assert not m.controller.members and snapshot(m)['physical_free_blocks'] == 4
        m.on_request_finished(late)
    new, nk, state = start(m, 'new-generation', 'ended', 4, generation=1, path=ak)
    assert state.max_offload_tokens == 64
    m.end_session('ended', 0);m.end_session('ended', 0)
    store(m, new, nk)
    assert snapshot(m)['physical_ready_blocks'] == 4
    snapshots['ended_collected_reset_new_generation'] = snapshot(m)
    checks.append('end_collect_reset_same_generation_denied_new_generation_legal_old_end_noop')

    # Both kinds of in-flight native references span end/new-generation.
    m = M1CPUManager(4, 16, 1)
    old, ok, _ = start(m, 'read-old', 'read-end', 4)
    store(m, old, ok);m.prepare_load(ok, old)
    m.end_session('read-end', 0);m.on_request_finished(old)
    late, _, state = start(m, 'read-late', 'read-end', 4, path=ok)
    assert state.max_offload_tokens == 0 and not store(m, late, ok).keys_to_store
    # Terminal logical status does not invalidate legal native read identity.
    m.prepare_load(ok, late)
    new, nk, _ = start(m, 'read-new', 'read-end', 4, generation=1, path=keys('read-new', 4))
    assert snapshot(m)['physical_load_protected_blocks'] == 4
    assert not store(m, new, nk).keys_to_store
    m.complete_load(ok, old)
    assert snapshot(m)['physical_free_blocks'] == 0
    m.complete_load(ok, late);m.on_request_finished(late)
    assert snapshot(m)['physical_free_blocks'] == 4
    store(m, new, nk);m.complete_load(ok, old)
    assert all(m.lookup(k, new) for k in nk)
    m.end_session('read-end', 1);m.on_request_finished(new)
    assert snapshot(m)['physical_free_blocks'] == 4
    assert snapshot(m)['tracked_request_contexts'] == 0
    checks.append('end_new_generation_load_refs_settle_late_read_no_early_free_duplicate_safe')

    # Membership finish, elastic borrowing, next-normal-request rebuild.
    m = M1CPUManager(10, 16, 1)
    a, ak, _=start(m,'elasticA','a',10)
    b, bk, _=start(m,'elasticB','b',1)
    store(m,a,ak);store(m,b,bk)
    assert snapshot(m)['physical_ready_blocks']==10
    m.on_request_finished(a);m.end_session('b',0);m.on_request_finished(b)
    next_a, _, state=start(m,'elasticA1','a',10)
    assert state.max_offload_tokens==160
    store(m,next_a,ak)
    assert all(m.lookup(k,next_a) for k in ak)
    checks.append('short_lends_then_end_releases_next_request_expands')

    # Random interleaving on actual manager: no force-free while refs protect.
    rng=random.Random(611214)
    m=M1CPUManager(13,16,1)
    for i in range(120):
        sid=str(i%7);generation=i//7
        n=rng.randrange(1,18)
        ctx,path,_=start(m,f'random{i}',sid,n,generation,path=keys(f'{sid}-{generation}',n))
        out=m.prepare_store(path,ctx)
        if out.keys_to_store:
            if rng.randrange(2):
                m.end_session(sid,generation)
            m.complete_store(out.keys_to_store,ctx,success=rng.randrange(5)!=0)
        ready=[k for k in path if m.lookup(k,ctx) is True]
        if ready:
            m.prepare_load(ready,ctx)
            m.end_session(sid,generation)
            m.complete_load(ready,ctx)
        m.on_request_finished(ctx)
        snapshot(m)
    for sid,generation in list(m.controller.latest.items()):m.end_session(sid,generation)
    assert snapshot(m)['physical_free_blocks']==13
    assert snapshot(m)['tracked_request_contexts']==0 and not m.controller.plans
    checks.append('120_native_interleavings_recover_all_physical_slots')

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=1) as pool:
        try:pool.submit(m.end_session,'0',100).result()
        except RuntimeError:pass
        else:raise AssertionError('cross-thread membership must be rejected')
    m=M1CPUManager(2,16,1)
    ctx,ks,_=start(m,'reset','reset',2,end=True)
    pending=m.prepare_store(ks,ctx);m.on_request_finished(ctx)
    m.reset_cache()
    m.complete_store(pending.keys_to_store,ctx)
    assert snapshot(m)['tracked_request_contexts']==0 and snapshot(m)['physical_free_blocks']==2
    checks.append('scheduler_thread_membership_guard_reset_late_completion')

    # Real factory and connector hook, not a pretend existing API.
    from vllm.v1.kv_cache_interface import FullAttentionSpec,KVCacheConfig,KVCacheTensor,KVCacheGroupSpec
    from vllm.v1.kv_offload.factory import OffloadingSpecFactory
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import OffloadingConnectorScheduler
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import OffloadingWorkerMetadata
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import TransferJobStatus
    shape=FullAttentionSpec(block_size=16,num_kv_heads=4,head_size=128,dtype=torch.float16)
    cache=KVCacheConfig(num_blocks=8,kv_cache_tensors=[KVCacheTensor(size=8*shape.page_size_bytes*64,shared_by=['layer'])],
                        kv_cache_groups=[KVCacheGroupSpec(['layer'],shape)])
    extra=dict(spec_name='SlackKVM1Spec',spec_module_path='slackkv.m1.spec',block_size=16,cpu_bytes_to_use=10*4*2**20)
    config=SimpleNamespace(kv_transfer_config=SimpleNamespace(kv_connector_extra_config=extra),
                           parallel_config=SimpleNamespace(world_size=2,decode_context_parallel_size=1,prefill_context_parallel_size=1),
                           cache_config=SimpleNamespace(block_size=16,enable_prefix_caching=True),kv_events_config=None)
    spec=OffloadingSpecFactory.create_spec(config,cache)
    assert spec.get_manager()._num_blocks==10 and spec.kv_bytes_per_offloaded_block==4*2**20
    connector=OffloadingConnectorScheduler(spec)
    req=SimpleNamespace(request_id='connector-real',kv_transfer_params={'slackkv_m1':dict(session_id='real',generation=0)},
                        num_prompt_tokens=64,num_tokens=64,block_hashes=[hashlib.sha256(f'real{i}'.encode()).digest() for i in range(4)],
                        is_finished=lambda:False)
    connector.on_new_request(req)
    assert connector.get_num_new_matched_tokens(req,0)==(0,False)
    state=connector._req_status[req.request_id]
    actual_keys=state.group_states[0].offload_keys
    output=connector.manager.prepare_store(actual_keys,state.req_context)
    connector._jobs[0]=TransferJobStatus(req.request_id,2,set(output.keys_to_store),True)
    state.transfer_jobs.add(0)
    connector.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=OffloadingWorkerMetadata(completed_jobs={0:1})))
    assert snapshot(connector.manager)['physical_pending_store_blocks']==4
    connector.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=OffloadingWorkerMetadata(completed_jobs={0:1})))
    assert snapshot(connector.manager)['physical_ready_blocks']==4
    snapshots['factory_connector_tp2']=snapshot(connector.manager)
    extra['slackkv_m1_enabled']=False
    disabled=OffloadingSpecFactory.create_spec(config,cache).get_manager()
    assert type(disabled) is CPUOffloadingManager and not hasattr(disabled,'slackkv_m1_observe')
    checks.append('real_factory_hooks_tp2_all_workers_complete_disabled_native')

    extra.pop('slackkv_m1_enabled')
    connector=OffloadingConnectorScheduler(OffloadingSpecFactory.create_spec(config,cache))
    from vllm.v1.core.sched.output import NewRequestData,CachedRequestData,SchedulerOutput
    def request(label,session):
        return SimpleNamespace(request_id=label,kv_transfer_params={'slackkv_m1':dict(session_id=session,generation=0)},
                               num_prompt_tokens=160,num_tokens=160,num_computed_tokens=0,
                               block_hashes=[hashlib.sha256(f'{session}-{i}'.encode()).digest() for i in range(10)],is_finished=lambda:False)
    def frame(req,ids,tokens,first):
        output=SchedulerOutput.make_empty()
        output.num_scheduled_tokens={req.request_id:tokens};output.total_num_scheduled_tokens=tokens
        if first:
            output.scheduled_new_reqs=[NewRequestData(req.request_id,None,[],None,None,(ids,),req.num_computed_tokens,None)]
        else:
            output.scheduled_cached_reqs=CachedRequestData([req.request_id],set(),[],{},[None],[req.num_computed_tokens],[0])
        return connector.build_connector_meta(output)
    def complete(meta):
        if meta.store_jobs:
            connector.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=OffloadingWorkerMetadata(
                completed_jobs={jid:2 for jid in meta.store_jobs})))
    ra=request('builder-A','builderA');connector.on_new_request(ra);connector.get_num_new_matched_tokens(ra,0)
    first=frame(ra,list(range(1,11)),32,True);complete(first);ra.num_computed_tokens=32
    rb=request('builder-B','builderB');connector.on_new_request(rb);connector.get_num_new_matched_tokens(rb,0)
    second=frame(rb,list(range(11,21)),80,True);complete(second)
    third=frame(ra,[],96,False)
    assert len(third.store_jobs)==1
    job=next(iter(third.store_jobs.values()));source,destination=job.transfer_spec
    assert source.block_ids.tolist()==[3,4,5] and len(destination.block_ids)==3
    complete(third);ra.num_computed_tokens=128
    assert connector._req_status[ra.request_id].group_states[0].next_stored_block_idx==8
    connector.manager.end_session('builderB',0);connector.request_finished(rb)
    fourth=frame(ra,[],32,False)
    assert not fourth.store_jobs and snapshot(connector.manager)['physical_ready_blocks']==5
    connector.request_finished(ra)
    next_req=request('builder-A1','builderA');connector.on_new_request(next_req)
    # Exercise the second hook: first effective observation in build metadata.
    rebuilt=frame(next_req,list(range(21,31)),160,True)
    assert len(next(iter(rebuilt.store_jobs.values())).transfer_spec[1].block_ids)==5
    complete(rebuilt)
    assert snapshot(connector.manager)['physical_ready_blocks']==10
    snapshots['native_builder_partial_cursor_next_request']=snapshot(connector.manager)
    checks.append('real_store_builder_partial_src_dst_no_cursor_rewind_next_request_rebuild')

    extra['eviction_policy']='arc'
    try:OffloadingSpecFactory.create_spec(config,cache)
    except ValueError:pass
    else:raise AssertionError('unsupported policy accepted')
    extra.pop('eviction_policy')
    from dataclasses import replace
    unsupported=replace(cache,kv_cache_groups=[KVCacheGroupSpec(['layer'],replace(shape,sliding_window=16))])
    try:OffloadingSpecFactory.create_spec(config,unsupported)
    except ValueError:pass
    else:raise AssertionError('unsupported sliding attention accepted')
    checks.append('unsupported_layout_and_policy_rejected_before_handlers')
    return dict(status='PASS',checks=checks,snapshots=snapshots,
                native_manager_sha256=hashlib.sha256(Path(native_module.__file__).read_bytes()).hexdigest(),
                torch=torch.__version__,scope='CPU/native metadata only; synthetic request metadata; no GPU tensors/DMA/content/performance validation')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();result=dict(status='RUNNING',gpu_count=0,gpu_hours=0)
    try:
        result.update(run())
    except BaseException:
        result.update(status='FAILED',error=traceback.format_exc())
        raise
    finally:
        import sys
        result['source_sha256']={name:hashlib.sha256(Path(sys.modules[name].__file__).read_bytes()).hexdigest()
                                 for name in ('slackkv.m1.controller','slackkv.m1.native','slackkv.m1.spec') if name in sys.modules}
        result['audit_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        result['cpu_wall_seconds']=time.monotonic()-started
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))


if __name__=='__main__':main()
