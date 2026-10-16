"""No CUDA tensors: real native terminal/load/store settlement contract."""
import hashlib
import json


def run():
    import torch
    def forbid(*args,**kwargs):raise AssertionError('native CPU audit attempted CUDA init')
    torch.cuda._lazy_init=forbid
    from vllm.v1.kv_offload.base import ReqContext,make_offload_key
    from slackkv.infra.native import TerminalManager
    keys=[make_offload_key(hashlib.sha256(str(i).encode()).digest(),0) for i in range(4)]
    manager=TerminalManager(4,cache_policy='lru')
    ordinary=ReqContext('ordinary',{})
    manager.on_new_request(ordinary)
    stored=manager.prepare_store(keys,ordinary)
    manager.complete_store(stored.keys_to_store,ordinary)
    terminal=ReqContext('terminal',{'slackkv_terminal':dict(independent=True,end_after_request=True)})
    manager.on_new_request(terminal)
    assert all(manager.lookup(k,terminal) for k in keys)
    manager.prepare_load(keys,terminal)
    assert all(manager._policy.get(k).ref_cnt>0 for k in keys)
    assert not manager.prepare_store(keys,terminal).keys_to_store
    manager.on_request_finished(terminal)
    assert all(manager._policy.get(k).ref_cnt>0 for k in keys)
    manager.complete_load(keys,terminal)
    assert all(manager._policy.get(k).ref_cnt==0 for k in keys)
    manager.reset_cache();assert manager._get_num_free_blocks()==4
    pending=manager.prepare_store(keys,ordinary)
    manager.complete_store(pending.keys_to_store,ordinary,success=False)
    assert manager._get_num_free_blocks()==4
    return dict(status='PASS',gpu_hours=0,checks=['native_store_lookup','terminal_skips_optional_writes',
        'finish_does_not_release_inflight_load','load_settles','native_reset','failed_store_returns_all_blocks'],
        scope='real native CPU metadata only; not GPU content or concurrency proof')


if __name__=='__main__':print(json.dumps(run(),indent=2))
