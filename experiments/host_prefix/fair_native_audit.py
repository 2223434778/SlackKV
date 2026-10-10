"""CPU metadata proof for static distinct-prefix admission and native references."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    from vllm.v1.kv_offload.base import ReqContext
    from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
    import vllm.v1.kv_offload.cpu.manager as module
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    ctx = ReqContext('fair-metadata')
    native = CPUOffloadingManager(7)
    a, b, c = [[prefix + bytes([i]) for i in range(size)] for prefix, size in ((b'a', 3), (b'b', 4), (b'c', 3))]
    def store(keys):
        job = native.prepare_store(keys, ctx)
        assert job is not None
        native.complete_store(job.keys_to_store, ctx)
        return job
    store(a)
    store(b)
    for _ in range(6):
        for path in (a, b):
            assert all(native.lookup(k, ctx) for k in path)
            native.prepare_load(path, ctx)
            native.complete_load(path, ctx)
            native.touch(path, ctx)
            assert not store(path).keys_to_store
    # Replace retired A with C while B remains in-flight. A real C request lists
    # only C keys: protect B through its read reference, not an artificial union.
    native.prepare_load(a, ctx)
    native.prepare_load(b, ctx)
    assert native.prepare_store(c, ctx) is None
    assert all(native.lookup(k, ctx) for k in a+b)
    native.complete_load(a, ctx)
    replacement = store(c)
    assert set(replacement.evicted_keys) == set(a)
    assert all(native.lookup(k, ctx) for k in b+c)
    assert all(native.lookup(k, ctx) is False for k in a)
    native.complete_load(b, ctx)
    assert all(block.ref_cnt == 0 for block in native._policy.blocks.values())
    cancel = CPUOffloadingManager(2)
    pending = cancel.prepare_store([b'x'], ctx)
    assert cancel.lookup(b'x', ctx) is None
    cancel.complete_store(pending.keys_to_store, ctx, success=False)
    assert cancel._get_num_free_blocks() == 2 and not cancel._policy.blocks
    result = dict(status='PASS', alternating_retained_blocks=[3, 4],
                  in_flight_replacement='deferred; no forced free', retired_replacement='PASS',
                  cancelled_store_free_blocks=2, gpu_hours=0,
                  manager_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
                  qualification='metadata only, static disjoint membership; no GPU cancellation or tensor-content proof')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
