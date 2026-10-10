"""CPU-only differential audit against pinned real vLLM offloading manager."""
import argparse
import hashlib
import json
import random
from pathlib import Path
from experiments.host_prefix.model import Cache, Tree


def main():
    from vllm.v1.kv_offload.base import ReqContext
    from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
    import vllm.v1.kv_offload.cpu.manager as module
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    checks = 0
    for capacity in (16, 32, 64, 256):
        tree = Tree()
        local = Cache(capacity, 'block_lru', tree)
        native = CPUOffloadingManager(capacity)
        rng = random.Random(901)
        histories = [[root] for root in range(4)]
        for request in range(80):
            branch = rng.randrange(4)
            histories[branch].extend(rng.randrange(10, 1000000) for _ in range(rng.randrange(1, 40)))
            path = tree.path(histories[branch])
            keys = [n.to_bytes(8, 'big') for n in path]
            ctx = ReqContext(str(request))
            real_hits = 0
            for key in keys:
                if not native.lookup(key, ctx):
                    break
                real_hits += 1
            assert real_hits == local.hit(path)
            local.touch(path)
            native.touch(keys, ctx)
            for start in range(0, len(keys), 32):
                result = native.prepare_store(keys[start:start + 32], ctx)
                if result is not None and result.keys_to_store:
                    native.touch(keys, ctx)
                    native.complete_store(result.keys_to_store, ctx)
            local.store(path)
            actual = [int.from_bytes(key, 'big') for key in native._policy.blocks]
            assert actual == list(local.blocks), (capacity, request, actual, list(local.blocks))
            checks += 1
    # Real refcounts: in-flight load blocks cannot be evicted; cancellation releases.
    ctx = ReqContext('lifecycle')
    native = CPUOffloadingManager(2)
    first = native.prepare_store([b'a', b'b'], ctx)
    assert native.lookup(b'a', ctx) is None
    native.complete_store(first.keys_to_store, ctx)
    native.prepare_load([b'a', b'b'], ctx)
    assert native.prepare_store([b'c'], ctx) is None
    native.complete_load([b'a', b'b'], ctx)
    cancelled = native.prepare_store([b'c'], ctx)
    native.complete_store(cancelled.keys_to_store, ctx, success=False)
    assert native.lookup(b'c', ctx) is False
    assert native._get_num_free_blocks() == 1
    result = dict(differential_requests=checks, exact_order_and_hits='PASS',
                  native_refcount_and_cancel='PASS', gpu_hours=0,
                  manager_path=module.__file__,
                  manager_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
                  qualification='metadata only, no tensor allocation, not scheduler validation')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
