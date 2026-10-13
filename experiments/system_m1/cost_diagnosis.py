"""Bounded CPU/native replay, not a measurement of the GPU critical path."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import statistics
import textwrap
import time
from types import SimpleNamespace


OLD_LOOP = 'while key is not None and key in parents:'
NEW_LOOP = 'while key is not None and key in parents and key not in keep:'


def closure(parents, needed, protected, stop):
    """Test oracle: needed must already be ancestor closed."""
    keep = set(needed)
    visits = 0
    for key in protected:
        while key is not None and key in parents and (not stop or key not in keep):
            visits += 1
            keep.add(key)
            key = parents[key]
    return keep, visits


def run(reference, repeats, batches):
    import torch
    def forbid_cuda(*args, **kwargs):
        raise AssertionError('CPU diagnosis attempted CUDA initialization')
    torch.cuda._lazy_init = forbid_cuda
    from vllm.v1.kv_offload.base import ReqContext, make_offload_key
    from slackkv.m1.native import M1CPUManager
    import slackkv.m1.native as native
    from experiments.system_m1 import native_audit

    source = reference.read_text(encoding='utf-8')
    tree = ast.parse(source)
    node = next(n for c in tree.body if isinstance(c, ast.ClassDef)
                and c.name == 'M1CPUManager' for n in c.body
                if isinstance(n, ast.FunctionDef) and n.name == '_reclaim_shared')
    original = textwrap.dedent(ast.get_source_segment(source, node))
    if original.count(OLD_LOOP) != 1:
        raise ValueError('reference ancestor loop differs')
    candidate = original.replace(OLD_LOOP, NEW_LOOP)
    functions = {}
    for name, code in [('reference', original), ('candidate', candidate)]:
        scope = dict(vars(native))
        exec(compile(code, f'<{name}-reclaim>', 'exec'), scope)
        functions[name] = scope['_reclaim_shared']
    installed = M1CPUManager._reclaim_shared
    result = dict(gpu_hours=0, gpu_count=0, repeats=repeats, batches=batches,
                  reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                  installed_sha256=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
                  candidate_change=dict(old=OLD_LOOP, new=NEW_LOOP),
                  scope='CPU metadata only; synthetic native audit/state replay; no GPU critical-path attribution')
    try:
        audit_results = {}
        for name, func in functions.items():
            calls = []
            def timed(manager, protected):
                size = dict(members=len(manager.controller.members),
                            path_blocks=sum(len(m.path) for m in manager.controller.members.values()),
                            physical_blocks=len(manager._policy.blocks),
                            explicit_protected_blocks=len(set(protected)),
                            ref_protected_blocks=sum(b.ref_cnt != 0 for b in manager._policy.blocks.values()))
                started = time.perf_counter_ns()
                removed = func(manager, protected)
                calls.append(dict(**size, ns=time.perf_counter_ns()-started, removed_blocks=len(removed)))
                return removed
            M1CPUManager._reclaim_shared = timed
            audit_results[name] = native_audit.run()
            result.setdefault('native_audit', {})[name] = dict(
                checks=audit_results[name]['checks'], calls=calls,
                call_count=len(calls), cumulative_ms=sum(c['ns'] for c in calls)/1e6)
        # Timing wrappers do not change allocation, events, refs or returned snapshots.
        if audit_results['reference'] != audit_results['candidate']:
            raise AssertionError('native audit outcomes differ')
        result['native_audit_equivalent'] = True

        def keys(label, n):
            return tuple(make_offload_key(hashlib.sha256(f'{label}-{i}'.encode()).digest(), 0) for i in range(n))
        def build(lengths, mode):
            m = M1CPUManager(512, 16, 458752, enable_events=True, shared_prefix_blocks=32)
            root = keys('root', 32)
            contexts = []
            paths = []
            for i, length in enumerate(lengths):
                path = root + keys(str(i), length-32)
                ctx = ReqContext(f'r{i}', {'slackkv_m1':dict(session_id=str(i), generation=0)})
                m.on_new_request(ctx)
                state = SimpleNamespace(req_context=ctx, req=SimpleNamespace(num_prompt_tokens=length*16),
                    config=SimpleNamespace(kv_group_configs=[SimpleNamespace(sliding_window_size_in_blocks=None)]),
                    group_states=[SimpleNamespace(offload_keys=path)], max_offload_tokens=None)
                m.slackkv_m1_observe(state)
                out = m.prepare_store(path, ctx)
                if mode != 'pending':
                    m.complete_store(out.keys_to_store, ctx)
                    m.on_request_finished(ctx)
                contexts.append(ctx); paths.append(path)
            # Materialize all current quotas once; timing then observes stable metadata.
            if len(lengths) > 1:
                for ctx, path in zip(contexts, paths):
                    # Existing frozen plans may underfill an expanded quota, intentionally.
                    out = m.prepare_store(path, ctx)
                    if out.keys_to_store:m.complete_store(out.keys_to_store, ctx)
            explicit = paths[-1] if mode == 'incoming' else ()
            if mode == 'load':
                ready = [k for k in paths[0] if m._policy.get(k) is not None]
                m.prepare_load(ready, contexts[0])
            return m, explicit

        cases = [('single_idle', [512], 'idle'), ('single_load', [512], 'load'),
                 ('single_pending', [512], 'pending'), ('single_incoming', [512], 'incoming'),
                 ('four_idle', [512,384,64,256], 'idle'),
                 ('four_load', [512,384,64,256], 'load'),
                 ('four_incoming', [512,384,64,256], 'incoming')]
        result['benchmarks'] = []
        for label, lengths, mode in cases:
            M1CPUManager._reclaim_shared = functions['reference']
            manager, protected = build(lengths, mode)
            before = manager.snapshot_metrics()
            timings = {name:[] for name in functions}
            # Alternate order to avoid treating a cold/late process as policy benefit.
            for batch in range(batches):
                for name in (['reference','candidate'] if batch%2 == 0 else ['candidate','reference']):
                    M1CPUManager._reclaim_shared = functions[name]
                    for _ in range(3):
                        assert manager.reclaim_tails(protected) == []
                    started = time.perf_counter_ns()
                    for _ in range(repeats):
                        assert manager.reclaim_tails(protected) == []
                    timings[name].append((time.perf_counter_ns()-started)/repeats/1e6)
                    assert manager.snapshot_metrics() == before
            result['benchmarks'].append(dict(label=label, lengths=lengths, mode=mode,
                snapshot=before, explicit_protected_blocks=len(protected),
                mean_call_ms_by_batch=timings,
                median_call_ms={k:statistics.median(v) for k,v in timings.items()}))
        result['status'] = 'PASS'
        result['python'] = __import__('sys').version
        result['torch'] = torch.__version__
        return result
    finally:
        M1CPUManager._reclaim_shared = installed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=100)
    parser.add_argument('--batches', type=int, default=5)
    args = parser.parse_args()
    if args.repeats <= 0 or args.batches <= 0:raise ValueError('positive repetition counts required')
    if args.output.exists():raise FileExistsError(args.output)
    started = time.monotonic()
    result = dict(status='FAILED', gpu_hours=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = run(args.reference, args.repeats, args.batches)
    except BaseException:
        result['error'] = __import__('traceback').format_exc()
        raise
    finally:
        result['cpu_wall_seconds'] = time.monotonic()-started
        result['diagnosis_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
        print(json.dumps({k:v for k,v in result.items() if k not in ('native_audit','benchmarks')}, indent=2))


if __name__ == '__main__':main()
