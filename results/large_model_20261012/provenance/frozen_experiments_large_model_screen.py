"""Frozen synthetic long-history closed-loop screen; never a production trace."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import threading
import time

from experiments.autonomous_fork.run import fixed_tokens

SESSIONS, TURNS, OUTPUT, SEED = 21, 3, 128, 914477
GPU_BYTES_PER_RANK = 70000 * 131072


def quotas(cpu_bytes):
    blocks = cpu_bytes // (16 * 262144)
    per = blocks // SESSIONS
    return [per * 16] * (SESSIONS-1) + [(blocks-per*(SESSIONS-1))*16]


def workload(out, runtime, policy, cpu_bytes):
    roots = [fixed_tokens(28672, SEED+s*10007) for s in range(SESSIONS)]
    assert len({tuple(root[:16]) for root in roots}) == SESSIONS
    shares = quotas(cpu_bytes)
    lock, ready = threading.Lock(), threading.Event()
    rows, inputs = [], []
    first_bg_ns = []
    save = lambda name, value: (out/name).write_text(json.dumps(value, indent=2) + '\n')
    (out/'before.prom').write_text(runtime.metric_text())

    def request(role, session, turn, on_first=None):
        if role == 'agent':
            prompt = roots[session] + fixed_tokens(turn*512, SEED+session+123)
            output = OUTPUT
            limit = len(prompt) if policy == 'full' else shares[session] if policy == 'fair' else 0
        else:
            prompt, output, limit = fixed_tokens(8192, SEED+800000+turn), 128, 0
        response = runtime.send(prompt, output, stream=True,
                                params=None if policy == 'nohost' else {'max_offload_tokens': limit}, on_first=on_first)
        if len(response['tokens']) != output:
            raise ValueError('prescribed output count mismatch')
        row = dict(role=role, session=session, turn=turn, length=len(prompt), max_offload_tokens=limit,
                   prompt_sha256=hashlib.sha256(json.dumps(prompt).encode()).hexdigest(), **response)
        with lock:
            rows.append(row)
            inputs.append(dict(role=role, session=session, turn=turn, prompt=prompt))
            save('requests.json', rows)
            save('inputs.json', inputs)
        print(json.dumps(dict(role=role, session=session, turn=turn,
                              ttft_ms=(response['first_ns']-response['issue_ns'])/1e6)), flush=True)

    def first(_):
        first_bg_ns.append(time.monotonic_ns())
        ready.set()

    def background():
        for turn in range(3):
            request('bg', -1, turn, first if turn == 0 else None)

    def agent(session):
        if not ready.wait(900):
            raise TimeoutError('background never started')
        deadline_ns = first_bg_ns[0] + session*20_000_000
        time.sleep(max(0, (deadline_ns-time.monotonic_ns())/1e9))
        for turn in range(TURNS):
            if turn:
                time.sleep(5)
            request('agent', session, turn)

    with ThreadPoolExecutor(max_workers=SESSIONS+1) as pool:
        futures = [pool.submit(background)] + [pool.submit(agent, s) for s in range(SESSIONS)]
        for future in futures:
            future.result()
    time.sleep(2)
    (out/'after.prom').write_text(runtime.metric_text())
    save('workload.json', dict(kind='synthetic capacity stress, not production demand', policy=policy,
                              sessions=SESSIONS, turns=TURNS, output=OUTPUT, seed=SEED,
                              cpu_bytes_budget=cpu_bytes, quotas_tokens=shares, tool_gap_s=5,
                              first_release_spacing_ms=20, gpu_bytes_per_rank=GPU_BYTES_PER_RANK))
