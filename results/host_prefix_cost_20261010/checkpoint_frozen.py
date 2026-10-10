"""One-path host-overflow mechanism test with a matched decode background."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import threading
import time
import urllib.request

CPU_BYTES = 256 * 2**20
TOKEN_BYTES = 28672
FRONTIER = CPU_BYTES // (16 * TOKEN_BYTES) * 16


def run(out, runtime, mode):
    from experiments.autonomous_fork.run import fixed_tokens
    save = lambda name, obj: (out / name).write_text(json.dumps(obj, indent=2) + '\n')
    records, backgrounds = [], []
    root = fixed_tokens(28672, 773311)
    inputs = []
    before = runtime.metric_text()
    (out / 'before.prom').write_text(before)
    for turn in range(6):
        time.sleep(.2)
        reset = urllib.request.Request('http://127.0.0.1:8000/reset_prefix_cache?reset_external=false', data=b'', method='POST')
        with urllib.request.urlopen(reset, timeout=20) as response:
            if response.status != 200:
                raise RuntimeError('GPU reset request rejected')
        ready = threading.Event()
        bg_prompt = fixed_tokens(8192, 6100 + turn)
        prompt = root + fixed_tokens(turn * 64, 1200)
        inputs.extend((dict(turn=turn, role='agent', prompt=prompt), dict(turn=turn, role='bg', prompt=bg_prompt)))
        save('inputs.json', inputs)
        with ThreadPoolExecutor(max_workers=1) as pool:
            bg = pool.submit(runtime.send, bg_prompt, 256, stream=True,
                             params={'max_offload_tokens': 0}, on_first=lambda _: ready.set())
            if not ready.wait(timeout=60):
                raise TimeoutError('background failed to reach decode')
            params = {'max_offload_tokens': len(prompt) if mode == 'full' else FRONTIER}
            response = runtime.send(prompt, 16, stream=True, params=params)
            background = bg.result(timeout=120)
        for role, result, expected, full_input in (
                ('agent', response, 16, prompt), ('bg', background, 256, bg_prompt)):
            if len(result['tokens']) != expected:
                raise RuntimeError('fixed model work mismatch')
            row = dict(turn=turn, role=role,
                       prompt_sha256=hashlib.sha256(json.dumps(full_input).encode()).hexdigest(),
                       length=len(full_input), output_count=len(result['tokens']),
                       ttft_ms=(result['first_ns'] - result['issue_ns']) / 1e6,
                       e2e_ms=(result['done_ns'] - result['issue_ns']) / 1e6,
                       issue_ns=result['issue_ns'], first_ns=result['first_ns'], done_ns=result['done_ns'],
                       output_tokens=result['tokens'])
            (records if role == 'agent' else backgrounds).append(row)
        save('requests.json', records)
        save('backgrounds.json', backgrounds)
        print(json.dumps(dict(turn=turn, mode=mode, ttft_ms=records[-1]['ttft_ms'], bg_ms=backgrounds[-1]['e2e_ms'])), flush=True)
    # Allow native logger and store completion counters to publish.
    time.sleep(2)
    after = runtime.metric_text()
    (out / 'after.prom').write_text(after)
    (out / 'checkpoint_config.json').write_text(json.dumps(dict(
        cpu_bytes=CPU_BYTES, bytes_per_token=TOKEN_BYTES, prefix_cap_tokens=FRONTIER,
        iterations=6, explicit_gpu_resets=6, background_output_tokens=1536,
        agent_output_tokens=96, workload='synthetic host-overflow forced-GPU-cold diagnostic',
        qualification='single growing path; not representative production or algorithm fairness proof'), indent=2) + '\n')
