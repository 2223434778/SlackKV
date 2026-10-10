"""Matched synthetic screens, reusing native admission without physical pins."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import threading
import time
import traceback
import urllib.request

TOKEN_BYTES = 28672
BLOCK = 16
FAMILY = 'host_prefix_followup_20261010'


def shares(cpu_bytes, sessions):
    if sessions < 1 or cpu_bytes < BLOCK * TOKEN_BYTES * sessions:
        raise ValueError('at least one physical block per session required')
    blocks = cpu_bytes // (BLOCK * TOKEN_BYTES)
    each = blocks // sessions
    return [each * BLOCK] * (sessions - 1) + [(blocks - each * (sessions - 1)) * BLOCK]


def cap(policy, length, quota):
    if policy == 'nohost':
        return 0
    if policy == 'full':
        return length
    if policy == 'fair':
        return min(length, quota)
    raise ValueError('unknown policy')


def budget(root):
    costs = {}
    for family in ('autonomous_fork', 'autonomous_round4', 'host_prefix_cost_20261010', FAMILY):
        rows = [json.loads(f.read_text()) for f in (root / 'results' / family).glob('*/execution.json')]
        if any(r['status'] == 'RUNNING' for r in rows):
            raise RuntimeError('unresolved owned execution')
        field = 'gpu_hours_conservative' if family in ('autonomous_fork', 'autonomous_round4') else 'gpu_hours'
        costs[family] = sum(r[field] for r in rows)
    return costs


def workload(out, runtime, cfg):
    from experiments.autonomous_fork.run import fixed_tokens
    quotas = shares(cfg['cpu_bytes'], cfg['sessions'])
    roots = [fixed_tokens(28672, cfg['seed'] + s * 10007) for s in range(cfg['sessions'])]
    rows, inputs = [], []
    lock = threading.Lock()
    save = lambda name, value: (out / name).write_text(json.dumps(value, indent=2) + '\n')
    (out / 'before.prom').write_text(runtime.metric_text())

    def request(role, session, turn, on_first=None):
        if role == 'agent':
            prompt = roots[session] + fixed_tokens(turn * 64, cfg['seed'] + session + 1200)
            output = 16
            limit = cap(cfg['policy'], len(prompt), quotas[session])
        else:
            prompt = fixed_tokens(8192, cfg['seed'] + 6100 + turn)
            output, limit = 256, 0
        before = runtime.metric_text()
        response = runtime.send(prompt, output, stream=True,
                                params={'max_offload_tokens': limit}, on_first=on_first)
        after = runtime.metric_text()
        if len(response['tokens']) != output:
            raise RuntimeError('prescribed work mismatch')
        row = dict(role=role, session=session, turn=turn, length=len(prompt),
                   prompt_sha256=hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
                   max_offload_tokens=limit, output_count=output, output_tokens=response['tokens'],
                   issue_ns=response['issue_ns'], first_ns=response['first_ns'], done_ns=response['done_ns'],
                   metrics_before=before, metrics_after=after)
        with lock:
            rows.append(row)
            inputs.append(dict(role=role, session=session, turn=turn, prompt=prompt))
            save('requests.json', rows)
            save('inputs.json', inputs)
        print(json.dumps(dict(role=role, session=session, turn=turn,
                              ttft_ms=(row['first_ns']-row['issue_ns'])/1e6)), flush=True)

    if cfg['reset_gpu']:
        for turn in range(6):
            for session in range(cfg['sessions']):
                time.sleep(.2)
                req = urllib.request.Request('http://127.0.0.1:8000/reset_prefix_cache?reset_external=false', data=b'', method='POST')
                with urllib.request.urlopen(req, timeout=20) as response:
                    if response.status != 200:
                        raise RuntimeError('GPU reset rejected')
                ready = threading.Event()
                with ThreadPoolExecutor(max_workers=1) as pool:
                    bg = pool.submit(request, 'bg', -1, turn * cfg['sessions'] + session,
                                     lambda _: ready.set())
                    if not ready.wait(60):
                        raise TimeoutError('BG did not reach decode')
                    request('agent', session, turn)
                    bg.result(timeout=120)
    else:
        ready = threading.Event()

        def background():
            for turn in range(12):
                request('bg', -1, turn, (lambda _: ready.set()) if turn == 0 else None)

        def agent(session):
            if not ready.wait(60):
                raise TimeoutError('BG did not reach decode')
            for turn in range(6):
                if turn:
                    time.sleep(.2)
                request('agent', session, turn)

        with ThreadPoolExecutor(max_workers=cfg['sessions'] + 1) as pool:
            futures = [pool.submit(background)] + [pool.submit(agent, s) for s in range(cfg['sessions'])]
            for future in futures:
                future.result(timeout=1200)
    time.sleep(2)
    (out / 'after.prom').write_text(runtime.metric_text())
    save('workload.json', dict(**cfg, quotas_tokens=quotas, turns=6,
                              kind='synthetic diagnostic' if cfg['reset_gpu'] else 'synthetic closed-loop'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--policy', required=True, choices=('full', 'fair', 'nohost'))
    p.add_argument('--sessions', type=int, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--cpu-mib', type=int, required=True)
    p.add_argument('--gpu-mib', type=int, default=0, help='0: natural85%% allocation')
    p.add_argument('--reset-gpu', action='store_true')
    args = p.parse_args()
    if Path(args.name).name != args.name or args.name in ('.', '..'):
        p.error('unique plain run name required')
    cfg = dict(policy=args.policy, sessions=args.sessions, seed=args.seed,
               cpu_bytes=args.cpu_mib*2**20, gpu_bytes=args.gpu_mib*2**20, reset_gpu=args.reset_gpu)
    shares(cfg['cpu_bytes'], cfg['sessions'])
    root = Path(__file__).resolve().parents[2]
    costs = budget(root)
    spent = sum(costs.values())
    if spent + .5 > 4:
        raise RuntimeError('continuous budget insufficient')
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1', 8000)) == 0:
            raise RuntimeError('occupied port; do not disturb service')
    out = root / 'results' / FAMILY / args.name
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    meta = dict(status='RUNNING', started_unix=time.time(), gpu_count=1, config=cfg,
                budget_before=costs, authorization_gpu_hours_before=spent,
                code_git='67eff77', source=str(args.source),
                runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scheduler_sha256=hashlib.sha256((args.source/'vllm/v1/core/sched/scheduler.py').read_bytes()).hexdigest())
    save = lambda: (out/'execution.json').write_text(json.dumps(meta, indent=2) + '\n')
    save()
    proc = None
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('1710s watchdog')))
    signal.alarm(1710)
    try:
        os.environ.update(VLLM_BATCH_INVARIANT='0', SLACKKV_CONTINUATION_CREDIT='0',
                          SLACKKV_WORKING_SET_EPOCHS='0', SLACKKV_CAPACITY_BACKFILL_LIMIT='0',
                          SLACKKV_RETENTION_CONTROL='0', SLACKKV_RETENTION_PIN='0',
                          SLACKKV_ROUND4_RESOURCE_TRACE='0', SLACKKV_FORK_TRACE='0')
        from experiments.phase4d_a import runtime
        from experiments.autonomous_fork.run import fixed_tokens
        runtime.SRC = args.source
        original = runtime.server_command

        def command(connector, gpu_bytes):
            argv = original(connector, cfg['gpu_bytes'])
            argv.remove('--enforce-eager')
            if not cfg['gpu_bytes']:
                index = argv.index('--kv-cache-memory-bytes')
                del argv[index:index+2]
                argv.extend(('--gpu-memory-utilization', '0.85'))
            index = argv.index('--kv-transfer-config')
            argv[index+1] = json.dumps(dict(kv_connector='OffloadingConnector', kv_role='kv_both',
                kv_connector_extra_config=dict(block_size=BLOCK, cpu_bytes_to_use=cfg['cpu_bytes'])))
            return argv

        runtime.server_command = command
        meta['server_command'] = command(True, cfg['gpu_bytes'])
        save()
        proc, log = runtime.start_server(args.name, gpu_bytes=cfg['gpu_bytes'], out=out)
        runtime.wait_ready(proc)
        runtime.send(fixed_tokens(256, 998), 16, stream=True, params={'max_offload_tokens': 0})
        time.sleep(1)
        workload(out, runtime, cfg)
        meta['status'] = 'COMPLETE'
    except Exception:
        meta['status'] = 'FAILED'
        (out/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        if proc is not None:
            meta['server_exit_code'] = runtime.stop_server(proc, log, out, args.name)
        signal.alarm(0)
        meta['wall_seconds_including_setup_cleanup'] = time.monotonic()-started
        meta['gpu_hours'] = meta['wall_seconds_including_setup_cleanup']/3600
        meta['authorization_total_gpu_hours'] = spent+meta['gpu_hours']
        save()


if __name__ == '__main__':
    main()
