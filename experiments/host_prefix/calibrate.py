"""Owned GPU cost calibration, not an algorithm performance comparison."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import time
import traceback
import urllib.request


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--mode', choices=('calibration', 'full', 'frontier', 'oracle'), default='calibration')
    args = p.parse_args()
    if Path(args.name).name != args.name or args.name in ('.', '..'):
        p.error('unique plain run name required')
    root = Path(__file__).resolve().parents[2]
    ledger = root / 'results/host_prefix_cost_20261010'
    previous = [json.loads(f.read_text()) for f in ledger.glob('*/execution.json')]
    if any(r['status'] == 'RUNNING' for r in previous):
        raise RuntimeError('previous owned run unresolved')
    spent = sum(r['gpu_hours'] for r in previous)
    # Conservatively carry all work since the latest explicit Round3 authorization.
    # A new directory is not a fresh resource allowance.
    prior_scope = []
    for family in ('autonomous_fork', 'autonomous_round4'):
        prior_scope.extend(json.loads(f.read_text()) for f in (root / 'results' / family).glob('*/execution.json'))
    if any(r['status'] == 'RUNNING' for r in prior_scope):
        raise RuntimeError('historical owned run unresolved')
    prior_hours = sum(r['gpu_hours_conservative'] for r in prior_scope)
    if prior_hours + spent + .5 > 4:
        raise RuntimeError('insufficient authorized GPU budget')
    with socket.socket() as s:
        if s.connect_ex(('127.0.0.1', 8000)) == 0:
            raise RuntimeError('port occupied; do not disturb service')
    out = ledger / args.name
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    meta = dict(status='RUNNING', mode=args.mode, batch_invariant=args.mode == 'oracle', gpu_count=1, gpu_hours_before=spent,
                prior_scope_gpu_hours=prior_hours, authorization_gpu_hours_before=prior_hours+spent,
                started_unix=time.time(), source=str(args.source), baseline_git='685ac5c',
                runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scheduler_sha256=hashlib.sha256((args.source / 'vllm/v1/core/sched/scheduler.py').read_bytes()).hexdigest())
    save = lambda name, value: (out / name).write_text(json.dumps(value, indent=2) + '\n')
    save('execution.json', meta)
    proc = None
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('1710s watchdog')))
    signal.alarm(1710)
    try:
        os.environ.update(VLLM_BATCH_INVARIANT=str(int(args.mode == 'oracle')), SLACKKV_CONTINUATION_CREDIT='0',
                          SLACKKV_WORKING_SET_EPOCHS='0', SLACKKV_CAPACITY_BACKFILL_LIMIT='0',
                          SLACKKV_RETENTION_CONTROL='0', SLACKKV_RETENTION_PIN='0',
                          SLACKKV_ROUND4_RESOURCE_TRACE='0', SLACKKV_FORK_TRACE='0')
        from experiments.phase4d_a import runtime
        from experiments.autonomous_fork.run import fixed_tokens
        runtime.SRC = args.source
        original = runtime.server_command

        def command(connector, gpu_bytes):
            argv = original(connector, gpu_bytes)
            argv.remove('--enforce-eager')
            index = argv.index('--kv-cache-memory-bytes')
            del argv[index:index+2]
            argv.extend(('--gpu-memory-utilization', '0.85'))
            if args.mode != 'calibration':
                from experiments.host_prefix.checkpoint import CPU_BYTES
                index = argv.index('--kv-transfer-config')
                argv[index+1] = json.dumps(dict(kv_connector='OffloadingConnector', kv_role='kv_both',
                    kv_connector_extra_config=dict(block_size=16, cpu_bytes_to_use=CPU_BYTES)))
            return argv

        runtime.server_command = command
        meta['server_command'] = command(True, 0)
        save('execution.json', meta)
        proc, log = runtime.start_server(args.name, gpu_bytes=0, out=out)
        runtime.wait_ready(proc)
        runtime.send(fixed_tokens(256, 998), 16, stream=True, params={'max_offload_tokens': 0})
        if args.mode == 'oracle':
            from experiments.host_prefix.oracle import run
            run(out, runtime)
            meta['status'] = 'COMPLETE'
            return
        if args.mode != 'calibration':
            from experiments.host_prefix.checkpoint import run
            run(out, runtime, args.mode)
            meta['status'] = 'COMPLETE'
            return
        records = []
        # The same full prompt is used under every prefix state within a length/rep.
        for length in (8192, 16384, 28672):
            for rep in range(3):
                prompt = fixed_tokens(length, 8000 + length + rep)
                for fraction in (0, .5, .875, 1):
                    time.sleep(.2)
                    reset = urllib.request.Request('http://127.0.0.1:8000/reset_prefix_cache?reset_external=false', data=b'', method='POST')
                    with urllib.request.urlopen(reset, timeout=20) as r:
                        body = r.read().decode()
                        if r.status != 200:
                            raise RuntimeError(f'GPU reset failed: {r.status} {body}')
                    prefix = int(length * fraction)
                    params = {'max_offload_tokens': 0}
                    if prefix:
                        runtime.send(prompt[:prefix], 1, stream=True, params=params)
                    before = runtime.metric_text()
                    response = runtime.send(prompt, 16, stream=True, params=params)
                    after = runtime.metric_text()
                    if len(response['tokens']) != 16:
                        raise RuntimeError('model work mismatch')
                    row = dict(length=length, rep=rep, prefix=prefix,
                               prompt_sha256=hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
                               ttft_ms=(response['first_ns'] - response['issue_ns']) / 1e6,
                               e2e_ms=(response['done_ns'] - response['issue_ns']) / 1e6,
                               output_count=len(response['tokens']), output_tokens=response['tokens'],
                               metrics_before=before, metrics_after=after)
                    records.append(row)
                    save('requests.json', records)
                    print(json.dumps({k: row[k] for k in ('length', 'rep', 'prefix', 'ttft_ms')}), flush=True)
        meta['status'] = 'COMPLETE'
    except Exception:
        meta['status'] = 'FAILED'
        (out / 'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        if proc is not None:
            meta['server_exit_code'] = runtime.stop_server(proc, log, out, args.name)
        signal.alarm(0)
        meta['wall_seconds_including_setup_cleanup'] = time.monotonic() - start
        meta['gpu_hours'] = meta['wall_seconds_including_setup_cleanup'] / 3600
        meta['total_gpu_hours'] = spent + meta['gpu_hours']
        meta['authorization_total_gpu_hours'] = prior_hours + meta['total_gpu_hours']
        save('execution.json', meta)


if __name__ == '__main__':
    main()
