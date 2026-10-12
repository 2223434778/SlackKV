"""Owned compiled TP2 feasibility screen with conservative continuous accounting."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
import traceback
import urllib.request

from experiments.deployment_opportunity.pin_probe import budget as historical_budget, memory

FAMILY = 'large_model_20261012'
TOKEN_BYTES = 262144


def group_memory(pgid):
    rows = []
    for path in Path('/proc').iterdir():
        if not path.name.isdecimal():
            continue
        try:
            fields = (path/'stat').read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) != pgid or fields[0] == 'Z':
                continue
            record = dict(pid=int(path.name), state=fields[0])
            for line in (path/'status').read_text().splitlines():
                key, value = line.split(':', 1)
                if key in ('VmRSS', 'VmLck', 'VmPin'):
                    record[key + '_bytes'] = int(value.split()[0]) * 1024
            rows.append(record)
        except (OSError, IndexError, ValueError):
            continue
    return rows


def budget(root):
    costs = historical_budget(root)
    rows = [json.loads(p.read_text()) for p in (root / 'results' / FAMILY).glob('*/execution.json')]
    if any(row['status'] == 'RUNNING' for row in rows):
        raise RuntimeError('unresolved large model execution')
    costs[FAMILY] = sum(row['gpu_hours'] for row in rows)
    return costs


def command(model, cpu_bytes, gpu_bytes=None, connector=True):
    env_root = Path('/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix')
    argv = [str(env_root / 'bin/vllm'), 'serve', str(model), '--dtype', 'float16',
            '--tensor-parallel-size', '2', '--enable-prefix-caching',
            '--gpu-memory-utilization', '0.85', '--max-model-len', '32768',
            '--max-num-seqs', '16', '--max-num-batched-tokens', '8192',
            '--host', '127.0.0.1', '--port', '8000']
    if gpu_bytes is not None:
        argv += ['--kv-cache-memory-bytes', str(gpu_bytes)]
    if connector:
        argv += ['--kv-transfer-config', json.dumps(dict(kv_connector='OffloadingConnector', kv_role='kv_both',
                              kv_connector_extra_config=dict(block_size=16, cpu_bytes_to_use=cpu_bytes)))]
    return argv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--acquisition', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--oracle', action='store_true')
    parser.add_argument('--screen-policy', choices=('nohost', 'full', 'fair'))
    parser.add_argument('--authorized-gpu-hours', type=float, choices=(4., 4.5), default=4.)
    args = parser.parse_args()
    if args.oracle and args.screen_policy:
        raise ValueError('correctness and performance execution modes are separate')
    if Path(args.name).name != args.name or args.name in ('.', '..'):
        raise ValueError('unique plain name required')
    costs = budget(args.root)
    wall_limit = min(1800, int((args.authorized_gpu_hours-sum(costs.values()))*1800)-2)
    if wall_limit < (1200 if args.screen_policy else 400):
        raise RuntimeError('insufficient continuous budget for planned run')
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1', 8000)) == 0:
            raise RuntimeError('occupied port; do not disturb service')
    gpu_before = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                                          '--format=csv,noheader,nounits'], text=True)
    if len(gpu_before.splitlines()) != 2 or any(int(line.split(',')[0]) > 512 or int(line.split(',')[1]) != 0
                                              for line in gpu_before.splitlines()):
        raise RuntimeError('GPU occupied beyond known desktop allowance')
    acquisition = json.loads(args.acquisition.read_text())
    if acquisition['status'] != 'PASS' or len(acquisition['verified_files']) != 24:
        raise RuntimeError('model acquisition not validated')
    out = args.root / 'results' / FAMILY / args.name
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    cpu_bytes = (64 if args.screen_policy else 32) * 2**30
    gpu_bytes = 70000 * 131072 if args.screen_policy else None
    argv = command(args.model, cpu_bytes, gpu_bytes, args.screen_policy != 'nohost')
    metadata = dict(status='RUNNING', gpu_count=2, started_unix=time.time(), budget_before=costs,
                    parent_and_system_memory_before=memory(),
                    acquisition=str(args.acquisition), model=str(args.model), source=str(args.source),
                    mode='oracle' if args.oracle else args.screen_policy or 'feasibility',
                    authorized_gpu_hours=args.authorized_gpu_hours,
                    batch_invariant=args.oracle, wall_limit_seconds=wall_limit,
                    authorization_gpu_hours_before=sum(costs.values()), gpu_before=gpu_before,
                    server_command=argv, cpu_bytes=cpu_bytes,
                    kind='matched synthetic screen' if args.screen_policy else 'correctness oracle' if args.oracle else 'feasibility, not matched speedup',
                    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    scheduler_sha256=hashlib.sha256((args.source/'vllm/v1/core/sched/scheduler.py').read_bytes()).hexdigest())
    save = lambda name, value: (out/name).write_text(json.dumps(value, indent=2) + '\n')
    save('execution.json', metadata)
    proc = None
    log = None
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('bounded workload watchdog; cleanup reserve160s')))
    signal.alarm(wall_limit-160)
    try:
        env_root = Path('/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix')
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '0,1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
               'VLLM_SERVER_DEV_MODE': '1', 'PYTHONPATH': f'{args.source}:{args.root}',
               'PATH': f'{env_root}/bin:' + os.environ.get('PATH', ''),
               'LD_LIBRARY_PATH': f'{env_root}/targets/x86_64-linux/lib:{env_root}/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:{env_root}/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:' + os.environ.get('LD_LIBRARY_PATH', ''),
               'VLLM_BATCH_INVARIANT': str(int(args.oracle))}
        for key in ('SLACKKV_CONTINUATION_CREDIT', 'SLACKKV_WORKING_SET_EPOCHS', 'SLACKKV_CAPACITY_BACKFILL_LIMIT',
                    'SLACKKV_RETENTION_CONTROL', 'SLACKKV_RETENTION_PIN', 'SLACKKV_ROUND4_RESOURCE_TRACE', 'SLACKKV_FORK_TRACE'):
            env[key] = '0'
        log = (out/'server.log').open('w')
        proc = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        metadata['owned_pid'] = proc.pid
        save('execution.json', metadata)
        from experiments.phase4d_a import runtime
        from experiments.autonomous_fork.run import fixed_tokens
        runtime.MODEL = args.model
        runtime.wait_ready(proc, timeout=1100)
        metadata['ready_seconds'] = time.monotonic() - started
        metadata['parent_and_system_memory_ready'] = memory()
        metadata['owned_group_rss_ready'] = group_memory(proc.pid)
        save('execution.json', metadata)
        if args.screen_policy:
            from experiments.large_model.screen import workload
            runtime.send(fixed_tokens(256, 71099), 16, stream=True,
                         params=None if args.screen_policy == 'nohost' else {'max_offload_tokens': 0})
            time.sleep(2)
            workload(out, runtime, args.screen_policy, cpu_bytes)
            pools = re.findall(r'GPU KV cache size: ([\d,]+) tokens', (out/'server.log').read_text(errors='replace'))
            metadata['actual_gpu_pool_tokens'] = [int(x.replace(',', '')) for x in pools]
            if not pools or set(metadata['actual_gpu_pool_tokens']) != {70000}:
                raise RuntimeError('matched actual pool differs')
            metadata['status'] = 'COMPLETE'
            return
        records = []
        cases = [(256, 331144, 'short', 0), (28672, 331145, 'long', 0), (28672, 331145, 'gpu_reuse', 0),
                 (8192, 331146, 'cpu_store', 8192), (8192, 331146, 'cpu_resume', 8192)]
        inputs = []
        oracle_tokenizer = None
        if args.oracle:
            from experiments.large_model.oracle import inputs as oracle_inputs
            prepared, oracle_tokenizer = oracle_inputs(args.model)
            cases = [(28672, key, f'{stage}_{key}', 12480 if stage != 'independent_cold' else 0)
                     for stage in ('store', 'resume', 'independent_cold') for key in ('a', 'b')]
        for length, seed, label, store_limit in cases:
            if label == 'cpu_resume' or args.oracle:
                request = urllib.request.Request('http://127.0.0.1:8000/reset_prefix_cache?reset_external=false',
                                                 data=b'', method='POST')
                with urllib.request.urlopen(request, timeout=20) as response:
                    if response.status != 200:
                        raise RuntimeError('diagnostic reset rejected')
            prompt = prepared[seed]['prompt'] if args.oracle else fixed_tokens(length, seed)
            before = runtime.metric_text()
            if args.oracle:
                from experiments.large_model.oracle import send as oracle_send
                salt = ('independent_' if label.startswith('independent') else 'stored_') + seed
                response = oracle_send(runtime, prompt, store_limit, salt)
                response['decoded'] = oracle_tokenizer.decode(response['tokens'])
                response['expected_marker'] = prepared[seed]['marker']
            else:
                salt = None
                response = runtime.send(prompt, 16, stream=True, params={'max_offload_tokens': store_limit})
            if len(response['tokens']) != 16:
                raise RuntimeError('prescribed output work mismatch')
            time.sleep(2)
            records.append(dict(label=label, length=length, prompt_sha256=hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
                                max_offload_tokens=store_limit,
                                cache_salt=salt,
                                response=response, before=before, after=runtime.metric_text()))
            save('requests.json', records)
            inputs.append(dict(label=label, prompt=prompt))
            save('inputs.json', inputs)
        text = (out/'server.log').read_text(errors='replace')
        pools = re.findall(r'GPU KV cache size: ([\d,]+) tokens', text)
        if not pools:
            raise RuntimeError('actual pool unobserved')
        metadata['actual_gpu_pool_tokens'] = [int(x.replace(',', '')) for x in pools]
        (out/'after.prom').write_text(runtime.metric_text())
        metadata['status'] = 'COMPLETE'
    except BaseException:
        metadata['status'] = 'FAILED'
        (out/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        signal.alarm(0)
        if proc is not None:
            # Signal only our owned session, including TP children, never GPU users.
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=15)
            metadata['server_exit_code'] = proc.returncode
            until = time.monotonic() + 10
            while group_memory(proc.pid) and time.monotonic() < until:
                time.sleep(.2)
            if group_memory(proc.pid):
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                time.sleep(1)
            metadata['owned_live_pids_after_cleanup'] = group_memory(proc.pid)
            if metadata['owned_live_pids_after_cleanup']:
                metadata['status'] = 'RUNNING'
                metadata['cleanup_error'] = 'owned workers remain; keep budget unresolved'
        if log is not None:
            log.close()
        # Process exit precedes asynchronous CUDA/host-registration teardown.
        # Count until device memory returns, rather than just parent termination.
        release_deadline = time.monotonic() + 60
        while True:
            gpu_after = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                                                '--format=csv,noheader,nounits'], text=True)
            released = all(int(line.split(',')[0]) <= 512 for line in gpu_after.splitlines())
            if released or time.monotonic() >= release_deadline:
                break
            time.sleep(1)
        metadata['gpu_after'] = gpu_after
        metadata['gpu_release_verified'] = released
        if not released:
            metadata['status'] = 'RUNNING'
            metadata['cleanup_error'] = 'device teardown not verified; budget unresolved'
        metadata['parent_and_system_memory_after'] = memory()
        metadata['wall_seconds_including_setup_cleanup'] = time.monotonic() - started
        metadata['gpu_hours'] = 2 * (metadata['wall_seconds_including_setup_cleanup'] + 1) / 3600
        metadata['authorization_total_gpu_hours'] = sum(costs.values()) + metadata['gpu_hours']
        metadata['finished_unix'] = time.time()
        save('execution.json', metadata)
        print(json.dumps(metadata), flush=True)


if __name__ == '__main__':
    main()
