"""One owned 32 GiB host allocation; conservatively charge one GPU for all time."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def memory():
    result = {}
    for path, fields in (('/proc/meminfo', ('MemAvailable', 'MemFree', 'SwapFree')),
                         ('/proc/self/status', ('VmRSS', 'VmLck', 'VmPin'))):
        for line in Path(path).read_text().splitlines():
            key, value = line.split(':', 1)
            if key in fields:
                result[key + '_bytes'] = int(value.split()[0]) * 1024
    return result


def budget(root):
    families = ('autonomous_fork', 'autonomous_round4', 'host_prefix_cost_20261010',
                'host_prefix_followup_20261010', 'deployment_opportunity_20261011', 'large_model_20261012')
    costs = {}
    for family in families:
        executions = [json.loads(p.read_text()) for p in (root / 'results' / family).glob('*/execution.json')]
        if not executions and family in families[:4]:
            raise RuntimeError('historical budget ledger unavailable: ' + family)
        if any(row['status'] == 'RUNNING' for row in executions):
            raise RuntimeError('unresolved owned run')
        key = 'gpu_hours_conservative' if family in families[:2] else 'gpu_hours'
        costs[family] = sum(row[key] for row in executions)
    return costs


def selected_gpu_idle(text, gpu):
    rows = text.splitlines()
    if gpu < 0 or gpu >= len(rows):
        raise ValueError('selected GPU missing')
    used, utilization = map(int, rows[gpu].split(','))
    return used <= 64 and utilization == 0


def allocate():
    baseline = memory()
    size = 32 * 2**30
    if baseline['MemAvailable_bytes'] < size + 48 * 2**30:
        raise RuntimeError('host headroom changed before worker import')
    import torch
    torch.set_num_threads(4)
    before_allocate = memory()
    t0 = time.monotonic()
    tensor = torch.zeros(size, dtype=torch.uint8, device='cpu', pin_memory=True)
    allocation_s = time.monotonic() - t0
    if not tensor.is_pinned() or tensor.numel() != size:
        raise AssertionError('allocation is not the prescribed pinned bytes')
    if tensor[:1024].sum().item() != 0 or tensor[-1024:].sum().item() != 0:
        raise AssertionError('sampled zero initialization failed')
    allocated = memory()
    del tensor
    gc.collect()
    print(json.dumps(dict(torch=torch.__version__, baseline=baseline, before_allocate=before_allocate,
                          allocated=allocated, after_del_gc=memory(), allocation_s=allocation_s,
                          is_pinned=True, sampled_zero_check=True,
                          caveats=['VmLck/VmPin may not expose CUDA registration; zero does not mean unpinned.',
                                   'Pinned allocator caching may retain RSS until process exit.',
                                   'One allocation does not establish acceptable serving overhead or production budget.'])))


def main():
    if sys.argv[1:] == ['--allocation-worker']:
        allocate()
        return
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in ('.', '..'):
        raise ValueError('plain unique name required')
    costs = budget(args.root)
    if sum(costs.values()) + 61 / 3600 > 4:
        raise RuntimeError('insufficient continuous budget')
    gpu_before = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                                          '--format=csv,noheader,nounits'], text=True)
    if not selected_gpu_idle(gpu_before, args.gpu):
        raise RuntimeError('GPU not idle; do not disturb other services')
    size = 32 * 2**30
    baseline = memory()
    if baseline['MemAvailable_bytes'] < size + 48 * 2**30:
        raise RuntimeError('insufficient safe host headroom')
    out = args.root / 'results/deployment_opportunity_20261011' / args.name
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    metadata = dict(status='RUNNING', gpu_count=1, gpu_hours_before=sum(costs.values()),
                    gpu_index=args.gpu, gpu_before=gpu_before,
                    budget_families=costs, started_unix=time.time(), bytes=size,
                    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    kind='allocation diagnostic, no model or GPU tensor')
    save = lambda name, value: (out / name).write_text(json.dumps(value, indent=2) + '\n')
    save('execution.json', metadata)
    try:
        # Parent waits for worker exit, including driver cleanup. subprocess.run
        # kills and waits for this owned worker on timeout, even during C++ calls.
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--allocation-worker'],
                                env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(args.gpu)},
                                capture_output=True, text=True, timeout=60)
        (out / 'worker.stdout.txt').write_text(result.stdout)
        (out / 'worker.stderr.txt').write_text(result.stderr)
        if result.returncode != 0:
            raise RuntimeError(f'owned allocation worker exit {result.returncode}')
        save('allocation.json', json.loads(result.stdout))
        save('after_worker_exit.json', memory())
        metadata['status'] = 'PASS'
    except BaseException:
        (out / 'failure.txt').write_text(traceback.format_exc())
        metadata['status'] = 'FAIL'
        raise
    finally:
        metadata.update(elapsed_s=time.monotonic() - started, finished_unix=time.time())
        metadata['gpu_hours'] = (metadata['elapsed_s'] + 1) / 3600
        metadata['exit_margin_s'] = 1
        save('execution.json', metadata)
        print(json.dumps(metadata), flush=True)


if __name__ == '__main__':
    main()
