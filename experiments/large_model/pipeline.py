"""One bounded acquisition retry followed by an owned feasibility run."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--checkout', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    family = root/'results/large_model_20261012'
    state_path = family/'pipeline.json'
    if state_path.exists():
        raise RuntimeError('pipeline already exists; inspect before restarting')
    state = dict(status='WAITING_ACQUISITION', pid=os.getpid(), started_unix=time.time(),
                 gpu_reserved_while_waiting=0)
    save = lambda: state_path.write_text(json.dumps(state, indent=2) + '\n')
    save()
    try:
        acquisition = family/'acquire2/acquisition.json'
        deadline = time.monotonic() + 2100
        while json.loads(acquisition.read_text())['status'] == 'RUNNING':
            if time.monotonic() > deadline:
                raise TimeoutError('acquisition wait deadline')
            time.sleep(5)
        if json.loads(acquisition.read_text())['status'] != 'PASS':
            state['status'] = 'RESUMING_ACQUISITION'
            save()
            argv = ['/usr/bin/python3', str(root/'.tmp/large_model_acquire_20261012.py'),
                    '--manifest', str(root/'.tmp/large_model_manifest_20261012.json'),
                    '--model-dir', '/media/a6000/D/ZhaoBinHe/slackkv-models/qwen2p5-32b-5ede1c97',
                    '--out', str(family/'acquire3'), '--endpoint', 'https://hf-mirror.com']
            with (family/'acquire3-controller.log').open('w') as log:
                code = subprocess.call(argv, stdout=log, stderr=subprocess.STDOUT)
            acquisition = family/'acquire3/acquisition.json'
            if code != 0 or json.loads(acquisition.read_text())['status'] != 'PASS':
                raise RuntimeError('bounded acquisition retry failed; no GPUs started')
        state.update(status='STARTING_FEASIBILITY', acquisition=str(acquisition))
        save()
        argv = ['/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python',
                '-m', 'experiments.large_model.run', '--root', str(root),
                '--source', '/media/a6000/D/ZhaoBinHe/slackkv-autonomous-round4-source',
                '--model', '/media/a6000/D/ZhaoBinHe/slackkv-models/qwen2p5-32b-5ede1c97',
                '--acquisition', str(acquisition), '--name', 'feasibility1']
        env = {**os.environ, 'PYTHONPATH': str(args.checkout)}
        with (family/'feasibility1-controller.log').open('w') as log:
            code = subprocess.call(argv, env=env, cwd=args.checkout, stdout=log, stderr=subprocess.STDOUT)
        execution = json.loads((family/'feasibility1/execution.json').read_text())
        state.update(status='FEASIBILITY_COMPLETE' if code == 0 and execution['status']=='COMPLETE' else 'FEASIBILITY_FAILED',
                     gpu_hours=execution['gpu_hours'])
    except BaseException:
        state['status'] = 'FAILED'
        (family/'pipeline-failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        state['finished_unix'] = time.time()
        save()


if __name__ == '__main__':
    main()
