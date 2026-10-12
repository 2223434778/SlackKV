"""Bounded public-model download, validating official LFS or Git blob hashes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import threading
import time
import traceback


def verify(path, record):
    size = path.stat().st_size
    if size != record['bytes']:
        return False
    digest = hashlib.sha256() if record['sha256'] else hashlib.sha1()
    if not record['sha256']:
        digest.update(f'blob {size}\0'.encode())
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            digest.update(block)
    return digest.hexdigest() == (record['sha256'] or record['git_blob_sha1'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--endpoint', choices=('https://huggingface.co', 'https://hf-mirror.com'), required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=False)
    total = sum(row['bytes'] for row in manifest['files'])
    if shutil.disk_usage(args.model_dir).free < total + 20 * 2**30:
        raise RuntimeError('insufficient research-disk space')
    start = time.monotonic()
    result = dict(status='RUNNING', model=manifest['model'], revision=manifest['revision'],
                  endpoint=args.endpoint, gpu_hours=0, gpu_count=0, verified_files=[],
                  workers=2, started_unix=time.time(), runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  kind='CPU/network acquisition; no GPUs reserved or model process started')
    lock = threading.Lock()
    save = lambda: (args.out / 'acquisition.json').write_text(json.dumps(result, indent=2) + '\n')
    save()
    def download(row):
            if Path(row['name']).name != row['name']:
                raise ValueError('unexpected model path')
            target = args.model_dir / row['name']
            if target.exists() and verify(target, row):
                with lock:
                    result['verified_files'].append(row['name'])
                    save()
                return
            temporary = target.with_suffix(target.suffix + '.download')
            url = f"{args.endpoint}/{manifest['model']}/resolve/{manifest['revision']}/{row['name']}"
            remaining = int(1800 - (time.monotonic() - start))
            if remaining <= 0:
                raise TimeoutError('30min acquisition deadline; preserve partial files')
            command = ['curl', '-L', '--fail', '--retry', '2', '--connect-timeout', '15',
                       '--max-time', str(remaining), '--continue-at', '-', '--output', str(temporary), url]
            # curl progress goes to an owned log; URL parameters/credentials are
            # not extracted or logged. Files must match official published hashes.
            with (args.out / (row['name'] + '.download.log')).open('a') as log:
                process = subprocess.run(command, stdout=log, stderr=log, timeout=remaining + 5)
            if process.returncode != 0:
                raise RuntimeError(f"download failed {row['name']}, curl exit {process.returncode}")
            if not verify(temporary, row):
                raise ValueError('official hash mismatch: ' + row['name'])
            temporary.rename(target)
            with lock:
                result['verified_files'].append(row['name'])
                save()
            print(json.dumps(dict(file=row['name'], bytes=row['bytes'], status='VERIFIED')), flush=True)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(download, manifest['files']))
        result['verified_files'].sort()
        result['status'] = 'PASS'
    except BaseException:
        result['status'] = 'FAIL'
        (args.out / 'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        result['wall_seconds'] = time.monotonic() - start
        save()


if __name__ == '__main__':
    main()
