"""Fetch official, revision-pinned Qwen metadata only; never model weights."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


def get(url):
    with urlopen(url, timeout=30) as response:
        raw = response.read(4 * 1024 * 1024 + 1)
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError('metadata exceeds size limit')
    return raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    model = 'Qwen/Qwen2.5-32B-Instruct'
    api = f'https://huggingface.co/api/models/{model}'
    revision = json.loads(get(api))['sha']
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {'model': model, 'revision': revision, 'revision_lookup_url': api, 'files': {}}
    for name in ('config.json', 'model.safetensors.index.json'):
        url = f'https://huggingface.co/{model}/resolve/{revision}/{name}'
        raw = get(url)
        (args.output / name).write_bytes(raw)
        manifest['files'][name] = {'url': url, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
