"""Acquire official pinned metadata only, never credentials or model weights."""
import argparse
import json
from pathlib import Path
from urllib.request import urlopen

MODEL = 'Qwen/Qwen2.5-32B-Instruct'
REVISION = '5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    url = f'https://huggingface.co/api/models/{MODEL}/revision/{REVISION}?blobs=true'
    with urlopen(url, timeout=30) as response:
        raw = response.read(2**20)
    metadata = json.loads(raw)
    if metadata['sha'] != REVISION:
        raise ValueError('official revision mismatch')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'official_model_metadata.json').write_bytes(raw)
    files = []
    small = {'config.json', 'generation_config.json', 'model.safetensors.index.json',
             'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt'}
    for record in metadata['siblings']:
        name = record['rfilename']
        if name in small or name.endswith('.safetensors'):
            lfs = record.get('lfs')
            files.append(dict(name=name, bytes=record['size'],
                              sha256=lfs['sha256'] if lfs else None,
                              git_blob_sha1=record['blobId']))
    if not files or not all(row['sha256'] for row in files if row['name'].endswith('.safetensors')):
        raise ValueError('missing official weight hashes')
    manifest = dict(model=MODEL, revision=REVISION, official_url=url, files=files)
    (args.output / 'model_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(revision=REVISION, files=len(files), bytes=sum(row['bytes'] for row in files))))


if __name__ == '__main__':
    main()
