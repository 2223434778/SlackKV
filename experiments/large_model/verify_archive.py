"""Verify decompressed raw-byte identity without accessing a GPU or model."""
import argparse
import gzip
import hashlib
import json
import lzma
from pathlib import Path, PurePosixPath

def verify(directory):
    manifest=json.loads((directory/'raw_manifest.json').read_text())
    total=0
    for name,record in manifest.items():
        path,stored=PurePosixPath(name),PurePosixPath(record['stored'])
        if any(p.is_absolute() or '..' in p.parts for p in (path,stored)):
            raise ValueError('unexpected archive path')
        # Use the manifest-selected immutable copy even if a stale local raw
        # snapshot from an earlier archive remains alongside it.
        raw=(directory/stored).read_bytes()
        if stored.suffix=='.xz':
            raw=lzma.decompress(raw)
        elif stored.suffix=='.gz':
            raw=gzip.decompress(raw)
        if len(raw)!=record['bytes'] or hashlib.sha256(raw).hexdigest()!=record['sha256']:
            raise ValueError('raw-byte identity mismatch: '+name)
        total+=len(raw)
    return dict(validated_files=len(manifest),uncompressed_bytes=total)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(verify(args.directory),indent=2))


if __name__=='__main__':
    main()
