"""Lossless compact evidence; keep original files untouched on the server."""
import argparse
import gzip
import io
import lzma
from pathlib import Path
import tarfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with tarfile.open(args.output, 'w:gz') as archive:
        for file in sorted(args.directory.rglob('*')):
            if not file.is_file():
                continue
            name = file.relative_to(args.directory).as_posix()
            data = file.read_bytes()
            if file.name == 'inputs.json':
                data = lzma.compress(data)
                name += '.xz'
            elif file.name == 'requests.json' or file.suffix == '.log':
                data = gzip.compress(data, mtime=0)
                name += '.gz'
            elif file.suffix not in ('.json', '.prom', '.txt', '.py', '.md'):
                continue
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


if __name__ == '__main__':
    main()
