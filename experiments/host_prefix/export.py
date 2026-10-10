"""Compact copies only; leave original server evidence intact."""
import argparse
import gzip
import io
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
            # Original large JSON/log retained server-side; deterministic compact copy.
            if file.name in ('requests.json', 'inputs.json', 'backgrounds.json', 'oracle.json') or file.suffix == '.log':
                data = gzip.compress(file.read_bytes(), mtime=0)
                name += '.gz'
            elif file.suffix in ('.json', '.prom', '.txt', '.py') or file.name.endswith('.py.gz'):
                data = file.read_bytes()
            else:
                continue
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


if __name__ == '__main__':
    main()
