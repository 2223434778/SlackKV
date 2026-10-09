"""Lossless export without removing or rewriting server originals."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    records = []
    with tarfile.open(args.archive, "w:gz") as archive:
        for path in sorted(args.directory.rglob("*")):
            if not path.is_file() or path.name == "SHA256SUMS.json":
                continue
            raw = path.read_bytes()
            relative = path.relative_to(args.directory)
            name = "results/autonomous_capacity/" + relative.as_posix()
            records.append({"path": relative.as_posix(), "bytes": len(raw),
                            "sha256": hashlib.sha256(raw).hexdigest()})
            if path.suffix in {".log", ".jsonl"} or path.name in {"run.json", "cold_oracle.json"}:
                name += ".gz"
                raw = gzip.compress(raw, mtime=0)
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
        raw = (json.dumps(records, indent=2) + "\n").encode()
        info = tarfile.TarInfo("results/autonomous_capacity/SHA256SUMS.json")
        info.size = len(raw)
        archive.addfile(info, io.BytesIO(raw))
    print(json.dumps({"files": len(records), "archive_bytes": args.archive.stat().st_size}))


if __name__ == "__main__":
    main()
