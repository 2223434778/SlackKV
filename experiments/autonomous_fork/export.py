"""Lossless compact export; server originals are read, never modified."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    executions = [json.loads(p.read_text()) for p in args.directory.glob("*/execution.json")]
    if not executions or any(e["status"] == "RUNNING" for e in executions):
        raise ValueError("export requires a finalized ledger")
    records = []
    with tarfile.open(args.archive, "w:gz") as archive:
        for path in sorted(args.directory.rglob("*")):
            if not path.is_file() or path.name in {"SHA256SUMS.json", "protocol.md", "summary.md", "README.md", "commands.sh", ".gitattributes"}:
                continue
            raw = path.read_bytes()
            relative = path.relative_to(args.directory).as_posix()
            records.append(dict(path=relative, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
            if path.suffix in {".log", ".jsonl"} or path.name in {"run.json", "requests.json", "cold_oracle.json"}:
                relative += ".gz"
                raw = gzip.compress(raw, mtime=0)
            info = tarfile.TarInfo("results/autonomous_fork/" + relative)
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
        raw = (json.dumps(records, indent=2) + "\n").encode()
        info = tarfile.TarInfo("results/autonomous_fork/SHA256SUMS.json")
        info.size = len(raw)
        archive.addfile(info, io.BytesIO(raw))
    print(json.dumps(dict(files=len(records), archive_bytes=args.archive.stat().st_size)))


if __name__ == "__main__":
    main()
