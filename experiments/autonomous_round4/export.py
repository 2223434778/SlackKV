"""Compact lossless evidence export; preserve every original on the server."""
import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path


def export(directory: Path, archive_path: Path):
    ledger = [json.loads(p.read_text()) for p in directory.glob("*/execution.json")]
    if not ledger or any(e["status"] == "RUNNING" for e in ledger):
        raise ValueError("ledger must be finalized before export")
    records = []
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.name == "SHA256SUMS.json":
                continue
            raw = path.read_bytes()
            relative = path.relative_to(directory).as_posix()
            records.append({"path": relative, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
            if path.suffix in {".log", ".jsonl"} or path.name in {"run.json", "cold_oracle.json"}:
                relative += ".gz"
                raw = gzip.compress(raw, mtime=0)
            info = tarfile.TarInfo("results/autonomous_round4/" + relative)
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
        raw = (json.dumps(records, indent=2) + "\n").encode()
        info = tarfile.TarInfo("results/autonomous_round4/SHA256SUMS.json")
        info.size = len(raw)
        archive.addfile(info, io.BytesIO(raw))
    return {"files": len(records), "archive_bytes": archive_path.stat().st_size}


def audit(directory: Path):
    records = json.loads((directory / "SHA256SUMS.json").read_text())
    for record in records:
        path = directory / record["path"]
        raw = path.read_bytes() if path.exists() else gzip.decompress(Path(str(path) + ".gz").read_bytes())
        if len(raw) != record["bytes"] or hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError(f'lossless evidence mismatch: {record["path"]}')
    return {"verified_files": len(records)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.directory, args.archive) if args.archive else audit(args.directory)))
