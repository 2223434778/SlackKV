"""Read live JSON or losslessly archived gzip JSON without double counting."""
from __future__ import annotations

import gzip
import json
from pathlib import Path


def read_json(path: Path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(gzip.decompress(Path(str(path) + ".gz").read_bytes()))


def run_paths(directory: Path) -> list[Path]:
    paths = list(directory.glob("runs/*/run.json"))
    paths.extend(p.with_suffix("") for p in directory.glob("runs/*/run.json.gz")
                 if not p.with_suffix("").exists())
    return paths
