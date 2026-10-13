"""Lossless compact copy; original remote results remain untouched."""
import gzip
import hashlib
import json
from pathlib import Path
import zipfile

root = Path('/media/a6000/D/ZhaoBinHe/SlackKV')
out = root/'results/system_m1_cost'
targets = [out]+[root/'results/system_m1_smoke'/n for n in
    ('m1_cost_ref1','m1_cost_content1','m1_cost_opt1','m1_cost_fair1')]
destination = root/'.tmp/m1-cost-artifacts.zip'
if destination.exists():raise FileExistsError(destination)
records = []
with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED) as archive:
    for directory in targets:
        for path in sorted(directory.rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts:continue
            if not path.resolve().is_relative_to(directory.resolve()):raise ValueError(path)
            raw = path.read_bytes()
            name = str(path.relative_to(root)).replace('\\','/')
            compact = path.suffix in ('.log','.prom') or path.name=='inputs.json'
            stored = gzip.compress(raw,mtime=0) if compact else raw
            target = name+'.gz' if compact else name
            archive.writestr(target,stored)
            records.append(dict(source=name,stored=target,bytes=len(raw),
                source_sha256=hashlib.sha256(raw).hexdigest(),stored_sha256=hashlib.sha256(stored).hexdigest()))
    archive.writestr('results/system_m1_cost/raw_manifest.json',json.dumps(records,indent=2)+'\n')
print(json.dumps(dict(files=len(records),archive_bytes=destination.stat().st_size)))
