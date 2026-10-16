"""Lossless archival of one engineering ledger; never mutate its raw files."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--manifest-name',default='manifest.json')
    args=p.parse_args();base=args.directory.resolve()
    if Path(args.manifest_name).name!=args.manifest_name:raise ValueError('plain manifest filename required')
    if args.output.exists():raise FileExistsError(args.output)
    if any(json.loads(p.read_text())['status']=='RUNNING' for p in base.glob('runs/*/execution.json')):
        raise RuntimeError('wait for owned execution settlement before archival')
    records=[]
    with zipfile.ZipFile(args.output,'w',zipfile.ZIP_DEFLATED) as out:
        for path in sorted(base.rglob('*')):
            if not path.is_file() or path.name=='budget.lock' or '__pycache__' in path.parts or path.suffix=='.tmp':continue
            if not path.resolve().is_relative_to(base) or path.suffix in ('.so','.zip','.whl','.safetensors') or path.stat().st_size>5*2**20:
                raise ValueError('unexpected/large artifact: '+str(path))
            raw=path.read_bytes();relative=path.relative_to(base).as_posix()
            compressed=path.suffix in ('.log','.prom')
            stored=relative+'.gz' if compressed else relative
            data=gzip.compress(raw,mtime=0) if compressed else raw
            out.writestr('results/'+base.name+'/'+stored,data)
            records.append(dict(source=relative,stored=stored,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),stored_sha256=hashlib.sha256(data).hexdigest()))
        out.writestr('results/'+base.name+'/'+args.manifest_name,json.dumps(records,indent=2)+'\n')
    print(json.dumps(dict(files=len(records),bytes=args.output.stat().st_size)))


if __name__=='__main__':main()
