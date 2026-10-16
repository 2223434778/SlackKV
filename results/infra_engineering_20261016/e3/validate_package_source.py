import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

root=Path.cwd();scratch=root/'.tmp/e3-package';scratch.mkdir(exist_ok=True)
out=root/'results/infra_engineering_20261016/e3';out.mkdir(exist_ok=True)
temporary=scratch/'temp';temporary.mkdir(exist_ok=True)
environment=dict(os.environ,TEMP=str(temporary),TMP=str(temporary),TMPDIR=str(temporary))
commands=[]
def run(args,cwd=root,env=environment):
    process=subprocess.run(args,cwd=cwd,env=env,text=True,encoding='utf-8',errors='replace',capture_output=True)
    commands.append(dict(command=args,cwd=str(cwd),returncode=process.returncode,stdout=process.stdout,stderr=process.stderr))
    if process.returncode:
        (out/'package_validation.json').write_text(json.dumps(dict(status='FAILED',commands=commands),indent=2),encoding='utf-8')
        raise RuntimeError(process.stderr)
    return process.stdout
run([sys.executable,'-m','pip','wheel','--no-index','--no-deps','--no-build-isolation','--wheel-dir',str(scratch/'wheels'),'.'])
wheel=next((scratch/'wheels').glob('slackkv-*.whl'))
with zipfile.ZipFile(wheel) as z:
    assert 'slackkv/infra/documents.json' in z.namelist()
    entry=z.read('slackkv-0.1.0.dist-info/entry_points.txt').decode()
    assert 'slackkv = slackkv.infra.cli:main' in entry
run([sys.executable,'-m','pip','install','--no-index','--no-deps','--target',str(scratch/'installed'),str(wheel)])
empty=scratch/'outside-source';empty.mkdir(exist_ok=True)
installed=dict(environment,PYTHONPATH=str(scratch/'installed'),PYTHONIOENCODING='utf-8')
value=run([sys.executable,'-c','import json,pathlib,slackkv.infra;print(json.dumps({"module":slackkv.infra.__file__,"documents":(pathlib.Path(slackkv.infra.__file__).parent/"documents.json").is_file()}))'],empty,installed)
metadata=json.loads(value);assert str(scratch/'installed') in metadata['module'] and metadata['documents']
run([sys.executable,'-m','slackkv.infra','doctor','--config',str(root/'configs/local.json'),'--config-only'],empty,installed)
run([sys.executable,'-m','slackkv.infra','report','--directory',str(root/'results/infra_engineering_20261016/runs/e1_demo2'),'--output',str(scratch/'installed-report')],empty,installed)
run([sys.executable,'-m','slackkv.infra','--help'],empty,installed)
run([sys.executable,'-m','slackkv.infra','report','--help'],empty,installed)
record=dict(status='PASS',python=sys.version,wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
    scope='offline wheel build/install in isolated target; installed CLI outside source tree; no CUDA or model load',commands=commands)
(out/'package_validation.json').write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(dict(status='PASS',wheel_sha256=record['wheel_sha256'],commands=len(commands))))
