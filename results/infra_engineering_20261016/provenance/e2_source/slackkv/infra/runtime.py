"""Explicit offline overlay from pinned upstream source + matching compiled libs."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil

COMMIT='0fc695fc6'
SOURCE_TREE_SHA256='a4dfca00a96977e7068774fedf254d98af11e36fac4863e7feedae23305dbeb8'
PINNED={'vllm':'0.23.0+cu126','torch':'2.11.0+cu126','transformers':'5.17.0'}
SCHEDULER='vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temporary.replace(path)


def versions():
    result={}
    for key in PINNED:
        try:result[key]=importlib.metadata.version(key)
        except importlib.metadata.PackageNotFoundError:result[key]=None
    return result


def prepare(source,compiled,target):
    source,compiled,target=map(lambda p:Path(p).resolve(),(source,compiled,target))
    if source==target or compiled==target or source in target.parents or compiled in target.parents:
        raise ValueError('target must be a separate new overlay')
    stamp=target/'slackkv-runtime.json'
    if stamp.exists():
        record=json.loads(stamp.read_text(encoding='utf-8'))
        if record['source']!=str(source) or record['compiled']!=str(compiled):raise ValueError('overlay already belongs to a different source')
        check_runtime(target)
        return record
    if target.exists():raise FileExistsError('refuse to overwrite partial/unknown target: '+str(target))
    package=source/'vllm'
    rows=[p.relative_to(package).as_posix()+'\0'+hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()+'\n'
          for p in sorted(package.rglob('*.py'),key=lambda p:p.relative_to(package).as_posix())]
    if hashlib.sha256(''.join(rows).encode()).hexdigest()!=SOURCE_TREE_SHA256:
        raise ValueError('upstream Python tree does not match pinned commit; no research or unreviewed patches accepted')
    generated=compiled/'vllm/_version.py'
    if COMMIT not in generated.read_text(encoding='utf-8'):raise ValueError('compiled snapshot commit does not match '+COMMIT)
    original=(source/SCHEDULER).read_text(encoding='utf-8')
    if 'max_offload_tokens' not in original or 'slackkv' in original.lower() or 'experiments.' in (source/'vllm/v1/core/sched/scheduler.py').read_text(encoding='utf-8'):
        raise ValueError('provide pristine pinned upstream source; not historical research source')
    # Prefer an actual git revision; codeload tar directories embed the pinned ref.
    import subprocess
    git=subprocess.run(['git','-C',str(source),'rev-parse','HEAD'],capture_output=True,text=True)
    if not (git.returncode==0 and git.stdout.startswith(COMMIT)) and source.name!='vllm-'+COMMIT:
        raise ValueError('source must be pinned Git checkout or verified codeload tar directory')
    target.mkdir(parents=True)
    shutil.copytree(source/'vllm',target/'vllm',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(generated,target/'vllm/_version.py')
    libraries=[]
    for file in (compiled/'vllm').rglob('*.so'):
        destination=target/file.relative_to(compiled)
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(file.resolve(),destination)
        libraries.append(str(file.relative_to(compiled)))
    if not libraries:raise ValueError('matching compiled vLLM shared libraries missing')
    files={str(p.relative_to(target)).replace('\\','/'):digest(p) for p in (target/'vllm').rglob('*.py')}
    record=dict(upstream_commit=COMMIT,upstream_python_tree_sha256=SOURCE_TREE_SHA256,source=str(source),compiled=str(compiled),python_sha256=files,
        shared_libraries={name:digest(target/name) for name in libraries},
        patch='none: native max_offload_tokens; project InfraCPUSpec loaded via native factory')
    save(stamp,record)
    return record


def check_runtime(target):
    target=Path(target);record=json.loads((target/'slackkv-runtime.json').read_text(encoding='utf-8'))
    if record['upstream_commit']!=COMMIT:raise ValueError('unsupported runtime commit')
    for name,expected in record['python_sha256'].items():
        if digest(target/name)!=expected:raise ValueError('runtime source changed: '+name)
    for name,expected in record['shared_libraries'].items():
        if digest(target/name)!=expected:raise ValueError('compiled library changed: '+name)
    return record
