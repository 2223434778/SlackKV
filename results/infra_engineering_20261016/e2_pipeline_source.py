"""Frozen9 purpose checks plus real disconnect/owned stop; same engineering grant."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path('/media/a6000/D/ZhaoBinHe/SlackKV');out=root/'results/infra_engineering_20261016'
code=root/'.tmp/infra-e2-code-20261016';sys.path.insert(0,str(code))
from slackkv.infra.benchmark import prepare
from slackkv.infra.config import Config
from slackkv.infra.runtime import save

python='/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python'
env=dict(os.environ,PYTHONPATH=str(code));prefix=[python,'-m','slackkv.infra']
base=json.loads((out/'demo_config.json').read_text());commands=[]
for scenario in ('hot','host-enough','host-limited'):
    cfg=dict(base)
    if scenario=='host-enough':cfg['host_kv_bytes']=2**30
    cfg_path=out/('bench-'+scenario+'-config.json')
    if cfg_path.exists():raise FileExistsError(cfg_path)
    save(cfg_path,cfg)
    plan_dir=out/'plans'/scenario
    prepare(Config(**cfg),scenario,plan_dir)
    for policy in ('nohost','full','fair'):
        name='e2_'+scenario+'_'+policy+'1'
        commands.append(prefix+['bench','--config',str(cfg_path),'--plan',str(plan_dir),
            '--policy',policy,'--ledger',str(out),'--name',name])
record=dict(base_commit='b6773c7318c06e7e429a87b6b23ededefb45196f',commands=commands,completed=[],
    selection='one predeclared run per case/policy, no outcome-based retuning; same budget including failures')
for cmd in commands:
    save(out/'e2_launch.json',record)
    with (out/(cmd[-1]+'_launcher.log')).open('w') as log:
        rc=subprocess.call(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
    record['completed'].append(dict(name=cmd[-1],exit_code=rc));save(out/'e2_launch.json',record)
    if rc:raise SystemExit(rc)
for scenario in ('hot','host-enough','host-limited'):
    dirs=[str(out/'runs'/('e2_'+scenario+'_'+policy+'1')) for policy in ('nohost','full','fair')]
    subprocess.check_call(prefix+['bench-compare','--directories',*dirs,'--output',str(out/('comparison-'+scenario+'.json'))],env=env)

# Validate actual live-service disconnect and public stop command, not a fake
# inference request or direct cross-process mutation of the native manager.
name='e2_reliability1';serve=prefix+['serve','--config',str(out/'demo_config.json'),'--ledger',str(out),'--name',name]
save(out/'e2_reliability_launch.json',dict(serve=serve))
with (out/(name+'_launcher.log')).open('w') as log:
    proc=subprocess.Popen(serve,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    execution=out/'runs'/name/'execution.json';deadline=time.monotonic()+280
    while time.monotonic()<deadline:
        if proc.poll() is not None:raise RuntimeError('serve exited before exercise')
        if execution.exists() and json.loads(execution.read_text()).get('ready_seconds'):break
        time.sleep(.5)
    else:raise TimeoutError('owner readiness unconfirmed; its watchdog remains responsible for cleanup')
    try:
        subprocess.check_call(prefix+['reliability','--config',str(out/'demo_config.json'),'--directory',str(execution.parent)],env=env)
    finally:
        subprocess.check_call(prefix+['stop','--directory',str(execution.parent)],env=env)
        proc.wait(timeout=60)
    e=json.loads(execution.read_text());assert e['status']=='STOPPED' and e['cleanup_verified']
    save(out/'e2_reliability_stop.json',dict(status=e['status'],cleanup_verified=True,owner_exit_code=proc.returncode))
