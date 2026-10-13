"""Frozen, sequential validation cohort; uses the original smoke budget ledger."""
import json
import os
from pathlib import Path
import subprocess

ROOT = Path('/media/a6000/D/ZhaoBinHe/SlackKV')
CODE = ROOT/'.tmp/system-m1-code'
PY = '/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python'
OUT = ROOT/'results/system_m1_cost'
COMMON = [PY, '-m', 'experiments.system_m1.smoke', '--config', str(CODE/'experiments/system_m1/shared_config.json'),
          '--execute', '--root', str(ROOT), '--plan', str(ROOT/'results/system_m1_shared/preparation5'),
          '--authorized-additional-gpu-hours', '0.5']
RUNS = [('m1_cost_content1','m1','content'), ('m1_cost_opt1','m1','mechanism'),
        ('m1_cost_fair1','fair','mechanism')]
commands = [COMMON+['--name',n,'--policy',p,'--mode',m] for n,p,m in RUNS]
record = dict(base_commit='5495a3636f6fb37196c6b9375a7abb4d0ba1d072',
              prior_reference_run='m1_cost_ref1', commands=commands, completions=[],
              scope='one equivalent CPU fix validation cohort; no new policy/parameter search')
if (OUT/'launch.json').exists():raise FileExistsError(OUT/'launch.json')
env = dict(os.environ, PYTHONPATH=str(CODE))
for cmd in commands:
    (OUT/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
    with (OUT/(cmd[-5]+'_launcher.log')).open('w') as log:
        code = subprocess.call(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
    record['completions'].append(dict(command=cmd,exit_code=code))
    (OUT/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
    if code:raise SystemExit(code)
