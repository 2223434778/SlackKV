"""python -m slackkv.infra / installed slackkv command."""
import argparse
import json
from pathlib import Path
import time

from slackkv.infra.config import Config
from slackkv.infra.runtime import prepare,save
from slackkv.infra.service import ManagedService,doctor,init_budget,stop_owned


def main(argv=None):
    parser=argparse.ArgumentParser(description='SlackKV native-offload infra prototype')
    sub=parser.add_subparsers(dest='action',required=True)
    for name in ('doctor','serve','demo','verify-host'):
        p=sub.add_parser(name);p.add_argument('--config',required=True);p.add_argument('--policy',choices=('fair','full','nohost'))
        if name=='doctor':p.add_argument('--config-only',action='store_true')
        else:
            p.add_argument('--ledger',required=True);p.add_argument('--name',required=True)
    p=sub.add_parser('prepare-runtime');p.add_argument('--source',required=True);p.add_argument('--compiled',required=True);p.add_argument('--target',required=True)
    p=sub.add_parser('init-budget');p.add_argument('--ledger',required=True)
    p=sub.add_parser('report');p.add_argument('--directory',required=True)
    p=sub.add_parser('stop');p.add_argument('--directory',required=True)
    args=parser.parse_args(argv)
    try:
        if args.action=='prepare-runtime':
            record=prepare(args.source,args.compiled,args.target)
            print(json.dumps(dict(status='PASS',upstream_commit=record['upstream_commit'],python_files=len(record['python_sha256']))));return
        if args.action=='init-budget':init_budget(args.ledger);print('Engineering grant ledger initialized; does not create user authorization.');return
        if args.action=='stop':print(stop_owned(args.directory));return
        if args.action=='report':
            from slackkv.infra.report import create
            print(json.dumps(create(args.directory),ensure_ascii=False,indent=2));return
        config=Config.load(args.config,args.policy)
        if args.action=='doctor':print(json.dumps(doctor(config,args.config_only),ensure_ascii=False,indent=2));return
        from slackkv.infra.workloads import run_demo,verify_host
        with ManagedService(config,args.ledger,args.name) as service:
            if args.action=='serve':
                print(f'Ready at {config.url}; Ctrl+C or slackkv stop --directory {service.out}',flush=True)
                while service.proc.poll() is None:time.sleep(.5)
                raise RuntimeError('owned server exited unexpectedly')
            if args.action=='demo':
                save(service.out/'validation.json',run_demo(config,service.out))
                from slackkv.infra.report import create
                create(service.out)
            else:
                verify_host(config,service.out)
                save(service.out/'validation.json',dict(status='PASS',scope='content IDs verified; use raw prom to verify actual transfers'))
        print(str(service.out))
    except (ValueError,RuntimeError,OSError,TimeoutError) as error:
        parser.exit(1,f'{type(error).__name__}: {error}\n')
