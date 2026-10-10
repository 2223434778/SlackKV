"""Sequential frozen small screens; each child owns its server and budget entry."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=('a', 'b', 'hot', 'pressure', 'ample'), required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--name-prefix', default='', help='unique prefix for safe reruns in the same continuous budget')
    args = p.parse_args()
    specs = {
        'a': [(1, 256, 0, True, seed, policy, 'a'+str(i)+policy)
              for i, seed, policies in ((1, 773311, ('nohost', 'fair')), (2, 884422, ('fair', 'nohost')))
              for policy in policies],
        'b': [(2, 256, 0, True, 773311, policy, 'b'+policy) for policy in ('fair', 'full', 'nohost')],
        'hot': [(4, 256, 0, False, 773311, policy, 'hot'+policy) for policy in ('nohost', 'fair', 'full')],
        'pressure': [(4, 512, 2048, False, 773311, policy, 'pressure'+policy) for policy in ('full', 'fair', 'nohost')],
        'ample': [(4, 8192, 2048, False, 773311, policy, 'ample'+policy) for policy in ('fair', 'nohost', 'full')],
    }
    root = Path(__file__).resolve().parents[2]
    for sessions, cpu, gpu, reset, seed, policy, name in specs[args.stage]:
        argv = [sys.executable, '-m', 'experiments.host_prefix.followup_screen', '--source', args.source,
                '--name', args.name_prefix+name, '--sessions', str(sessions), '--cpu-mib', str(cpu), '--gpu-mib', str(gpu),
                '--seed', str(seed), '--policy', policy] + (['--reset-gpu'] if reset else [])
        print('LAUNCH', argv, flush=True)
        subprocess.run(argv, cwd=root, check=True)


if __name__ == '__main__':
    main()
