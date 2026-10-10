"""Sequential frozen reverse pair and separate correctness run; same GPU ledger."""
import subprocess
import sys

SOURCE = '/media/a6000/D/ZhaoBinHe/slackkv-autonomous-round4-source'


if __name__ == '__main__':
    for mode, name in (('frontier', 'frontier2'), ('full', 'full2'), ('oracle', 'oracle1')):
        subprocess.run([sys.executable, '-m', 'experiments.host_prefix.calibrate',
                        '--mode', mode, '--name', name, '--source', SOURCE], check=True, timeout=1785)
