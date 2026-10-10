"""Bounded public-source prefix acquisition; no model download or GPU use."""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

REVISION = '23f152f6f0f9399a85901b89a6458def0ef16729'
URL = f'https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126/resolve/{REVISION}/traces.jsonl'
EXPECTED_PREFIX = 'd078503145b09533298928438fe186bbbc40c726901f920872510df04cd47994'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        if hashlib.sha256(args.output.read_bytes()).hexdigest() != EXPECTED_PREFIX:
            raise ValueError('existing source prefix hash mismatch; do not overwrite')
        print('existing source prefix hash verified')
        return
    request = urllib.request.Request(URL, headers={'Range': 'bytes=0-67108863'})
    lines, size = [], 0
    with urllib.request.urlopen(request, timeout=30) as response:
        while len(lines) < 16:
            line = response.readline(67108864 - size + 1)
            if not line or size + len(line) > 67108864 or not line.endswith(b'\n'):
                break
            json.loads(line)
            lines.append(line)
            size += len(line)
    raw = b''.join(lines)
    if hashlib.sha256(raw).hexdigest() != EXPECTED_PREFIX:
        raise ValueError('pinned source prefix differs from recorded evidence')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(raw)
    print(json.dumps({'complete_rows': len(lines), 'bytes': len(raw), 'sha256': EXPECTED_PREFIX}))


if __name__ == '__main__':
    main()
