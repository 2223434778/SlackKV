"""Create an isolated Python overlay; reuse built libraries via read-only links."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import os
import shutil
import time
import traceback

RELATIVE = 'vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py'
ANCHORS = (
    '        req_status.update_offload_keys()\n        req_status.num_locally_computed_tokens = num_computed_tokens',
    '            req_status.update_offload_keys()\n\n            if preempted:',
)


def hooked(text):
    for old, indent in zip(ANCHORS, (8, 12)):
        if text.count(old) != 1:
            raise ValueError('native hook anchor mismatch; do not edit an unknown version')
        first, rest = old.split('\n', 1)
        spaces = ' ' * indent
        insertion = (f'\n{spaces}if (observe := getattr(self.manager, "slackkv_m1_observe", None)) is not None:'
                     f'\n{spaces}    observe(req_status)')
        text = text.replace(old, first + insertion + '\n' + rest)
    ast.parse(text)
    return text


def disabled_ast(text):
    class RemoveHooks(ast.NodeTransformer):
        def visit_If(self, node):
            if any(isinstance(n, ast.Constant) and n.value == 'slackkv_m1_observe' for n in ast.walk(node.test)):
                return None
            return self.generic_visit(node)
    return ast.dump(RemoveHooks().visit(ast.parse(text)), include_attributes=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base, target = args.base.resolve(), args.target.resolve()
    if not base.is_dir() or target.exists() or target == base or base in target.parents or target in base.parents:
        raise ValueError('new separate target required; never overwrite historical native source')
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(status='RUNNING', gpu_count=0, gpu_hours=0, base=str(base), target=str(target))
    started = time.monotonic()
    try:
        original = (base/RELATIVE).read_text()
        patched = hooked(original)
        if disabled_ast(patched) != ast.dump(ast.parse(original), include_attributes=False):
            raise ValueError('disabled-hook AST differs from original')
        target.mkdir()
        def copy(source, destination):
            src = Path(source)
            if src.suffix == '.py':
                shutil.copyfile(src, destination)
            else:
                os.symlink(str(src.resolve()), destination)
            return destination
        shutil.copytree(base/'vllm', target/'vllm', copy_function=copy,
                        ignore=shutil.ignore_patterns('__pycache__', '.git'))
        (target/RELATIVE).write_text(patched)
        result.update(status='COMPLETE', original_scheduler_sha256=hashlib.sha256((base/RELATIVE).read_bytes()).hexdigest(),
                      patched_scheduler_sha256=hashlib.sha256((target/RELATIVE).read_bytes()).hexdigest(),
                      disabled_ast_equal=True,
                      scope='two opt-in scheduler metadata hooks; no worker/DMA/GPU allocator changes')
    except BaseException:
        result.update(status='FAILED', error=traceback.format_exc())
        raise
    finally:
        result['cpu_wall_seconds'] = time.monotonic()-started
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result))


if __name__ == '__main__':
    main()
