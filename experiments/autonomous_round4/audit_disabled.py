"""Reject instrumentation patches that alter native control flow when disabled."""
import argparse
import ast
import json
from pathlib import Path


class DisabledResources(ast.NodeTransformer):
    def visit_If(self, node):
        expression = ast.unparse(node.test)
        if "SLACKKV_RESOURCE_TRACE" in expression or expression == "_ROUND4_RESOURCE_TRACE":
            return None
        return self.generic_visit(node)

    def visit_Import(self, node):
        node.names = [n for n in node.names if n.name != "os"]
        return node if node.names else None

    def visit_Assign(self, node):
        if any(isinstance(t, ast.Name) and t.id == "_ROUND4_RESOURCE_TRACE" for t in node.targets):
            return None
        return self.generic_visit(node)


def equivalent(original: str, patched: str) -> bool:
    original_ast = DisabledResources().visit(ast.parse(original))
    patched_ast = DisabledResources().visit(ast.parse(patched))
    return ast.dump(original_ast, include_attributes=False) == ast.dump(patched_ast, include_attributes=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("original", type=Path)
    parser.add_argument("patched", type=Path)
    args = parser.parse_args()
    results = {}
    for relative in ("vllm/v1/core/sched/scheduler.py", "vllm/v1/core/kv_cache_manager.py"):
        results[relative] = equivalent((args.original / relative).read_text(), (args.patched / relative).read_text())
    print(json.dumps(results, indent=2))
    if not all(results.values()):
        raise SystemExit("disabled instrumentation changes native AST")
