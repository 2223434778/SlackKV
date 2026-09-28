"""Fresh full-lifecycle checks and CPU gate-cost diagnostics for the K2 fix."""
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import timeit
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/k2_lifecycle_fix"
SRC = Path("/media/a6000/D/ZhaoBinHe/slackkv-phase2c-fix-sourceprobe")
SCHED = SRC / "vllm/v1/core/sched/scheduler.py"
BASELINE = Path("/tmp/slackkv-k2-lifecycle-baseline.py")
OLD_DIFF = "f0f8e327cc7ff4eb1797852a24e56e0e7f4a77b62e0f9439e88405fd5b470ae4"
WORKLOAD_SHA = "9b381336a70e73b0efbddbb79d2c88608c19663247e819c1e7a364efd2971105"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def audit_gate(events):
    seen, violations, readmissions = set(), [], []
    for event in events:
        kind, rid = event["event"], event["request_id"]
        owners = set(event.get("gate_owner_request_ids", []))
        stale = owners & seen
        if stale:
            violations.append({"event": kind, "session": event["session"], "stale_owners": sorted(stale)})
        if kind == "RESUME_ADMITTED" and event.get("num_output_tokens", 0) > 0:
            readmissions.append({"session": event["session"], "output_tokens": event["num_output_tokens"], "owns_gate": rid in owners})
        if kind == "FIRST_TOKEN":
            seen.add(rid)  # FIRST_TOKEN trace is emitted immediately before release.
        if kind == "REQUEST_FINISHED" and rid in owners:
            violations.append({"event": kind, "session": event["session"], "finished_owner": rid})
    return {"post_first_readmissions": readmissions, "gate_lifecycle_violations": violations}


def worker(policy, rep):
    from experiments.phase4g_a_fix import run_phase4g_a_fix as harness
    manifest = json.loads((OUT / "manifest.json").read_text())
    harness.OUT = OUT / "integration"
    harness.TEMP_LOGS = Path("/tmp/slackkv-k2-lifecycle-validation")
    harness.VLLM_EXPECTED_DIFF_SHA = manifest["fixed_vllm_diff_sha256"]
    result = harness.run_agent(policy, "smoke", rep)
    audit = audit_gate(result["resume_trace"])
    row = {k: result[k] for k in ("policy", "valid", "turns_complete", "boundaries", "max_pre_first", "logical_leaks", "lease_leaks", "capacity_valid", "OOM", "server_alive", "BG_total_generated_tokens", "BG_tail_request_finish_ms")}
    row.update(audit)
    row["repetition"] = rep
    save(OUT / f"validation_{rep}_{policy}.json", row)
    if not row["valid"] or audit["gate_lifecycle_violations"]:
        raise RuntimeError("integration validity or gate lifecycle violation")


def helpers(text):
    tree = ast.parse(text)
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"_slackkv_first_token_gate_limit", "_slackkv_first_token_gate_can_admit"}]
    ns = {"Request": object}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "gate_cost", "exec"), ns)
    return ns["_slackkv_first_token_gate_can_admit"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "manifest.json").exists():
        raise RuntimeError("validation already initialized; preserve existing artifacts")
    workload = ROOT / "results/phase4c/hybrid_workload.csv"
    assert hashlib.sha256(workload.read_bytes()).hexdigest() == WORKLOAD_SHA
    old, new = BASELINE.read_text(), SCHED.read_text()
    patch = "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), "a/vllm/v1/core/sched/scheduler.py", "b/vllm/v1/core/sched/scheduler.py"))
    (OUT / "vllm_k2_lifecycle_delta.patch").write_text(patch)
    save(OUT / "manifest.json", {
        "start_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "workload_sha256": WORKLOAD_SHA, "baseline_vllm_diff_sha256": OLD_DIFF,
        "fixed_vllm_diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "--binary"], cwd=SRC)).hexdigest(),
        "baseline_scheduler_sha256": hashlib.sha256(old.encode()).hexdigest(),
        "fixed_scheduler_sha256": hashlib.sha256(new.encode()).hexdigest(),
        "validation_order": ["D", "A", "D", "D"],
        "purpose": "correctness/liveness validation; not a performance matrix"})
    tests = {}
    for name, path in [("baseline", BASELINE), ("fixed", SCHED)]:
        env = dict(os.environ, SLACKKV_SCHEDULER_SOURCE=str(path))
        check = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_k2_lifecycle_fix.py"], cwd=ROOT, env=env, text=True, capture_output=True)
        (OUT / f"tests_{name}.txt").write_text(check.stdout)
        tests[name] = {"returncode": check.returncode, "summary": check.stdout.strip().splitlines()[-1]}
    save(OUT / "tests.json", tests)
    assert tests["baseline"]["returncode"] != 0 and tests["fixed"]["returncode"] == 0
    cost = []
    fns = {"baseline": helpers(old), "fixed": helpers(new)}
    request = SimpleNamespace(request_id="new", kv_transfer_params={"slackkv_first_token_gate_limit": 2})
    for count in (0, 1, 2):
        owners = set(map(str, range(count)))
        values = {name: [] for name in fns}
        for repeat in range(10):
            for name in (["baseline", "fixed"] if repeat % 2 == 0 else ["fixed", "baseline"]):
                fn = fns[name]
                values[name].append(timeit.timeit(lambda: fn(owners, 2, request), number=200000) * 1e9 / 200000)
        cost.append({"owners": count, "baseline_median_ns": statistics.median(values["baseline"]), "fixed_median_ns": statistics.median(values["fixed"]), "samples_ns": values})
    save(OUT / "gate_helper_cost.json", {"scope": "CPU helper microbenchmark only; not end-to-end throughput", "cases": cost})
    completed = []
    for rep, policy in enumerate(("D", "A", "D", "D"), 1):
        save(OUT / "progress.json", {"completed": completed, "active": [rep, policy]})
        with (OUT / f"driver_{rep}_{policy}.log").open("w") as log:
            proc = subprocess.Popen([sys.executable, "-u", "-m", "experiments.k2_lifecycle_fix.validate", "--worker", policy, "--rep", str(rep)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = proc.wait(timeout=540)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                code = -1
            if code:
                save(OUT / "validation_report.json", {"status": "FAIL", "completed": completed, "failed": [rep, policy], "returncode": code})
                raise RuntimeError(f"integration {rep}/{policy} failed")
        completed.append(json.loads((OUT / f"validation_{rep}_{policy}.json").read_text()))
    save(OUT / "validation_report.json", {"status": "PASS", "runs": completed, "tests": tests, "performance_superiority_evaluated": False})
    save(OUT / "progress.json", {"completed": completed, "active": None})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=["A", "D"])
    parser.add_argument("--rep", type=int)
    args = parser.parse_args()
    worker(args.worker, args.rep) if args.worker else main()
