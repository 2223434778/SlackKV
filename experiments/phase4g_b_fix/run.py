"""Run Phase 4G-B afresh after the guarded-native allocation liveness repair.

The old Phase 4G-B result remains immutable. This runner reuses its frozen
workload, order, metrics, and classification without merging old samples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from experiments.phase4d_a import runtime
from experiments.phase4g_a.workload import ROOT, SOURCE, SOURCE_SHA, build_manifest, load_source
from experiments.phase4g_b import run_phase4g_b as original

START_COMMIT = "0a32b037641449462699cbe1dc7edfb199369a4d"
OUT = ROOT / "results/phase4g_b_fix"
TMP_LOGS = Path("/tmp/slackkv-phase4g-b-fix-server-logs")
PYTHON = runtime.ENV_ROOT / "bin/python"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _pre_fix_evidence(path: Path = Path(
    "/tmp/slackkv-phase4g-b-fix-diagnostic/server_logs/measured_rep101_d.log"
)) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("missing retained pre-fix diagnostic log")
    first = None
    last = None
    count = 0
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            marker = "SLACKKV_PREFETCH_ALLOC_TRACE "
            if marker not in line:
                continue
            event = json.loads(line.split(marker, 1)[1])
            if event.get("event") != "ALLOCATE_NONE":
                continue
            first = first or event
            last = event
            count += 1
    if not first or not last:
        raise RuntimeError("pre-fix diagnostic did not capture allocation failure")
    return {
        "source_log": str(path), "rate_limited_events": count,
        "first": first, "last": last,
        "schedulable_blocks_at_first": first["free_blocks"] - first["reserved_blocks"],
        "hypothesis": "connector-first freezes normal in-flight prefills while a guarded native prefetch repeatedly fails allocation",
    }


def _context() -> dict[str, Any]:
    if _sha256(SOURCE) != SOURCE_SHA:
        raise RuntimeError("EXPERIMENT_INVALID_WORKLOAD: Phase 4C SHA mismatch")
    original.START_COMMIT = START_COMMIT
    frozen = original.frozen_checks()
    if frozen["start_commit"] != START_COMMIT:
        raise RuntimeError("EXPERIMENT_INVALID_ENVIRONMENT: unexpected SlackKV HEAD")
    return frozen


def _annotate_context(frozen: dict[str, Any], *, result: str | None = None) -> None:
    cfg = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
    cfg.update({
        "phase": "4G-B-FIX", "repair_scope": "guarded native prefetch allocation failure only",
        "frozen_phase4g_b_order_unchanged": True,
        "frozen_workload_sha256": SOURCE_SHA,
        "vllm_working_diff_sha256": frozen["vllm_current_diff_sha256"],
    })
    old_patch_sha = cfg.pop("instrumentation_patch_sha256", None)
    if old_patch_sha is not None:
        cfg["accumulated_vllm_patch_sha256"] = old_patch_sha
    if result:
        cfg["classification"] = result
    _write_json(OUT / "config.json", cfg)
    manifest = json.loads((OUT / "source_manifest.json").read_text(encoding="utf-8"))
    manifest.update({
        "phase": "Phase 4G-B-FIX", "repair_start_commit": START_COMMIT,
        "previous_incomplete_result": "results/phase4g_b/summary.md",
        "vllm_mechanism_changed": True,
        "vllm_change": "guarded native prefetch returns capacity_rejected when actual allocate_slots returns None before connector load",
        "vllm_working_diff_sha256": frozen["vllm_current_diff_sha256"],
    })
    old_patch_sha = manifest.pop("new_trace_only_instrumentation_patch_sha256", None)
    if old_patch_sha is not None:
        manifest["accumulated_vllm_patch_sha256"] = old_patch_sha
    manifest.pop("no_scheduler_algorithm_changes", None)
    _write_json(OUT / "source_manifest.json", manifest)
    (OUT / "commands.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "cd /media/a6000/D/ZhaoBinHe/SlackKV\n"
        f"{PYTHON} -m pytest -q tests/test_phase4g_b.py tests/test_phase4g_b_fix.py\n"
        f"{PYTHON} -m experiments.phase4g_b_fix.run --run\n"
        f"{PYTHON} -m experiments.phase4g_b_fix.audit\n",
        encoding="utf-8",
    )


def _write_summary(report: dict[str, Any], evidence: dict[str, Any], smoke: dict[str, Any]) -> None:
    summary = OUT / "summary.md"
    first = evidence["first"]
    cond = report.get("condition_summaries", [])
    lines = [
        "# Phase 4G-B-FIX — guarded-native allocation liveness and fresh mechanism matrix", "",
        f"**Result:** `{report['result']}`. This is a mechanism RCA, not a performance comparison.", "",
        "## Objective and repair", "",
        "The preceding Phase 4G-B matrix remains `INCOMPLETE_INVALID_RUN`; its data were not reused.",
        "The failed pre-fix run repeatedly admitted the same guarded prefetch to the connector-first pass,",
        "but actual KV allocation returned None. Connector-first suppressed ordinary running prefill",
        "steps, so their in-flight reservation could not shrink.", "",
        f"A fresh pre-fix diagnostic recorded free={first['free_blocks']} blocks, in-flight reserved={first['reserved_blocks']},",
        f"schedulable={evidence['schedulable_blocks_at_first']}, local={first['local_tokens']} tokens,",
        f"external={first['external_tokens']} tokens, and async_load={first['load_kv_async']}.",
        "The repair applies only to guarded native-prefetch requests: on actual `allocate_slots(None)`,",
        "return a structured `capacity_rejected` with `INSUFFICIENT_SCHEDULABLE_BLOCKS`, no connector load,",
        "no lease, and normal OnDemand resume fallback. Ordinary requests and unguarded historical prefetches are unchanged.", "",
        "## Frozen setup", "",
        f"- SlackKV start commit: `{START_COMMIT}`; workload SHA256: `{SOURCE_SHA}`.",
        "- Qwen2.5-1.5B-Instruct, vLLM 0.23.0, RTX A6000 GPU0, FP16, 2 GiB GPU KV / 8 GiB CPU KV, 4681 blocks of 16 tokens.",
        "- K2, P_SAFE=102 ms, resume guard=110 ms, logical budget=3510 blocks; all 40 turns and 32 dynamic boundaries per run.",
        f"- Fresh instrumentation smoke: valid={smoke['valid']}, probe CPU duty={smoke['probe_cost_summary']['probe_cpu_duty_fraction']:.4f} (<0.05).", "",
        "## Compared conditions", "",
        "| Condition | Policy | BGv2 | Valid runs | Source initially not ready | Ready by safe start | Actionable windows ≥20 ms | Capacity rejects | PSA rejects | Reservation skips |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in cond:
        lines.append(
            f"| {row['condition']} | {row['policy']} | {row['background']} | {row['valid_runs']}/{row['runs']} | "
            f"{row['initial_not_ready']} | {row['full_ready_by_safe_start']} | {row['actionable_ge_20ms']} | "
            f"{row['capacity_rejects']} | {row['progress_rejects']} | {row['reservation_skips']} |"
        )
    lines.extend([
        "", "## Interpretation and limitations", "",
        "The frozen A–E classification is computed from the new complete matrix only. Readiness, physical fit,",
        "PSA-safe headroom, JIT timing, and logical reservation remain separate observed constraints.",
        "A `capacity_rejected` from actual allocation pressure is a safe promotion fallback, not a failed real resume.",
        "Counterfactual source/headroom samples do not demonstrate a successful promotion or performance gain.",
        "There is no request-timestamp-faithful trace here: token structure is from coding-agent data and timing is TraceToolProxy.",
        "No latency winner, SOTA, or statistical-significance claim is made.", "",
        "## Next-step status", "", "This repair and frozen mechanism matrix are complete; no later phase was started.", "",
    ])
    summary.write_text("\n".join(lines), encoding="utf-8")


def execute() -> dict[str, Any]:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"refusing to mix runs in non-empty result directory: {OUT}")
    OUT.mkdir(parents=True)
    original.OUT = OUT
    original.TMP_LOGS = TMP_LOGS
    os.environ["PATH"] = f"{runtime.ENV_ROOT / 'bin'}:{os.environ.get('PATH', '')}"
    frozen = _context()
    evidence = _pre_fix_evidence()
    _write_json(OUT / "pre_fix_diagnostic.json", evidence)
    original.write_initial_context(frozen)
    _annotate_context(frozen)
    manifest = build_manifest(load_source())
    results: list[dict[str, Any]] = []
    smoke: dict[str, Any] | None = None
    try:
        smoke = original.run_one("M2", "A", True, "smoke", 0, manifest)
        duty = smoke.get("probe_cost_summary", {}).get("probe_cpu_duty_fraction")
        if smoke.get("valid") is not True or duty is None or float(duty) >= .05:
            raise RuntimeError("EXPERIMENT_INVALID_INSTRUMENTATION_SMOKE")
        for repetition, (condition, policy, background) in enumerate(original.RUN_ORDER, start=1):
            run = original.run_one(condition, policy, background, "measured", repetition, manifest)
            if run.get("valid") is not True:
                raise RuntimeError(f"INCOMPLETE_INVALID_RUN:{run.get('run_id')}")
            results.append(run)
            _write_json(OUT / "progress.json", {"completed_runs": len(results),
                "expected_runs": len(original.RUN_ORDER), "run_ids": [row["run_id"] for row in results]})
        report = original.write_artifacts(results, manifest, frozen, smoke)
        _annotate_context(frozen, result=report["result"])
        _write_summary(report, evidence, smoke)
        report.update({"phase": "4G-B-FIX", "previous_result_unchanged": True,
                       "repair": "guarded prefetch allocation-fast-reject", "vllm_working_diff_sha256": frozen["vllm_current_diff_sha256"]})
        _write_json(OUT / "validation_report.json", report)
        return report
    except BaseException as exc:
        _write_json(OUT / "validation_report.json", {"phase": "4G-B-FIX",
            "start_commit": START_COMMIT, "workload_sha256": SOURCE_SHA,
            "smoke_valid": bool(smoke and smoke.get("valid")),
            "measured_valid": len(results), "measured_expected": len(original.RUN_ORDER),
            "result": "INCOMPLETE_INVALID_RUN", "error": repr(exc)})
        (OUT / "summary.md").write_text(
            "# Phase 4G-B-FIX\n\n**INCOMPLETE_INVALID_RUN**. The previous Phase 4G-B result remains unchanged. "
            f"Fresh valid measured runs: {len(results)}/{len(original.RUN_ORDER)}. Error: `{exc!r}`. "
            "No partial-matrix classification or performance claim.\n", encoding="utf-8")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        print(json.dumps(_context(), indent=2, sort_keys=True))
    elif args.run:
        print(json.dumps(execute(), indent=2, sort_keys=True))
    else:
        parser.error("choose --preflight or --run")


if __name__ == "__main__":
    main()
