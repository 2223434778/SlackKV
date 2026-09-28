"""Audit K2 ownership across the completed Phase 4G-A-FIX rerun traces."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/phase4g_a_fix_k2repair"


def audit_run(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    seen_first_token: set[str] = set()
    violations = []
    readmissions = []
    for event in result.get("resume_trace", []):
        kind = event.get("event")
        request_id = event.get("request_id")
        owners = set(event.get("gate_owner_request_ids") or [])
        stale_owners = owners & seen_first_token
        if stale_owners:
            violations.append({
                "event": kind,
                "session": event.get("session"),
                "violation": "post_first_token_gate_owner",
            })
        if kind == "RESUME_ADMITTED" and int(event.get("num_output_tokens") or 0) > 0:
            readmissions.append({
                "session": event.get("session"),
                "output_tokens": int(event["num_output_tokens"]),
                "owns_gate": request_id in owners,
            })
        if kind == "FIRST_TOKEN":
            seen_first_token.add(request_id)
        if kind == "REQUEST_FINISHED" and request_id in owners:
            violations.append({
                "event": kind,
                "session": event.get("session"),
                "violation": "finished_request_still_owns_gate",
            })
    return {
        "policy": result["policy"],
        "repetition": result["repetition"],
        "valid": result.get("valid") is True,
        "turns_complete": result.get("turns_complete"),
        "boundaries": result.get("boundaries"),
        "max_pre_first": result.get("max_pre_first"),
        "post_first_readmissions": readmissions,
        "lifecycle_violations": violations,
    }


def main() -> None:
    paths = sorted((OUT / "runs").glob("measured_rep*/run.json"))
    if len(paths) != 16:
        raise RuntimeError(f"expected 16 measured traces, found {len(paths)}")
    runs = [audit_run(path) for path in paths]
    violations = sum(len(run["lifecycle_violations"]) for run in runs)
    readmissions = [item for run in runs for item in run["post_first_readmissions"]]
    report = {
        "measured_runs": len(runs),
        "all_run_gates_valid": all(
            run["valid"] and run["turns_complete"] == 40
            and run["boundaries"] == 32 and run["max_pre_first"] <= 2
            for run in runs
        ),
        "post_first_readmission_count": len(readmissions),
        "post_first_readmissions_reacquired_gate": sum(
            item["owns_gate"] for item in readmissions
        ),
        "lifecycle_violation_count": violations,
        "runs": runs,
    }
    report["pass"] = (
        report["all_run_gates_valid"]
        and report["post_first_readmissions_reacquired_gate"] == 0
        and violations == 0
    )
    (OUT / "gate_lifecycle_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    validation_path = OUT / "validation_report.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["K2_lifecycle_audit"] = {
        key: value for key, value in report.items() if key != "runs"
    }
    validation_path.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not report["pass"]:
        raise RuntimeError("K2 lifecycle trace audit failed")


if __name__ == "__main__":
    main()
