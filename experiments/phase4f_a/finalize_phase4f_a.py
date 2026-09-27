from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.phase4f_a.logic import WORKLOAD_SHA, psa_trace_valid

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/phase4f_a"


def rows(name: str) -> list[dict[str, str]]:
    path = OUT / name
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def stage_rows(all_rows: list[dict[str, str]], stage: str, policy: str) -> list[dict[str, str]]:
    return [row for row in all_rows if row["stage"] == stage and row["policy"] == policy]


def all_valid(group: list[dict[str, str]], expected: int, classification: str) -> bool:
    return len(group) == expected and all(row.get("classification") == classification
                                          if "classification" in row else row.get("valid") == "True"
                                          for row in group)


def main() -> None:
    liveness = rows("liveness_runs.csv")
    micro = rows("micro_runs.csv")
    admissions = rows("progress_admission.csv")
    waiting = rows("waiting_trace.csv")
    lease_events = rows("lease_events.csv")
    resume_events = rows("resume_trace.csv")
    arrivals = {(row["stage"], row["repetition"], row["policy"], row["session"]):
                float(row["timestamp_monotonic"])
                for row in resume_events if row.get("event") == "RESUME_ARRIVED"}
    remote_waits = {(row["stage"], row["repetition"], row["policy"], row["session"])
                    for row in resume_events if row.get("event") == "WAITING_FOR_REMOTE_KVS"}
    confirmed_useful = Counter()
    for row in lease_events:
        if row.get("event") != "LEASE_ACQUIRED":
            continue
        key = (row["stage"], row["repetition"], row["policy"], row["lease_key"])
        arrival = arrivals.get(key)
        if arrival is not None and float(row["timestamp_monotonic"]) < arrival and key not in remote_waits:
            confirmed_useful[row["stage"]] += 1
    p1_off = stage_rows(liveness, "P1", "PSA_OFF")
    p1_on = stage_rows(liveness, "P1", "PSA1")
    p2_on = stage_rows(liveness, "P2", "PSA1")
    p3 = stage_rows(liveness, "P3", "PSA1")
    p4 = stage_rows(liveness, "P4", "GUARDED_LEASE_PSA1")
    p5 = stage_rows(liveness, "P5", "FULL_PSA1")
    psa_events = [row for row in admissions if row.get("decision") in
                  {"ADMIT", "CAPACITY_REJECT", "PROGRESS_REJECT"}]
    violations = sum(not psa_trace_valid(row) for row in psa_events)
    p0_valid = len(micro) == 3 and all(row["valid"] == "True" for row in micro)
    p1_valid = all_valid(p1_on, 2, "PROGRESS")
    p2_valid = all_valid(p2_on, 3, "PROGRESS")
    p3_valid = all_valid(p3, 2, "PROGRESS")
    p4_valid = len(p4) == 8 and all(row["valid"] == "True" for row in p4)
    p5_valid = len(p5) == 4 and all(row["valid"] == "True" for row in p5)
    baseline = all_valid(p1_off, 1, "LEASE_HOL_NONPROGRESS")
    p3_e04 = [row for row in admissions if row["stage"] == "P3" and
              row["candidate_episode"] == "E04"]
    candidate_valid = len(p3_e04) == 2 and all(row["decision"] == "ADMIT"
                                                    and int(row["progress_reserve_blocks"]) == 7
                                                    for row in p3_e04)
    for repetition in (1, 2):
        p3_lease = [row for row in lease_events if row["stage"] == "P3"
                    and row["repetition"] == str(repetition)
                    and row["lease_key"] == "E04:turn:1"]
        p3_resume = [row for row in resume_events if row["stage"] == "P3"
                     and row["repetition"] == str(repetition)
                     and row.get("session") == "E04:turn:1"]
        candidate_valid &= (any(row["event"] == "LEASE_ACQUIRED" for row in p3_lease)
                            and any(row["event"] == "LEASE_RELEASED" for row in p3_lease)
                            and any(row["event"] == "RESUME_ADMITTED" for row in p3_resume))
    p1_e02 = [row for row in admissions if row["stage"] == "P1" and
              row["policy"] == "PSA1" and row["candidate_episode"] == "E02"]
    p2_e02 = [row for row in admissions if row["stage"] == "P2" and
              row["policy"] == "PSA1" and row["candidate_episode"] == "E02"]
    e02_valid = (len(p1_e02) == 2 and len(p2_e02) == 3 and
                 all(row["decision"] == "PROGRESS_REJECT"
                     for row in p1_e02 + p2_e02))
    completed_lease_events = [row for row in lease_events
                              if row["stage"] != "P0" and not (row["stage"] in {"P1", "P2"}
                                      and row["policy"] == "PSA_OFF")]
    lifecycle: dict[tuple[str, str, str, str], Counter[str]] = {}
    for row in completed_lease_events:
        key = (row["stage"], row["repetition"], row["policy"], row["lease_key"])
        lifecycle.setdefault(key, Counter())[row["event"]] += 1
    lease_leaks = sum(max(0, counts["LEASE_ACQUIRED"] - counts["LEASE_RELEASED"])
                      for counts in lifecycle.values())
    logical_leaks = sum(int(row.get("logical_leaks") or 0) for row in p5)
    unsafe_releases = sum(int(row.get("unsafe_release") or 0) for row in p5)
    if not p0_valid or violations or (not candidate_valid and p3) or lease_leaks or logical_leaks or unsafe_releases:
        result = "EXPERIMENT_INVALID_IMPLEMENTATION"
    elif any(row.get("classification") == "INVALID" for row in p1_on + p2_on + p4):
        result = "RESULT_C_PSA1_LIVENESS_FAILURE"
    elif baseline and p1_valid and p2_valid and p3_valid and p4_valid and p5_valid and e02_valid:
        if all(int(row["promotion_admits"]) >= 1 for row in p4):
            result = "RESULT_A_PSA1_RESTORES_LIVENESS"
        else:
            result = "RESULT_B_PSA1_RESTORES_LIVENESS_BUT_OVERCONSERVATIVE"
    else:
        result = "INCOMPLETE"
    report: dict[str, Any] = {
        "phase": "4F-A", "start_commit": "5e94128bb69e13bf3c7772ed1e5de32f56f0701b",
        "workload_sha256": WORKLOAD_SHA, "gpu_blocks": 4681,
        "P0_micro_valid": sum(row["valid"] == "True" for row in micro),
        "P1_baseline_nonprogress": baseline,
        "P1_PSA_valid": sum(row["classification"] == "PROGRESS" for row in p1_on),
        "P2_PSA_valid": sum(row["classification"] == "PROGRESS" for row in p2_on),
        "P3_valid": sum(row["classification"] == "PROGRESS" for row in p3),
        "P4_valid": sum(row["valid"] == "True" for row in p4),
        "P5_valid": sum(row["valid"] == "True" for row in p5),
        "P4_promotion_admits_per_run": [int(row["promotion_admits"]) for row in p4],
        "P4_progress_rejects_per_run": [int(row["progress_rejects"]) for row in p4],
        "P5_reservation_skips": sum(int(row["reservation_skips"]) for row in p5),
        "P5_progress_rejects": sum(int(row["progress_rejects"]) for row in p5),
        "P5_capacity_rejects": sum(int(row["capacity_rejects"]) for row in p5),
        "P4_confirmed_useful_promotions": confirmed_useful["P4"],
        "P5_confirmed_useful_promotions": confirmed_useful["P5"],
        "P1_E02_progress_rejects": len([row for row in p1_e02
                                          if row["decision"] == "PROGRESS_REJECT"]),
        "P2_E02_progress_rejects": len([row for row in p2_e02
                                          if row["decision"] == "PROGRESS_REJECT"]),
        "P3_E04_candidate_admits": len([row for row in p3_e04
                                         if row["decision"] == "ADMIT"]),
        "PSA_invariant_violations": violations,
        "server_crashes": sum(row.get("server_alive") == "False" for row in liveness),
        "OOM": sum(row.get("OOM") == "True" for row in liveness),
        "final_lease_leaks": lease_leaks,
        "final_logical_leaks": logical_leaks,
        "unsafe_logical_releases": unsafe_releases,
        "result": result,
    }
    (OUT / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    p0_progress = next((row for row in admissions if row["stage"] == "P0" and
                        row["policy"] == "progress_reject" and
                        row["candidate_episode"] == "E02"), None)
    example = (f"free={p0_progress['free_blocks_before']}, required={p0_progress['candidate_required_blocks']}, "
               f"after={p0_progress['free_blocks_after_candidate']}, reserve={p0_progress['progress_reserve_blocks']}"
               if p0_progress else "not observed")
    p4_admits = report["P4_promotion_admits_per_run"]
    p4_rejects = report["P4_progress_rejects_per_run"]
    wait_none = Counter((row.get("stage"), row.get("policy")) for row in waiting
                        if row.get("event") == "WAITING_ALLOCATE_NONE" and
                        row.get("session") == "E04:turn:1")
    summary = f"""# Phase 4F-A — PSA-1 liveness restoration

Result: **{result}**. This is a correctness/liveness experiment, not a performance comparison.

The admission rule is `m <= F` and `F - m >= H`, with `H = max(0, P_h - max(L_h, m if candidate==protected else 0))`. Here `F` is the scheduler's exact free-block count, `m` is candidate prefix blocks, and `L_h` is the engine's actual matching protected lease. The protected job is the earliest-deadline request without a first token, including already-sent requests. `P_h = ceil((Turn1 ISL + 1)/16)` includes new input and first-token progress; cached-prefix blocks alone would omit that footprint.

In U3, old physical admission can accept E02 because its prefix fits at that instant, leaving less than E04's 1278-block progress footprint. The observed PSA micro state was {example}; its E02 decision was `{p0_progress['decision'] if p0_progress else 'NA'}`. The `progress_rejected` path performs no transfer, lease, model forward, or generation and falls back to OnDemand.

| Gate | Valid / expected | E04 waiting-allocation failures |
|---|---:|---:|
| P0 micro | {report['P0_micro_valid']}/3 | NA |
| P1 baseline no X | {int(baseline)}/1 nonprogress reproduced | {wait_none[('P1','PSA_OFF')]} |
| P1 PSA no X | {report['P1_PSA_valid']}/2, {sum(int(row['finished']) for row in p1_on)}/16 completed | {wait_none[('P1','PSA1')]} |
| P2 PSA + X | {report['P2_PSA_valid']}/3, {sum(int(row['finished']) for row in p2_on)}/24 completed | {wait_none[('P2','PSA1')]} |
| P3 protected E04 | {report['P3_valid']}/2, {sum(int(row['finished']) for row in p3)}/16 completed | {wait_none[('P3','PSA1')]} |
| P4 real JIT | {report['P4_valid']}/8, {sum(int(row['finished']) for row in p4)}/64 completed | {wait_none[('P4','GUARDED_LEASE_PSA1')]} |
| P5 FullSlackKV | {report['P5_valid']}/4, {sum(int(row['finished']) for row in p5)}/32 completed | {wait_none[('P5','FULL_PSA1')]} |

P1 E02 progress-rejects: {report['P1_E02_progress_rejects']}/2. P2 E02 progress-rejects: {report['P2_E02_progress_rejects']}/3. P3 candidate E04 admitted: {report['P3_E04_candidate_admits']}/2 with 7-block incremental reserve.

P4 promotion admits per run: {p4_admits}; progress rejects per run: {p4_rejects}. Lease-acquired-before-arrival promotions without remote wait, a conservative confirmed-useful count, were {report['P4_confirmed_useful_promotions']} in P4. P5 reservation skips: {report['P5_reservation_skips']}; progress rejects: {report['P5_progress_rejects']}; capacity rejects: {report['P5_capacity_rejects']}; confirmed useful promotions: {report['P5_confirmed_useful_promotions']}. If PSA never triggers in P5, the fixed 3510-block logical reservation may overlap with the progress-safe condition on this workload; that does not establish equivalence.

PSA trace invariant violations: {violations}. Server crashes: {report['server_crashes']}; OOM: {report['OOM']}; final physical lease leaks: {lease_leaks}; logical leaks: {logical_leaks}; unsafe logical releases: {unsafe_releases}.

Limitation: only the frozen eight first-boundary hybrid jobs, oracle TraceToolProxy timing, pinned vLLM scheduler, and GPU0 were tested. The retained compact artifacts establish usefulness via lease acquisition before arrival; they do not retain the endpoint's exact T6 timestamp for each promotion, so this count is conservative. No latency winner, performance matrix, predictor, or 32-boundary replay is claimed.
"""
    (OUT / "summary.md").write_text(summary, encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
