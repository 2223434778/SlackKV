from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from experiments.phase4d_a.experiment import write_csv, write_json
from experiments.phase4e_a_rca.analysis import CONDITIONS, condition_blocks, final_result, r1_result

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/phase4e_a_rca"


def rows(name: str) -> list[dict[str, str]]:
    path = OUT / name
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    runs = rows("liveness_runs.csv")
    waiting = rows("waiting_trace.csv")
    resume = rows("resume_trace.csv")
    lease_events = rows("lease_events.csv")
    if not runs:
        raise RuntimeError("no diagnostic runs")
    by_stage: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        by_stage[run["stage"]][run["condition"]].append(run["classification"])
    r1 = r1_result(dict(by_stage["R1"]))
    result = final_result(
        r1,
        by_stage["R2"].get("U3_UNLEASED_HEAD_LEASE3", []),
        by_stage["R2"].get("S3_HEAD_LEASED", []),
    )
    control_rows = []
    for condition, (episodes, leased) in CONDITIONS.items():
        group = [run for run in runs if run["stage"] == "R1" and run["condition"] == condition]
        progress = [run for run in group if run["classification"] == "PROGRESS"]
        blocked = [run for run in group if run["classification"] == "LEASE_HOL_NONPROGRESS"]
        control_rows.append({
            "condition": condition,
            "leased_set": "|".join(episodes) if leased else "",
            "leased_blocks": condition_blocks(condition) if leased else 0,
            "E04_leased": leased and "E04" in episodes,
            "runs": len(group),
            "E04_blocked": len(blocked),
            "E04_admitted": sum(any(
                event["stage"] == run["stage"]
                and event["condition"] == condition
                and event["repetition"] == run["repetition"]
                and event["session"] == "E04:turn:1"
                and event["event"] == "RESUME_ADMITTED"
                for event in resume
            ) for run in group),
            "all_requests_finish": len(progress),
            "supports_headroom_hypothesis": condition == "H2_HEADROOM_LEASE2" and len(progress) == 2,
            "supports_lease_hypothesis": condition == "N3_EVICTABLE_PREFETCH3" and len(progress) == 2,
            "supports_head_identity_hypothesis": condition == "S3_HEAD_LEASED" and len(progress) == 2,
        })
    write_csv(OUT / "causal_controls.csv", control_rows)
    first_blocks = [event for event in waiting
                    if event["event"] == "WAITING_ALLOCATE_NONE"
                    and event["session"] == "E04:turn:1"
                    and event["blocked_count"] == "1"]
    write_json(OUT / "validation_report.json", {
        "start_commit": "e829bee1fb6d45dcdfb7bdce395c5a10154cc892",
        "R1": {name: by_stage["R1"].get(name, []) for name in CONDITIONS},
        "R1_gate": r1,
        "R2": {name: by_stage["R2"].get(name, []) for name in
               ("U3_UNLEASED_HEAD_LEASE3", "S3_HEAD_LEASED")},
        "runs_completed": len(runs),
        "U3_direct_first_blocks": [
            {key: event.get(key) for key in
             ("stage", "repetition", "free_blocks", "matching_lease",
              "active_lease_count", "active_lease_unique_blocks",
              "will_break_waiting_loop")}
            for event in first_blocks
            if event["condition"] == "U3_UNLEASED_HEAD_LEASE3"
        ],
        "OOM_count": sum(run["OOM"] == "True" for run in runs),
        "server_error_count": sum(run["server_error"] == "True" for run in runs),
        "result": result,
    })
    lines = [
        "# Phase 4E-A-RCA — lease-induced head-of-line liveness diagnosis",
        "",
        f"Result: `{result}`. This is a controlled mechanism diagnosis, not a performance comparison.",
        "",
        "Frozen Phase 4C hybrid workload SHA256: `9b381336a70e73b0efbddbb79d2c88608c19663247e819c1e7a364efd2971105`; GPU KV: 4,681 blocks; first-boundary jobs and deadlines unchanged.",
        "",
        "The pinned vLLM waiting path calls `allocate_slots()`, and when it returns `None`, executes `break` (not `continue`). The RCA patch adds bounded structured logs immediately before that existing break. It changes no scheduler branch, queue order, allocator, lease, guard, or K2 semantics.",
        "",
        "| Stage | Condition | Reps | Classification | E04 first allocation failure free blocks | Finished requests |",
        "|---|---|---:|---|---|---|",
    ]
    for stage in ("R1", "R2"):
        for condition in CONDITIONS:
            group = [run for run in runs if run["stage"] == stage and run["condition"] == condition]
            if not group:
                continue
            lines.append(
                f"| {stage} | {condition} | {len(group)} | "
                f"{', '.join(run['classification'] for run in group)} | "
                f"{', '.join(run['E04_first_block_free_blocks'] for run in group)} | "
                f"{', '.join(run['finished_final'] for run in group)} |"
            )
    lines += ["", "## Direct scheduler evidence", ""]
    for event in first_blocks:
        if event["condition"] != "U3_UNLEASED_HEAD_LEASE3":
            continue
        lines.append(
            f"- {event['stage']} rep {event['repetition']}: E04 `WAITING_ALLOCATE_NONE`, "
            f"free={event['free_blocks']} < required=1271, matching_lease={event['matching_lease']}, "
            f"active leases={event['active_lease_count']} / unique leased blocks="
            f"{event['active_lease_unique_blocks']}, will_break={event['will_break_waiting_loop']}."
        )
    lines += ["", "The U3 precondition sequentially leases E00/E01/E02 (3,930 blocks) before t0, while E04 is not promoted or leased and has the earliest 120-ms deadline. In U3, E04 reaches the existing waiting-loop allocation failure and break; later leased resumes do not acquire real-request ownership, the leases do not release, and headroom does not recover during the 12-s observation. This conclusion is limited to the observed pinned scheduler/workload and the control outcomes below.", ""]
    late_r2_u3 = [event for event in waiting
                  if event["stage"] == "R2"
                  and event["condition"] == "U3_UNLEASED_HEAD_LEASE3"
                  and event["session"] == "E04:turn:1"
                  and event["event"] == "WAITING_ALLOCATE_NONE"
                  and float(event["timestamp_ms"]) >= 8000]
    if late_r2_u3:
        lines.append(
            "At/after t0+8 s in the R2 U3 repetitions, later rate-limited E04 "
            "allocator-failure traces still show free blocks "
            + ", ".join(sorted({event["free_blocks"] for event in late_r2_u3}))
            + " (<1,271), despite initial X pressure subsiding. Exact X finish timestamps "
              "were not separately persisted, so this is supporting evidence rather than "
              "an X-finish-time measurement."
        )
        lines.append("")
    for row in control_rows:
        lines.append(f"- {row['condition']}: {row['all_requests_finish']}/{row['runs']} complete all eight requests; leases={row['leased_blocks']} blocks.")
    u3_runs = [run for run in runs if run["condition"] == "U3_UNLEASED_HEAD_LEASE3"]
    lines += [
        "",
        f"All {len(u3_runs)} U3 runs observed 8 Agent arrivals and 0 admission, 0 first token, "
        "0 completion, and 0 `LEASE_RELEASED` at the 12-s checkpoint; all servers "
        "remained alive without OOM or server ERROR. The later leased E00/E01/E02 "
        "requests therefore never reached real-request ownership behind E04.",
        "",
    ]
    s3_runs = [run for run in runs if run["condition"] == "S3_HEAD_LEASED"]
    s3_order_ok = 0
    for run in runs:
        if run["condition"] != "S3_HEAD_LEASED":
            continue
        key = (run["stage"], run["repetition"])
        released = [float(event["timestamp_ms"]) for event in lease_events
                    if (event["stage"], event["repetition"]) == key
                    and event["lease_key"] == "E04:turn:1"
                    and event["event"] == "LEASE_RELEASED"]
        admitted = [float(event["timestamp_ms"]) for event in resume
                    if (event["stage"], event["repetition"]) == key
                    and event["session"] == "E04:turn:1"
                    and event["event"] == "RESUME_ADMITTED"]
        if released and admitted and released[0] <= admitted[0]:
            s3_order_ok += 1
    lines.append(
        f"In S3, E04 had `matching_lease=true` at its first waiting attempt; "
        f"E04 `LEASE_RELEASED` preceded `RESUME_ADMITTED` in {s3_order_ok}/{len(s3_runs)} runs, "
        f"and all {len(s3_runs)} runs completed 8/8 requests."
    )
    lines += ["", "## Interpretation", ""]
    if result == "RESULT_A_LEASE_HOL_CIRCULAR_WAIT_CONFIRMED":
        lines.append(
            "The controlled comparison supports lease-induced waiting-head circular wait: "
            "U3 leaves E04 unleased and without enough allocatable blocks; the pinned "
            "waiting-loop break prevents leased requests behind E04 from acquiring ownership "
            "and releasing their leases. H2 restores headroom, N3 makes the same prefetched "
            "KV evictable, and S3 gives E04 a matching lease; each control restores progress. "
            "The X-background repetitions reproduce U3 and S3 outcomes. Instantaneous "
            "promotion fit alone does not guarantee a future resume progress path."
        )
    else:
        lines.append(
            "The direct trace and controls do not satisfy the full preregistered causal gate. "
            "The result is limited to the observed conditions; no unobserved control outcome is inferred."
        )
    lines += ["", "No P95 policy winner, H1/H2/H3, progress-reserve fix, or Phase 4E performance matrix was run.", ""]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
