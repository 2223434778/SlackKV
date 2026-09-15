#!/usr/bin/env python3
"""Phase 1C-Fix2: event-correct QueueJIT selective-offload replay."""
import csv
import heapq
import json
import math
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase1b_fix.run_phase1b_fix import (MAX_CONTEXT, extract_events,
    poisson_workload, quantiles, session_workload, write_csv)
from slackkv.simulator.admission import admit, slack_ratio
from slackkv.simulator.policies import queue_safe_idle_s
from slackkv.simulator.selective_offload import local_metrics
from slackkv.simulator.simulator import EPS
from slackkv.simulator.transfer_model import TransferModel

KIB = 1024
PRIMARY, STRESS = 56 * KIB, 128 * KIB
POLICIES = ["KeepAll", "OffloadAll-QueueJIT", "FeasibilityOracle"] + \
    [f"FixedTTL-{value}" for value in (.1, .25, .5, 1, 2, 5, 10)] + \
    [f"SlackRatio-{value}" for value in (1, 1.25, 1.5, 2, 3, 4)]
PREFIX = ("workload", "footprint_bytes_per_token", "concurrency", "rho", "seed", "policy")


def run_sessions(groups, offsets, footprint, policy, model):
    """Closed-loop replay with explicit session, ready, H2D, and idle-expiry events."""
    states, future, pending, records = {}, [], [], []
    serial = 0
    now = math.inf
    busy = 0.0
    diagnostics = Counter()
    for order, (sid, source_events) in enumerate(sorted(groups.items())):
        events = [dict(event) for event in source_events]
        if not events:
            continue
        events[0]["actual"] = offsets[sid]
        states[sid] = {"events": events, "index": 0, "origin": events[0]["release_epoch_s"],
                       "offset": offsets[sid], "shift": 0.0, "order": order}
        heapq.heappush(future, (offsets[sid], order, sid))
        now = min(now, offsets[sid])

    def advance(sid, record, completion):
        state = states[sid]
        stall = max(0.0, completion - record["deadline"])
        record["completion"] = completion
        record["stall"] = stall
        state["shift"] += stall
        state["index"] += 1
        if state["index"] == len(state["events"]):
            return
        event = state["events"][state["index"]]
        nominal = state["offset"] + event["release_epoch_s"] - state["origin"] + state["shift"]
        event["actual"] = max(nominal, completion)
        heapq.heappush(future, (event["actual"], state["order"], sid))

    def release(sid):
        nonlocal serial
        state = states[sid]
        event = state["events"][state["index"]]
        size = event["reusable_tokens"] * footprint
        down = model[0].piecewise_service_time_s(size)
        up = model[1].piecewise_service_time_s(size)
        ratio = slack_ratio(event["pause_s"], down, up)
        offload = admit(policy, event["pause_s"], down, up)
        record = {"sid": sid, "idx": state["index"], "release": event["actual"],
                  "deadline": event["actual"] + event["pause_s"], "pause": event["pause_s"],
                  "size": size, "size_gb": size / 1e9, "down": down, "up": up, "ratio": ratio,
                  "offload": offload, "ready_time": event["actual"] + down, "start": None,
                  "completion": event["actual"] + event["pause_s"], "stall": 0.0}
        records.append(record)
        if offload:
            heapq.heappush(pending, (record["ready_time"], serial, record))
            serial += 1
        else:
            advance(sid, record, record["deadline"])

    while future or pending:
        while future and future[0][0] <= now + EPS:
            _, _, sid = heapq.heappop(future)
            release(sid)
        ready = [entry[2] for entry in pending if entry[0] <= now + EPS]
        if not ready:
            wakeups = []
            if future:
                wakeups.append((future[0][0], "session_release"))
            if pending:
                wakeups.append((pending[0][0], "promotion_ready"))
            if not wakeups:
                break
            next_time = min(value for value, _ in wakeups)
            for value, kind in wakeups:
                if abs(value - next_time) <= EPS:
                    diagnostics[f"{kind}_wakeups"] += 1
            now = next_time
            continue
        safe_idle = queue_safe_idle_s(now, [SimpleNamespace(deadline_s=item["deadline"],
            service_time_s=item["up"], job_id=f'{item["sid"]}:{item["idx"]}') for item in ready])
        if safe_idle > EPS:
            idle_end = now + safe_idle
            # A long closed-loop replay can accumulate a large absolute clock.
            # If a positive slack is below that clock's ULP, adding it leaves
            # ``now`` unchanged and would otherwise create an infinite idle loop.
            if idle_end <= now:
                safe_idle = 0.0
            else:
                wakeups = [(idle_end, "safe_idle_expiry")]
                if future and future[0][0] > now + EPS:
                    wakeups.append((future[0][0], "session_release"))
                later_ready = [entry[0] for entry in pending if entry[0] > now + EPS]
                if later_ready:
                    wakeups.append((min(later_ready), "promotion_ready"))
                next_time = min(value for value, _ in wakeups)
                diagnostics["intentional_idle_time_total"] += next_time - now
                diagnostics["intentional_idle_events"] += 1
                for value, kind in wakeups:
                    if abs(value - next_time) <= EPS:
                        diagnostics[f"{kind}_wakeups"] += 1
                now = next_time
                continue
        record = min(ready, key=lambda item: (item["deadline"], item["release"], item["sid"], item["idx"]))
        pending.remove(next(entry for entry in pending if entry[2] is record))
        # Removing a non-root EDF choice invalidates the ready-time heap used by
        # the next D2H-completion wakeup calculation.
        heapq.heapify(pending)
        record["start"] = now
        completion = now + record["up"]
        busy += record["up"]
        diagnostics["H2D_busy_time"] += record["up"]
        now = completion
        advance(record["sid"], record, completion)

    offloaded = [record for record in records if record["offload"]]
    assert all(record["start"] is not None for record in offloaded)
    assert all(record["completion"] + EPS >= record["start"] for record in offloaded)
    assert all(record["stall"] >= -EPS for record in records)
    metrics = local_metrics(records)
    span = max(max(record["deadline"] for record in records) - min(record["release"] for record in records), EPS)
    metrics["promotion_queue_utilization"] = busy / span
    for key in ("intentional_idle_time_total", "intentional_idle_events", "promotion_ready_wakeups",
                "session_release_wakeups", "safe_idle_expiry_wakeups", "H2D_busy_time"):
        metrics[key] = diagnostics[key]
    early = [record["deadline"] - record["completion"] for record in offloaded if record["completion"] < record["deadline"] - EPS]
    metrics["mean_promotion_lead_time"] = sum(early) / len(early) if early else 0.0
    return records, metrics


def static(events, footprint, policy, model):
    groups = {"p": [{**event, "release_epoch_s": event["release_time_s"]} for event in events]}
    return run_sessions(groups, {"p": 0.0}, footprint, policy, model)


def decisions(records, policy):
    values = Counter()
    for record in records:
        if policy.startswith("SlackRatio-"):
            gamma = float(policy.split("-", 1)[1])
            name = "intrinsically_nonoffloadable" if record["ratio"] < 1 else \
                "kept_by_safety_margin" if record["ratio"] < gamma else "admitted_for_offload"
        elif policy.startswith("FixedTTL-"):
            name = "admitted_by_ttl" if record["offload"] else "kept_by_ttl"
        else:
            name = "admitted_for_offload" if record["offload"] else "kept"
        values[name] += 1
    return values


def main():
    out = ROOT / "results" / "phase1c_fix2"
    out.mkdir(parents=True, exist_ok=True)
    events, skips, _ = extract_events(ROOT / "data/tracelab/syfi_coding_trace.duckdb")
    events = [event for event in events if 0 < event["reusable_tokens"] <= MAX_CONTEXT]
    model = (TransferModel.from_phase0_direction(ROOT, "gpu0_to_pinned_cpu"),
             TransferModel.from_phase0_direction(ROOT, "pinned_cpu_to_gpu0"))
    rows, admission_rows, miss_rows, c0_rows = [], [], [], []
    for footprint, seeds, concurrencies in ((PRIMARY, range(10), (4, 8, 16, 32, 64)), (STRESS, range(5), (16, 32, 64))):
        for concurrency in concurrencies:
            for seed in seeds:
                groups, offsets = session_workload(events, concurrency, seed)
                for policy in POLICIES:
                    records, metrics = run_sessions(groups, offsets, footprint, policy, model)
                    context = {"workload": "closed_loop_session", "footprint_bytes_per_token": footprint,
                               "concurrency": concurrency, "rho": "", "seed": seed, "policy": policy}
                    rows.append({**context, **metrics})
                    if policy == "FeasibilityOracle" or policy.startswith("SlackRatio-"):
                        for record in records:
                            if record["offload"] and record["pause"] + EPS < record["down"] + record["up"]:
                                c0_rows.append({**context, "job_id": f'{record["sid"]}:{record["idx"]}',
                                                "pause_s": record["pause"], "d2h_plus_h2d_s": record["down"] + record["up"]})
                    for reason, count in decisions(records, policy).items():
                        admission_rows.append({**context, "decision": reason, "count": count})
                    for record in records:
                        if record["stall"] > EPS and record["offload"]:
                            reason = "transfer_model_boundary_or_rounding" if abs(record["ratio"] - 1) < 1e-9 else \
                                "large_KV_blocking" if record["size"] >= 4 * (1024 ** 3) else "promotion_queue_collision"
                            miss_rows.append({**context, "reason": reason, "count": 1})
        for rho in (.25, .5, .75, .9):
            for seed in range(10):
                replay = poisson_workload(events, rho, 1000 + seed, footprint, model[1])
                for policy in POLICIES:
                    records, metrics = static(replay, footprint, policy, model)
                    context = {"workload": "poisson_tuple", "footprint_bytes_per_token": footprint,
                               "concurrency": "", "rho": rho, "seed": 1000 + seed, "policy": policy}
                    rows.append({**context, **metrics})
                    if policy == "FeasibilityOracle" or policy.startswith("SlackRatio-"):
                        for record in records:
                            if record["offload"] and record["pause"] + EPS < record["down"] + record["up"]:
                                c0_rows.append({**context, "job_id": f'{record["sid"]}:{record["idx"]}',
                                                "pause_s": record["pause"], "d2h_plus_h2d_s": record["down"] + record["up"]})
                    for reason, count in decisions(records, policy).items():
                        admission_rows.append({**context, "decision": reason, "count": count})
    fields = PREFIX + tuple(key for key in rows[0] if key not in PREFIX)
    write_csv(out / "aggregate_metrics.csv", fields, rows)
    write_csv(out / "admission_metrics.csv", PREFIX + ("decision", "count"), admission_rows)
    write_csv(out / "miss_analysis.csv", PREFIX + ("reason", "count"), miss_rows)
    write_csv(out / "c0_violations.csv", PREFIX + ("job_id", "pause_s", "d2h_plus_h2d_s"), c0_rows)
    assert not c0_rows
    characterization = []
    for footprint in (PRIMARY, STRESS):
        values = {"pause_ms": [event["pause_s"] * 1000 for event in events],
                  "kv_gb": [event["reusable_tokens"] * footprint / 1e9 for event in events],
                  "d2h_ms": [model[0].piecewise_service_time_s(event["reusable_tokens"] * footprint) * 1000 for event in events],
                  "h2d_ms": [model[1].piecewise_service_time_s(event["reusable_tokens"] * footprint) * 1000 for event in events]}
        values["roundtrip_ms"] = [a + b for a, b in zip(values["d2h_ms"], values["h2d_ms"])]
        values["slack_ratio"] = [pause / (roundtrip / 1000) for pause, roundtrip in zip(values["pause_ms"], values["roundtrip_ms"])]
        for name, value in values.items():
            characterization.extend({"footprint_bytes_per_token": footprint, "measure": name, "statistic": key, "value": stat}
                                   for key, stat in quantiles(value).items())
    write_csv(out / "characterization.csv", ("footprint_bytes_per_token", "measure", "statistic", "value"), characterization)
    sanity = []
    for size in (256 << 20, 1 << 30, 4 << 30, max(event["reusable_tokens"] for event in events) * PRIMARY):
        d2h, h2d = model[0].piecewise_service_time_s(size), model[1].piecewise_service_time_s(size)
        sanity.append({"size_bytes": size, "d2h_ms": d2h * 1000, "h2d_ms": h2d * 1000, "roundtrip_ms": (d2h + h2d) * 1000})
    write_csv(out / "transfer_sanity.csv", ("size_bytes", "d2h_ms", "h2d_ms", "roundtrip_ms"), sanity)
    config = {"phase0_calibration_commit": model[0].phase0_commit, "base_commit": "a7b6d4df5ae33fe0ac50d39aebcab23216a674df",
              "d2h_direction": model[0].direction, "h2d_direction": model[1].direction, "retained_events": len(events),
              "context_cap": MAX_CONTEXT, "footprints": [PRIMARY, STRESS], "policies": POLICIES, "c0": True,
              "d2h_assumption": "immediate and non-contended; optimistic", "matrix": "identical Phase 1C-Fix seeds, M, rho, policies, and footprints",
              "skips": dict(skips)}
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print("rows", len(rows), "C0", not c0_rows)


if __name__ == "__main__":
    main()
