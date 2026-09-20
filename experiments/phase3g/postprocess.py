from __future__ import annotations

import csv
import json
import re
import statistics
from pathlib import Path

from experiments.phase3f.helpers import write_csv

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase3g"
POLICY_NAMES = {
    "ondemand_trace": "OnDemand-Trace",
    "resumeaware_trace": "ResumeAware-Trace",
    "inenginefirsttokengate_resumeaware": "InEngineFirstTokenGate-ResumeAware",
}


def read_csv(path):
    return list(csv.DictReader(path.open()))


def percentile(values, q):
    values = sorted(values)
    if not values:
        return 0.0
    rank = (len(values) - 1) * q / 100
    lo, hi = int(rank), min(int(rank) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (rank - lo)


def overlap(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def interval_stats(intervals):
    events = sorted([(a, 1) for a, _ in intervals] + [(b, -1) for _, b in intervals])
    current = maximum = 0
    area = 0.0
    previous = events[0][0] if events else 0.0
    for timestamp, delta in events:
        area += (timestamp - previous) * current
        current += delta
        maximum = max(maximum, current)
        previous = timestamp
    return maximum, area


def parse_logs(context):
    events = []
    for path in (OUT / "server_logs").glob("*.log"):
        match = re.match(r"(.+)_(\d+)\.log$", path.name)
        if not match or match.group(1) == "control":
            continue
        raw_name = match.group(1).lower()
        policy = POLICY_NAMES.get(raw_name)
        if policy is None:
            continue
        rep = int(match.group(2))
        t0 = float(context[(policy, rep)])
        for line in path.read_text(errors="replace").splitlines():
            marker = "SLACKKV_RESUME_TRACE "
            if marker not in line:
                continue
            try:
                event = json.loads(line.split(marker, 1)[1].strip())
            except Exception:
                continue
            event.update({"policy": policy, "repetition": rep,
                          "timestamp_monotonic_raw": event["timestamp_monotonic"],
                          "timestamp_ms_from_t0": (event["timestamp_monotonic"] - t0) * 1000})
            events.append(event)
    return events


def main():
    context = {}
    with (OUT / "run_context.csv").open() as stream:
        for row in csv.DictReader(stream):
            context[(row["policy"], int(row["repetition"]))] = row["t0_monotonic_s"]
    events = parse_logs(context)
    fields = ["policy", "repetition", "session", "request_id", "event", "step_id",
              "timestamp_monotonic_raw", "timestamp_ms_from_t0", "request_status",
              "num_running", "num_waiting", "num_skipped_waiting",
              "gate_owner_request_id", "num_slackkv_pre_first_token",
              "num_scheduled_requests", "total_num_scheduled_tokens",
              "num_computed_tokens", "num_output_tokens"]
    write_csv(OUT / "scheduler_trace.csv", [{k: e.get(k, "NA") for k in fields} for e in events])

    steps = {}
    for e in events:
        steps.setdefault((e["policy"], e["repetition"], e["step_id"]), []).append(e)
    step_rows = []
    for (policy, rep, step), es in steps.items():
        step_rows.append({"policy": policy, "repetition": rep, "step_id": step,
            "step_timestamp_ms": min(e["timestamp_ms_from_t0"] for e in es),
            "num_running": max(e.get("num_running", 0) for e in es),
            "num_waiting": max(e.get("num_waiting", 0) for e in es),
            "num_slackkv_pre_first_token": max(e.get("num_slackkv_pre_first_token", 0) for e in es),
            "new_slackkv_resumes_admitted": sum(e["event"] == "RESUME_ADMITTED" for e in es),
            "num_scheduled_requests": ";".join(str(e.get("num_scheduled_requests", "NA")) for e in es),
            "total_num_scheduled_tokens": ";".join(str(e.get("total_num_scheduled_tokens", "NA")) for e in es),
            "sessions_admitted": ";".join(e.get("session", "") for e in es if e["event"] == "RESUME_ADMITTED"),
            "sessions_gate_blocked": ";".join(e.get("session", "") for e in es if e["event"] == "GATE_BLOCKED"),
            "sessions_first_token": ";".join(e.get("session", "") for e in es if e["event"] == "FIRST_TOKEN")})
    write_csv(OUT / "step_metrics.csv", step_rows)

    session_rows = read_csv(OUT / "session_metrics.csv")
    by = {}
    for event in events:
        by.setdefault((event["policy"], int(event["repetition"]), event.get("session")), []).append(event)
    for row in session_rows:
        es = by.get((row["policy"], int(row["repetition"]), row["session"]), [])
        def first(name):
            matches = [e for e in es if e["event"] == name]
            return min(matches, key=lambda e: e["timestamp_ms_from_t0"]) if matches else None
        arrival, wait, ready = first("RESUME_ARRIVED"), first("WAITING_FOR_REMOTE_KVS"), first("REMOTE_KV_READY")
        admission, model, token, finished = (first(name) for name in
                                              ("RESUME_ADMITTED", "FIRST_MODEL_SCHEDULED", "FIRST_TOKEN", "REQUEST_FINISHED"))
        blocked = [e for e in es if e["event"] == "GATE_BLOCKED"]
        adm_ms = admission["timestamp_ms_from_t0"] if admission else "NA"
        fin_ms = finished["timestamp_ms_from_t0"] if finished else "NA"
        row.update({
            "request_id": arrival.get("request_id", "NA") if arrival else "NA",
            "scheduler_arrival_ms": arrival["timestamp_ms_from_t0"] if arrival else "NA",
            "scheduler_arrival_step": arrival["step_id"] if arrival else "NA",
            "arrival_transport_delay_ms": (float(arrival["timestamp_ms_from_t0"]) - float(row["client_request_sent_ms"])) if arrival else "NA",
            "remote_kv_wait_start_ms": wait["timestamp_ms_from_t0"] if wait else "NA",
            "remote_kv_wait_start_step": wait["step_id"] if wait else "NA",
            "remote_kv_ready_ms": ready["timestamp_ms_from_t0"] if ready else "NA",
            "remote_kv_ready_step": ready["step_id"] if ready else "NA",
            "gate_first_blocked_ms": blocked[0]["timestamp_ms_from_t0"] if blocked else "NA",
            "gate_wait_ms": max(0.0, float(adm_ms) - float(blocked[0]["timestamp_ms_from_t0"])) if blocked and admission else 0.0,
            "gate_blocked_steps": len({e["step_id"] for e in blocked}),
            "scheduler_admission_ms": adm_ms,
            "scheduler_admission_step": admission["step_id"] if admission else "NA",
            "first_model_scheduled_ms": model["timestamp_ms_from_t0"] if model else "NA",
            "first_model_scheduled_step": model["step_id"] if model else "NA",
            "first_token_time_ms": token["timestamp_ms_from_t0"] if token else "NA",
            "first_token_step": token["step_id"] if token else "NA",
            "request_finished_ms": fin_ms,
            "request_finished_step": finished["step_id"] if finished else "NA",
            "gpu_ready_residency_ms": max(0.0, float(adm_ms) - float(row["promotion_completion_ms"])) if row["promotion_completion_ms"] != "NA" and admission else 0.0,
        })
        if admission and token:
            row["effective_TTFT_ms"] = float(token["timestamp_ms_from_t0"]) - float(row["original_deadline_ms"])
            row["admission_to_first_token_ms"] = float(token["timestamp_ms_from_t0"]) - float(admission["timestamp_ms_from_t0"])
        if token and finished:
            row["decode_duration_ms"] = float(finished["timestamp_ms_from_t0"]) - float(token["timestamp_ms_from_t0"])
            row["TPOT_ms"] = float(row["decode_duration_ms"]) / 31
            row["completion_latency_ms"] = float(finished["timestamp_ms_from_t0"]) - float(row["original_deadline_ms"])
    write_csv(OUT / "session_metrics.csv", session_rows)

    runs = read_csv(OUT / "run_metrics.csv")
    for run in runs:
        policy, rep = run["policy"], int(run["repetition"])
        rows = [r for r in session_rows if r["policy"] == policy and int(r["repetition"]) == rep]
        abc = [r for r in rows if r["session"] in "ABC"]
        pre = [(float(r["scheduler_admission_ms"]), float(r["first_token_time_ms"])) for r in abc]
        dec = [(float(r["first_token_time_ms"]), float(r["request_finished_ms"])) for r in abc]
        pre_pairs = [overlap(pre[i], pre[j]) for i in range(3) for j in range(i + 1, 3)]
        dec_pairs = [overlap(dec[i], dec[j]) for i in range(3) for j in range(i + 1, 3)]
        pre_max, pre_area = interval_stats([(float(r["scheduler_admission_ms"]), float(r["first_token_time_ms"])) for r in rows])
        dec_max, dec_area = interval_stats([(float(r["first_token_time_ms"]), float(r["request_finished_ms"])) for r in rows])
        admissions = {r["session"]: float(r["scheduler_admission_ms"]) for r in abc}
        tokens = {r["session"]: float(r["first_token_time_ms"]) for r in abc}
        finishes = {r["session"]: float(r["request_finished_ms"]) for r in abc}
        order_ok = admissions.get("A", 1e99) < admissions.get("B", 1e99) < admissions.get("C", 1e99)
        gate_rows = [r for r in rows if r["session"] in "ABC"]
        align = all(0 <= float(r["arrival_transport_delay_ms"]) < 50 for r in rows)
        release_overlap = (tokens.get("A", 1e99) < admissions.get("B", 0) < finishes.get("A", -1)
                           and tokens.get("B", 1e99) < admissions.get("C", 0) < finishes.get("B", -1))
        run.update({
            "ABC_pre_first_overlap_pair_count": sum(x > 0 for x in pre_pairs),
            "ABC_pre_first_pairwise_overlap_ms": sum(pre_pairs),
            "max_concurrent_pre_first_token": pre_max,
            "area_under_pre_first_token_concurrency_ms": pre_area,
            "ABC_total_decode_overlap_ms": sum(dec_pairs),
            "max_concurrent_post_first_token_decode": dec_max,
            "area_under_post_first_token_decode_concurrency_ms": dec_area,
            "post_first_token_release_overlap_validation": "PASS" if release_overlap else "FAIL",
            "ABC_total_gate_wait_ms": sum(float(r["gate_wait_ms"]) for r in gate_rows),
            "ABC_P50_TTFT_ms": percentile([float(r["effective_TTFT_ms"]) for r in abc], 50),
            "ABC_P95_TTFT_ms": percentile([float(r["effective_TTFT_ms"]) for r in abc], 95),
            "ABC_P50_completion_latency_ms": percentile([float(r["completion_latency_ms"]) for r in abc], 50),
            "ABC_P95_completion_latency_ms": percentile([float(r["completion_latency_ms"]) for r in abc], 95),
            "ABC_completion_makespan_ms": max(finishes.values()) - min(float(r["original_deadline_ms"]) for r in abc),
            "ABC_decode_throughput_tok_s": 93 / max((max(finishes.values()) - min(tokens.values())) / 1000, 1e-9),
            "overall_P95_TTFT_ms": percentile([float(r["effective_TTFT_ms"]) for r in rows], 95),
            "overall_P95_completion_latency_ms": percentile([float(r["completion_latency_ms"]) for r in rows], 95),
            "idle_gpu_memory_time_mib_s": sum(float(r["gpu_ready_residency_ms"]) for r in rows) * 112 / 1000,
            "ABC_same_step_admission_pairs": sum(float(a["scheduler_admission_step"]) == float(b["scheduler_admission_step"]) for i, a in enumerate(abc) for b in abc[i+1:]),
            "timeline_validation": "PASS" if all(float(r["client_request_sent_ms"]) >= float(r["original_deadline_ms"]) for r in rows) else "FAIL",
            "time_alignment_validation": "PASS" if align else "FAIL",
            "gate_invariant_validation": "PASS" if (policy != "InEngineFirstTokenGate-ResumeAware" or (order_ok and pre_max <= 1)) else "FAIL",
            "decode_overlap_validation": "PASS" if (policy != "InEngineFirstTokenGate-ResumeAware" or (dec_max >= 2 and sum(dec_pairs) > 0)) else "FAIL",
        })
    write_csv(OUT / "run_metrics.csv", runs)
    print("TRACE_EVENTS", len(events), "TIME_ALIGN", all(r["time_alignment_validation"] == "PASS" for r in runs))


if __name__ == "__main__":
    main()
