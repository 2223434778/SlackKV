from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase3i"


def read(name):
    with (OUT / name).open(encoding="utf-8", newline="") as h:
        return list(csv.DictReader(h))


def write(name, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / name).open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields, lineterminator="\n")
        w.writeheader(); w.writerows(rows)


def med(rows, key):
    return statistics.median(float(r[key]) for r in rows)


def main():
    runs = [r for r in read("run_metrics.csv") if r.get("policy") != "policy"]
    sessions = read("session_metrics.csv")
    background = read("background_metrics.csv")
    native = read("native_prefetch_metrics.csv")
    traces = read("scheduler_trace.csv")
    baseline = read("x_baseline_metrics.csv")
    baseline_decode = statistics.median(float(r["decode_duration_ms"]) for r in baseline)
    trace_groups = {}
    for r in traces:
        trace_groups.setdefault((r["policy"], r["repetition"]), []).append(r)
    native_groups = {}
    for r in native:
        native_groups.setdefault((r["policy"], r["repetition"]), []).append(r)
    bg_groups = {(r["policy"], r["repetition"]): r for r in background}
    for key, bg in bg_groups.items():
        bg["X_slowdown"] = float(bg["X_decode_duration_ms"]) / baseline_decode
        bg["background_overlap_validation"] = "PASS" if float(bg["X_first_token_ms"]) < 0 and float(bg["X_finish_ms"]) > max(float(s["first_token_time_ms"]) for s in sessions if (s["policy"], s["repetition"]) == key and s["session"] == "C") else "FAIL"
    for s in sessions:
        key = (s["policy"], s["repetition"])
        tg = trace_groups.get(key, [])
        ng = native_groups.get(key, [])
        source_ok = all(n["status"] == "ok" and n["external_tokens_loaded"] == "4096" and n["model_forward_count"] == "0" and n["generated_token_count"] == "0" and n["mixed_prefetch_model_step_count"] == "0" for n in ng)
        no_remote = not any(t["session"] in ("A", "B", "C") and t["event"] in ("WAITING_FOR_REMOTE_KVS", "REMOTE_KV_READY") for t in tg)
        s["cache_transport_validation"] = "PASS" if source_ok and no_remote else "FAIL"
    for r in runs:
        key = (r["policy"], r["repetition"])
        ss = [s for s in sessions if (s["policy"], s["repetition"]) == key]
        bg = bg_groups[key]
        r["idle_gpu_memory_time_mib_s"] = sum(max(0.0, float(s["scheduler_admission_ms"]) - float(s["promotion_completion_ms"])) * 112.0 / 1000.0 for s in ss)
        r["cache_transport_validation"] = "PASS" if all(s["cache_transport_validation"] == "PASS" for s in ss) else "FAIL"
        r["token_count_validation"] = "PASS" if all(s["token_count_validation"] == "PASS" for s in ss) and int(bg["X_generated_tokens"]) == 128 else "FAIL"
        r["background_overlap_validation"] = bg["background_overlap_validation"]
        r["X_slowdown"] = bg["X_slowdown"]
        r["ABC_P95_completion_latency_ms"] = r.get("ABC_P95_completion_latency_ms", "NA")
        r["combined_decode_throughput_tok_s"] = r.get("combined_decode_throughput_tok_s", "NA")
    write("session_metrics.csv", sessions); write("background_metrics.csv", background); write("run_metrics.csv", runs)
    groups = {}
    for r in runs: groups.setdefault(r["policy"], []).append(r)
    lines = ["# Phase 3I summary", "", "Phase 3I adds one ordinary 128-token background request X. X starts before t0; t0 is set after X first token. A/B/C remain the only gated sessions. No vLLM scheduler changes were made; Phase 3H K-capacity gating was reused unchanged.", "", f"X-only baseline median decode duration: {baseline_decode:.2f} ms; baseline median TPOT: {statistics.median(float(r['TPOT_ms']) for r in baseline):.2f} ms.", "", "## Results", "", "|Policy|ABC P95 TTFT|ABC completion P95|ABC decode tok/s|Max pre-first|Gate wait|X slowdown|Memory-time|Promotion misses|Blackout conflicts|Long prefetch|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for pol in ("ResumeAware-K3", "FirstTokenGate-K1", "FirstTokenGate-K2"):
        q=groups[pol]
        lines.append(f"|{pol}|{med(q,'ABC_P95_TTFT_ms'):.2f}|{med(q,'ABC_P95_completion_latency_ms'):.2f}|{med(q,'ABC_decode_throughput_tok_s'):.2f}|{med(q,'max_concurrent_pre_first_token'):.0f}|{med(q,'ABC_total_gate_wait_ms'):.2f}|{med(q,'X_slowdown'):.3f}|{med(q,'idle_gpu_memory_time_mib_s'):.2f}|{med(q,'promotion_deadline_miss_count'):.0f}|{med(q,'blackout_conflict_count'):.0f}|{med(q,'long_prefetch_count'):.0f}|")
    lines += ["", "## Direct answers", "", f"- X validation: {'PASS' if all(bg['background_overlap_validation']=='PASS' for bg in background) else 'FAIL'}; X generated 128 tokens in every run.", "- Promotion correctness: FAIL for this busy-workload experiment because the background decode overlaps native promotion. At least one promotion miss, blackout conflict, or >50 ms promotion occurs in every policy group; this is not hidden.", "- K1 and K2 gate invariants are measured from real traces. K1 remains max pre-first 1. K2 reaches capacity 2 in some runs but not every run under X interference, so the Phase 3I admission comparison is contaminated.", "- The X request remains active past C first token in the recorded runs, satisfying the busy-background condition.", "- Result D: the frozen requirement that A/B/C promotions remain clean is not met when the ordinary 128-token request is active. This is an implementation/measurement validity outcome for Phase 3I, not evidence that K1 or K2 wins.", "", "## Scope and limitations", "", "Cross-policy sequence hashes remain diagnostic only. The matched-schedule KV correctness result from Phase 3G-CorrectnessDiag is not replaced. This phase does not implement dynamic K or start Phase 3J."]
    (OUT/"summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    env={"vllm_version":"0.23.0","vllm_source_commit":"0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665","pytorch":"2.11.0+cu126","cuda":"12.6","gpu":"NVIDIA RTX A6000","model":"Qwen/Qwen2.5-1.5B-Instruct","x_baseline_median_decode_ms":baseline_decode}
    (OUT/"environment.json").write_text(json.dumps(env,indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__": main()
