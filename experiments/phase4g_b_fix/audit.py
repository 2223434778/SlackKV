"""Post-run audit of the frozen Phase 4G-B-FIX matrix and vLLM delta."""
from __future__ import annotations

import csv
import difflib
import gzip
import hashlib
import json
from pathlib import Path

from experiments.phase4g_b_fix.run import OUT, TMP_LOGS, _annotate_context, _write_json
from experiments.phase4g_b import run_phase4g_b as original


def _delta_patch() -> str:
    source = original.VLLM_SOURCE
    scheduler_path = source / "vllm/v1/core/sched/scheduler.py"
    current = scheduler_path.read_text(encoding="utf-8")
    helper_start = current.index("def _slackkv_reject_blocked_native_prefetch(")
    helper_end = current.index("def _slackkv_psa_decision(", helper_start)
    previous = current[:helper_start] + current[helper_end:]
    branch_start = previous.index("                    if _slackkv_reject_blocked_native_prefetch(request):")
    branch_end = previous.index("                    if _is_slackkv_trace_resume(request):", branch_start)
    assert 'reason="INSUFFICIENT_SCHEDULABLE_BLOCKS"' in previous[branch_start:branch_end]
    previous = previous[:branch_start] + previous[branch_end:]

    test_path = source / "tests/v1/kv_connector/unit/test_slackkv_connector_first.py"
    current_test = test_path.read_text(encoding="utf-8")
    previous_test = current_test.replace("    _slackkv_reject_blocked_native_prefetch,\n", "", 1)
    test_start = previous_test.index("def test_only_guarded_native_prefetch_uses_allocation_failure_fallback():")
    test_end = previous_test.index("def test_special_request_model_tokens_violate_invariant():", test_start)
    previous_test = previous_test[:test_start] + previous_test[test_end:]

    chunks = []
    for name, old, new in (
        ("vllm/v1/core/sched/scheduler.py", previous, current),
        ("tests/v1/kv_connector/unit/test_slackkv_connector_first.py", previous_test, current_test),
    ):
        chunks.extend(difflib.unified_diff(
            old.splitlines(keepends=True), new.splitlines(keepends=True),
            fromfile=f"a/{name}", tofile=f"b/{name}",
        ))
    patch = "".join(chunks)
    assert "INSUFFICIENT_SCHEDULABLE_BLOCKS" in patch
    return patch


def audit() -> dict[str, object]:
    report = json.loads((OUT / "validation_report.json").read_text(encoding="utf-8"))
    assert report["measured_runs_valid"] == report["expected_measured_runs"] == 12
    raw_paths = sorted((OUT / "runs").glob("measured_*/run.json"))
    compressed_paths = sorted((OUT / "runs").glob("measured_*/run.json.gz"))
    paths = raw_paths if raw_paths else compressed_paths
    runs = []
    for path in paths:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                runs.append(json.load(stream))
        else:
            runs.append(json.loads(path.read_text(encoding="utf-8")))
    assert len(runs) == 12
    assert all(row["valid"] and row["turns_complete"] == 40 and row["boundaries"] == 32 for row in runs)
    assert all(row["max_pre_first"] <= 2 and not row["OOM"] and row["server_alive"] for row in runs)
    assert all(row["lease_leaks"] == row["logical_leaks"] == 0 for row in runs)
    assert all(row["resume_trace_counts"] == {
        "RESUME_ARRIVED": 32, "FIRST_TOKEN": 32, "REQUEST_FINISHED": 32,
    } for row in runs)
    background = [{"run_id": row["run_id"], "condition": row["condition"],
                   "tokens": row["background_validation"]["tokens"],
                   "max_inter_chunk_idle_gap_ms": row["background_validation"]["max_inter_chunk_idle_gap_ms"],
                   "tail_last_first_token_ms": row["background_validation"]["tail_last_first_token_ms"],
                   "tail_last_request_finish_ms": row["background_validation"]["tail_last_request_finish_ms"],
                   "valid": row["background_validation"]["valid"]}
                  for row in runs if row["background_enabled"]]
    assert len(background) == 8
    assert all(row["valid"] and row["tokens"] == 8192 and
               row["max_inter_chunk_idle_gap_ms"] <= 50 and
               row["tail_last_first_token_ms"] > 0 and
               row["tail_last_request_finish_ms"] > 0 for row in background)
    with (OUT / "background_audit.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(background[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(background)

    events = []
    for run in runs:
        log = TMP_LOGS / "server_logs" / f"measured_rep{run['repetition']}_{run['policy'].lower()}.log"
        assert log.is_file(), log
        for line in log.open(encoding="utf-8", errors="replace"):
            marker = "SLACKKV_PREFETCH_ALLOC_TRACE "
            if marker not in line:
                continue
            event = json.loads(line.split(marker, 1)[1])
            if event.get("event") != "ALLOCATION_REJECT":
                continue
            matches = [(key, value["response"]) for key, value in run["promotions"].items()
                       if value.get("response", {}).get("reason") == "INSUFFICIENT_SCHEDULABLE_BLOCKS"]
            assert len(matches) == 1, (run["run_id"], matches)
            boundary_id, response = matches[0]
            episode_id, boundary_index = boundary_id.split(":b", 1)
            lease_key = f"{episode_id}:turn:{int(boundary_index) + 1}"
            for field in ("transfer_bytes", "external_tokens_loaded", "model_forward_count",
                          "generated_token_count", "mixed_prefetch_model_step_count"):
                assert response[field] == 0, (run["run_id"], field, response[field])
            assert response["status"] == "capacity_rejected"
            assert not any(e.get("event") == "LEASE_ACQUIRED" and e.get("lease_key") == lease_key
                           for e in run["lease_events"])
            events.append({
                "run_id": run["run_id"], "boundary_id": boundary_id, "lease_key": lease_key,
                "request_id": event["request_id"], "free_blocks": event["free_blocks"],
                "inflight_reserved_blocks": event["inflight_reserved_blocks"],
                "schedulable_blocks": event["schedulable_blocks"],
                "local_tokens": event["local_tokens"], "external_tokens": event["external_tokens"],
                "required_blocks": response["required_blocks"], "status": response["status"],
                "reason": response["reason"], "client_response_ms": response["client_response_ms"],
                "transfer_bytes": response["transfer_bytes"], "lease_acquired": False,
                "turns_complete": run["turns_complete"], "boundaries_complete": run["boundaries"],
            })
    assert events, "full matrix did not exercise repaired allocation branch"
    statuses: dict[str, int] = {}
    for run in runs:
        for promotion in run["promotions"].values():
            status = promotion["response"]["status"]
            statuses[status] = statuses.get(status, 0) + 1
    with (OUT / "allocation_reject_events.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(events[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(events)

    with (OUT / "boundary_metrics.csv").open(newline="", encoding="utf-8") as stream:
        milestones = [row for row in csv.DictReader(stream) if row["condition"] == "M2" and row["source_in_time"] == "True"]
    with (OUT / "counterfactual_constraints.csv").open(newline="", encoding="utf-8") as stream:
        fit_keys = {(row["run_id"], row["boundary_id"]) for row in csv.DictReader(stream)
                    if row["condition"] == "M2" and row["source_fully_ready"] == "True"
                    and row["physical_fit_full_prefix"] == "True"}
    physically_fit = [row for row in milestones if (row["run_id"], row["boundary_id"]) in fit_keys]
    no_safe = sum(float(row["full_psa_safe_window_ms"]) < 20 for row in physically_fit)
    assert len(milestones) == 128 and no_safe > len(physically_fit) / 2

    result = {
        "measured_runs_valid": len(runs), "turns_complete": sum(row["turns_complete"] for row in runs),
        "dynamic_boundaries_complete": sum(row["boundaries"] for row in runs),
        "allocation_reject_branch_events": len(events),
        "BGv2_valid_runs": len(background),
        "BGv2_min_tail_last_first_token_ms": min(row["tail_last_first_token_ms"] for row in background),
        "BGv2_max_inter_chunk_idle_gap_ms": max(row["max_inter_chunk_idle_gap_ms"] for row in background),
        "promotion_response_statuses": statuses,
        "M2_source_in_time": len(milestones),
        "M2_source_in_time_and_physical_fit": len(physically_fit),
        "M2_physical_fit_without_20ms_PSA_safe_window": no_safe,
        "M2_max_boundary_probe_gap_ms": max(float(row["source_probe_max_gap_ms"]) for row in milestones),
        "classification": report["result"],
        "caveat": "Sampled counterfactual windows are not a continuous-time proof; gaps may hide shorter opportunities.",
    }
    _write_json(OUT / "repair_audit.json", result)
    patch = _delta_patch()
    (OUT / "vllm_phase4g_b_fix_delta.patch").write_text(patch, encoding="utf-8")
    _annotate_context({"vllm_current_diff_sha256": report["vllm_working_diff_sha256"]}, result=report["result"])
    patch_sha = hashlib.sha256(patch.encode("utf-8")).hexdigest()
    for name in ("config.json", "source_manifest.json"):
        path = OUT / name
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata["phase4g_b_fix_incremental_patch_sha256"] = patch_sha
        _write_json(path, metadata)
    summary = OUT / "summary.md"
    prior = summary.read_text(encoding="utf-8")
    marker = "\n## Independent completion and classification audit\n"
    summary.write_text(prior.split(marker, 1)[0] + "\n", encoding="utf-8")
    with summary.open("a", encoding="utf-8") as stream:
        stream.write("\n## Independent completion and classification audit\n\n")
        stream.write(f"All {len(runs)}/12 fresh measured runs completed: {result['turns_complete']}/480 turns and "
                     f"{result['dynamic_boundaries_complete']}/384 dynamic boundaries. K2 stayed ≤2, with no final "
                     "lease or logical leaks, server crash, or OOM.\n\n")
        stream.write(f"BGv2 passed {len(background)}/8 measured runs: each produced 8192 tokens; "
                     f"minimum tail past the last Agent first token was "
                     f"{result['BGv2_min_tail_last_first_token_ms']:.2f} ms, and maximum inter-chunk gap was "
                     f"{result['BGv2_max_inter_chunk_idle_gap_ms']:.2f} ms (<50 ms). "
                     "The inherited `run_metrics.csv` tail columns are `NA`; exact values retained in each `run.json` "
                     "are extracted into `background_audit.csv`.\n\n")
        stream.write(f"The repaired allocation branch was exercised {len(events)} time(s). "
                     "The guarded request returned `capacity_rejected`, zero transfer/model work, and no lease; "
                     "its real-resume run completed. Exact evidence is in `allocation_reject_events.csv`.\n\n")
        stream.write(f"Across the D runs, promotion endpoint responses were {statuses}; "
                     "this is mechanism accounting, not a latency comparison.\n\n")
        stream.write(f"Frozen Result D criterion: in M2, {len(milestones)} source-in-time boundaries included "
                     f"{len(physically_fit)} with at least one sampled physical fit; "
                     f"{no_safe}/{len(physically_fit)} had no sampled ≥20-ms full-prefix PSA-safe window (>50%). "
                     "This supports PSA-safe headroom as the classified blocker, not a performance winner.\n\n")
        stream.write(f"Sampling limitation: per-boundary probe gaps reached "
                     f"{result['M2_max_boundary_probe_gap_ms']:.2f} ms. A short unobserved window cannot be excluded; "
                     "the classification is conditional on the frozen sampled-window method.\n")
    # Keep the complete per-run records recoverable without committing their
    # uncompressed duplicate text alongside the compact CSV artifacts.
    for path in sorted((OUT / "runs").glob("*/run.json")):
        data = path.read_bytes()
        target = path.with_suffix(path.suffix + ".gz")
        with target.open("wb") as raw_stream:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw_stream,
                               compresslevel=9, mtime=0) as stream:
                stream.write(data)
        with gzip.open(target, "rb") as stream:
            assert stream.read() == data, target
        path.unlink()
    return result


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, sort_keys=True))
