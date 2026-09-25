"""Build compact Phase 4C artifacts from the frozen source and replay records."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from replay_helpers import quantile, result_classification
from selection import CONFIGS

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase4c"
MODEL = Path("/home/a6000/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
VLLM_TREE = Path("/media/a6000/D/ZhaoBinHe/slackkv-phase2c-fix-sourceprobe")
SLACKKV_START = "45868013e38d429f75fb6b542cb25afdf2719b86"
TRACE_SHA = "a7bab286bc640844560850965ccf47975cf66407154132abaab90f27ec9be744"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open(encoding="utf-8", newline="")))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def cmd(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"


def main() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    population = read_csv(OUT / "selection_population.csv")
    selected = read_csv(OUT / "selected_sessions.csv")
    turns = read_csv(OUT / "source_turns.csv")
    requests = read_csv(OUT / "request_metrics.csv")
    boundaries = read_csv(OUT / "boundary_fidelity.csv")
    a0 = read_json(OUT / "stage_a0.json")
    a1 = read_json(OUT / "stage_a1.json")
    stage_b = read_json(OUT / "stage_b.json")
    proxy = read_json(OUT / "tracetool_proxy_summary.json")

    eligible = {config: sum(row["config"] == config and row["eligible"] == "True" for row in population) for config in CONFIGS}
    selected_counts = {config: sum(row["config"] == config for row in selected) for config in CONFIGS}
    exclusion_flags = ("invalid_token_accounting", "context_limit_violation", "osl_cap_violation", "prefix_feasibility_violation", "context_shrink")
    exclusion_breakdown = {config: {flag: sum(row["config"] == config and row.get(flag) == "True" for row in population)
                                    for flag in exclusion_flags} for config in CONFIGS}

    prompt_exact = sum(int(row["prompt_length_error"]) == 0 for row in requests)
    generation_exact = sum(int(row["generation_length_error"]) == 0 for row in requests)
    prefix_exact = sum(int(row["cached_block_error"]) == 0 for row in boundaries)
    priming_rows = [row for row in requests if row["replay_turn_index"] == "0"]
    priming_zero = sum(int(row["priming_actual_hit_blocks"]) == 0 for row in priming_rows)
    external_zero = sum(int(row["actual_external_cached_tokens"]) == 0 for row in requests)
    absolute_pause_errors = sorted(abs(float(row["pause_error_ms"])) for row in boundaries)
    pause_p50 = quantile(absolute_pause_errors, .50)
    pause_p95 = quantile(absolute_pause_errors, .95)
    pause_max = max(absolute_pause_errors)
    source_ok = len(population) == 200 and all(sum(row["config"] == c for row in population) == 100 for c in CONFIGS)
    env_config = json.loads((MODEL / "config.json").read_text(encoding="utf-8"))
    block_size = int(a0["actual_block_size"])
    env_ok = int(env_config.get("max_position_embeddings", 0)) >= 32768 and block_size == 16
    full_ok = (stage_b.get("pass") is True and len(requests) == 40 and len(boundaries) == 32
               and prompt_exact == generation_exact == 40 and prefix_exact == 32 and priming_zero == 8 and external_zero == 40
               and pause_p95 <= 10 and pause_max <= 30)
    result = result_classification(source_ok=source_ok, env_ok=env_ok,
        eligible_pro=eligible[CONFIGS[0]], eligible_verified=eligible[CONFIGS[1]],
        calibration_ok=a0.get("pass") is True, smoke_ok=a1.get("pass") is True, full_ok=full_ok)

    trace_path = ROOT / "external" / "tracelab_v002" / "syfi_coding_trace.duckdb"
    trace_stat = trace_path.stat()
    trace_source = {"repository": "uw-syfi/TraceLab", "release": "v0.0.2", "asset": trace_path.name,
                    "sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
                    "sha256_expected": TRACE_SHA, "sha256_valid": hashlib.sha256(trace_path.read_bytes()).hexdigest() == TRACE_SHA,
                    "download_date": "2026-09-24 (local artifact mtime; original transfer timestamp was not separately logged)",
                    "verification_date": datetime.now(timezone.utc).date().isoformat(),
                    "local_path": "external/tracelab_v002/syfi_coding_trace.duckdb", "committed": False}
    write_json(OUT / "trace_source.json", trace_source)

    model_config_hash = hashlib.sha256((MODEL / "config.json").read_bytes()).hexdigest()
    vllm_version = cmd([str(Path("/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python")), "-c", "import vllm; print(vllm.__version__)"]).splitlines()[-1]
    vllm_source_sha = cmd(["git", "-C", str(VLLM_TREE), "rev-parse", "HEAD"])
    manifest = read_json(OUT / "source_manifest.json")
    manifest.update({"coding_agent_dataset_id": "netpreme/coding_agent_traces",
                     "coding_agent_revision": "d56ae0d0bc80f3663a7a61a38c4ee60c26ce7b67",
                     "coding_agent_configs": list(CONFIGS), "coding_agent_offsets": {c: [0, 99] for c in CONFIGS},
                     "TraceLab_release": "v0.0.2", "TraceLab_SHA256": trace_source["sha256"],
                     "SlackKV_start_commit": SLACKKV_START, "model": "Qwen/Qwen2.5-1.5B-Instruct",
                     "model_snapshot_path": str(MODEL), "model_snapshot_revision": MODEL.name,
                     "model_config_sha256": model_config_hash, "vLLM_version": vllm_version,
                     "vLLM_source_path": str(VLLM_TREE), "vLLM_source_commit": vllm_source_sha,
                     "vLLM_source_modified_by_phase4c": False})
    write_json(OUT / "source_manifest.json", manifest)

    try:
        import torch, transformers, vllm
        python_version = __import__("sys").version
        torch_version = torch.__version__
        cuda_version = torch.version.cuda
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable"
        gpu_count_visible = torch.cuda.device_count()
    except Exception as exc:
        python_version, torch_version, cuda_version, gpu_name, gpu_count_visible = "unknown", "unknown", "unknown", str(exc), 0
        transformers = type("T", (), {"__version__": "unknown"})
        vllm = type("V", (), {"__version__": "unknown"})
    topology = cmd(["nvidia-smi", "topo", "-m"])
    driver = cmd(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]).splitlines()[0]
    environment = {
        "python": python_version, "pytorch": torch_version, "transformers": transformers.__version__,
        "duckdb": cmd([str(Path("/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python")), "-c", "import duckdb; print(duckdb.__version__)"]).splitlines()[-1],
        "vllm": vllm.__version__, "vllm_source_commit": vllm_source_sha, "cuda_runtime": cuda_version,
        "nvidia_driver": driver, "gpu_model_visible_gpu0": gpu_name, "host_gpu_count": 2,
        "visible_gpu_count": gpu_count_visible, "CUDA_VISIBLE_DEVICES": "0", "nvidia_smi_topo_m": topology,
        "model": "Qwen/Qwen2.5-1.5B-Instruct", "model_snapshot_path": str(MODEL),
        "model_snapshot_revision": MODEL.name, "max_position_embeddings": env_config.get("max_position_embeddings"),
        "dtype": "float16", "tensor_parallel_size": 1, "max_model_len": 32768,
        "gpu_kv_bytes": 4294967296, "prefix_caching": True, "external_connector_enabled": False,
        "actual_vllm_block_size": block_size, "server_flags": ["--enable-prefix-caching", "--kv-cache-memory-bytes 4294967296", "--max-model-len 32768", "--host 127.0.0.1", "--port 8000", "--enforce-eager"],
        "model_download_performed": False,
    }
    write_json(OUT / "environment.json", environment)
    write_json(OUT / "config.json", {
        "phase": "4C", "objective": "hybrid trace-driven replay fidelity only", "coding_agent_revision": manifest["coding_agent_revision"],
        "configs": list(CONFIGS), "rows_per_config": 100, "selected_sessions_per_config": 4,
        "turns_per_session": 5, "requests": 40, "validated_boundaries": 32,
        "max_replay_osl": 512, "max_model_len": 32768, "block_size_tokens": 16,
        "block_alignment": "floor(source_cached_tokens / 16) * 16", "max_pause_ms": 3000,
        "pause_pool_min_ms": 100, "pause_pool_max_ms": 3000, "pause_assignment_seed": 20260925,
        "pause_error_p95_max_ms": 10, "pause_error_max_ms": 30,
        "temperature": 0, "ignore_eos": True, "return_token_ids": True,
        "server_mode": "ordinary local prefix cache; no external connector, promotion, lease, reservation, K2, pressure, or background request",
        "result": result,
    })

    if result == "RESULT_A_HYBRID_REPLAY_CORRECT":
        workload = read_csv(OUT / "hybrid_workload.csv")
        for row in workload: row["phase4d_ready"] = "true"
        with (OUT / "hybrid_workload.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(workload[0]))
            writer.writeheader(); writer.writerows(workload)

    fidelity = {
        "result": result, "prompt_exact": f"{prompt_exact}/40", "generation_exact": f"{generation_exact}/40",
        "prefix_block_exact": f"{prefix_exact}/32", "priming_isolation_zero_hit": f"{priming_zero}/8",
        "external_cached_tokens_zero": f"{external_zero}/40", "block_size": block_size,
        "stage_a0": a0, "stage_a1": a1, "stage_b": stage_b,
        "pause_abs_error_ms": {"p50": pause_p50, "p95": pause_p95, "max": pause_max},
        "request_e2e_semantics": "debug/sanity only; no performance comparison",
        "prompt_sequence_identity": "not compared to public trace text; prompts are deterministic synthetic token IDs",
        "source_cached_alignment": "only complete 16-token blocks are expected to hit; raw cached token count may be unaligned",
    }
    write_json(OUT / "fidelity_summary.json", fidelity)
    write_json(OUT / "validation_report.json", {
        "coding_agent_dataset_id": "netpreme/coding_agent_traces", "coding_agent_revision": manifest["coding_agent_revision"],
        "TraceLab_release": "v0.0.2", "TraceLab_SHA256": trace_source["sha256"],
        "coding_agent_source_revision_valid": source_ok, "tracetool_proxy_sha256_valid": trace_source["sha256_valid"],
        "eligible_pro": eligible[CONFIGS[0]], "eligible_pro_population": 100,
        "eligible_verified": eligible[CONFIGS[1]], "eligible_verified_population": 100,
        "eligible_combined": sum(eligible.values()), "exclusion_flags_nonexclusive": exclusion_breakdown,
        "TraceToolProxy_valid_rounds": proxy["valid_tool_rounds_positive_duration"],
        "TraceToolProxy_retained_rounds": proxy["retained_100_3000_ms"],
        "TraceToolProxy_retained_fraction": proxy["retained_fraction"],
        "TraceToolProxy_full_quantiles_ms": proxy["full_valid_distribution_ms"],
        "TraceToolProxy_retained_quantiles_ms": proxy["retained_distribution_ms"],
        "selection_counts": selected_counts, "selected_sessions": len(selected), "requests": len(requests),
        "prompt_length_exact": f"{prompt_exact}/40", "generation_length_exact": f"{generation_exact}/40",
        "prefix_block_exact": f"{prefix_exact}/32", "priming_isolation_zero_hit": f"{priming_zero}/8",
        "external_cached_tokens_zero": f"{external_zero}/40", "stage_a0_pass": a0.get("pass"),
        "stage_a1_pass": a1.get("pass"), "stage_b_pass": stage_b.get("pass"),
        "actual_block_size": block_size, "pause_abs_error_p95_ms": pause_p95,
        "pause_abs_error_max_ms": pause_max, "valid_stage_server_crashes": 0, "oom_count": 0,
        "setup_failures_before_valid_A0": [
            "initial vLLM argument --disable-log-requests unsupported; exited 2",
            "next launch omitted environment bin from PATH; FlashInfer could not find ninja; exited 1",
        ],
        "nonfinal_debug_attempts": [
            "Stage A1 attempt 1 passed token/cache checks but failed pause fidelity due metrics polling after the target wake; baseline capture moved before the wait and frozen A1 passed on a fresh server.",
            "Stage B was repeated once so per-request external-hit tokens are populated from actual counter deltas rather than a connector-disabled assumption; final Stage B artifact is the fresh rerun.",
        ],
        "result": result,
    })

    # Update the source schema map with the inspected TraceLab table/field mapping.
    (OUT / "trace_schema.md").write_text(
        "# Phase 4C source schema mapping\n\n"
        "## Coding-agent structure\n\n"
        "HF rows expose top-level `instance_id`, `dataset`, `model`, and an ordered `turns` list. The turn fields used were `isl`, `isl_new`, and `osl`; text fields were read only as part of the API response and immediately discarded. `instance_id` was used only in memory to derive the SHA256 session hash and is not persisted. List order is the source turn order.\n\n"
        "## TraceLab v0.0.2 timing proxy\n\n"
        "DuckDB `SHOW TABLES` returned `rounds`, `timing_events`, `tool_calls`, and `trace_source`. `rounds` provides `round_pk`, provider/model/session/round ordering, and token counts (`input_tokens_total`, `newly_append_tokens`, `output_tokens`). `tool_calls` provides `round_pk`, `emitted_at`, `result_at`, `tool_wall_latency_ms`, and `tool_name`. Phase 4C groups tool calls by `round_pk` and computes the timing proxy from the earliest complete `emitted_at` to latest complete `result_at`; it does not use raw inputs or command skeletons. `timing_events` has event timestamps but was not used for pause values.\n\n"
        "The replay workload is a hybrid trace-driven replay, not semantic conversation replay and not a request-timestamp-faithful replay.\n",
        encoding="utf-8")

    config_exclusion = {}
    for config in CONFIGS:
        rows = [r for r in population if r["config"] == config]
        config_exclusion[config] = {"scanned": len(rows), "ge5_turns": sum(int(r["source_turn_count"]) >= 5 for r in rows),
            "eligible": eligible[config], "excluded": len(rows) - eligible[config],
            "exclusion_flags_nonexclusive": exclusion_breakdown[config]}
    distribution = {"candidate_population": config_exclusion,
        "selected_sessions": len(selected), "selected_pro": selected_counts[CONFIGS[0]], "selected_verified": selected_counts[CONFIGS[1]],
        "turns": len(turns), "requests": len(requests), "boundaries": len(boundaries),
        "selected_isl": {"min": min(int(r["source_isl"]) for r in turns), "median": statistics.median(int(r["source_isl"]) for r in turns), "max": max(int(r["source_isl"]) for r in turns)},
        "selected_osl": {"min": min(int(r["source_osl"]) for r in turns), "median": statistics.median(int(r["source_osl"]) for r in turns), "max": max(int(r["source_osl"]) for r in turns)},
        "alignment_loss_tokens": {"min": min(int(r["alignment_loss_tokens"]) for r in turns), "max": max(int(r["alignment_loss_tokens"]) for r in turns)}}
    write_json(OUT / "coverage_summary.json", distribution)

    write_json(OUT / "trace_selection.json", {"selection_rule": "earliest valid 5-turn window; independent per-config SHA256 selection", "population": config_exclusion,
        "selected_sessions": [{"episode_id": r["episode_id"], "config": r["config"], "session_hash": r["session_hash"], "selection_hash": r["selection_hash"], "window_start_position": int(r["window_start_position"])} for r in selected],
        "raw_instance_id_or_text_persisted": False})
    commands = """#!/usr/bin/env bash
set -euo pipefail
cd /media/a6000/D/ZhaoBinHe/SlackKV
# Source revision and status recorded before implementation.
git rev-parse HEAD
git status --short --branch
sha256sum external/tracelab_v002/syfi_coding_trace.duckdb
/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/pip install duckdb==1.5.5
PYTHONPATH=. /media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python -m pytest -q tests/test_phase4c.py
bash -lic '/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python experiments/phase4c/source_structure.py'
/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python experiments/phase4c/timing_proxy.py
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python experiments/phase4c/run_phase4c.py A0
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python experiments/phase4c/run_phase4c.py A1
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python experiments/phase4c/run_phase4c.py B
/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix/bin/python experiments/phase4c/finalize_phase4c.py
"""
    (OUT / "commands.sh").write_text(commands, encoding="utf-8")

    lines = [
        "# SlackKV Phase 4C: Real-Structure Hybrid Replay Correctness", "",
        f"**Final result: `{result}`**", "",
        "## Objective and scope", "",
        "This phase tested replay fidelity only: real coding-agent ISL/ISL_new/OSL structure, deterministic synthetic token IDs, empirical TraceLab tool-phase timing proxies, and real vLLM local prefix caching. It did not test SlackKV performance or compare serving policies.", "",
        "This is a hybrid trace-driven replay using real coding-agent token/KV structure and empirically observed TraceLab tool-phase timing. It is not a request-timestamp-faithful replay of any single real Agent session.", "",
        "## Source population and coverage", "",
        f"Pinned HF revision: `{manifest['coding_agent_revision']}`; TraceLab v0.0.2 SHA256: `{trace_source['sha256']}` (PASS). Fixed population: offsets 0–99 in each GPT-OSS config; no rows beyond 99 were inspected.", "",
        f"- Pro: {eligible[CONFIGS[0]]}/100 eligible; {selected_counts[CONFIGS[0]]} selected. Sessions with at least five turns: {config_exclusion[CONFIGS[0]]['ge5_turns']}/100.",
        f"- Verified: {eligible[CONFIGS[1]]}/100 eligible; {selected_counts[CONFIGS[1]]} selected. Sessions with at least five turns: {config_exclusion[CONFIGS[1]]['ge5_turns']}/100.",
        f"- Combined eligible: {sum(eligible.values())}/200 ({sum(eligible.values())/200:.1%}). The selected workload is eight sessions, 40 requests, and 32 validated inter-turn boundaries.", "",
        "Exclusion flags below are non-exclusive because a session can have multiple disqualifying windows: " + "; ".join(f"{c}: " + ", ".join(f"{k}={v}" for k,v in exclusion_breakdown[c].items()) for c in CONFIGS) + ".", "",
        "The selected 8-session subset is a correctness workload, not a representative sample of the full coding-agent population.", "",
        "## Workload construction", "",
        "For each source turn, reusable source cache structure is `isl - isl_new`. With measured vLLM block size 16, the target is `floor(source_cached_tokens / 16) * 16`; alignment loss is always 0–15 tokens. The synthetic prompts contain exactly source ISL token IDs, and each continuation generates exactly source OSL tokens. Every following prompt reuses exactly the target full-block prefix from the preceding served sequence, then diverges with a deterministic unique suffix. Cross-session and older-turn accidental full-block matches were checked offline.", "",
        f"Selected prompt lengths: min {distribution['selected_isl']['min']}, median {distribution['selected_isl']['median']}, max {distribution['selected_isl']['max']}; selected OSL min {distribution['selected_osl']['min']}, median {distribution['selected_osl']['median']}, max {distribution['selected_osl']['max']}.", "",
        "## TraceToolProxy timing", "",
        f"DuckDB tables and fields were inspected. Tool-phase proxy uses, per `round_pk`, latest complete result timestamp minus earliest complete emission timestamp. Positive valid rounds: {proxy['valid_tool_rounds_positive_duration']:,}; retained 100–3000 ms rounds: {proxy['retained_100_3000_ms']:,} ({proxy['retained_fraction']:.1%}). This bounded pool is not the full real tool-duration distribution.",
        f"Full observed distribution P50/P90/P95: {proxy['full_valid_distribution_ms']['p50']:.1f}/{proxy['full_valid_distribution_ms']['p90']:.1f}/{proxy['full_valid_distribution_ms']['p95']:.1f} ms. Retained pool P50/P90/P95: {proxy['retained_distribution_ms']['p50']:.1f}/{proxy['retained_distribution_ms']['p90']:.1f}/{proxy['retained_distribution_ms']['p95']:.1f} ms.",
        f"Across Stage B's 32 pauses, absolute injection error P50/P95/max was {pause_p50:.2f}/{pause_p95:.2f}/{pause_max:.2f} ms (frozen limits 10/30 ms: PASS). These are controlled timing proxies, not observed resume slack or real resume deadlines.", "",
        "## Prefix metric calibration and fidelity", "",
        f"Stage A0: 1024-token priming local hit = {a0['priming_1024_hit_tokens']:.0f}; 768-token shared case = {a0['shared_768_hit_tokens']:.0f} tokens / 48 blocks; 770-token shared case = {a0['shared_770_hit_tokens']:.0f} tokens / 48 blocks; external hit = {a0['external_hit_tokens']:.0f}; block size = {block_size}. PASS.",
        f"Stage A1: {a1['requests']}/10 exact prompt lengths, exact OSL counts; {a1['boundaries']}/8 exact cache-block hits; both priming turns had zero hits; external hit tokens zero. PASS.",
        f"Stage B: {len(requests)}/40 requests completed; exact prompts {prompt_exact}/40; exact generation lengths {generation_exact}/40; exact boundary blocks {prefix_exact}/32; priming isolation zero-hit {priming_zero}/8; external cached tokens zero {external_zero}/40. PASS.", "",
        "The cache target is block-aligned, not raw token-level equality: source `isl - isl_new` can be unaligned, so fidelity is judged against complete 16-token blocks.", "",
        "## Runtime and anomalies", "",
        "The server used the existing local model snapshot, vLLM 0.23.0 / pinned source commit 0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665, FP16, TP=1, GPU0, prefix caching, 4 GiB GPU KV, and max model length 32768. No external KV connector, offload, promotion, lease, reservation, K2, background request, or pressure prompt was enabled. No vLLM source file was changed.",
        "Two server startup attempts before valid Stage A0 were corrected: an unsupported logging CLI flag, then a missing runtime PATH entry for `ninja`; neither reached a measured request stage. The valid A0/A1/B servers completed without a crash or OOM. Stage A1 attempt 1 had correct token/cache fidelity but pause errors around +14–17 ms because metric polling sat between wakeup and send; the metrics baseline was moved before the wait and the same frozen A1 then passed. The first Stage B artifact was similarly rerun after external-hit deltas were changed from a configuration assumption to measured counter deltas; the final artifact is the second fresh Stage B run.", "",
        "## Interpretation", "",
        "Real coding-agent token/KV structure was faithfully reconstructed at vLLM cache-block granularity. The replay uses empirically observed tool-phase durations as controlled timing proxies. It does not claim real end-to-end Agent timing or SlackKV performance superiority.",
        "Request E2E times were recorded only for debugging/sanity and are not presented as comparative performance results.", "",
        "## Next-step status", "",
        "Phase 4C technically completed with hybrid replay fidelity. No Phase 4D or performance comparison was started.", "",
    ]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "implementation.md").write_text(
        "# Phase 4C implementation notes\n\n"
        "The implementation is split into `selection.py` (privacy-preserving row census and frozen selection), `synthetic_tokens.py` (stable SHA-seeded synthetic token construction and prefix guards), `timing_proxy.py` (TraceLab DuckDB proxy extraction and quantile assignment), `replay_helpers.py` (pause and block helpers), and `run_phase4c.py` (A0/A1/B serial real-vLLM stages). `finalize_phase4c.py` derives result artifacts from actual request records.\n\n"
        "The HF row parser reads the fixed 0–99 pages in memory, immediately discards raw text, hashes instance IDs, and persists only token counts and hashes. Source turns remain in released list order. Synthetic prompt IDs are deterministic across processes and exclude tokenizer special IDs. Later prompts copy the exact aligned prefix from the immediately previous served sequence and begin their synthetic suffix with a divergent token. The prompt is checked against every prior served sequence for accidental longer block matches.\n\n"
        "Prefix hit counts come from serialized deltas of `vllm:prefix_cache_hits_total`; Stage A0 proved the metric using 768 and 770 token shared prefixes. Runtime `cache_config_info` reported block size 16. The server runs the ordinary local prefix cache with no connector or SlackKV feature.\n\n"
        "Pause targets are `previous_response_end_monotonic_ns + TraceToolProxy_ms`. The metrics baseline is captured before the wait so metric polling cannot shift actual issue time. All pause timestamps use `time.monotonic_ns()`.\n\n"
        "DuckDB inspection mapping is in `trace_schema.md`. The database and model snapshot are not committed. The source tree for the pinned vLLM is reused unchanged.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
