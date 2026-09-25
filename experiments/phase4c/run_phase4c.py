"""Run Phase 4C's serial local-prefix-cache fidelity gates."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import signal
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from replay_helpers import actual_pause_ms, pause_error_ms, quantile
from synthetic_tokens import deterministic_token_ids, longest_full_block_match, make_next_prompt, token_hash

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase4c"
MODEL = Path("/home/a6000/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
ENV_ROOT = Path("/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix")
VLLM = ENV_ROOT / "bin" / "vllm"
SERVER = [str(VLLM), "serve", str(MODEL), "--dtype", "float16", "--tensor-parallel-size", "1",
          "--enable-prefix-caching", "--kv-cache-memory-bytes", "4294967296", "--max-model-len", "32768",
          "--host", "127.0.0.1", "--port", "8000", "--enforce-eager"]
BASE_ENV = {"CUDA_VISIBLE_DEVICES": "0", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "PATH": str(ENV_ROOT / "bin") + ":" + os.environ.get("PATH", "")}
API = "http://127.0.0.1:8000/v1/completions"
METRICS = "http://127.0.0.1:8000/metrics"
BLOCK = 16


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def get_text(url: str, timeout: float = 10) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


def parse_samples(text: str) -> dict[str, float]:
    samples: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#") or " " not in line:
            continue
        key, value = line.rsplit(None, 1)
        try:
            samples[key] = float(value)
        except ValueError:
            continue
    return samples


def sum_metric(samples: dict[str, float], family: str) -> float:
    return sum(value for key, value in samples.items() if key.split("{", 1)[0] in {family, family + "_total"})


def metric_block_size(text: str) -> int | None:
    for line in text.splitlines():
        if line.startswith("vllm:cache_config_info{"):
            labels = dict(re.findall(r'([A-Za-z_][A-Za-z0-9_]*)="([^"]*)"', line))
            if "block_size" in labels:
                return int(labels["block_size"])
    return None


def wait_ready(proc: subprocess.Popen, timeout_s: float = 300) -> tuple[str, int]:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited before ready: {proc.returncode}")
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2) as response:
                if response.status == 200:
                    metrics = get_text(METRICS)
                    block_size = metric_block_size(metrics)
                    if block_size is None:
                        raise RuntimeError("cache_config_info block_size missing")
                    if block_size != BLOCK:
                        raise RuntimeError(f"EXPERIMENT_INVALID_ENVIRONMENT:block_size:{block_size}")
                    return metrics, block_size
        except (OSError, urllib.error.URLError):
            time.sleep(2)
    raise TimeoutError("vLLM did not become healthy")


def start_server(stage: str) -> tuple[subprocess.Popen, Path]:
    fd, name = tempfile.mkstemp(prefix=f"phase4c_{stage}_", suffix=".log", dir="/tmp")
    os.close(fd)
    log_path = Path(name)
    env = os.environ.copy()
    env.update(BASE_ENV)
    env["LD_LIBRARY_PATH"] = (f"{ENV_ROOT}/targets/x86_64-linux/lib:"
        f"{ENV_ROOT}/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:"
        f"{ENV_ROOT}/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:" + env.get("LD_LIBRARY_PATH", ""))
    log = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(SERVER, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    return proc, log_path


def stop_server(proc: subprocess.Popen, tmp_log: Path, final_name: str) -> None:
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=40)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
    # Retain only compact startup/warning/error lines, never full request logs.
    keep = []
    for line in tmp_log.read_text(encoding="utf-8", errors="replace").splitlines():
        if re.search(r"(ERROR|WARNING|cache_config_info|KV cache|block_size|Maximum concurrency|Loading model|Starting vLLM|Application startup complete)", line, re.I):
            keep.append(line[:1000])
    (OUT / "server_logs" / final_name).write_text("\n".join(keep[-250:]) + "\n", encoding="utf-8")
    tmp_log.unlink(missing_ok=True)


def token_settings(tokenizer):
    return len(tokenizer), set(tokenizer.all_special_ids)


def make_prompt0(tokenizer, config: str, session_hash: str, turn: int, length: int):
    vocab, specials = token_settings(tokenizer)
    return deterministic_token_ids(length, phase="Phase4C", config=config, session_hash=session_hash,
                                   turn_index=turn, vocab_size=vocab, special_ids=specials)


def send(prompt: list[int], max_tokens: int) -> dict[str, Any]:
    payload = {"model": str(MODEL), "prompt": prompt, "max_tokens": max_tokens,
               "temperature": 0, "ignore_eos": True, "stream": False, "return_token_ids": True}
    request = urllib.request.Request(API, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    issue_ns = time.monotonic_ns()
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.loads(response.read())
    complete_ns = time.monotonic_ns()
    choice = result["choices"][0]
    output = choice.get("token_ids")
    if output is None:
        raise RuntimeError("vLLM response did not return token IDs")
    usage = result.get("usage") or {}
    actual_prompt = usage.get("prompt_tokens")
    if actual_prompt is None:
        raise RuntimeError("vLLM response omitted actual prompt token count")
    return {"issue_ns": issue_ns, "complete_ns": complete_ns, "prompt_tokens": int(actual_prompt),
            "output_ids": [int(token) for token in output], "e2e_ms": (complete_ns - issue_ns) / 1e6,
            "request_error": ""}


def wait_metric_update(before: dict[str, float], timeout_s: float = 5.0) -> dict[str, float]:
    end = time.monotonic() + timeout_s
    before_q = sum_metric(before, "vllm:prefix_cache_queries")
    while time.monotonic() < end:
        after = parse_samples(get_text(METRICS))
        if sum_metric(after, "vllm:prefix_cache_queries") != before_q:
            return after
        time.sleep(.1)
    return parse_samples(get_text(METRICS))


def request_with_metrics(prompt: list[int], max_tokens: int, index: str,
                         before_text: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    if before_text is None:
        before_text = get_text(METRICS)
    before = parse_samples(before_text)
    result = send(prompt, max_tokens)
    after = wait_metric_update(before)
    cached = sum_metric(after, "vllm:prefix_cache_hits") - sum_metric(before, "vllm:prefix_cache_hits")
    queries = sum_metric(after, "vllm:prefix_cache_queries") - sum_metric(before, "vllm:prefix_cache_queries")
    return result, {"request_index": index, "prefix_cache_queries_delta": queries,
                    "prefix_cache_hits_delta": cached, "cached_blocks": cached / BLOCK,
                    "external_metric_present": any("external_prefix_cache" in key for key in after),
                    "external_hit_metric_delta": sum_metric(after, "vllm:external_prefix_cache_hits") - sum_metric(before, "vllm:external_prefix_cache_hits")}


def calibration_prompt(tokenizer, label: str, length: int) -> list[int]:
    return make_prompt0(tokenizer, "phase4c-calibration", hashlib.sha256(label.encode()).hexdigest(), 0, length)


def run_a0() -> bool:
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    proc, tmp = start_server("A0")
    rows = []
    try:
        ready_text, block_size = wait_ready(proc)
        (OUT / "metric_calibration.csv").write_text("", encoding="utf-8")
        p0 = calibration_prompt(tokenizer, "P0", 1024)
        r0, m0 = request_with_metrics(p0, 16, "priming_1024")
        s0 = p0 + r0["output_ids"]
        p1 = s0[:768] + calibration_prompt(tokenizer, "768-divergent", 256)
        # Force the first suffix token to differ from the next previous-sequence token.
        if p1[768] == s0[768]: p1[768] = (p1[768] + 1) % len(tokenizer)
        p2 = s0[:770] + calibration_prompt(tokenizer, "770-divergent", 254)
        if p2[770] == s0[770]: p2[770] = (p2[770] + 1) % len(tokenizer)
        r1, m1 = request_with_metrics(p1, 1, "shared_768")
        r2, m2 = request_with_metrics(p2, 1, "shared_770")
        rows = [
            {**m0, "prompt_tokens": r0["prompt_tokens"], "generated_tokens": len(r0["output_ids"]), "expected_hit_blocks": 0,
             "raw_metric_names": "\\n".join(line.split("{", 1)[0] for line in ready_text.splitlines() if any(t in line for t in ("prefix", "prompt_tokens", "cache", "external")))},
            {**m1, "prompt_tokens": r1["prompt_tokens"], "generated_tokens": len(r1["output_ids"]), "expected_hit_blocks": 48, "raw_metric_names": ""},
            {**m2, "prompt_tokens": r2["prompt_tokens"], "generated_tokens": len(r2["output_ids"]), "expected_hit_blocks": 48, "raw_metric_names": ""},
        ]
        write_csv(OUT / "metric_calibration.csv", rows)
        passed = (block_size == 16 and len(r0["output_ids"]) == 16 and m0["prefix_cache_hits_delta"] == 0
                  and m1["prefix_cache_hits_delta"] == 768 and m2["prefix_cache_hits_delta"] == 768
                  and all(r["prompt_tokens"] == 1024 for r in (r0, r1, r2))
                  and all(r["external_hit_metric_delta"] == 0 for r in (m0, m1, m2)))
        report = {"stage": "A0", "actual_block_size": block_size, "priming_1024_hit_tokens": m0["prefix_cache_hits_delta"],
                  "shared_768_hit_tokens": m1["prefix_cache_hits_delta"], "shared_770_hit_tokens": m2["prefix_cache_hits_delta"],
                  "external_hit_tokens": sum(r["external_hit_metric_delta"] for r in (m0, m1, m2)), "pass": bool(passed)}
        (OUT / "stage_a0.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return bool(passed)
    finally:
        stop_server(proc, tmp, "stage_a0_compact.log")


def load_workload():
    sessions = list(csv.DictReader((OUT / "selected_sessions.csv").open(encoding="utf-8")))
    turns = list(csv.DictReader((OUT / "hybrid_workload.csv").open(encoding="utf-8")))
    for row in turns:
        for key in ("replay_turn_index", "source_isl", "source_isl_new", "source_osl", "source_cached_tokens", "target_cached_tokens", "target_cached_blocks"):
            row[key] = int(row[key])
        if row["proxy_pause_ms"]:
            row["proxy_pause_ms"] = float(row["proxy_pause_ms"])
    return sessions, turns


def replay(stage: str, sessions: list[dict[str, str]], assignments: dict[tuple[str, int], float]) -> tuple[bool, list[dict[str, Any]], list[dict[str, Any]]]:
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    proc, tmp = start_server(stage)
    request_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    snapshots = []
    try:
        _, block_size = wait_ready(proc)
        if block_size != 16:
            raise RuntimeError("EXPERIMENT_INVALID_ENVIRONMENT:block size changed")
        histories: list[list[int]] = []
        for session in sessions:
            session_hash = session["session_hash"]
            episode_turns = [r for r in load_workload()[1] if r["session_hash"] == session_hash]
            episode_turns.sort(key=lambda row: row["replay_turn_index"])
            previous_served: list[int] | None = None
            previous_response_end_ns: int | None = None
            for turn in episode_turns:
                index = turn["replay_turn_index"]
                config = turn["config"]
                expected_cached = turn["target_cached_tokens"] if index else 0
                before_metric_text = get_text(METRICS)
                if previous_served is None:
                    prompt = make_prompt0(tokenizer, config, session_hash, index, turn["source_isl"])
                    if histories and longest_full_block_match(prompt, histories, BLOCK) > 0:
                        raise RuntimeError("cross-session accidental full-block match")
                else:
                    vocab, specials = token_settings(tokenizer)
                    prompt = make_next_prompt(previous_served, turn["source_isl"], expected_cached,
                                              phase="Phase4C", config=config, session_hash=session_hash,
                                              turn_index=index, vocab_size=vocab, special_ids=specials)
                    if longest_full_block_match(prompt, histories, BLOCK) > expected_cached:
                        raise RuntimeError("accidental old-prefix match exceeds target")
                if index and previous_response_end_ns is not None:
                    proxy_ms = assignments[(session_hash, index - 1)]
                    target_ns = previous_response_end_ns + int(proxy_ms * 1_000_000)
                    delay = target_ns - time.monotonic_ns()
                    if delay > 0:
                        time.sleep(delay / 1_000_000_000)
                result, metric = request_with_metrics(prompt, turn["source_osl"],
                    f"{stage}_{session_hash[:8]}_t{index}", before_metric_text)
                prompt_error = result["prompt_tokens"] - turn["source_isl"]
                generation_error = len(result["output_ids"]) - turn["source_osl"]
                actual_cached = int(round(metric["prefix_cache_hits_delta"]))
                cached_blocks = actual_cached // BLOCK
                issue_ns, complete_ns = result["issue_ns"], result["complete_ns"]
                actual_pause = ""
                error_ms = ""
                if index and previous_response_end_ns is not None:
                    actual_pause = actual_pause_ms(previous_response_end_ns, issue_ns)
                    error_ms = pause_error_ms(actual_pause, assignments[(session_hash, index - 1)])
                    boundary_rows.append({
                        "config": config, "session_hash": session_hash, "boundary_index": index - 1,
                        "source_cached_tokens": turn["source_cached_tokens"], "target_cached_tokens": expected_cached,
                        "target_cached_blocks": expected_cached // BLOCK, "actual_cached_tokens": actual_cached,
                        "actual_cached_blocks": cached_blocks, "cached_block_error": cached_blocks - expected_cached // BLOCK,
                        "proxy_pause_ms": assignments[(session_hash, index - 1)], "actual_pause_ms": actual_pause,
                        "pause_error_ms": error_ms, "previous_source_isl": episode_turns[index - 1]["source_isl"],
                        "previous_source_osl": episode_turns[index - 1]["source_osl"],
                        "current_source_isl": turn["source_isl"], "current_source_isl_new": turn["source_isl_new"],
                    })
                request_rows.append({
                    "stage": stage, "config": config, "session_hash": session_hash,
                    "replay_turn_index": index, "source_isl": turn["source_isl"],
                    "source_isl_new": turn["source_isl_new"], "source_osl": turn["source_osl"],
                    "source_cached_tokens": turn["source_cached_tokens"],
                    "target_cached_tokens": expected_cached, "target_cached_blocks": expected_cached // BLOCK,
                    "alignment_loss_tokens": turn["source_cached_tokens"] - expected_cached,
                    "actual_prompt_tokens": result["prompt_tokens"], "actual_generated_tokens": len(result["output_ids"]),
                    "actual_local_cached_tokens": actual_cached, "actual_local_cached_blocks": cached_blocks,
                    "actual_external_cached_tokens": int(round(metric["external_hit_metric_delta"])), "prompt_length_error": prompt_error,
                    "generation_length_error": generation_error, "cached_block_error": cached_blocks - expected_cached // BLOCK,
                    "prompt_sha256": token_hash(prompt), "output_sha256": token_hash(result["output_ids"]),
                    "request_issue_monotonic_ns": issue_ns, "request_complete_monotonic_ns": complete_ns,
                    "request_e2e_ms": result["e2e_ms"], "request_error": result["request_error"],
                    "priming_expected_hit_blocks": 0 if index == 0 else "",
                    "priming_actual_hit_blocks": cached_blocks if index == 0 else "",
                    "proxy_pause_ms": assignments.get((session_hash, index), ""),
                    "actual_pause_ms": actual_pause, "pause_error_ms": error_ms,
                })
                previous_served = prompt + result["output_ids"]
                histories.append(previous_served)
                previous_response_end_ns = complete_ns
                snapshots.append({"session_hash": session_hash, "turn": index, **metric})
        pause_errors = [abs(float(r["pause_error_ms"])) for r in boundary_rows]
        passed = (block_size == 16 and all(int(r["actual_prompt_tokens"]) == int(r["source_isl"]) for r in request_rows)
                  and all(int(r["actual_generated_tokens"]) == int(r["source_osl"]) for r in request_rows)
                  and all(int(r["cached_block_error"]) == 0 for r in boundary_rows)
                  and all(int(r["priming_actual_hit_blocks"]) == 0 for r in request_rows if r["replay_turn_index"] == 0)
                  and all(int(r["actual_external_cached_tokens"]) == 0 for r in request_rows)
                  and (not pause_errors or (quantile(pause_errors, .95) <= 10 and max(pause_errors) <= 30)))
        return bool(passed), request_rows, boundary_rows
    finally:
        stop_server(proc, tmp, f"stage_{stage.lower()}_compact.log")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("A0", "A1", "B"))
    args = parser.parse_args()
    if args.stage == "A0":
        print(json.dumps({"A0_pass": run_a0()}))
        return
    sessions, _ = load_workload()
    pauses = list(csv.DictReader((OUT / "pause_assignments.csv").open(encoding="utf-8")))
    assignments = {(row["session_hash"], int(row["boundary_index"])): float(row["proxy_pause_ms"]) for row in pauses}
    if args.stage == "A1":
        chosen = []
        for config in ("gpt-oss-120b-pro", "gpt-oss-120b-verified"):
            chosen.append(min((s for s in sessions if s["config"] == config), key=lambda row: row["selection_hash"]))
        # Keep deterministic config order and corresponding 8 frozen proxy pauses.
        chosen.sort(key=lambda row: (row["config"], row["selection_hash"]))
        a1_assignments = {(s["session_hash"], b): assignments[(s["session_hash"], b)] for s in chosen for b in range(4)}
        passed, req, bnd = replay("A1", chosen, a1_assignments)
        write_csv(OUT / "stage_a1_requests.csv", req)
        write_csv(OUT / "stage_a1_boundaries.csv", bnd)
        (OUT / "stage_a1.json").write_text(json.dumps({"pass": passed, "requests": len(req), "boundaries": len(bnd)}, indent=2) + "\n")
    else:
        passed, req, bnd = replay("B", sessions, assignments)
        write_csv(OUT / "request_metrics.csv", req)
        write_csv(OUT / "boundary_fidelity.csv", bnd)
        (OUT / "stage_b.json").write_text(json.dumps({"pass": passed, "requests": len(req), "boundaries": len(bnd)}, indent=2) + "\n")
    print(json.dumps({"stage": args.stage, "pass": passed, "requests": len(req), "boundaries": len(bnd)}))


if __name__ == "__main__":
    main()
