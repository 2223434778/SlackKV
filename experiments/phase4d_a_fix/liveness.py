from __future__ import annotations

import concurrent.futures
import json
import math
import os
import re
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

from experiments.phase3f.native_client import native_prefetch
from experiments.phase4d_a import runtime
from experiments.phase4d_a.experiment import (
    OUT as OLD_OUT, ROOT, build_resume_prompts, first_boundary_jobs, materialize_and_evict,
    write_csv, write_json,
)
from experiments.phase4d_a_fix.release_fix import (
    EXPECTED_GPU_BLOCKS, WATCHDOG_SECONDS, issue_spread_ms, issue_spread_valid,
    smallest_prefix_groups, watchdog_classification, stalled_for_five_seconds,
)

OUT = ROOT / "results/phase4d_a_fix"


def free_memory_bytes() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable missing")


def kv_usage_percent(timeout_s: float = 0.5) -> float | str:
    try:
        with urllib.request.urlopen(runtime.METRICS, timeout=timeout_s) as response:
            body = response.read().decode("utf-8", errors="replace")
        values = []
        for line in body.splitlines():
            if line.startswith("vllm:kv_cache_usage_perc") and not line.startswith("#"):
                match = re.search(r"\s([-+0-9.eE]+)\s*$", line)
                if match:
                    values.append(float(match.group(1)))
        return max(values) if values else "NA"
    except Exception:
        return "NA"


def health_ok(timeout_s: float = 0.15) -> bool:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=timeout_s) as response:
            return response.status == 200
    except Exception:
        return False


class ConnectorStepCounter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0
        self.pending = b""
        self.count = 0

    def sample(self) -> int:
        if not self.path.exists():
            return self.count
        with self.path.open("rb") as handle:
            handle.seek(self.offset)
            chunk = handle.read()
            self.offset = handle.tell()
        data = self.pending + chunk
        parts = data.split(b"\n")
        self.pending = parts.pop() if parts else b""
        self.count += sum(part.count(b"SLACKKV_CONNECTOR_ONLY_STEP") for part in parts)
        return self.count


def prefetch_complete(record: dict[str, Any], required_tokens: int) -> bool:
    timestamps = record.get("phase3m_timestamps") or {}
    return (
        record.get("status") == "ok"
        and int(record.get("external_tokens_loaded", -1)) == required_tokens
        and int(record.get("transfer_bytes", 0)) > 0
        and all(name in timestamps for name in ("T0", "T6", "T7"))
        and int(record.get("model_forward_count", -1)) == 0
        and int(record.get("generated_token_count", -1)) == 0
        and int(record.get("mixed_prefetch_model_step_count", -1)) == 0
    )


def run_liveness(stage: str, n: int, repetition: int, jobs: list[dict[str, Any]],
                 data: dict[str, dict[str, Any]], *, with_x: bool = False) -> dict[str, Any]:
    selected = smallest_prefix_groups(jobs)["ordered_jobs"][:n]
    selected_ids = [row["episode_id"] for row in selected]
    total_blocks = sum(int(row["required_blocks"]) for row in selected)
    label = f"liveness_{stage.lower()}_n{n}_rep{repetition}" + ("_x" if with_x else "")
    proc, log = runtime.start_server(label, connector=True, out=OUT)
    xpool = xfuture = None
    futures: dict[str, concurrent.futures.Future] = {}
    issue_ns: dict[str, int] = {}
    samples: list[dict[str, Any]] = []
    returned: dict[str, dict[str, Any]] = {}
    future_errors: dict[str, str] = {}
    try:
        _, block = runtime.wait_ready(proc)
        if block != 16:
            raise RuntimeError("EXPERIMENT_INVALID_ENVIRONMENT:block_size")
        from experiments.phase4d_a.experiment import gpu_capacity_blocks
        gpu_blocks = gpu_capacity_blocks(log, 16)
        if gpu_blocks != EXPECTED_GPU_BLOCKS:
            raise RuntimeError("EXPERIMENT_INVALID_ENVIRONMENT:GPU KV blocks changed")
        if free_memory_bytes() < 16 * 1024**3:
            raise RuntimeError("EXPERIMENT_INVALID_ENVIRONMENT:MemAvailable below 16 GiB")
        subset_jobs = [row for row in jobs if row["episode_id"] in set(selected_ids)]
        subset_data = {episode: data[episode] for episode in selected_ids}
        materialize_and_evict(proc, log, label, subset_jobs, subset_data, OUT)

        x_first_ns = None
        if with_x:
            from experiments.phase4d_a.run_policy import start_background
            xpool, xfuture, xmeta = start_background()
            x_first_ns = int(xmeta["first_ns"])
            if xfuture.done():
                raise RuntimeError("L2 invalid: X completed before prefetch issue")

        barrier = threading.Barrier(n + 1)

        def issue_one(episode: str) -> dict[str, Any]:
            barrier.wait()
            issue_ns[episode] = time.monotonic_ns()
            record = native_prefetch(subset_data[episode]["prefix"], timeout_s=WATCHDOG_SECONDS)
            return {"episode_id": episode, "client_issue_ns": issue_ns[episode], "response": record}

        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
            for episode in selected_ids:
                futures[episode] = pool.submit(issue_one, episode)
            # The workers are blocked at the barrier, so this is the single GO.
            go_ns = time.monotonic_ns()
            barrier.wait()

            started = time.monotonic()
            last_completion_ms: float | None = None
            processed: set[str] = set()
            success_by_episode: dict[str, bool] = {}
            connector_counter = ConnectorStepCounter(log)
            while True:
                now_elapsed_ms = (time.monotonic() - started) * 1000
                for episode, future in futures.items():
                    if future.done() and episode not in processed:
                        processed.add(episode)
                        try:
                            result = future.result()
                            returned[episode] = result["response"]
                            success = prefetch_complete(result["response"], int(subset_data[episode]["job"]["required_prefix_tokens"]))
                            success_by_episode[episode] = success
                            if success:
                                last_completion_ms = now_elapsed_ms
                        except Exception as error:
                            future_errors[episode] = f"{type(error).__name__}: {error}"
                            success_by_episode[episode] = False
                completed_count = sum(success_by_episode.values())
                pending_count = sum(not future.done() for future in futures.values())
                alive = proc.poll() is None
                samples.append({
                    "stage": stage, "N": n, "repetition": repetition, "episode_ids": "|".join(selected_ids),
                    "elapsed_ms": now_elapsed_ms, "completed_prefetch_count": completed_count,
                    "pending_prefetch_count": pending_count, "kv_cache_usage_perc": kv_usage_percent(),
                    "server_alive": alive, "health_ok": health_ok() if alive else False,
                    "connector_only_steps": connector_counter.sample(),
                    "last_completion_ms": last_completion_ms if last_completion_ms is not None else "NA",
                    "stalled_5s": stalled_for_five_seconds(last_completion_ms, now_elapsed_ms, pending_count),
                })
                all_done = all(future.done() for future in futures.values())
                if all_done or now_elapsed_ms >= WATCHDOG_SECONDS * 1000:
                    break
                time.sleep(0.1)
            watchdog_elapsed = (time.monotonic() - started)
            completed_count = sum(success_by_episode.values())
            pending_count = sum(not future.done() for future in futures.values())
            alive_at_watchdog = proc.poll() is None
            connector_steps = connector_counter.sample()
            log_text = log.read_text(encoding="utf-8", errors="replace")
            oom = bool(re.search(r"CUDA out of memory|OutOfMemoryError|out of memory", log_text, re.I))
            server_crash = proc.poll() is not None or bool(re.search(r"EngineCore.*exited|Traceback.*EngineCore", log_text))
            spread = issue_spread_ms(issue_ns.values()) if len(issue_ns) == n else float("inf")
            issue_valid = len(issue_ns) == n and issue_spread_valid(spread)
            if not issue_valid:
                classification = "RUN_INVALID_START_SKEW"
            elif oom or server_crash:
                classification = "INVALID_SERVER_FAILURE"
            elif pending_count > 0 and alive_at_watchdog and watchdog_elapsed >= WATCHDOG_SECONDS:
                classification = "NON_PROGRESS"
            elif completed_count == n and all(success_by_episode.values()):
                classification = "COMPLETE"
            else:
                classification = "PREFETCH_INCOMPLETE"
            non_progress = classification == "NON_PROGRESS"
            stalled = pending_count > 0 and samples and bool(samples[-1]["stalled_5s"])
            x_active_at_end = None if not with_x else (not xfuture.done())
            # Freeze the watchdog classification before stopping the server. Shutdown-generated
            # HTTP errors are recorded separately and are not used as evidence of non-progress.
            run_row = {
                "stage": stage, "N": n, "repetition": repetition, "episode_ids": "|".join(selected_ids),
                "total_required_blocks": total_blocks, "gpu_total_blocks": gpu_blocks,
                "X_enabled": with_x, "resume_enabled": False, "issue_spread_ms": spread,
                "prefetch_issued": len(issue_ns), "prefetch_completed": completed_count,
                "prefetch_pending": pending_count,
                "HTTP_200": len(returned),
                "timeouts": sum("timeout" in value.lower() for value in future_errors.values()),
                "watchdog_seconds": WATCHDOG_SECONDS if non_progress else round(watchdog_elapsed, 3),
                "non_progress": non_progress, "stalled_5s": stalled,
                "connector_only_steps": connector_steps,
                "kv_usage_peak": max((float(s["kv_cache_usage_perc"]) for s in samples if s["kv_cache_usage_perc"] != "NA"), default="NA"),
                "server_alive_at_watchdog": alive_at_watchdog,
                "OOM": oom, "server_crash": server_crash, "classification": classification,
                "X_first_token_ns": x_first_ns if x_first_ns is not None else "NA",
                "X_active_at_watchdog": x_active_at_end if non_progress else "NA",
                "future_errors": json.dumps(future_errors, sort_keys=True),
            }
        # Stop the server after the watchdog decision; pending HTTP 500s after this point are
        # shutdown effects and do not alter the classification above.
        runtime.stop_server(proc, log, OUT, label)
        compact = OUT / "server_logs" / f"{label}_compact.log"
        compact_text = compact.read_text(encoding="utf-8", errors="replace") if compact.exists() else ""
        raw_text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        shutdown_http_500 = len(re.findall(r"POST /slackkv/prefetch.*500", raw_text))
        run_row["shutdown_http_500"] = shutdown_http_500
        run_row["server_errors"] = sum(1 for line in raw_text.splitlines() if " ERROR " in line)
        for episode, result in returned.items():
            timestamps = result.get("phase3m_timestamps") or {}
            run_row.setdefault("prefetch_rows", []).append({
                "stage": stage, "N": n, "repetition": repetition, "episode_id": episode,
                "required_blocks": subset_data[episode]["job"]["required_blocks"],
                "required_tokens": subset_data[episode]["job"]["required_prefix_tokens"],
                "client_issue_ns": issue_ns.get(episode, "NA"),
                "sent_monotonic_s": result.get("sent_monotonic_s", "NA"),
                "T0": timestamps.get("T0", "NA"), "T6": timestamps.get("T6", "NA"),
                "T7": timestamps.get("T7", "NA"),
                "issue_to_T6_ms": ((float(timestamps["T6"]) - float(result["sent_monotonic_s"])) * 1000
                                   if "T6" in timestamps and result.get("sent_monotonic_s") is not None else "NA"),
                "external_tokens_loaded": result.get("external_tokens_loaded", "NA"),
                "transfer_bytes": result.get("transfer_bytes", "NA"),
                "status": result.get("status", "NA"),
                "model_forward_count": result.get("model_forward_count", "NA"),
                "generated_token_count": result.get("generated_token_count", "NA"),
                "mixed_prefetch_model_step_count": result.get("mixed_prefetch_model_step_count", "NA"),
                "prefetch_complete": prefetch_complete(result, int(subset_data[episode]["job"]["required_prefix_tokens"])),
            })
        samples_path = OUT / "liveness_samples.csv"
        existing_samples = []
        if samples_path.exists():
            import csv
            existing_samples = list(csv.DictReader(samples_path.open(encoding="utf-8")))
        write_csv(samples_path, existing_samples + samples)
        run_path = OUT / "liveness_runs.csv"
        existing_runs = []
        if run_path.exists():
            import csv
            existing_runs = list(csv.DictReader(run_path.open(encoding="utf-8")))
        prefetch_rows = run_row.pop("prefetch_rows", [])
        write_csv(run_path, existing_runs + [run_row])
        pref_path = OUT / "liveness_prefetch_events.csv"
        existing_pref = []
        if pref_path.exists():
            import csv
            existing_pref = list(csv.DictReader(pref_path.open(encoding="utf-8")))
        write_csv(pref_path, existing_pref + prefetch_rows)
        log.unlink(missing_ok=True)
        if xpool is not None:
            # X is allowed to finish only after liveness classification is frozen.
            xpool.shutdown(wait=False, cancel_futures=True)
        return run_row
    finally:
        if proc.poll() is None:
            runtime.stop_server(proc, log, OUT, label)
        if log.exists():
            log.unlink(missing_ok=True)
        if xpool is not None:
            xpool.shutdown(wait=False, cancel_futures=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "server_logs").mkdir(exist_ok=True)
    jobs = first_boundary_jobs(ROOT / "results/phase4c/hybrid_workload.csv")
    data = build_resume_prompts(jobs)
    if len(jobs) != 8 or sum(j["required_blocks"] for j in jobs) != 10501:
        raise RuntimeError("frozen boundary-0 job invariant failed")
    plan = smallest_prefix_groups(jobs)
    if (plan["N1"], plan["N2"], plan["N3"], plan["N4"]) != (1271, 2545, 3821, 5098):
        raise RuntimeError("mechanical N1-N4 block totals changed")
    if EXPECTED_GPU_BLOCKS != 4681 or not (plan["N3"] < EXPECTED_GPU_BLOCKS < plan["N4"]):
        raise RuntimeError("expected physical capacity crossing absent")
    write_json(OUT / "liveness_plan.json", {
        "selection_rule": "ascending required_blocks, then episode_id",
        "N1_N2_N3_N4": plan["episode_ids"],
        "totals_blocks": {"N1": plan["N1"], "N2": plan["N2"], "N3": plan["N3"], "N4": plan["N4"]},
        "gpu_total_blocks": EXPECTED_GPU_BLOCKS,
        "crossing": "3821 < 4681 < 5098",
        "no_resumes": True, "no_X": True, "no_K2": True, "no_lease": True, "no_reservation": True,
        "watchdog_seconds": WATCHDOG_SECONDS,
    })
    l0 = [run_liveness("L0", n, 0, jobs, data) for n in (1, 2)]
    l1 = [run_liveness("L1", n, rep, jobs, data) for n in (3, 4) for rep in range(3)]
    n3 = [row for row in l1 if row["N"] == 3]
    n4 = [row for row in l1 if row["N"] == 4]
    isolated_l1 = all(row["classification"] == "COMPLETE" for row in n3) and all(
        row["classification"] == "NON_PROGRESS" and row["server_alive_at_watchdog"] and
        not row["OOM"] and not row["server_crash"] and row["issue_spread_ms"] <= 20
        for row in n4
    )
    summary = {
        "L0": l0, "L1": l1, "L1_capacity_crossing_isolated": isolated_l1,
        "L2_started": False, "L3_started": False,
        "minimal_liveness_conclusion": "NAIVE_CAPACITY_CROSSING_CONFIRMED" if isolated_l1 else "L1_NOT_ISOLATED",
    }
    write_json(OUT / "liveness_summary.json", summary)
    print(json.dumps({"L0": [{k: r[k] for k in ("N", "classification", "issue_spread_ms")} for r in l0],
                      "L1": [{k: r[k] for k in ("N", "repetition", "classification", "issue_spread_ms",
                                                "prefetch_completed", "prefetch_pending", "connector_only_steps",
                                                "server_alive_at_watchdog", "OOM", "server_crash")} for r in l1],
                      "isolated": isolated_l1}, indent=2))


if __name__ == "__main__":
    main()
