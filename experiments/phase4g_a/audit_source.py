"""One-request natural offload probe; no prefix pre-materialization."""
from __future__ import annotations

import json
import time

from experiments.phase4d_a import runtime
from experiments.phase4g_a.workload import OUT, build_manifest, load_source


def main() -> None:
    manifest = build_manifest(load_source())
    source = load_source()
    OUT.mkdir(parents=True, exist_ok=True)
    probe_dir = OUT / "source_probe"
    probe_dir.mkdir(exist_ok=True)
    proc, log = runtime.start_server("source_probe", out=probe_dir)
    result: dict[str, object] = {}
    try:
        _, block_size = runtime.wait_ready(proc)
        if block_size != 16:
            raise RuntimeError("EXPERIMENT_INVALID_ENVIRONMENT:block size")
        probes = []
        for episode_id in ("E00", "E01"):
            row = next(item for item in source if item["episode_id"] == episode_id and item["replay_turn_index"] == "0")
            boundary = next(item for item in manifest if item["boundary_id"] == f"{episode_id}:b0")
            prompt = runtime.tokenizer_ids(row["config"], row["session_hash"], 0, int(row["source_isl"]))
            response = runtime.send(prompt, int(row["source_osl"]), params={
                "slackkv_session": f"{episode_id}:turn:0",
                "slackkv_trace_next_boundary": f"{episode_id}:b0",
                "slackkv_trace_source_required_tokens": int(boundary["required_prefix_tokens"]),
            })
            if len(response["tokens"]) != int(row["source_osl"]):
                raise RuntimeError("natural offload probe output mismatch")
            probes.append({"episode_id": episode_id, "prompt_tokens": len(prompt),
                           "generated_tokens": len(response["tokens"]),
                           "next_required_tokens": boundary["required_prefix_tokens"]})
        time.sleep(2)
        events = []
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            if "SLACKKV_OFFLOAD_TRACE " in line:
                events.append(json.loads(line.split("SLACKKV_OFFLOAD_TRACE ", 1)[1]))
        result = {"block_size": block_size, "probes": probes,
                  "events": events, "pre_materialized_boundary_count": 0}
        if len(events) < 2:
            raise RuntimeError("EXPERIMENT_INVALID_LIFECYCLE_OBSERVABILITY:no offload event")
    finally:
        runtime.stop_server(proc, log, probe_dir, "source_probe")
        if log.exists():
            lines = [line[:2000] for line in log.read_text(encoding="utf-8", errors="replace").splitlines()
                     if any(marker in line for marker in ("SLACKKV_OFFLOAD_TRACE", "ERROR", "OOM", "GPU KV cache size"))]
            (probe_dir / "source_probe_compact.log").write_text("\n".join(lines[-100:]) + "\n", encoding="utf-8")
    (OUT / "natural_source_probe.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
