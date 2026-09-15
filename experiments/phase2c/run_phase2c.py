#!/usr/bin/env python3
"""Phase 2C capability gate; refuses to run invalid selective-offload comparisons."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "phase2c"
VENV = Path("/home/a6000/.venvs/slackkv-phase2c")
VLLM_ROOT = VENV / "lib/python3.11/site-packages/vllm"
REQUIRED = {
    "offloading_connector": VLLM_ROOT
    / "distributed/kv_transfer/kv_connector/v1/offloading_connector.py",
    "max_offload_tokens": VLLM_ROOT
    / "distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py",
}


def runtime_info() -> dict[str, object]:
    """Read the isolated runtime without importing its CUDA stack in this process."""
    program = (
        "import json, sys, torch, vllm; "
        "print(json.dumps({'python_version': sys.version.split()[0], "
        "'vllm_version': vllm.__version__, 'torch_version': torch.__version__, "
        "'torch_cuda_version': torch.version.cuda, "
        "'gpu_name': torch.cuda.get_device_name(0), "
        "'gpu_total_memory_bytes': torch.cuda.get_device_properties(0).total_memory}))"
    )
    command = [str(VENV / "bin/python"), "-c", program]
    return json.loads(subprocess.check_output(command, text=True))


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    runtime = runtime_info()
    version = str(runtime["vllm_version"])
    connector_present = REQUIRED["offloading_connector"].is_file()
    scheduler_path = REQUIRED["max_offload_tokens"]
    scheduler_text = scheduler_path.read_text() if scheduler_path.is_file() else ""
    cap_present = "max_offload_tokens" in scheduler_text
    version_029_error = "libcudart.so.13: cannot open shared object file"
    status = {
        **runtime,
        "available_vllm_version": version,
        "offloading_connector_present": connector_present,
        "max_offload_tokens_present": cap_present,
        "selective_offload_runnable": connector_present and cap_present,
        "blocker": (
            "vLLM 0.11.2 imports on CUDA 12.4 and contains OffloadingConnector, "
            "but lacks max_offload_tokens; vLLM 0.29.0 contains the required "
            "per-request cap but cannot import because its wheel requires CUDA 13 "
            f"({version_029_error})."
        ),
    }
    (RESULTS / "config.json").write_text(
        json.dumps(
            {
                "base_commit": "0b85a5a5a5df8068f17bfade142e8ef406ab91c3",
                "model": "Qwen/Qwen2.5-1.5B-Instruct",
                "target_gpu_kv_capacity": "320 MiB",
                "target_cpu_offload_capacity": "2 GiB",
                "phase_status": "INCOMPLETE: capability gate failed",
            },
            indent=2,
        )
        + "\n"
    )
    (RESULTS / "environment.json").write_text(json.dumps(status, indent=2) + "\n")
    (RESULTS / "cache_metrics.csv").write_text(
        "vllm_version,offloading_connector_present,max_offload_tokens_present,selective_offload_runnable\n"
        f"{version},{connector_present},{cap_present},{status['selective_offload_runnable']}\n"
    )
    (RESULTS / "resume_metrics.csv").write_text(
        "repetition,resume_order,session,admission,resume_ttft_ms,resume_e2e_ms,"
        "local_prefix_hit_tokens,external_prefix_hit_tokens,recomputed_prompt_tokens,"
        "gpu_kv_usage_before_resume,gpu_kv_usage_after_resume\n"
        "NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA\n"
    )
    (RESULTS / "server_logs" / "environment_probe.txt").write_text(
        json.dumps(status, indent=2) + "\n"
    )
    (RESULTS / "commands.sh").write_text(
        "python -m pytest -q tests/test_phase2c.py\n"
        "python experiments/phase2c/run_phase2c.py\n"
    )
    print(json.dumps(status, indent=2))
    if not status["selective_offload_runnable"]:
        raise SystemExit("Phase 2C capability gate failed; server experiment not started.")


if __name__ == "__main__":
    main()
