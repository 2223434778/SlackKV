#!/usr/bin/env python3
"""Collect only Phase 0 hardware/software facts; no secrets are included."""
import json, os, platform, shutil, subprocess, sys
from pathlib import Path

OUT = Path("results/phase0")
def run(cmd):
    try:
        p = subprocess.run(cmd, text=True, capture_output=True, check=False)
        return {"command": " ".join(cmd), "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except FileNotFoundError:
        return {"command": " ".join(cmd), "unavailable": True}
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    queries = {"nvidia_smi_L": run(["nvidia-smi","-L"]), "nvidia_smi_topo_m": run(["nvidia-smi","topo","-m"]),
               "nvidia_smi_gpu_query": run(["nvidia-smi","--query-gpu=index,name,uuid,pci.bus_id,memory.total,driver_version","--format=csv,noheader"]),
               "lspci": run(["lspci"]), "numactl_hardware": run(["numactl","--hardware"]), "nvcc_version": run(["/usr/local/cuda-12/bin/nvcc","--version"])}
    env = {"os": platform.platform(), "kernel": platform.release(), "python_version": sys.version,
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"), "commands": queries}
    try:
        import torch
        env["pytorch_version"] = torch.__version__; env["torch_cuda_version"] = torch.version.cuda
        env["cuda_available"] = torch.cuda.is_available(); env["cuda_device_count"] = torch.cuda.device_count()
    except Exception as e: env["pytorch_error"] = str(e)
    (OUT / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
    topo = "# nvidia-smi -L\n" + queries["nvidia_smi_L"].get("stdout", "") + "\n# nvidia-smi topo -m\n" + queries["nvidia_smi_topo_m"].get("stdout", "")
    for name in ("lspci", "numactl_hardware"):
        topo += f"\n# {name}\n" + queries[name].get("stdout", queries[name].get("stderr", "unavailable"))
    (OUT / "topology.txt").write_text(topo)
if __name__ == "__main__": main()
