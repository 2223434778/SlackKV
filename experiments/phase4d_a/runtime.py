from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from experiments.phase2d.helpers import parse_prometheus_samples, source_tokens_delta

ROOT = Path(__file__).resolve().parents[2]
MODEL = Path("/home/a6000/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
ENV_ROOT = Path("/media/a6000/D/ZhaoBinHe/.conda-envs/slackkv-phase2c-fix")
SRC = Path("/media/a6000/D/ZhaoBinHe/slackkv-phase2c-fix-sourceprobe")
VLLM = ENV_ROOT / "bin" / "vllm"
API = "http://127.0.0.1:8000/v1/completions"
METRICS = "http://127.0.0.1:8000/metrics"
TOKENIZER = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)

def server_command(connector: bool, gpu_bytes: int) -> list[str]:
    command = [str(VLLM), "serve", str(MODEL), "--dtype", "float16",
        "--tensor-parallel-size", "1", "--enable-prefix-caching",
        "--kv-cache-memory-bytes", str(gpu_bytes), "--max-model-len", "32768",
        "--host", "127.0.0.1", "--port", "8000", "--enforce-eager"]
    if connector:
        command += ["--kv-transfer-config",
            '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"block_size":16,"cpu_bytes_to_use":8589934592}}']
    return command

def start_server(label: str, connector: bool = True, gpu_bytes: int = 2147483648, out: Path | None = None):
    log_dir = out / "server_logs" if out else ROOT / "results/phase4d_a/server_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{label}.log"
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES":"0","HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1",
                "VLLM_SERVER_DEV_MODE":"1","PYTHONPATH":f"{SRC}:{ROOT}",
                "PATH":f"{ENV_ROOT}/bin:"+os.environ.get("PATH",""),
                "LD_LIBRARY_PATH":f"{ENV_ROOT}/targets/x86_64-linux/lib:{ENV_ROOT}/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:{ENV_ROOT}/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:"+os.environ.get("LD_LIBRARY_PATH","")})
    log = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(server_command(connector,gpu_bytes), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    return proc, log_path

def metric_text() -> str:
    with urllib.request.urlopen(METRICS, timeout=20) as response:
        return response.read().decode()

def wait_ready(proc, timeout: float = 360.0) -> tuple[str,int]:
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited before ready ({proc.returncode})")
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/health",timeout=3) as response:
                if response.status==200:
                    raw=metric_text()
                    import re
                    for line in raw.splitlines():
                        if line.startswith("vllm:cache_config_info{"):
                            m=re.search(r'block_size="(\d+)"',line)
                            if m: return raw,int(m.group(1))
        except OSError:
            pass
        time.sleep(2)
    raise TimeoutError("vLLM readiness timeout")

def stop_server(proc, log_path: Path, out: Path, label: str) -> int:
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try: proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait(timeout=15)
    code=int(proc.returncode or 0)
    try:
        lines=log_path.read_text(encoding="utf-8",errors="replace").splitlines()
        keep=[line[:1000] for line in lines if any(x in line for x in ("ERROR","WARNING","GPU KV cache size","cache_config_info","OOM"))]
        (out/"server_logs"/f"{label}_compact.log").write_text("\n".join(keep[-250:])+"\n",encoding="utf-8")
    except OSError: pass
    return code

def send(prompt: list[int], max_tokens: int, *, stream: bool=False, params: dict[str,Any]|None=None, on_first=None) -> dict[str,Any]:
    payload={"model":str(MODEL),"prompt":prompt,"max_tokens":max_tokens,"temperature":0,
             "ignore_eos":True,"stream":stream,"return_token_ids":True}
    if params: payload["kv_transfer_params"]=params
    issue=time.monotonic_ns()
    request=urllib.request.Request(API,data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"})
    tokens=[]; first=None; prompt_tokens=None
    with urllib.request.urlopen(request,timeout=900) as response:
        if not stream:
            data=json.loads(response.read()); choice=data["choices"][0]
            tokens=[int(x) for x in (choice.get("token_ids") or [])]
            prompt_tokens=(data.get("usage") or {}).get("prompt_tokens")
            first=time.monotonic_ns()
        else:
            for line in response:
                if not line.startswith(b"data: "): continue
                part=line[6:].strip()
                if part==b"[DONE]": continue
                choice=(json.loads(part).get("choices") or [{}])[0]
                ids=[int(x) for x in (choice.get("token_ids") or [])]
                if ids:
                    tokens.extend(ids)
                    if first is None:
                        first=time.monotonic_ns()
                        if on_first is not None:
                            on_first(first)
    done=time.monotonic_ns()
    if first is None: raise RuntimeError("response had no generated token")
    return {"issue_ns":issue,"first_ns":first,"done_ns":done,"tokens":tokens,
            "prompt_tokens":prompt_tokens,"e2e_ms":(done-issue)/1e6}

def tokenizer_ids(config: str, session_hash: str, turn: int, length: int) -> list[int]:
    from experiments.phase4c.synthetic_tokens import deterministic_token_ids
    return deterministic_token_ids(length,phase="Phase4C",config=config,session_hash=session_hash,
        turn_index=turn,vocab_size=len(TOKENIZER),special_ids=TOKENIZER.all_special_ids)

def source_samples(raw: str) -> dict:
    return parse_prometheus_samples(raw)

def source_delta(before: str, after: str) -> dict:
    return source_tokens_delta(source_samples(before),source_samples(after))

def snapshot_sources() -> tuple[str,dict]:
    raw=metric_text()
    return raw,source_samples(raw)

def tokenizer_specials() -> set[int]:
    return set(TOKENIZER.all_special_ids)
