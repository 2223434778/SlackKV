from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.phase4c.synthetic_tokens import make_next_prompt, token_hash
from experiments.phase4d_a import runtime
from experiments.phase4d_a.helpers import FROZEN_SHA256, first_boundary_jobs, minimum_h2d_mib, sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/phase4d_a"
WORKLOAD = ROOT / "results/phase4c/hybrid_workload.csv"
REFERENCE = ROOT / "results/phase4c/request_metrics.csv"
START_COMMIT = "2fea2d810305ef94d65068c812a236b08a19e73c"
VLLM_COMMIT = "0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665"

def write_rows(path: Path, rows: list[dict[str,Any]]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as f:
        fields=list(dict.fromkeys(k for row in rows for k in row))
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)

def prepare_frozen_inputs() -> list[dict[str,Any]]:
    digest=sha256_file(WORKLOAD)
    if digest != FROZEN_SHA256:
        raise RuntimeError(f"EXPERIMENT_INVALID_WORKLOAD: sha256={digest}")
    jobs=first_boundary_jobs(WORKLOAD)
    write_rows(OUT/"boundary0_jobs.csv",jobs)
    (OUT/"source_manifest.json").write_text(json.dumps({
        "slackkv_start_commit":START_COMMIT,"phase4c_workload_path":"results/phase4c/hybrid_workload.csv",
        "phase4c_workload_sha256":digest,"vllm_source_commit":VLLM_COMMIT,
        "boundary_index":0,"deadline_source":"Turn0.proxy_pause_ms",
        "resume_prefix_source":"Turn1.target_cached_tokens/blocks and reconstructed Turn1 prompt",
        "gpu_kv_bytes":2147483648,"cpu_external_kv_bytes":8589934592,
        "bytes_per_kv_token":28672,"block_size_tokens":16,"k2":2
    },indent=2)+"\n",encoding="utf-8")
    return jobs

def stage0a(jobs: list[dict[str,Any]]) -> bool:
    reference={}
    with REFERENCE.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            reference[(row["session_hash"],int(row["replay_turn_index"]))]=row
    proc,log=runtime.start_server("stage0a_manifest",connector=False,gpu_bytes=4294967296,out=OUT)
    manifest=[]
    try:
        _,block=runtime.wait_ready(proc)
        if block!=16: raise RuntimeError(f"EXPERIMENT_INVALID_ENVIRONMENT:block_size={block}")
        for job in jobs:
            h=job["session_hash"]; config=job["config"]
            t0=runtime.tokenizer_ids(config,h,0,job["turn0_isl"])
            result=runtime.send(t0,job["turn0_osl"])
            out=result["tokens"]
            if len(out)!=job["turn0_osl"]: raise RuntimeError(f"Turn0 token count mismatch:{job['episode_id']}")
            served=t0+out
            t1=make_next_prompt(served,job["turn1_isl"],job["required_prefix_tokens"],
                phase="Phase4C",config=config,session_hash=h,turn_index=1,
                vocab_size=len(runtime.TOKENIZER),special_ids=runtime.TOKENIZER.all_special_ids)
            r0=reference[(h,0)]; r1=reference[(h,1)]
            hashes=(token_hash(t0),token_hash(out),token_hash(t1))
            expected=(r0["prompt_sha256"],r0["output_sha256"],r1["prompt_sha256"])
            if hashes!=expected: raise RuntimeError(f"EXPERIMENT_INVALID_WORKLOAD:hash mismatch:{job['episode_id']}")
            manifest.append({"episode_id":job["episode_id"],"session_hash":h,"turn0_prompt_sha256":hashes[0],
                "turn0_output_sha256":hashes[1],"turn0_output_ids":out,"turn1_prompt_sha256":hashes[2],
                "target_cached_tokens":job["required_prefix_tokens"],"target_cached_blocks":job["required_blocks"],
                "resume_osl":job["resume_osl"],"proxy_pause_ms":job["proxy_pause_ms"]})
    finally:
        runtime.stop_server(proc,log,OUT,"stage0a_manifest")
    (OUT/"token_manifest.json").write_text(json.dumps(manifest,separators=(",",":"))+"\n",encoding="utf-8")
    return len(manifest)==8

def stage_a0() -> dict[str,Any]:
    import sys
    sys.path.insert(0,str(ROOT/"experiments/phase4c"))
    import experiments.phase4c.run_phase4c as calibration
    calibration.OUT=OUT
    passed=calibration.run_a0()
    result=json.loads((OUT/"stage_a0.json").read_text(encoding="utf-8"))
    return result

def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True); (OUT/"server_logs").mkdir(exist_ok=True)
    jobs=prepare_frozen_inputs()
    if not stage0a(jobs):
        raise SystemExit("Stage 0A failed")
    report=stage_a0()
    print(json.dumps({"workload_sha256":FROZEN_SHA256,"jobs":len(jobs),
        "minimum_h2d_mib":minimum_h2d_mib(jobs),"stage0a":"PASS","stage_a0":report},indent=2))

if __name__=="__main__":
    main()
