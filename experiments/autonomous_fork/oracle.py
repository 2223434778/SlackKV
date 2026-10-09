"""Serial cold-prefill identity oracle, separate from default-mode performance."""
from __future__ import annotations

import json
from pathlib import Path
import urllib.request

from experiments.autonomous_admission.oracle import cold_sources


def verify(runtime, directory: Path, log: Path) -> list[dict]:
    inputs = {r["session"]: r["prompt"] for r in
              map(json.loads, (directory / "inputs.jsonl").read_text().splitlines())}
    selected = [r for r in json.loads((directory / "requests.json").read_text())
                if r["session"].startswith("F0:")]
    if len(selected) != 16:
        raise ValueError("oracle requires every branch and both turns of episode zero")
    rows = []
    for record in sorted(selected, key=lambda r: r["session"]):
        session = "cold:" + record["session"]
        payload = dict(model=str(runtime.MODEL), prompt=inputs[record["session"]],
                       max_tokens=128, temperature=0, ignore_eos=True,
                       return_token_ids=True, cache_salt=session,
                       kv_transfer_params=dict(max_offload_tokens=0,
                                               slackkv_trace_resume=True,
                                               slackkv_session=session))
        request = urllib.request.Request(runtime.API, data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = json.load(response)
            tokens = body["choices"][0]["token_ids"]
            verified_cold, source = cold_sources(log.read_text(errors="replace"), session)
            rows.append(dict(session=record["session"], status="COMPLETE",
                             verified_cold=verified_cold, source_trace=source,
                             cold_tokens=tokens, reference_tokens=record["tokens"],
                             exact_match=tokens == record["tokens"],
                             cold_token_count=len(tokens)))
        except Exception as exc:
            rows.append(dict(session=record["session"], status="FAILED", error=repr(exc)))
        (directory / "cold_oracle.json").write_text(json.dumps(rows, indent=2) + "\n", newline="\n")
    return rows
