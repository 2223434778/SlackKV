"""Cold-prefill replay after timed workload, using supported cache_salt isolation."""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from experiments.phase4c.synthetic_tokens import token_hash


def run_oracle(runtime, directory: Path) -> None:
    records = [json.loads(line) for line in (directory / "output_tokens.jsonl").read_text().splitlines()]
    selected = [r for r in records if "prompt" in r]
    rows = []
    for record in selected:
        for repetition in range(2):
            try:
                payload = {
                    "model": str(runtime.MODEL), "prompt": record["prompt"],
                    "max_tokens": len(record["tokens"]), "temperature": 0,
                    "ignore_eos": True, "return_token_ids": True, "logprobs": 5,
                    "cache_salt": f'autonomous-cold-{record["session"]}-{repetition}',
                    "kv_transfer_params": {"max_offload_tokens": 0},
                }
                request = urllib.request.Request(runtime.API, data=json.dumps(payload).encode(),
                                                 headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=120) as response:
                    body = json.load(response)
                tokens = body["choices"][0]["token_ids"]
                pairs = zip(record["tokens"], tokens)
                divergence = next((i for i, (left, right) in enumerate(pairs) if left != right), None)
                row = {"session": record["session"], "repetition": repetition,
                       "status": "COMPLETE", "expected_tokens": len(record["tokens"]),
                       "actual_tokens": len(tokens), "reference_hash": token_hash(record["tokens"]),
                       "cold_hash": token_hash(tokens), "exact_match": tokens == record["tokens"],
                       "first_divergence": divergence, "usage": body.get("usage"),
                       "response": body}
            except Exception as exc:
                row = {"session": record["session"], "repetition": repetition,
                       "status": "FAILED", "error": repr(exc)}
            rows.append(row)
            (directory / "cold_oracle.json").write_text(json.dumps(rows, indent=2) + "\n")
