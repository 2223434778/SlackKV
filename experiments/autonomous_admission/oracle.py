"""Cold-prefill replay after timed workload, using supported cache_salt isolation."""
from __future__ import annotations

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from experiments.phase4c.synthetic_tokens import token_hash


def cold_sources(text: str, session: str) -> tuple[bool, list[dict]]:
    source = [json.loads(line.split("SLACKKV_SOURCE_TRACE ", 1)[1])
              for line in text.splitlines() if "SLACKKV_SOURCE_TRACE " in line]
    source = [event for event in source if event["session"] == session]
    verified = bool(source) and all(event["local_cached_tokens"] == 0 and
                                    event["external_cached_tokens"] == 0 for event in source)
    return verified, source


def run_oracle(runtime, directory: Path, log: Path) -> None:
    (directory / "metrics_after_timed_workload.prom").write_text(runtime.metric_text())
    records = [json.loads(line) for line in (directory / "output_tokens.jsonl").read_text().splitlines()]
    selected = [r for r in records if "prompt" in r]
    rows = []
    def query(record, repetition, mode):
        oracle_session = f'cold:{record["session"]}:{mode}:{repetition}'
        offset = log.stat().st_size
        try:
            payload = {
                "model": str(runtime.MODEL), "prompt": record["prompt"],
                "max_tokens": len(record["tokens"]), "temperature": 0,
                "ignore_eos": True, "return_token_ids": True, "logprobs": 5,
                "cache_salt": oracle_session,
                "kv_transfer_params": {"max_offload_tokens": 0,
                                       "slackkv_trace_resume": True,
                                       "slackkv_session": oracle_session},
            }
            request = urllib.request.Request(runtime.API, data=json.dumps(payload).encode(),
                                             headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=120) as response:
                body = json.load(response)
            tokens = body["choices"][0]["token_ids"]
            pairs = zip(record["tokens"], tokens)
            divergence = next((i for i, (left, right) in enumerate(pairs) if left != right), None)
            with log.open("rb") as stream:
                stream.seek(offset)
                new_log = stream.read().decode(errors="replace")
            verified_cold, source = cold_sources(new_log, oracle_session)
            row = {"session": record["session"], "repetition": repetition,
                   "mode": mode, "verified_cold": verified_cold, "source_trace": source,
                   "status": "COMPLETE", "expected_tokens": len(record["tokens"]),
                   "actual_tokens": len(tokens), "reference_hash": token_hash(record["tokens"]),
                   "cold_hash": token_hash(tokens), "exact_match": tokens == record["tokens"],
                   "first_divergence": divergence, "usage": body.get("usage"),
                   "response": body}
        except Exception as exc:
            row = {"session": record["session"], "repetition": repetition,
                   "mode": mode,
                   "status": "FAILED", "error": repr(exc)}
        return row

    for record in selected:
        for repetition in range(2):
            row = query(record, repetition, "serial")
            rows.append(row)
            (directory / "cold_oracle.json").write_text(json.dumps(rows, indent=2) + "\n")
        with ThreadPoolExecutor(max_workers=3) as pool:
            batch = list(pool.map(lambda index: query(record, index, "concurrent3"), range(3)))
        rows.extend(batch)
        (directory / "cold_oracle.json").write_text(json.dumps(rows, indent=2) + "\n")
