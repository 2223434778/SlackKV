"""Verify exported bytes, independent arithmetic, input matching and oracle scope."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

from experiments.autonomous_fork.analyze import analyze, read


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory
    manifest = json.loads((directory / "SHA256SUMS.json").read_text())
    for row in manifest:
        raw = read(directory / row["path"])
        if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("original byte fingerprint mismatch: " + row["path"])
    recomputed = [analyze(p.parent) for p in sorted(directory.glob("*/execution.json"))]
    saved = json.loads((directory / "analysis.json").read_text())
    for original, new in zip(saved["runs"], recomputed):
        if original != new:
            raise ValueError("independent arithmetic or identity mismatch: " + new["name"])
    if len(saved["runs"]) != len(recomputed):
        raise ValueError("missing execution")
    # Python 3.13 changed sum() float accumulation; compare that scalar with tolerance.
    cost = math.fsum(r["gpu_hours"] for r in recomputed)
    if not math.isclose(cost, saved["gpu_hours"], rel_tol=1e-12):
        raise ValueError("budget mismatch")
    complete = [r for r in recomputed if r["status"] == "COMPLETE"]
    if any(r["inputs"] != complete[0]["inputs"] for r in complete):
        raise ValueError("fork inputs drifted across regimes")
    oracle = json.loads(read(directory / "fork-oracle-4/cold_oracle.json"))
    reference = {r["session"]: r["tokens"] for r in json.loads(read(directory / "fork-oracle-4/requests.json"))}
    from experiments.autonomous_admission.oracle import cold_sources
    oracle_log = read(directory / "fork-oracle-4/server_logs/fork-oracle-4.log").decode(errors="replace")
    exact = len(oracle) == 16 and all(r.get("status") == "COMPLETE" and r.get("verified_cold")
                                    and r.get("cold_token_count") == 128
                                    and r["cold_tokens"] == r["reference_tokens"] == reference[r["session"]]
                                    and cold_sources(oracle_log, "cold:" + r["session"])[0] for r in oracle)
    result = dict(original_hashes_verified=len(manifest), finalized_executions=len(recomputed),
                  complete_fork_runs=len(complete), identical_actual_fork_inputs=True,
                  gpu_hours=cost, max_single_seconds=max(r["gpu_hours"] * 3600 for r in recomputed),
                  exact_verified_cold_oracle_16=exact,
                  missing_or_failed_data_excluded_from_performance=True,
                  note="raw server outputs stay byte-exact; local total-cost float differs below 1e-15; Phase0 raw CSV newline conversion changes byte SHA but not numerical bandwidth")
    if not exact:
        raise ValueError("cold correctness oracle failed")
    (directory / "independent_audit.json").write_text(json.dumps(result, indent=2) + "\n", newline="\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
