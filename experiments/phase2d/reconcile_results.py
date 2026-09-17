"""Reconcile cumulative source-token counters after a no-scrape JIT resume."""
from __future__ import annotations

import csv
import json

from experiments.phase2d.helpers import external_cache_hit_delta, source_tokens_delta
from experiments.phase2d.run_phase2d import RESULTS, write_summary
from experiments.phase2c_fix.helpers import parse_prometheus_samples


def main() -> None:
    raw_path = RESULTS / "raw_runs.csv"
    with raw_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    snapshots = json.loads((RESULTS / "metric_snapshots.json").read_text(encoding="utf-8"))
    by_stage = {item["stage"]: item for item in snapshots}

    for row in rows:
        policy = row["policy"]
        if not policy.startswith("JIT-"):
            continue
        prefix = f"{policy.lower().replace('-', '')}_rep{row['repetition']}"
        before_stage = by_stage[f"{prefix}_after_pressure"]
        after_stage = by_stage[f"{prefix}_after_resume"]
        before = parse_prometheus_samples(before_stage["raw_relevant"])
        after = parse_prometheus_samples(after_stage["raw_relevant"])

        external = source_tokens_delta(before, after, "external_kv_transfer")
        local = source_tokens_delta(before, after, "local_cache_hit")
        recomputed = source_tokens_delta(before, after, "local_compute")
        external_cache_hits = external_cache_hit_delta(before, after)

        # Since prefetch and resume are serialized with no intervening request,
        # exactly 4096 local-hit tokens after the pair belong to the resume.
        # The 4096 external-hit tokens therefore belong to the preceding proxy.
        if external != 4096.0 or local != 4096.0:
            raise ValueError(
                f"{prefix}: expected cumulative external/local hits 4096/4096, "
                f"observed {external}/{local}"
            )
        row.update({
            "prefetch_local_hit_tokens": 0.0,
            "prefetch_external_hit_tokens": external,
            "prefetch_external_cache_hit_tokens": external_cache_hits,
            "prefetch_recomputed_tokens": 0.0,
            "resume_local_hit_tokens": local,
            "resume_external_hit_tokens": 0.0,
            "resume_external_cache_hit_tokens": 0.0,
            "resume_recomputed_tokens": recomputed,
            "prefetch_validation": "PASS" if external == 4096.0 else "FAIL",
            "post_prefetch_local_validation": "PASS" if local == 4096.0 else "FAIL",
        })

    fieldnames = list(rows[0])
    with raw_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    control = json.loads((RESULTS / "control.json").read_text(encoding="utf-8"))
    result = write_summary(rows, control)
    print(json.dumps({"run_count": len(rows), **result}, indent=2))


if __name__ == "__main__":
    main()
