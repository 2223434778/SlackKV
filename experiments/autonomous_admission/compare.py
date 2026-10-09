"""Apply frozen screening thresholds and check every session, prompt and output."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.autonomous_admission.analyze import analyze


def compare(baseline: dict, candidate: dict) -> dict:
    a, b = analyze(baseline, []), analyze(candidate, [])
    def hashes(run):
        return {(t["episode_id"], t["turn_index"]): (t["prompt_sha256"], t["output_sha256"])
                for t in run["turn_rows"]}
    matching = hashes(baseline) == hashes(candidate)
    metrics = ("resume_p95_ms", "resume_median_ms", "session_e2e_median_ms",
               "BG_wall_ms", "BG_TPOT_ms", "agent_external_H2D_MiB",
               "mixed_generated_tokens_per_second")
    deltas = {key: 100 * (b[key] / a[key] - 1) for key in metrics}
    pass_screen = (a["valid"] and b["valid"] and matching
                   and deltas["resume_p95_ms"] <= -5
                   and deltas["session_e2e_median_ms"] <= 2
                   and deltas["BG_wall_ms"] <= 3)
    return {"baseline": a, "candidate": b, "percent_change": deltas,
            "prompt_and_output_hashes_match": matching, "pass_screen": pass_screen,
            "session_percent_change": {key: 100 * (b["session_e2e_ms"][key] / value - 1)
                                       for key, value in a["session_e2e_ms"].items()},
            "conclusion": "reverse-order replication warranted" if pass_screen else "stop bounded backfill direction"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    def load(directory):
        return json.loads((directory / "runs/measured_rep1_a/run.json").read_text())
    result = compare(load(args.baseline), load(args.candidate))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("percent_change", "prompt_and_output_hashes_match", "pass_screen", "conclusion")}, indent=2))


if __name__ == "__main__":
    main()
