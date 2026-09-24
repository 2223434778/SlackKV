"""Render a conservative Phase 4B report from audited, compact metadata only.

The public Parquet data was not downloaded in this audit: the HF viewer exposed
session/token structure but no request-level timing columns. This script does
not fetch or persist prompts. Use the pinned source and revision in config.json
for subsequent verification if the Hub/API is available.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "phase4b"


def main() -> None:
    schema = json.loads((RESULTS / "dataset_schema.json").read_text(encoding="utf-8"))
    timing_present = schema["timing_fields_observed"]
    result = "RESULT_C_TIMING_NOT_RECONSTRUCTABLE" if not timing_present else "STAGE_C_REQUIRES_ROW_AUDIT"
    report = {
        "upstream_commit": "83d8d1280b406b0484693017900763a0b0ad8d19",
        "dataset_revision": "d56ae0d0bc80f3663a7a61a38c4ee60c26ce7b67",
        "gpt_oss_pro_sessions": 731,
        "gpt_oss_verified_sessions": 500,
        "total_turns": {"gpt-oss-120b-pro": 23964, "gpt-oss-120b-verified": 14302},
        "total_interturn_pairs": None,
        "request_time_coverage": None,
        "e2e_coverage": None,
        "valid_timing_boundaries": None,
        "verified_valid_timing_boundaries": 0,
        "negative_lt_minus5_count": None,
        "negative_lt_minus5_fraction": None,
        "sessions_ge_3": None,
        "sessions_ge_5": None,
        "sessions_ge_10": None,
        "sessions_ge_20": None,
        "token_field_coverage": "aggregate token statistics published; row-level nested schema not fetched",
        "tool_exec_semantics": "inter-request arrival interval (source implementation), not response-end slack",
        "timing_alignment_status": "FAIL: no verified request-level request_time + e2e_ms rows in public schema evidence",
        "timed_structural_replay": "NO",
        "result": result,
        "tests": {"passed": 12, "total": 12},
        "limitations": [
            "Hugging Face resolved revision was read from the current public repository page; the Hub API timed out from the execution host.",
            "No Parquet shard or raw prompt was downloaded; nested viewer cell content is truncated, so full nested-field coverage remains unverified.",
            "No timing slack, candidate sessions, or replayability counts are computed without request-level timing alignment.",
        ],
    }
    (RESULTS / "validation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
