"""Render Phase 4B validation from compact, privacy-preserving audit artifacts.

Run audit_hf_live.py first when the HF endpoint is reachable. This script never
fetches or persists row text; the live audit stores only schema/count summaries.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "phase4b"


def main() -> None:
    schema = json.loads((RESULTS / "dataset_schema.json").read_text(encoding="utf-8"))
    source = json.loads((RESULTS / "dataset_source.json").read_text(encoding="utf-8"))
    live_path = RESULTS / "live_hf_recheck.json"
    live = json.loads(live_path.read_text(encoding="utf-8")) if live_path.exists() else {}
    sample100_path = RESULTS / "row_schema_sample_100.json"
    sample100 = json.loads(sample100_path.read_text(encoding="utf-8")) if sample100_path.exists() else {}
    sample100_configs = sample100.get("configs", {})
    sampled_sessions = sum(item.get("sessions_scanned", 0) for item in sample100_configs.values())
    sampled_turns = sum(item.get("turns_scanned", 0) for item in sample100_configs.values())
    sampled_fields = sorted({key for item in sample100_configs.values() for key in item.get("turn_fields", [])})
    sampled_timing_fields = sorted({key for item in sample100_configs.values() for key in item.get("tool_or_timing_named_fields", [])})
    if not sampled_sessions:
        configs = live.get("configs", [])
        sampled_sessions = sum(item.get("sample_rows_received", 0) for item in configs)
        sampled_turns = sum(item.get("sample_turns_received", 0) for item in configs)
        sampled_fields = sorted({key for item in configs for key in item.get("turn_fields", [])})
        sampled_timing_fields = sorted({key for item in configs for key in item.get("timing_or_tool_named_fields_in_sample", [])})
    result = "RESULT_C_TIMING_NOT_RECONSTRUCTABLE"
    report = {
        "upstream_commit": "83d8d1280b406b0484693017900763a0b0ad8d19",
        "dataset_revision": source["resolved_revision"],
        "hf_revision_api_verified": live.get("revision_verified_against_hub_api", False),
        "gpt_oss_pro_sessions": source["row_counts"]["gpt-oss-120b-pro"],
        "gpt_oss_verified_sessions": source["row_counts"]["gpt-oss-120b-verified"],
        "total_turns": source["public_card_turn_counts"],
        "sampled_sessions": sampled_sessions,
        "sampled_turns": sampled_turns,
        "sampled_turn_fields": sampled_fields,
        "sampled_tool_or_timing_fields": sampled_timing_fields,
        "raw_text_persisted": False,
        "full_dataset_scan": False,
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
        "token_field_coverage": (
            f"live row fields inspected for {sampled_sessions} sessions / {sampled_turns} turns; "
            "no full text persisted"
            if sampled_sessions
            else "public aggregate counts only; live rows not queried"
        ),
        "tool_exec_semantics": "inter-request arrival interval (source implementation), not response-end slack",
        "timing_alignment_status": "FAIL: no verified request-level request_time + e2e_ms rows in public schema evidence",
        "timed_structural_replay": "NO",
        "result": result,
        "tests": {"status": "see pytest output; test counts are not embedded by this postprocessor"},
        "limitations": [
            "The live row scan is bounded, not exhaustive; full GPT-OSS Parquet was not downloaded.",
            "No timing slack, candidate sessions, or replayability counts are computed without request-level timing alignment.",
        ],
    }
    (RESULTS / "validation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
