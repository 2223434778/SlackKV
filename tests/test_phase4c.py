from __future__ import annotations

from experiments.phase4c.selection import aligned_prefix, inspect_session, select_sessions, session_digest
from experiments.phase4c.synthetic_tokens import (
    common_prefix, deterministic_token_ids, longest_full_block_match, make_next_prompt,
    stable_seed, token_hash,
)
from experiments.phase4c.replay_helpers import (
    actual_pause_ms, block_alignment, pause_error_ms, result_classification,
    trace_tool_proxy_quantiles,
)


def fixture_turns(start: int = 100):
    return [{"isl": start + i * 8, "isl_new": 0, "osl": 32} for i in range(5)]


def test_source_cached_and_alignment():
    assert aligned_prefix(1003, 0) == (1003, 992, 11)
    assert block_alignment(1003) == (62, 11)


def test_invalid_isl_new_rejected():
    row, turns = inspect_session("gpt-oss-120b-pro", "private-id", [{"isl": 10, "isl_new": 11, "osl": 1}] * 5)
    assert not turns and row["invalid_token_accounting"]


def test_alignment_loss_bound():
    for value in range(1000):
        _, _, loss = aligned_prefix(value, 0)
        assert 0 <= loss < 16


def test_sha_seed_deterministic():
    assert stable_seed("p", "c", "s", 1) == stable_seed("p", "c", "s", 1)


def test_session_hash_is_revision_config_id_sha256():
    import hashlib
    expected = hashlib.sha256(b"d56ae0d0bc80f3663a7a61a38c4ee60c26ce7b67|cfg|id").hexdigest()
    assert session_digest("cfg", "id") == expected


def test_special_tokens_excluded_and_ids_reproducible():
    a = deterministic_token_ids(200, phase="p", config="c", session_hash="s", turn_index=0,
                                vocab_size=1000, special_ids={0, 1, 2})
    b = deterministic_token_ids(200, phase="p", config="c", session_hash="s", turn_index=0,
                                vocab_size=1000, special_ids={0, 1, 2})
    assert a == b and not ({0, 1, 2} & set(a))


def test_exact_prompt_length_and_shared_prefix():
    served = list(range(200))
    prompt = make_next_prompt(served, 150, 128, phase="p", config="c", session_hash="s", turn_index=1,
                              vocab_size=1000, excluded_ids=range(200))
    assert len(prompt) == 150 and common_prefix(prompt, served) == 128


def test_first_divergent_suffix_is_enforced():
    previous = [3] * 20 + [999]
    prompt = make_next_prompt(previous, 24, 20, phase="p", config="c", session_hash="s", turn_index=1,
                              vocab_size=1000, excluded_ids={999})
    assert prompt[20] != previous[20]


def test_exact_token_shared_prefix():
    previous = list(range(100))
    prompt = make_next_prompt(previous, 70, 48, phase="p", config="c", session_hash="s", turn_index=1,
                              vocab_size=1000, excluded_ids=range(100))
    assert common_prefix(prompt, previous) == 48


def test_accidental_old_prefix_detection():
    prior = [list(range(80))]
    prompt = list(range(64)) + [100, 101]
    assert longest_full_block_match(prompt, prior) == 64


def test_structural_prefix_feasibility():
    rows = fixture_turns()
    _, selected = inspect_session("gpt-oss-120b-pro", "id", rows)
    assert selected and selected[1]["target_cached_tokens"] <= rows[0]["isl"] + rows[0]["osl"]


def test_five_turn_candidate():
    row, selected = inspect_session("gpt-oss-120b-pro", "id", fixture_turns())
    assert row["eligible"] and len(selected) == 5


def test_earliest_valid_window():
    turns = [{"isl": 10, "isl_new": 11, "osl": 4}] + fixture_turns(100)
    row, selected = inspect_session("gpt-oss-120b-pro", "id", turns)
    assert row["selected_window_start"] == 1 and selected[0]["trace_turn_index"] == 1


def test_selection_deterministic_and_config_balanced():
    records = []
    for config in ("gpt-oss-120b-pro", "gpt-oss-120b-verified"):
        for i in range(5):
            records.append((config, str(i), fixture_turns()))
    pop_a, turns_a = select_sessions(records)
    pop_b, turns_b = select_sessions(records)
    assert [r["session_hash"] for r in pop_a if r["selected"]] == [r["session_hash"] for r in pop_b if r["selected"]]
    assert len(turns_a) == len(turns_b) == 40
    assert sum(r["selected"] and r["config"].endswith("pro") for r in pop_a) == 4
    assert sum(r["selected"] and r["config"].endswith("verified") for r in pop_a) == 4


def test_osl_cap():
    turns = fixture_turns()
    turns[2]["osl"] = 513
    row, selected = inspect_session("gpt-oss-120b-pro", "id", turns)
    assert not selected and row["osl_cap_violation"]


def test_context_limit():
    turns = fixture_turns(32760)
    row, selected = inspect_session("gpt-oss-120b-pro", "id", turns)
    assert not selected and row["context_limit_violation"]


def test_context_shrink_excluded():
    turns = fixture_turns()
    turns[3]["isl"] = turns[2]["isl"] - 1
    row, selected = inspect_session("gpt-oss-120b-pro", "id", turns)
    assert not selected and row["context_shrink"]


def test_trace_tool_proxy_quantile_and_permutation():
    pool = list(range(100, 301))
    a = trace_tool_proxy_quantiles(pool)
    assert a == trace_tool_proxy_quantiles(pool) and len(a) == 32
    assert sorted(a) == sorted(trace_tool_proxy_quantiles(pool))


def test_deterministic_pause_permutation():
    pool = [100, 250, 1000, 3000]
    assert trace_tool_proxy_quantiles(pool, count=32, seed=20260925) == trace_tool_proxy_quantiles(pool, count=32, seed=20260925)


def test_pause_error_uses_monotonic_ns_delta():
    actual = actual_pause_ms(1_000_000_000, 1_125_000_000)
    assert actual == 125 and pause_error_ms(actual, 120) == 5


def test_hash_reproducibility():
    ids = [1, 4, 9]
    assert token_hash(ids) == token_hash(ids)


def test_no_raw_identifier_or_text_in_population_record():
    row, _ = inspect_session("gpt-oss-120b-pro", "raw-instance-id", fixture_turns())
    assert "raw-instance-id" not in str(row)
    assert not any(key.endswith("_text") for key in row)


def test_result_classification_sparse():
    assert result_classification(source_ok=True, env_ok=True, eligible_pro=3, eligible_verified=4,
                                 calibration_ok=True, smoke_ok=True, full_ok=True) == "RESULT_B_NATIVE_FIT_SPARSE"


def test_result_classification_a():
    assert result_classification(source_ok=True, env_ok=True, eligible_pro=4, eligible_verified=4,
                                 calibration_ok=True, smoke_ok=True, full_ok=True) == "RESULT_A_HYBRID_REPLAY_CORRECT"


def test_final_phase4c_artifacts_when_present():
    import csv
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    result_dir = root / "results" / "phase4c"
    final_requests = result_dir / "request_metrics.csv"
    if not final_requests.exists():
        return
    requests = list(csv.DictReader(final_requests.open(encoding="utf-8")))
    boundaries = list(csv.DictReader((result_dir / "boundary_fidelity.csv").open(encoding="utf-8")))
    assert len(requests) == 40 and len(boundaries) == 32
    assert all(int(row["prompt_length_error"]) == 0 and int(row["generation_length_error"]) == 0 for row in requests)
    assert all(int(row["cached_block_error"]) == 0 for row in boundaries)
    assert all(int(row["actual_external_cached_tokens"]) == 0 for row in requests)
    report = json.loads((result_dir / "validation_report.json").read_text(encoding="utf-8"))
    assert report["result"] == "RESULT_A_HYBRID_REPLAY_CORRECT"
    for filename in ("source_turns.csv", "selection_population.csv", "selected_sessions.csv"):
        header = (result_dir / filename).read_text(encoding="utf-8").splitlines()[0].lower()
        assert "instance_id" not in header and "isl_text" not in header and "osl_text" not in header
