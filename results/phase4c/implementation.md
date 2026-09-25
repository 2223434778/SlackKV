# Phase 4C implementation notes

The implementation is split into `selection.py` (privacy-preserving row census and frozen selection), `synthetic_tokens.py` (stable SHA-seeded synthetic token construction and prefix guards), `timing_proxy.py` (TraceLab DuckDB proxy extraction and quantile assignment), `replay_helpers.py` (pause and block helpers), and `run_phase4c.py` (A0/A1/B serial real-vLLM stages). `finalize_phase4c.py` derives result artifacts from actual request records.

The HF row parser reads the fixed 0–99 pages in memory, immediately discards raw text, hashes instance IDs, and persists only token counts and hashes. Source turns remain in released list order. Synthetic prompt IDs are deterministic across processes and exclude tokenizer special IDs. Later prompts copy the exact aligned prefix from the immediately previous served sequence and begin their synthetic suffix with a divergent token. The prompt is checked against every prior served sequence for accidental longer block matches.

Prefix hit counts come from serialized deltas of `vllm:prefix_cache_hits_total`; Stage A0 proved the metric using 768 and 770 token shared prefixes. Runtime `cache_config_info` reported block size 16. The server runs the ordinary local prefix cache with no connector or SlackKV feature.

Pause targets are `previous_response_end_monotonic_ns + TraceToolProxy_ms`. The metrics baseline is captured before the wait so metric polling cannot shift actual issue time. All pause timestamps use `time.monotonic_ns()`.

DuckDB inspection mapping is in `trace_schema.md`. The database and model snapshot are not committed. The source tree for the pinned vLLM is reused unchanged.
