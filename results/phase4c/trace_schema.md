# Phase 4C source schema mapping

## Coding-agent structure

HF rows expose top-level `instance_id`, `dataset`, `model`, and an ordered `turns` list. The turn fields used were `isl`, `isl_new`, and `osl`; text fields were read only as part of the API response and immediately discarded. `instance_id` was used only in memory to derive the SHA256 session hash and is not persisted. List order is the source turn order.

## TraceLab v0.0.2 timing proxy

DuckDB `SHOW TABLES` returned `rounds`, `timing_events`, `tool_calls`, and `trace_source`. `rounds` provides `round_pk`, provider/model/session/round ordering, and token counts (`input_tokens_total`, `newly_append_tokens`, `output_tokens`). `tool_calls` provides `round_pk`, `emitted_at`, `result_at`, `tool_wall_latency_ms`, and `tool_name`. Phase 4C groups tool calls by `round_pk` and computes the timing proxy from the earliest complete `emitted_at` to latest complete `result_at`; it does not use raw inputs or command skeletons. `timing_events` has event timestamps but was not used for pause values.

The replay workload is a hybrid trace-driven replay, not semantic conversation replay and not a request-timestamp-faithful replay.

## Field-level notes

In `rounds`, logical grouping/order fields are `session_id`, `round_index`, and `ingest_seq`; no round-level timestamp column is present. `input_tokens_total`, `newly_append_tokens`, and `output_tokens` are the token-count fields inspected. `timing_events.timestamp` is the released event timestamp, keyed by `round_pk` and ordered with `event_index`; it was not treated as a reliable next-request start. Tool records are in `tool_calls`, keyed by `round_pk` and `tool_index`, with `emitted_at`, `result_at`, and `tool_wall_latency_ms`. `tool_internal_latency_ms` is also present but was not used. `trace_source` contains only `path`.

No explicit context-compaction/shrink field was found in the inspected tables. The selected HF windows were checked for non-decreasing `isl` and source-level structural validity; no compaction value was invented. Tool timing was reconstructed only from complete emitted/result timestamps, using earliest `emitted_at` to latest `result_at` per round to preserve overlapping calls.
