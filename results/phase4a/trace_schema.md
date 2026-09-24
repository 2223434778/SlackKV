# TraceLab v0.0.2 schema and time-semantics audit

The public dataset was inspected directly; all row counts and event frequencies are in `schema_snapshot.json`.

## Tables and identifiers

### `rounds` (665,453 rows)

- `round_pk`: `BIGINT`
- `ingest_seq`: `BIGINT`
- `provider`: `VARCHAR`
- `project`: `VARCHAR`
- `session_id`: `VARCHAR`
- `round_index`: `BIGINT`
- `round_id`: `VARCHAR`
- `model`: `VARCHAR`
- `input_tokens_total`: `BIGINT`
- `prefix_tokens`: `BIGINT`
- `newly_append_tokens`: `BIGINT`
- `claude_uncached_input_tokens`: `BIGINT`
- `claude_cache_creation_input_tokens`: `BIGINT`
- `claude_cache_read_input_tokens`: `BIGINT`
- `output_tokens`: `BIGINT`
- `current_input_event_count`: `BIGINT`
- `current_user_message_count`: `BIGINT`
- `current_tool_result_count`: `BIGINT`
- `current_user_message_chars`: `BIGINT`
- `current_tool_result_chars`: `BIGINT`
- `current_input_chars`: `BIGINT`
- `first_input_event_type`: `VARCHAR`
- `user`: `VARCHAR`
- `store`: `VARCHAR`
- `trace_key`: `VARCHAR`
- `turn_id`: `UUID`
- `reasoning_output_tokens`: `BIGINT`

### `tool_calls` (743,819 rows)

- `round_pk`: `BIGINT`
- `tool_index`: `BIGINT`
- `tool_name`: `VARCHAR`
- `tool_call_id`: `VARCHAR`
- `emitted_at`: `TIMESTAMP`
- `result_at`: `TIMESTAMP`
- `tool_wall_latency_ms`: `BIGINT`
- `tool_internal_latency_ms`: `BIGINT`
- `is_error`: `BOOLEAN`
- `input_chars`: `BIGINT`
- `result_chars`: `BIGINT`
- `continuation_of_tool_call_id`: `VARCHAR`
- `command_status`: `VARCHAR`
- `command_exit_code`: `HUGEINT`
- `executables`: `VARCHAR[]`
- `executable_parse_status`: `VARCHAR`
- `executable_parse_reason`: `VARCHAR`
- `command_skeleton`: `VARCHAR`

### `timing_events` (2,688,829 rows)

- `round_pk`: `BIGINT`
- `event_index`: `BIGINT`
- `event_type`: `VARCHAR`
- `source`: `VARCHAR`
- `timestamp`: `TIMESTAMP`
- `tool_call_id`: `VARCHAR`
- `tool_index`: `BIGINT`
- `tool_name`: `VARCHAR`
- `is_error`: `BOOLEAN`
- `result_chars`: `BIGINT`
- `content_chars`: `BIGINT`

### `trace_source` (1 rows)

- `path`: `VARCHAR`

## Mapping

- Logical session: `rounds.session_id`; round order: `rounds.round_index`; round joins: `round_pk`.
- Tool emission/completion: `tool_calls.emitted_at` / `result_at`.
- Token accounting: `input_tokens_total = prefix_tokens + newly_append_tokens`. `prefix_tokens` is trace-observed cached-input tokens, so it is the most relevant reusable-prefix measurement; it is not an exact vLLM KV block identity.
- `timing_events` contains input-ready (`user_message`, `tool_result`) and output-side (`reasoning`, `text`, `tool_call`, `usage_report`) observations, but no explicit next-LLM request/start event.
- `rounds` has no request/start timestamp column. `tool_calls.result_at` is tool completion, not proof that the next LLM request started at that instant.

## Actual resume time decision

`actual_resume_time = UNOBSERVABLE`. Accordingly `decision_time -> actual_resume_time` and `actual_slack_ms` cannot be measured without substituting a proxy forbidden by the Phase 4A protocol. No predictor or opportunity estimate is emitted.

The [TraceLab v0.0.2 data-format documentation](https://github.com/uw-syfi/TraceLab/blob/v0.0.2/README.md#data-format) lists these event types and describes input-ready-to-tool-input as a *proxy*, not an LLM request-start timestamp. The [prompt-cache accounting note](https://github.com/uw-syfi/TraceLab/blob/v0.0.2/docs/prompt_cache_accounting.md) defines `prefix_tokens` as cache-read/cached-input tokens.
