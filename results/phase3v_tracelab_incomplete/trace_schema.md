# TraceLab v0.0.1 schema mapping

The pinned DuckDB contains `rounds` (357,161 rows), `tool_calls` (432,510), `timing_events` (1,496,416), and `trace_source` (1).

| Phase 3V concept | Released field |
|---|---|
| logical session | `rounds.session_id` |
| provider/model | `rounds.provider`, `rounds.model` |
| round order | `rounds.round_index` |
| input/output tokens | `rounds.input_tokens_total`, `rounds.output_tokens` |
| reusable-prefix diagnostic | `rounds.prefix_tokens` |
| append tokens | `rounds.newly_append_tokens` (selection formula also evaluated directly) |
| tool identity | `tool_calls.tool_name` |
| emission/result timestamps | `tool_calls.emitted_at`, `tool_calls.result_at` |
| tool wall latency | `tool_calls.tool_wall_latency_ms` |
| compaction/shrink | no explicit field; conservatively require non-decreasing context over the window |

Tool pause was evaluated as `max(result_at) - min(emitted_at)` for each round's complete tool group. `session_id` is the logical session key (4,265 distinct values); `trace_key` is effectively round-specific (356,647 distinct values) and is not used as a session key.

## Eligibility gate outcome

Using the requested `input_tokens_total` mapping and every frozen filter, eligible four-boundary consecutive windows by median size class are: small 0, medium 11, large 2. Therefore the mandated 2 small + 2 medium + 2 large episode selection is impossible.

As a diagnostic only, substituting `prefix_tokens`—the closest released reusable-KV measure—produces small 0, medium 5, large 1. It does not resolve the blocker and was not adopted for an experiment.
