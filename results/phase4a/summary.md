# Phase 4A — real-trace opportunity and causal-predictability audit

## Objective and setup

Audit whether the sanitized TraceLab v0.0.2 coding-agent trace contains the *actual next LLM request/start time* needed to measure SlackKV inter-turn slack. This is a read-only, offline analysis; no vLLM source, controller, lease, GPU serving run, or Phase 4B work was changed or executed. The pinned DuckDB asset is 160,182,272 bytes and its SHA256 matches `a7bab286bc640844560850965ccf47975cf66407154132abaab90f27ec9be744`.

## Stage A result: timing-semantics gate

The database contains 665,453 `rounds`, 743,819 `tool_calls`, 2,688,829 `timing_events`, 8,058 logical sessions, and 574,485 rounds with at least one tool call. The actual schema and all event-type frequencies are saved in `schema_snapshot.json`; field mapping is in `trace_schema.md`.

There is **no explicit next-LLM request/start event or timestamp field** in this released schema. The only event types are `usage_report`, `tool_call`, `tool_result`, `reasoning`, `text`, and `user_message`. `rounds` has no timestamp. The earliest event of a next round is commonly a `tool_result` or `user_message`, which records an input-side event, not when the next model request starts. A `tool_calls.result_at` value measures tool completion. Neither is scientifically interchangeable with a request-start timestamp. The [pinned TraceLab data-format documentation](https://github.com/uw-syfi/TraceLab/blob/v0.0.2/README.md#data-format) itself describes an input-ready-to-tool-input interval as a *proxy* for LLM-side latency, not a direct request-start observation.

Therefore `actual_resume_time` is **UNOBSERVABLE** from the v0.0.2 public DuckDB under the required semantics. The frozen Stage A gate yields **`RESULT_C_NO_VALID_RESUME_SEMANTICS`**. We did not substitute the latest tool result time, first next-round event, or any output event for request start.

## Measurements intentionally not made

Without `actual_resume_time`, `actual_slack_ms = actual_resume_time - decision_time` is undefined. Thus valid timing/replayable boundary counts, joint slack–KV opportunity, 40/70/200/240-ms reference opportunities, GlobalPastMedian and ToolPastMedian MAE/P95/early/late rates, and a Phase 4B candidate pool are **not estimated**. Predictor evaluation count is zero; `future_leakage_count` is **NA**, not an asserted empirical zero. Focused synthetic tests verify that the online-predictor implementation would exclude unresolved future events, but no real-trace predictor result is claimed. Promotion service-model audit and Stage E replayability were not run after the mandatory timing gate. `VERBATIM_PROMPT_REPLAY = NO`; `STRUCTURAL_TRACE_REPLAY = NO` for a *timed* replay because the resume timeline is unavailable. Public token fields may support a separate prefix-structure analysis, but that does not repair the missing time semantics and was not used to bypass the stop.

The `prefix_tokens` column is trace-observed cached-input tokens, not `input_tokens_total`: [TraceLab's pinned cache-accounting note](https://github.com/uw-syfi/TraceLab/blob/v0.0.2/docs/prompt_cache_accounting.md) defines the split. This mapping was audited but not used to fabricate a serving workload.

## Validation, anomaly, conclusion

- Dataset checksum: PASS. Schema audit: PASS. Explicit request/start events: 0. Focused tests: 10/10 PASS.
- The Python serving environment emits a non-fatal startup warning about a missing optional `nvidia_cutlass_dsl` module; DuckDB audit and tests completed normally. DuckDB was installed only under `/tmp/slackkv-phase4a-deps`, not into the frozen vLLM environment.
- This is a **valid negative data-semantics finding**, not a SlackKV controller or predictor failure. It supersedes the previous uncommitted Phase 4A eligibility-only diagnostic, which assumed tool-result completion could define resume. No older phase result directory was changed.
- No Phase 4B workload selection or runtime experiment was started. Any future use of tool-result time must be labeled as a proxy or obtain a trace source that exposes actual next-LLM request start; no such change is made here.
