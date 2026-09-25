# Phase 4B — Public Trace Timing & Replayability Audit

## Objective and setup

Offline audit of `netpreme/coding_agent_traces` for request-level timing and structural replayability. No vLLM/SlackKV benchmark, GPU, SWE-Bench run, or paid API was used. The bounded rows API responses included public text fields, processed in memory only; no raw prompt/completion text or Parquet file was persisted.

**2026-09-25 live-access recheck:** the prior HF timeout was caused by the non-login remote command omitting the A6000 interactive shell's HTTPS proxy. The login-shell audit now verifies the revision and samples the live rows API; see `live_hf_recheck.json` and `row_schema_sample_100.json`.

Frozen upstream source is `netpreme/coding_agents@83d8d1280b406b0484693017900763a0b0ad8d19` (2026-07-08). Frozen HF page revision is `d56ae0d0bc80f3663a7a61a38c4ee60c26ce7b67`. Public card counts are 731 Pro sessions / 23,964 turns and 500 Verified sessions / 14,302 turns.

## Findings

1. The HF Hub API is reachable from the A6000 login shell when its HTTPS proxy environment is inherited. Hub metadata confirms the frozen revision and declares GPT-OSS `turns` as `list<JSON>`. A bounded first-100-row sample per GPT-OSS config (200 sessions, 5,550 turns) exposed only `isl`, `isl_new`, `osl` and their text fields, with no tool-, request-, timestamp-, duration-, or latency-named fields. This sample is not an exhaustive scan of every nested JSON row.
2. In pinned proxy source, `request_time` is stamped at proxy request entry, before sanitization and before forwarding.
3. The pinned metrics scraper derives `e2e_ms` as a delta of cumulative Prometheus E2E latency over adjacent scrape snapshots, divided by `n`. When multiple requests complete in a polling tick, this is a per-request average, not exact request latency. Source does not expose an exact request-ID join between that aggregate and each proxy arrival time.
4. Pinned recorder source calls `note_response_time()` on new request arrival and updates its stored timestamp there. Therefore `tool_exec_ms` is approximately current-arrival minus previous-arrival, not previous-response-completion to current-arrival. This is a source-code mismatch with README wording.
5. Because exact request/e2e alignment is not established, candidate slack `arrival_i - (arrival_(i-1) + e2e_(i-1))` is not computed. Negative slack is therefore not clipped or summarized. Row-level KV replayability and candidate-session counts are likewise not computed.

## Result

**RESULT_C_TIMING_NOT_RECONSTRUCTABLE** on available public timing evidence. The dataset contains useful aggregate token/session shape, but this audit did not establish a request-level timing ground truth suitable for timed replay. This is not evidence that SlackKV improves latency.

`TIMED_STRUCTURAL_REPLAY = NO` for this phase. Token-level structure may support a future structural-only workload, but not an oracle-timed replay without verified aligned request times and per-request completion durations.

## Anomalies and limitations

- The earlier timeout was specific to the non-login remote command not inheriting proxy variables. The 2026-09-25 recheck verified the exact HF API revision and sampled live rows through a login shell.
- The row sample is not exhaustive; the full GPT-OSS datasets total over 7 GB and were not downloaded. No conclusion is made about unobserved nested rows beyond the declared schema and sample.
- No Stage C slack distribution or row-level replay candidate count is reported.
- No raw prompts or large dataset files were stored.

## Validation

Focused helper tests cover session isolation/order, units, slack (including negative preservation), KV token accounting, block rounding, missing E2E exclusion, consecutive runs, determinism, and source revision pinning.

The live-access audit/privacy tests also passed. The original and live suites passed 14/14. A reproducible 10-row/config audit was saved; the larger 100-row/config field scan was inspected in a no-persistence console pipeline. No GPU or serving workload was run.
