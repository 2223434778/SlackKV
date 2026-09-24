# Phase 4B — Public Trace Timing & Replayability Audit

## Objective and setup

Offline audit of `netpreme/coding_agent_traces` for request-level timing and structural replayability. No vLLM/SlackKV benchmark, GPU, SWE-Bench run, paid API, prompt download, or Parquet download was used.

Frozen upstream source is `netpreme/coding_agents@83d8d1280b406b0484693017900763a0b0ad8d19` (2026-07-08). Frozen HF page revision is `d56ae0d0bc80f3663a7a61a38c4ee60c26ce7b67`. Public card counts are 731 Pro sessions / 23,964 turns and 500 Verified sessions / 14,302 turns.

## Findings

1. The HF viewer exposes session metadata (`instance_id`, `dataset`, `model`) and nested `turns`; the visible GPT-OSS turn preview includes token-count and text members. It does not provide verified row-level `request_time` + `e2e_ms` pairs in the schema evidence collected. Full nested schema could not be fetched because the Hub API timed out; no Parquet was downloaded. Hence request/e2e coverage and candidate-boundary statistics are **not measured**, rather than presumed zero in the underlying data.
2. In pinned proxy source, `request_time` is stamped at proxy request entry, before sanitization and before forwarding.
3. The pinned metrics scraper derives `e2e_ms` as a delta of cumulative Prometheus E2E latency over adjacent scrape snapshots, divided by `n`. When multiple requests complete in a polling tick, this is a per-request average, not exact request latency. Source does not expose an exact request-ID join between that aggregate and each proxy arrival time.
4. Pinned recorder source calls `note_response_time()` on new request arrival and updates its stored timestamp there. Therefore `tool_exec_ms` is approximately current-arrival minus previous-arrival, not previous-response-completion to current-arrival. This is a source-code mismatch with README wording.
5. Because exact request/e2e alignment is not established, candidate slack `arrival_i - (arrival_(i-1) + e2e_(i-1))` is not computed. Negative slack is therefore not clipped or summarized. Row-level KV replayability and candidate-session counts are likewise not computed.

## Result

**RESULT_C_TIMING_NOT_RECONSTRUCTABLE** on available public timing evidence. The dataset contains useful aggregate token/session shape, but this audit did not establish a request-level timing ground truth suitable for timed replay. This is not evidence that SlackKV improves latency.

`TIMED_STRUCTURAL_REPLAY = NO` for this phase. Token-level structure may support a future structural-only workload, but not an oracle-timed replay without verified aligned request times and per-request completion durations.

## Anomalies and limitations

- HF Hub API timed out from both local and A6000 hosts; dataset revision was taken from the current public HF Files page commit link. The pinned revision string is recorded but its API metadata could not be independently retrieved.
- Public viewer cell contents are truncated and nested schema completeness remains unverified. No claim is made that hidden Parquet fields are definitively absent; only that the required timing fields were not available as verified row-level evidence in the release view.
- No Stage C slack distribution or row-level replay candidate count is reported.
- No raw prompts or large dataset files were stored.

## Validation

Focused helper tests cover session isolation/order, units, slack (including negative preservation), KV token accounting, block rounding, missing E2E exclusion, consecutive runs, determinism, and source revision pinning.
