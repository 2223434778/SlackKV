# Dataset schema audit

Frozen HF page revision: `d56ae0d0bc80f3663a7a61a38c4ee60c26ce7b67`.

The HF Hub metadata API identifies the GPT-OSS train tables as 731 Pro rows and 500 Verified rows. Top-level fields are `instance_id: string`, `dataset: string`, `model: string`, and `turns: list<JSON>`. The public dataset card reports 23,964 Pro turns and 14,302 Verified turns. The JSON-valued turn type does not statically enumerate nested fields.

The original viewer-only check truncated nested turn values. On 2026-09-25 the HF Hub metadata and rows APIs were queried from the A6000 login shell (the proxy environment was not inherited by non-login remote commands). Hub metadata verifies the pinned SHA and declares GPT-OSS `turns` as `list<JSON>`. A bounded first-100-row sample from each GPT-OSS config (200 sessions; 5,550 turns) was read in memory. The observed nested fields were `isl`, `isl_new`, `osl`, `isl_text`, `isl_new_text`, and `osl_text`; the sample exposed no tool/request/timestamp/duration/latency-named members. Only field names, counts, and token summaries were retained. No IDs, prompt/completion text, or Parquet files were persisted.

The sample is not an exhaustive scan of all 1,231 sessions, and the two GPT-OSS configs total over 7 GB on the public card. Claims about the whole dataset are limited to the Hub-declared schema and sampled rows. The observed sample provides no tool completion or request timing fields, and the pinned source timing audit independently shows `tool_exec_ms` is inter-arrival time while scraper E2E can be a scrape-window average. This remains insufficient for Stage C alignment/slack calculations.

No raw prompt text or Parquet data was downloaded or persisted. Opus is listed for schema context only and is excluded from timing analysis.
