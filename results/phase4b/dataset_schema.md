# Dataset schema audit

Frozen HF page revision: `d56ae0d0bc80f3663a7a61a38c4ee60c26ce7b67`.

The public viewer identifies the GPT-OSS train tables as 731 Pro rows and 500 Verified rows. The displayed top-level fields are `instance_id: string`, `dataset: string`, `model: string`, and `turns: list<struct>`. The Pro preview exposes turn members including `isl_new`, `osl`, and text fields. The public dataset card reports 23,964 Pro turns and 14,302 Verified turns, but does not expose row-level request timing in its schema summary.

The viewer truncates nested turn values because they contain long text. The HF Hub metadata API timed out from the execution host and no Parquet shard was downloaded. Accordingly, nested schema completeness is marked **unverified**; this audit does not assert that fields hidden beyond the preview are absent. In particular, no row-level `request_time`, `e2e_ms`, `tool_exec_ms`, `n`, or `ts` values were observed or validated. This is insufficient evidence for Stage C alignment/slack calculations.

No raw prompt text or Parquet data was downloaded or persisted. Opus is listed for schema context only and is excluded from timing analysis.
