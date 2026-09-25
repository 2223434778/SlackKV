# Timing semantics audit

Pinned source: `netpreme/coding_agents@83d8d1280b406b0484693017900763a0b0ad8d19` (2026-07-08; the commit changes README only). The audit inspected the pinned proxy `app.py`, `recorder.py`, and `serving/metric_scraper.py` blobs.

Source evidence links: [app.py lines 95–107](https://github.com/netpreme/coding_agents/blob/83d8d1280b406b0484693017900763a0b0ad8d19/coding_agents/trace_collection/pipeline/proxy/app.py#L95-L107), [recorder.py lines 29–39](https://github.com/netpreme/coding_agents/blob/83d8d1280b406b0484693017900763a0b0ad8d19/coding_agents/trace_collection/pipeline/proxy/recorder.py#L29-L39), and [metric_scraper.py lines 237–271](https://github.com/netpreme/coding_agents/blob/83d8d1280b406b0484693017900763a0b0ad8d19/coding_agents/serving/metric_scraper.py#L237-L271).

## `request_time`

`app.py` sets `request_received_time = time.time()` at proxy request entry, before request sanitization/title filtering, and passes it to `MessageTrace`. This is a proxy-receive timestamp, not the vLLM engine arrival timestamp. For retained task requests it is captured before forwarding to vLLM.

## `e2e_ms`

`metric_scraper.py` timestamps each Prometheus scrape before the HTTP round trip, snapshots cumulative vLLM counters, then builds delta rows between adjacent snapshots. Its row builder documents per-request average measurements; when `n > 1`, the E2E and token/time fields are divided by `n`. Thus `e2e_ms` from this path is not an exact request-level latency for a multi-completion polling window. The field cannot be paired exactly with one proxy `request_time` solely by row order/count.

## `tool_exec_ms`

At proxy entry, `app.py` calls `recorder.note_response_time()` immediately before forwarding the new request. `recorder.py` computes `now - self._last_response_time`, then sets `_last_response_time = now`. Despite the variable name, this saved time is the previous call to `note_response_time`, also made at a prior request arrival; it is not set on inference response completion. Therefore the implementation's `tool_exec_ms` measures approximately current request arrival minus previous request arrival. It includes prior serving time (and any harness gap), so it is not response-end-to-next-arrival slack. This is a code-level semantic mismatch with the README description.

## Live HF row-schema recheck (2026-09-25)

Using the A6000 login shell (which supplies the configured HTTPS proxy), the Hub API confirmed revision `d56ae0d0bc80f3663a7a61a38c4ee60c26ce7b67`. The datasets-server rows endpoint returned GPT-OSS `turns` as JSON objects. In a fixed sample of 100 rows per config (5,550 turns), observed keys were only `isl`, `isl_new`, `osl`, `isl_text`, `isl_new_text`, and `osl_text`; no tool/request/timestamp/duration/latency-named members appeared. Text values and row IDs were inspected only in memory and not persisted. This is not an exhaustive scan of the JSON-valued column. The two GPT-OSS datasets total over 7 GB, so no full shard download was made.

## Alignment and limitations

Even if a release row contains both fields, exact alignment fails for `n > 1` polling windows unless a separate request-identity join is present. No such join is described in the source path inspected. The sampled HF rows also contain no tool/request timing fields. Consequently this phase does not calculate reconstructed slack, negative-slack fractions, or replay-candidate counts. Confidence is high for the pinned code's `tool_exec_ms` and scraper aggregation semantics; coverage of the published nested HF row schema is sample-based, not exhaustive.
