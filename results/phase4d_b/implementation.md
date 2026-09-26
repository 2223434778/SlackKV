# Implementation notes — Phase 4D-B

- `Scheduler._slackkv_capacity_required_blocks`: validates research-only prefetch metadata against actual prompt token count and runtime block size.
- `Scheduler._slackkv_capacity_trace`: samples `block_pool.get_num_free_blocks()` once at admission and emits `SLACKKV_CAPACITY_TRACE` with total/free/used blocks, de-duplicated running/waiting/lease ownership diagnostics, request counts, and decision.
- `Scheduler._slackkv_preflight_capacity_guards`: runs from the start of `Scheduler.schedule()` before connector-first preflight/load; impossible guarded requests are fast-finished without allocation or connector work.
- The waiting traversal contains the same exact-headroom guard immediately before normal allocation, protecting the allocation point if a guarded request reaches that path. Requests already waiting for remote KV or already in prefill are not rechecked.
- `Scheduler._slackkv_finish_capacity_guard_request`: routes `capacity_rejected` through `_slackkv_prefetch_finished`, frees request state, and reports zero external tokens/bytes, model forwards, and generated tokens.
- `SlackKVPrefetchRequest` and `/slackkv/prefetch`: add optional `capacity_guard` / `required_blocks`, validate 16-token alignment and the block count, and accept `capacity_rejected` only for guarded calls. Unguarded historical requests retain the previous status behavior.
- A small connector-only output filter avoids marking request IDs already removed by fast rejection as active connector-only work; the connector transfer algorithm itself is unchanged.
- `experiments/phase4d_b/capacity_client.py` is separate from the historical Phase 3 native client. `integration.py` harvests completed futures before each reservation decision, on wakeups, and at final drain; capacity rejection releases only the logical reservation and falls back to OnDemand.

The exact Phase 4D-B source delta relative to the captured pre-change SlackKV-vLLM tree is in `results/phase4d_b/vllm_phase4d_b_delta.patch`. Pinned vLLM baseline commit: `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`.
