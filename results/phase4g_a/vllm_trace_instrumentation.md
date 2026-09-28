# Phase 4G-A trace-only vLLM additions

Pinned upstream vLLM: `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`.
The frozen working-tree diff SHA256 for every Stage L and measured server was
`f0f8e327cc7ff4eb1797852a24e56e0e7f4a77b62e0f9439e88405fd5b470ae4`.
These are additive observations on top of the Phase 4F-A/4F-B patched source;
no scheduling, connector transfer, lease, PSA-1, K2, or eviction decision was
changed.

1. `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`
   reads only requests carrying `slackkv_trace_next_boundary` and
   `slackkv_trace_source_required_tokens`. At request finish and after an
   already-pending store completes, it checks each required 16-token CPU
   offload key with the existing `OffloadingManager.lookup` API. It logs
   `SLACKKV_OFFLOAD_TRACE` with `SOURCE_READY` or `SOURCE_NOT_READY`, the
   exact ready-prefix block count, and a monotonic timestamp. It does not
   create a store/load job or change connector policy. The engine's existing
   `offload_prompt_only=True` behavior remains in force.
2. `vllm/v1/core/sched/scheduler.py` emits `SLACKKV_SOURCE_TRACE` immediately
   after an existing successful `allocate_slots` on a traced real resume,
   reporting `num_new_local_computed_tokens` and
   `num_external_computed_tokens`. Re-admission after preemption may emit
   multiple events for the same request ID; the analysis retains the first
   event for initial source attribution and includes later external restores
   in H2D accounting.

The repository intentionally does not vendor the patched vLLM tree. The
source-path/fingerprint evidence is retained here and in `source_manifest.json`.
