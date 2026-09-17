# Phase 3A native-promotion implementation

## Runtime and patch scope

The research patch is against pinned vLLM commit
`0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` (vLLM 0.23.0). The exact
patch is preserved in `vllm_patch.diff`: 11 files, 332 additions, and 20
deletions. It does not change model math, attention kernels, KV layout,
sampling, weights, or `OffloadingWorker.transfer_async`.

Modified files:

- `vllm/distributed/kv_transfer/kv_connector/v1/offloading/common.py`
- `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`
- `vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py`
- `vllm/entrypoints/serve/__init__.py`
- `vllm/entrypoints/serve/dev/slackkv_prefetch/__init__.py`
- `vllm/entrypoints/serve/dev/slackkv_prefetch/api_router.py`
- `vllm/v1/core/sched/output.py`
- `vllm/v1/core/sched/scheduler.py`
- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/v1/worker/kv_connector_model_runner_mixin.py`
- `tests/v1/kv_connector/unit/test_slackkv_native_prefetch.py`

## Request and cache lifecycle

`POST /slackkv/prefetch` accepts only `prompt_token_ids`. The local
research route submits a normal engine request with
`slackkv_prefetch_only=true` and `max_offload_tokens=0`. Normal prefix hashes
and `OffloadingConnector` lookup are therefore used; no independent copy
engine or fabricated cache identity exists.

When the scheduler finds an external hit, its normal GPU block allocator
reserves the destination blocks. `OffloadingConnectorScheduler` builds the
ordinary CPU-load/GPU-store `TransferJob`, annotated only with the research
flag. The worker then executes the unchanged `start_kv_transfers()` and
`OffloadingWorker.transfer_async()` path.

For a load-only scheduling step, `KVConnectorModelRunnerMixin` enters
`kv_connector_no_forward()`. On completion, the main scheduler marks the
loaded blocks computed in the normal prefix cache, finishes the special
request without sampling, and releases request ownership. Freeing request
ownership does not invalidate computed prefix-cache blocks, so the completed
prefetch remains reusable by a later ordinary request.

The special request ID is explicitly forbidden from appearing in
`num_scheduled_tokens`. A connector-only load may coexist with model tokens
belonging to an unrelated normal request; that mixed step is logged with a
special-request forward count of zero rather than being rejected.

```text
POST /slackkv/prefetch
        |
        v
normal token hashes + external prefix lookup
        |
        v
normal GPU block allocation + connector load metadata
        |
        v
kv_connector_no_forward / special request has zero scheduled model tokens
        |
        v
OffloadingWorker.transfer_async (CPU -> GPU)
        |
        v
wait for connector completion -> mark blocks computed in prefix cache
        |
        v
finish special request, generated tokens = 0, release request ownership
        |
        v
later real request observes a GPU-local prefix hit
```

## Completion and instrumentation

The offloading worker reports measured transfer time and bytes through its
existing completion metadata. The scheduler attaches those observations and
the external-token count to the research-only completion output. The endpoint
returns status, external tokens, transfer bytes, connector transfer time,
endpoint elapsed time, `model_forward_count=0`, and
`generated_token_count=0`; it returns no generated text.

Zero forward is measured at the model-runner boundary. The connector-only
branch logs `SLACKKV_NATIVE_PREFETCH_NO_FORWARD`; mixed scheduling verifies
that special request IDs are absent from the model-token map and logs
`SLACKKV_NATIVE_PREFETCH_SPECIAL_NO_FORWARD`. Completion is separately logged
as `SLACKKV_NATIVE_PREFETCH_COMPLETE`. All instrumentation uses the
`SLACKKV_` namespace and does not redefine official metrics.

## Concurrency observation

The first implementation incorrectly rejected a valid mixed step containing
a special connector load and an unrelated real resume's model tokens. The
guard was corrected before the final matrix: it now rejects only model tokens
scheduled for the special request itself. The complete 12-run rerun had zero
engine failures, 54 successful native prefetches, zero special-request model
forwards, zero generated prefetch tokens, and valid cache/generation evidence.

Although the raw CPU-to-GPU counter remained about 5 ms, a native endpoint
that overlapped a real resume took up to about 93 ms to complete. This is
scheduler/serving interference, not extra transfer time, and explains the
six deadline misses in each JIT policy under the frozen 14-ms estimate.
