# Phase 3B implementation

Pinned vLLM source commit: `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`.

## Connector-first scheduling

The scheduler detects a `slackkv_prefetch_only` request before ordinary waiting-queue token scheduling. While one is active, it schedules connector metadata only and emits a `SchedulerOutput` with zero normal model tokens. The worker and `kv_connector_no_forward()` poll the existing OffloadingConnector load lifecycle until completion. Normal requests are deferred to the next scheduler step; an already-running CUDA forward is not preempted.

An in-flight load remains connector-first on subsequent engine iterations. Completion is transferred through the existing scheduler bookkeeping, loaded blocks remain prefix-cache blocks, and the special request is then finished without sampling. A one-shot emitted-ID set prevents completed-prefetch output from being delivered twice.

The scheduler and model runner enforce the research invariant that connector-only outputs contain zero scheduled model tokens. `mixed_prefetch_model_step_count` is reported from the completed request metadata and is zero in the measured runs.

## Before / after

```text
Phase 3A: load -> unrelated normal forward -> connector completion observed
Phase 3B: load -> connector-only poll(s) -> completion observed -> normal forward
```

The native endpoint remains a local research control surface. It uses the normal prefix identity, block allocation, OffloadingConnector lookup, and OffloadingWorker transfer path; it is not a separate memcpy engine.

## Service-time calibration

The frozen scheduling service time is `ceil(max(three warm client-observed HTTP E2E times))`, not the server-internal elapsed field. Server E2E, client E2E, and CPU-to-GPU transfer time are retained separately.

## Scope and limitation

The overlap experiment submits B's native prefetch at 300 ms and A's normal resume at 302 ms. Connector-first priority begins only at a scheduler step boundary; it does not preempt a forward already executing on the GPU. The six-session replay uses oracle deadlines, a serialized client promotion worker, and the normal request-based native endpoint; it does not claim a production promotion-only API.

## Modified vLLM files

The exact patch relative to the pinned source commit is stored in `vllm_patch.diff`. The Phase 3B changes are confined to scheduler/output invariants, worker no-forward assertions, and the prefetch completion handoff; the existing Phase 3A endpoint and connector path are retained.
