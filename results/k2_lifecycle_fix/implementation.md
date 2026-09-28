# K2 request lifecycle repair

The failed 32-boundary run re-admitted E03:turn:1 after 17 output tokens and E06:turn:1 after 6 output tokens. The scheduler unconditionally reacquired first-token gate ownership on waiting-to-running transitions. Release was conditional on the first output, which could not happen again. Normal request completion did not clear this reacquired ownership. Two stale owners exhausted K=2 permanently.

The repair changes four sites in the existing scheduler:

1. Waiting admission applies the research first-token gate only while `request.num_output_tokens == 0`.
2. Gate ownership acquisition uses the same condition. Post-first preemption and remote-KV restoration cannot reacquire ownership.
3. A pre-first request that already owns a slot may re-enter without acquiring another slot, even if the gate is full. New request IDs are still bounded by K.
4. `_free_request` discards gate ownership and first-token bookkeeping on terminal cleanup, before potentially delayed KV freeing. This covers normal completion and the existing abort path.

The canonical request output count is O(1). The new membership lookup occurs only when the capacity check finds the gate full. Terminal cleanup uses two O(1) set discards. No scan, synchronization, metric scrape, new trace, or sleep is added. No PSA, lease, reservation, connector, KV capacity, workload, deadline, or service-budget parameter is changed.

## Verification method

`tests/test_k2_lifecycle_fix.py` extracts and executes the actual production gate branches and terminal-cleanup method using minimal request/backend stubs. It does not duplicate the gate implementation. The pre-fix scheduler snapshot is tested first, then the fixed scheduler. This is a focused state-transition test, not a substitute for full EngineCore execution.

The GPU validation reuses the frozen Phase 4G-A-FIX lifecycle runner with output paths redirected to this repair's directory and its source fingerprint pinned to the new manifest. The fixed order is FullSlackKV, OnDemand, FullSlackKV, FullSlackKV, each with a fresh server, 40 turns, 32 causal boundaries, and the same four-chunk 8,192-token background. These runs validate liveness and gate ownership; they do not form a performance comparison and are not appended to any historical matrix.

The gate audit checks every retained runtime event for owners that already produced a first token, post-first readmissions that reacquire ownership, and terminal requests that retain ownership. The FIRST_TOKEN event is interpreted at its actual instrumentation point, immediately before ownership removal.

`gate_helper_cost.json` contains an alternating-order CPU microbenchmark of the old/new capacity helper (10 samples of 200,000 calls per case). It excludes the surrounding scheduler, additional request-output predicate, GPU compute, and terminal cleanup. Its results must not be reported as an end-to-end performance regression bound.

## Reproduction

The baseline scheduler must be captured before applying `vllm_k2_lifecycle_delta.patch`; its exact SHA256 and the baseline/full fixed vLLM diff fingerprints are in `manifest.json`. The runner defaults to `/tmp/slackkv-k2-lifecycle-baseline.py` for that snapshot. On a fixed checkout, reconstruct the baseline in an isolated copy by reversing the committed delta, not by changing a live serving tree.

Historical Phase 4G-A-FIX still pins the old source fingerprint. Its fingerprint test should reject this newly repaired runtime; historical artifacts and pins are preserved. Full historical performance aggregation remains invalid.
