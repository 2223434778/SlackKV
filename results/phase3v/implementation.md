# Phase 3V implementation notes

## Scope and preserved runtime

This is the Phase 3V comparison of `FixedCount2-Async40-K2` against `DynamicBlock512-Async40-K2` on the heterogeneous 12-session A–L workload, alongside OnDemand and NaiveAsync controls. The Phase 3S vLLM physical residency-lease primitive is reused unchanged; no vLLM source was modified. Phase 3W and prediction uncertainty were not run.

The dynamic admission decision uses only the in-memory reservation ledger, the QueueJIT/EDF ready order, the candidate's predicted `ceil(prefix_tokens / 16)` block demand, and the frozen 512-block budget. It does not query Prometheus, `nvidia-smi`, or live free GPU memory. The count-based control remains limited to fewer than two active reservations and separately records block demand. Each admitted lease is keyed by session and uses the existing Phase 3S acquire / hold-after-prefetch / real-resume handoff / release lifecycle. Controller reservations expire conservatively at the frozen resume-deadline plus handoff guard; this is not synchronized to physical lease release.

## Run stages

`run_phase3v.py` enforces the order of the experiment:

1. Three fresh X-only controls establish the median X decode duration and TPOT.
2. One 4096-token equal-size run each for FixedCount2 and DynamicBlock checks regression behavior.
3. One heterogeneous integration run for each of the four policies must pass before any measured run is allowed.
4. Three fresh measured runs per policy are stored separately from integration data.
5. Finalization derives CSV summaries and validation counts from raw per-run JSON records.

All session prompts use deterministic, distinct synthetic token IDs with exact 1024- or 4096-token lengths. They preserve the frozen request shape but do not represent semantic conversations. Background X is the fixed 9500-token pressure prompt with 256 generated tokens. `t0` is the real server-side X `FIRST_TOKEN` event. A–L deadlines and all scheduler/promotion timestamps use that common monotonic clock.

During setup, the offload metric is asynchronous. A measured attempt exposed that an immediate scrape could attribute one completed GPU→CPU increment to the next session's snapshot. The runner now polls the same cumulative metric against a fixed per-request baseline until exactly that request's expected byte increment is visible (10-second bounded timeout), then proceeds. This only stabilizes setup evidence and does not change any request or the post-t0 schedule. The incomplete attempt was not recorded as a run; the affected repetition was re-run after this instrumentation fix.

## Metrics and evidence

- Native prefetch measurements (`external_tokens_loaded`, bytes, transfer time, CPU→GPU time, request timings, connector-only steps, forward/token counters, and status) are persisted from the actual endpoint response.
- Source accounting is calculated from before/after source-counter snapshots and checked against the workload token total; it is never forced to expected values.
- H2D totals combine observed native transfer bytes with external-restore token counters converted using the measured 4096-token KV size. Duplicate traffic is the positive excess above one required transfer per session.
- Lease block counts, acquires, releases, and leaks come from the real vLLM lease trace. Predicted block demand is compared with acquired physical blocks for every admitted lease.
- X slowdown is each policy run's measured X decode duration divided by the median of the three fresh X-only controls.
- Integration runs are validation only and are excluded from policy medians.

## Arithmetic note

The frozen test text described `448 + 64 > 512`, but the arithmetic is equality. The explicit admission predicate is `reserved_blocks + required_blocks <= 512`; the implementation honors that rule. In this workload each cluster contains three 1024-token sessions (64 blocks each) and one 4096-token session (256 blocks), totaling 448 blocks. The observed DynamicBlock peak is therefore 448, not an artificially constrained value.

## Interpretation boundary

Phase 3V tests whether block-based logical reservation admits more useful heterogeneous promotions than a two-session count limit while preserving physical lease correctness. It does not establish a general online benefit predictor and makes no claim about prediction error. The observed NaiveAsync runs did not reproduce duplicate H2D or remote-wait failure: all 12 promotions per run remained useful in this 588-MiB workload. That negative observation is retained as measured, not treated as invalidity.
