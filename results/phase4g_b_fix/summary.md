# Phase 4G-B-FIX — guarded-native allocation liveness and fresh mechanism matrix

**Result:** `RESULT_D_PSA_HEADROOM_BLOCKER`. This is a mechanism RCA, not a performance comparison.

## Objective and repair

The preceding Phase 4G-B matrix remains `INCOMPLETE_INVALID_RUN`; its data were not reused.
The failed pre-fix run repeatedly admitted the same guarded prefetch to the connector-first pass,
but actual KV allocation returned None. Connector-first suppressed ordinary running prefill
steps, so their in-flight reservation could not shrink.

A fresh pre-fix diagnostic recorded free=1564 blocks, in-flight reserved=759,
schedulable=805, local=21264 tokens,
external=16 tokens, and async_load=True.
The repair applies only to guarded native-prefetch requests: on actual `allocate_slots(None)`,
return a structured `capacity_rejected` with `INSUFFICIENT_SCHEDULABLE_BLOCKS`, no connector load,
no lease, and normal OnDemand resume fallback. Ordinary requests and unguarded historical prefetches are unchanged.

## Frozen setup

- SlackKV start commit: `0a32b037641449462699cbe1dc7edfb199369a4d`; workload SHA256: `9b381336a70e73b0efbddbb79d2c88608c19663247e819c1e7a364efd2971105`.
- Qwen2.5-1.5B-Instruct, vLLM 0.23.0, RTX A6000 GPU0, FP16, 2 GiB GPU KV / 8 GiB CPU KV, 4681 blocks of 16 tokens.
- K2, P_SAFE=102 ms, resume guard=110 ms, logical budget=3510 blocks; all 40 turns and 32 dynamic boundaries per run.
- Fresh instrumentation smoke: valid=True, probe CPU duty=0.0402 (<0.05).

## Compared conditions

| Condition | Policy | BGv2 | Valid runs | Source initially not ready | Ready by safe start | Actionable windows ≥20 ms | Capacity rejects | PSA rejects | Reservation skips |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 | A | False | 2/2 | 44 | 64 | 0 | 0 | 0 | 0 |
| M1 | D | False | 2/2 | 44 | 64 | 0 | 16 | 4 | 0 |
| M2 | A | True | 4/4 | 88 | 128 | 0 | 0 | 0 | 0 |
| M3 | D | True | 4/4 | 88 | 128 | 0 | 32 | 6 | 0 |

## Interpretation and limitations

The frozen A–E classification is computed from the new complete matrix only. Readiness, physical fit,
PSA-safe headroom, JIT timing, and logical reservation remain separate observed constraints.
A `capacity_rejected` from actual allocation pressure is a safe promotion fallback, not a failed real resume.
Counterfactual source/headroom samples do not demonstrate a successful promotion or performance gain.
There is no request-timestamp-faithful trace here: token structure is from coding-agent data and timing is TraceToolProxy.
No latency winner, SOTA, or statistical-significance claim is made.

## Next-step status

This repair and frozen mechanism matrix are complete; no later phase was started.







## Independent completion and classification audit

All 12/12 fresh measured runs completed: 480/480 turns and 384/384 dynamic boundaries. K2 stayed ≤2, with no final lease or logical leaks, server crash, or OOM.

BGv2 passed 8/8 measured runs: each produced 8192 tokens; minimum tail past the last Agent first token was 94079.94 ms, and maximum inter-chunk gap was 34.57 ms (<50 ms). The inherited `run_metrics.csv` tail columns are `NA`; exact values retained in each `run.json` are extracted into `background_audit.csv`.

The repaired allocation branch was exercised 1 time(s). The guarded request returned `capacity_rejected`, zero transfer/model work, and no lease; its real-resume run completed. Exact evidence is in `allocation_reject_events.csv`.

Across the D runs, promotion endpoint responses were {'progress_rejected': 10, 'capacity_rejected': 48, 'ok': 2}; this is mechanism accounting, not a latency comparison.

Frozen Result D criterion: in M2, 128 source-in-time boundaries included 128 with at least one sampled physical fit; 128/128 had no sampled ≥20-ms full-prefix PSA-safe window (>50%). This supports PSA-safe headroom as the classified blocker, not a performance winner.

Sampling limitation: per-boundary probe gaps reached 171.64 ms. A short unobserved window cannot be excluded; the classification is conditional on the frozen sampled-window method.
