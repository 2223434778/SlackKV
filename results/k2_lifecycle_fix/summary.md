# K2 lifecycle repair — PASS

## Objective and root cause

Repair the stale first-token admission owners observed in Phase 4G-A-FIX without changing the workload or research policies. Start commit: `992561a661effdb96cd4c5d8e5357002885ae26f`.

The failed run re-admitted E03:turn:1 with 17 output tokens and E06:turn:1 with 6. Waiting-to-running transitions reacquired K2 ownership despite an already-produced first token. The first-output release branch could not run a second time, and normal completion did not discard these owners. This exhausted both gate slots. The request queue can temporarily reside in the step-skipped waiting list; a zero transient waiting counter did not prove absence of waiting work. Rep8 ran D before A; the historical report's position label is not used as causal evidence.

## Minimal implementation

Gate checking and acquisition now require zero output tokens. An existing pre-first owner can re-enter at full capacity without acquiring another slot. Terminal cleanup discards gate/first-token state even when KV freeing is delayed. Four scheduler sites change; no periodic repair scan, new tracing, synchronization, metric query, or waiting is introduced. PSA, leases, reservation, K=2, connector behavior, workload and timing remain unchanged. See `vllm_k2_lifecycle_delta.patch`.

## Setup and actual validation

Qwen2.5-1.5B-Instruct, vLLM 0.23.0, FP16/TP1/GPU0 RTX A6000; 2-GiB GPU KV, 8-GiB CPU KV, 4,681 blocks, block size16, max length32768. Frozen Phase 4C hash and both source fingerprints are in `manifest.json`. The reused Phase 4G-A-FIX harness executes 40 turns/32 causal boundaries and four 2,048-output-token background chunks, with a fresh server per run. Command: `python -u -m experiments.k2_lifecycle_fix.validate` (exact environment in `commands.sh`).

| Run | Policy | Turns / boundaries | Post-first re-admissions | Max K2 | Gate violations |
|---|---|---|---|---|---|
| 1 | FullSlackKV+PSA1 | 40 / 32 | 2 | 2 | 0 |
| 2 | OnDemand | 40 / 32 | 2 | 2 | 0 |
| 3 | FullSlackKV+PSA1 | 40 / 32 | 0 | 2 | 0 |
| 4 | FullSlackKV+PSA1 | 40 / 32 | 1 | 2 | 0 |

All four runs are valid: 160/160 turns and128/128 boundaries, zero final logical/lease leaks, zero OOM, healthy servers, valid capacity checks. All backgrounds produced8,192 tokens and outlasted the last Agent finish by90,847–94,533ms. Five real post-first re-admissions (at1,4,4,8,12 outputs) correctly held no gate slot. Retained per-run JSON contains the runtime traces, output lengths/hashes and source events.

Production-AST regressions: baseline **7 failed / 1 passed**, fixed **8/8 passed**. Existing native K2 capacity tests: **3/3 passed**. The terminal-cleanup tests cover delayed freeing and idempotence. The final driver also explicitly rejects a false inherited validity flag; every persisted run already has `valid=true`, so this reporting safeguard changes no executed serving behavior or acceptance outcome.

## Performance scope

The repair uses O(1) output-count/set operations. Alternating-order CPU helper measurements observed old/new medians of294.65/295.84ns (empty),296.51/297.17ns (one owner), and292.87/324.46ns (full gate). These numbers exclude surrounding scheduler predicates, terminal cleanup and GPU execution; they are **not an end-to-end overhead bound**. No latency superiority or throughput regression claim is made.

## Limitations and status

Four fresh runs are targeted regression coverage, not the full performance matrix or a proof over every preemption pattern. No lease lifecycle code changed; some runs acquired no leases, so zero leaks here is not a new comprehensive lease validation. Existing Python startup emits an unrelated `nvidia_cutlass_dsl` .pth warning; tests and GPU runs completed. Historical fingerprint tests intentionally reject the repaired source; old artifacts and pins are unchanged.

Repair validation completed successfully. Phase4G-A/FIX remain invalid historical matrices. No old/new performance samples were combined and no subsequent phase was started.
