# Phase 4G-A-FIX: sustained fixed-work background and 32-boundary rerun

Final result: **RESULT_C_MULTITURN_NO_MATERIAL_LATENCY_VALUE**. BG-CAL 4/4, BG-only 8/8, measured 16/16; partial matrices are never aggregated.

## Objective and protocol

This rerun fixes the Phase 4G-A background duration and its dependent metrics. It retains the same eight-session, 40-turn, 32-boundary frozen hybrid workload, controller, deadlines, model/runtime, KV capacities, K2, PSA-1, P_SAFE, and reservation budget.
BGv2 uses one active X request at a time: four sequential requests, each with the same 9500-token prompt and 2048 generated tokens. The per-request KV footprint therefore stays close to the old single-X request while the fixed work lasts longer. The background covers through the last Agent REQUEST_FINISHED; last-first-token tail is diagnostic only.
The old single X=2048 run ended before some Agent workflows. A single X=8192 would increase one request's KV footprint from about 11548 to 17692 tokens and confound duration with KV pressure, so the protocol uses four sequential 2048-token chunks.

## BG-CAL and BG-only controls

BG-CAL tail after the last Agent REQUEST_FINISHED (ms): 91755.30, 91081.52, 93420.24, 93453.95; minimum 91081.52 ms versus the frozen 30000-ms gate.
Maximum BG-CAL inter-chunk idle gap: 33.12 ms (limit 50 ms).
BG-only valid runs: pre 4/4 and post 4/4. Eight-run median BGWall 118407.77 ms; weighted decode TPOT median 14.432 ms. First-four / last-four wall medians: 118407.77 / 118658.77 ms; drift +0.21%.
Measured repeated-prompt local-cache hit medians by chunk:
OnDemand-K2: X0 median 0 local tokens, X1 median 0 local tokens, X2 median 0 local tokens, X3 median 0 local tokens
FullSlackKV+PSA1-K2: X0 median 0 local tokens, X1 median 0 local tokens, X2 median 0 local tokens, X3 median 0 local tokens

## Measured policy results (median of eight run-level values)

| Policy | Boundary P50 | P90 | P95 | TTFT P95 | Cum TTFT/session | Session E2E | BG slowdown | BG TPOT | H2D MiB | Duplicate MiB | Combined tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OnDemand-K2 | 3873.33 | 5699.51 | 6434.53 | 6408.80 | 15834.23 | 33474.42 | 1.103 | 15.916 | 19276.25 | 0.00 | 89.99 |
| FullSlackKV+PSA1-K2 | 3762.88 | 5758.50 | 6413.48 | 6354.70 | 15544.68 | 34075.78 | 1.104 | 15.935 | 19276.25 | 0.00 | 89.88 |

Paired FullSlackKV minus OnDemand boundary P95: +1.38% (range -19.31% to +10.31%). Cumulative TTFT delta -1.71%; session E2E delta +1.29%. H1 FAIL; H2 PASS; H3 PASS.
Bootstrap intervals are descriptive intervals for paired run-level medians, not significance tests. FullSlackKV paired P95 and X slowdown intervals are in bootstrap_intervals.csv.

## Lifecycle and tail decomposition

Measured runs: 40/40 Agent requests and 32/32 boundaries per run; aggregate 640/640 requests and 512/512 boundaries. BG coverage through last request finish: 16/16; maximum measured idle gap 39.38 ms.
For FullSlackKV's run-level P95 boundary, median scheduler wait was 6233.46 ms and post-admission-to-first-token time was 62.77 ms; the larger observed component was scheduler waiting. This is a descriptive decomposition, not a causal attribution.
Output identity matched the Phase 4C reference for A 224/256 and D 221/256 boundary requests; identity is diagnostic, while token length is the hard gate.

## Validation and interpretation

Algorithm fingerprint unchanged: True. K2 valid 16/16; pause fidelity 16/16; crashes 0; OOM 0; logical leaks 0; lease leaks 0; capacity invariant failures 0.
The post-run ownership audit found four real post-first-token re-admissions across the 16 traces. None reacquired gate ownership; there were zero stale/finished gate owners and zero lifecycle violations. See `gate_lifecycle_audit.json`.
This is a frozen hybrid trace-derived replay using real coding-agent token/KV structure and empirical TraceToolProxy pauses. It is not a request-timestamp-faithful replay or a production Agent benchmark. No earlier Phase 4G-A measured samples were reused.
