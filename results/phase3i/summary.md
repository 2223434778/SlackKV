# Phase 3I summary

Phase 3I adds one ordinary 128-token background request X. X starts before t0; t0 is set after X first token. A/B/C remain the only gated sessions. No vLLM scheduler changes were made; Phase 3H K-capacity gating was reused unchanged.

X-only baseline median decode duration: 1824.40 ms; baseline median TPOT: 14.37 ms.

## Results

|Policy|ABC P95 TTFT|ABC completion P95|ABC decode tok/s|Max pre-first|Gate wait|X slowdown|Memory-time|Promotion misses|Blackout conflicts|Long prefetch|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|ResumeAware-K3|96.16|564.20|186.63|2|0.00|1.052|14.27|1|2|1|
|FirstTokenGate-K1|104.29|579.11|172.88|1|77.16|1.062|22.51|1|2|1|
|FirstTokenGate-K2|90.82|551.59|183.45|2|0.00|1.054|13.13|1|2|1|

## Direct answers

- X validation: PASS; X generated 128 tokens in every run.
- Promotion correctness: FAIL for this busy-workload experiment because the background decode overlaps native promotion. At least one promotion miss, blackout conflict, or >50 ms promotion occurs in every policy group; this is not hidden.
- K1 and K2 gate invariants are measured from real traces. K1 remains max pre-first 1. K2 reaches capacity 2 in some runs but not every run under X interference, so the Phase 3I admission comparison is contaminated.
- The X request remains active past C first token in the recorded runs, satisfying the busy-background condition.
- Historical light-load ordering from Phase 3H was K2 (115.23 ms) < K3 (131.69 ms) < K1 (159.98 ms) for ABC P95 TTFT. Under this busy run, the median ordering was K2 (90.82 ms) < K3 (96.16 ms) < K1 (104.29 ms), but this cannot be promoted to a clean admission result because promotion invariants failed.
- The busy-load data do not establish a ranking reversal or a justified dynamic-K decision: X changes promotion service and therefore contaminates the comparison. The next direction remains KEEP_K2_AND_FIX_PROMOTION before any dynamic-K study.
- Result D: the frozen requirement that A/B/C promotions remain clean is not met when the ordinary 128-token request is active. This is an implementation/measurement validity outcome for Phase 3I, not evidence that K1 or K2 wins.

## Scope and limitations

Cross-policy sequence hashes remain diagnostic only. The matched-schedule KV correctness result from Phase 3G-CorrectnessDiag is not replaced. This phase does not implement dynamic K or start Phase 3J.
