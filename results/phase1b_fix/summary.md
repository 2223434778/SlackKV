# Phase 1B-Fix — closed-loop and target-model audit

## Objective

Re-evaluate the Phase 1B QueueJIT result after correcting its open-loop
session replay, restricting traces to a target-model context limit, and
auditing token and continuation semantics.  QueueJIT itself is unchanged.

## Setup

The TraceLab v0.0.2 database and Phase 0 CPU-to-GPU0 P50 transfer knots are
the same as Phase 1B (calibration commit
`0337837b43ddf78821839d86be6a89ae2a7c164a`).  Promotion latency uses all
measured knots, interpolation, and measured-size chunking above 8 GiB.

The primary configuration is Qwen-like BF16 GQA geometry, 56 KiB/token, with
`max_context_tokens=131072`.  The stress sensitivity is 128 KiB/token on the
**same** retained events and cap.  Neither primary nor sensitivity runs use
uncapped contexts.

For token accounting, reusable-context estimate is
`input_tokens_total + max(0, output_tokens - reasoning_output_tokens)`.
This follows TraceLab's documented Codex visible-output approximation and is
an audit-based estimate, not a claim about provider-internal KV allocation.

Session replay is closed loop: after a promotion completes after its deadline,
that session's following release times shift by the observed stall.  Each
policy therefore has the same sessions, order, offsets, and tool durations,
but its own causally propagated arrival times.  M=4/8/16/32/64 use ten seeded
session selections and U[0,60] second offsets.  Tuple-preserving Poisson
replay remains a separate open offered-load sensitivity (5,000 jobs,
rho=0.25/0.50/0.75/0.90, ten seeds).

## Trace audit

The continuation audit inspected 30 distinct Codex `exec_command` roots.  The
database contains 36,994 continuation roots and 77,564 `write_stdin` calls;
41,818 calls are more than one round after their root.  Continuation polls are
emitted by later rounds, so merging an entire chain would suppress intervening
LLM turns.  The Phase 1B per-round pause rule remains valid: final tool emit
to final tool result for that round, with no chain-latency sum.

`timing_events` does not expose the next GPU inference start.  Consequently
`result_at` remains a conservative earliest-result-availability deadline; this
fix does not invent an unmeasured post-result scheduling allowance.

## Context and token coverage

Of 563,199 Phase 1B-valid opportunities, 283,193 satisfy the corrected
reusable-context cap (50.283%).  For this subset, reusable-context P50/P90/
P95/P99 is 73,914 / 118,845 / 124,966 / 129,822 tokens.  At 56 KiB/token,
derived KV P50/P90/P95/P99 is 4.239 / 6.815 / 7.166 / 7.445 decimal GB; at
128 KiB/token it is 9.688 / 15.577 / 16.380 / 17.016 GB.

Pause P50/P90/P95/P99 is 203 ms / 10.098 s / 30.007 s / 150.845 s.  For the
56-KiB primary mapping, P50/P90/P95/P99 round-trip promotion service is
324.5 / 519.8 / 546.3 / 567.4 ms.  This makes 52.28% of retained pauses
individually transfer-infeasible (`pause / promotion_service < 1`); 34.38%
have ratio >=3.  At 128 KiB/token, 61.53% are below one and 25.59% are >=3.

## Results

Closed-loop primary 56-KiB/token results below are medians across ten seeds:

| M | Naive miss | QueueJIT miss | relative miss reduction | Naive P95 | QueueJIT P95 | relative P95 reduction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 55.57% | 55.07% | 0.90% | 225.6 ms | 220.3 ms | 2.38% |
| 32 | 56.90% | 54.75% | 3.77% | 234.7 ms | 229.9 ms | 2.03% |
| 64 | 60.44% | 56.94% | 5.79% | 281.5 ms | 258.1 ms | 8.31% |

At the 128-KiB stress mapping, M=32 changes from 69.02% to 66.57% miss rate
and 733.7 to 688.0 ms P95 stall.  At M=64 it changes from 78.00% to 75.54%
miss rate, while P95 is essentially unchanged (2,804.7 to 2,817.9 ms).

The open-loop audit confirms the original replay artifact at high stress.  For
128 KiB/token M=64 QueueJIT, open-loop P95 is 44,576.1 ms whereas closed-loop
P95 is 2,817.9 ms; miss rate falls from 77.55% to 75.54%.  Thus the old
uncapped/open-loop high-tail values are not suitable as the main research
conclusion.  The corresponding primary 56-KiB M=64 QueueJIT P95 changes from
276.5 to 258.1 ms, showing that the artifact is most visible in the
large-footprint, high-concurrency stress regime.

For the 56-KiB tuple-Poisson sensitivity, QueueJIT versus NaiveJIT miss rate
is 52.83% versus 56.97% at rho=0.25 and 65.10% versus 71.64% at rho=0.50;
P95 is 258.9 versus 266.5 ms and 414.5 versus 442.6 ms, respectively.

## Interpretation and limitations

The Phase 1B workload replay had a material open-loop defect.  Closed-loop
propagation fixes that defect and materially reduces the 128-KiB/M=64 tail.
The target-model cap also removes contexts that cannot map to the selected
model.  Nevertheless, in the capped 56-KiB primary session replay QueueJIT's
gain over NaiveJIT remains below 20% for all M>=16 points.  This is a negative
result for QueueJIT alone, not a conclusion about selective offload or a real
system.

This remains an oracle-deadline simulator with one serialized inbound resource,
no real KV tensors, no real inference, no learned prediction, and no measured
post-tool-result request queueing slack.

Phase 1B-Fix execution status: COMPLETE
