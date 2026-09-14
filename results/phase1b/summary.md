# Phase 1B — Trace-driven QueueJIT validation

## Hypothesis

Queue-aware just-in-time (JIT) promotion should retain a useful portion of the
synthetic-workload benefit when real coding-agent pause lengths and context
sizes are replayed with oracle resume deadlines.  This phase tests that
hypothesis in a simulator only.

## Dataset and calibration

- TraceLab release: public sanitized `v0.0.2`, asset
  `syfi_coding_trace.duckdb`, downloaded 2026-09-14.  SHA256:
  `a7bab286bc640844560850965ccf47975cf66407154132abaab90f27ec9be744`.
- The inspected DuckDB tables were `rounds`, `tool_calls`, `timing_events`,
  and `trace_source`.  Extraction uses `rounds.round_pk`, `provider`,
  `session_id`, `round_id`, `input_tokens_total`, `output_tokens` and
  `tool_calls.tool_index`, `tool_name`, `emitted_at`, `result_at`,
  `continuation_of_tool_call_id`.
- The database contains 665,453 rounds, 743,819 tool calls, and 8,058
  sessions.  Of 574,485 tool-containing rounds, 563,199 became promotion
  opportunities.  Skips were 751 rounds with missing tool timing, 10,313
  with non-positive pauses, and 222 with non-positive context tokens.
- Calibration is Phase 0 commit
  `0337837b43ddf78821839d86be6a89ae2a7c164a`.  CPU-to-GPU0 P50 latency is
  10.330 ms (256 MiB), 40.858 ms (1 GiB), and 164.427 ms (4 GiB).  The
  transfer model loads all Phase 0 CPU-to-GPU0 P50 knots, linearly
  interpolates between knots, and represents larger payloads as repeated
  8-GiB measured chunks plus an interpolated remainder.

## Extraction and workload characterization

Each valid tool round produces one opportunity: `release_time` is the latest
`emitted_at` across its calls and oracle `deadline` is the latest `result_at`.
All calls must have both timestamps and the resulting pause must be positive.
`write_stdin` continuation calls are retained as their own TraceLab rounds;
their durations are not latency-summed with their originating `exec_command`.
The hypothetical KV size is `(input_tokens_total + output_tokens) * 131072`
bytes (32 layers, 8 KV heads, head dimension 128, FP16), not an allocation
measurement for the provider model.

Pause P50/P90/P95/P99 are 257 ms / 14.044 s / 30.164 s / 180.016 s.  Context
token P50/P90/P95/P99 are 130,448 / 318,765 / 477,059 / 828,032.  Derived
128-KiB/token KV size is 17.098 / 41.781 / 62.529 / 108.532 decimal GB
(15.924 / 38.911 / 58.235 / 101.077 GiB).  34.47% of pauses are below 100 ms,
56.40% below 500 ms, and 61.60% below one second.  Pause duration and KV size
have Pearson correlation 0.0113.

Session-preserving replay samples complete 50--100-event sessions without
truncation, uses M=4/8/16/32/64 sessions and ten deterministic seeds, and
places session origins uniformly in [0, 60] seconds.  Tuple-preserving Poisson
replay samples whole TraceLab event tuples (not independent fields), runs
5,000 jobs for rho=0.25/0.50/0.75/0.90 and ten deterministic seeds.

## Policies

- **OnDemand:** enqueues at the oracle deadline; every transfer therefore
  stalls by its service time.
- **ASAP-EDF:** starts released work immediately and selects earliest deadline
  first.
- **NaiveJIT:** releases independently at `deadline - service_time` and uses
  EDF for ties, without cumulative queue accounting.
- **SlackKV-QueueJIT:** after each arrival, completion, or idle expiry,
  computes the minimum EDF-prefix slack and idles only for that safe amount;
  otherwise it starts the EDF job immediately.

## QueueJIT versus NaiveJIT

The table gives medians across ten primary 128-KiB/token session-replay seeds.

| M | jobs (range) | Naive miss rate | QueueJIT miss rate | Naive P95 stall | QueueJIT P95 stall |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 245--332 | 71.04% | 70.86% | 892.4 ms | 892.4 ms |
| 8 | 493--633 | 64.91% | 64.13% | 923.5 ms | 923.5 ms |
| 16 | 1,056--1,201 | 67.10% | 65.25% | 1,009.3 ms | 1,009.3 ms |
| 32 | 2,124--2,310 | 71.38% | 68.89% | 1,114.1 ms | 1,100.4 ms |
| 64 | 4,289--4,731 | 80.38% | 79.02% | 25,629.3 ms | 25,629.3 ms |

The representative M=32 result is a 3.49% relative miss-rate reduction and a
1.23% relative P95-stall reduction.  The best primary session-replay run is
M=32, seed 7 (2,229 jobs): QueueJIT reduces miss rate from 65.86% to 62.81%
(4.63% relative) and P95 stall from 852.1 to 831.7 ms (2.39%).  No
configuration reaches the 20% R1 threshold.  Tuple-preserving Poisson replay
also has modest gains: at rho=0.50, median miss rate is 78.32% versus 75.45%
and P95 stall is 4,084.3 versus 4,022.5 ms (NaiveJIT versus QueueJIT).

## QueueJIT versus ASAP-EDF and memory-time tradeoff

At representative M=32, ASAP-EDF has P95 stall 1,019.9 ms and prefetch
VRAM-time 579,891.2 GB*s; QueueJIT has P95 stall 1,100.4 ms (+80.5 ms),
prefetch VRAM-time 181.7 GB*s (99.969% lower), and peak prefetch memory 26.32
GB versus 147.03 GB (82.1% lower).  QueueJIT's prefetch-to-keep ratio is
0.000286 versus ASAP-EDF's 0.980922.  Thus QueueJIT sharply postpones
promotion, but in this trace-derived workload it does not retain comparable
tail latency to ASAP-EDF.

## Concurrency, offered-load, and footprint sensitivity

Across session M=4/8/16/32/64, QueueJIT median miss rate is
70.86%/64.13%/65.25%/68.89%/79.02%.  Across tuple-Poisson
rho=0.25/0.50/0.75/0.90, it is 65.94%/75.45%/86.45%/94.00%, with P95 stalls
2,739/4,023/7,852/19,434 ms.  The corresponding QueueJIT miss rates for the
M=32 footprint sensitivity at 64/128/256 KiB per token are 60.23%/69.17%/
80.97%; P95 stalls are 415.9/966.4/7,105.6 ms.  These are hypothetical KV
footprint sensitivities, not measurements of any provider model.

## Deadline-miss diagnosis

`miss_analysis.csv` classifies QueueJIT misses with documented approximate
rules: service time longer than pause is `late_arrival`; an earlier >=4e9-byte
transfer spanning a job's latest start is `large_kv_domination`; otherwise an
overloaded released EDF prefix is `inherently_infeasible`, and the remainder
is `queue_collision`.  The dominant result is late arrival: 51,550 of 64,284
primary session-replay misses and 113,868 of 160,910 Poisson misses.  Large
KV domination is second (12,464 and 46,834 respectively); queue collision is
rare (270 and 208).  No new scheduling mechanism is introduced from this
diagnostic.

## Evaluation

- **Criterion R1: FAIL.**  There are not two >=1,000-job real-trace
  configurations with a >=20% QueueJIT reduction in miss rate or P95 stall
  versus NaiveJIT.
- **Criterion R2: FAIL.**  No >=1,000-job QueueJIT configuration jointly has
  miss rate <=10%, P95 stall <=100 ms, and prefetch-to-keep ratio <=0.10.

## Limitations

Phase 1B remains a simulator with oracle resume deadlines, maps trace tokens
onto a hypothetical target-model KV footprint, uses no real KV tensors or
inference, includes no predictor error, and abstracts the inbound path as one
serialized transfer resource.  It does not modify vLLM or integrate a real
LLM.

Reference: Zhu et al., *TraceLab: Characterizing Coding Agent Workloads for
LLM Serving*, 2026, arXiv:2606.30560.

Phase 1B execution status: COMPLETE
