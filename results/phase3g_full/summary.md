# Phase 3G-Full — multi-token gated resume performance

## Frozen setup

- Base: `b0ca5fd755f4ec47c2991d4836fecc2d40956d99`; vLLM source `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`.
- vLLM 0.23.0, PyTorch 2.11.0+cu126, CUDA 12.6, RTX A6000 GPU0, Qwen2.5-1.5B-Instruct FP16/TP1, 1 GiB GPU KV, 2 GiB CPU external KV.
- A–F each use 4096-token / 117,440,512-byte KV prefixes; deadlines are 300/320/340/650/900/1300 ms. Real resumes use `max_tokens=32`, `temperature=0`, `ignore_eos=true`.
- Phase 3B connector-first native promotion and the Phase 3F-Fix in-engine FirstTokenGate were reused unchanged; no new vLLM algorithm change was made.

## Main medians across three independent restarts

| Policy | ABC TTFT P95 (ms) | Overall TTFT P95 (ms) | ABC completion P95 (ms) | ABC makespan (ms) | ABC decode tok/s | Max pre-first | Max decode | Memory-time (MiB*s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OnDemand-Trace | 194.77 | 191.80 | 673.28 | 695.41 | 171.54 | 2.00 | 4.00 | 0.00 |
| ResumeAware-Trace | 134.08 | 133.77 | 651.36 | 667.61 | 173.23 | 3.00 | 4.00 | 39.57 |
| InEngineFirstTokenGate-ResumeAware | 157.56 | 156.04 | 667.18 | 708.22 | 162.35 | 1.00 | 4.00 | 53.94 |

## Mechanism and correctness

- Token-count validation: PASS; every measured request is required to produce exactly 32 tokens.
- Cache transport validation: PASS; native endpoint runtime records: PASS.
- FirstTokenGate invariant / multi-token release: PASS. The gated median pre-first overlap is 0.00 ms; median post-first ABC decode overlap is 1410.74 ms.
- Phase 3G-CorrectnessDiag separately established LocalKeep-Solo == Prefetch-Solo and LocalKeep-Gated == Prefetch-Gated under matched execution. Cross-policy sequence-hash differences here are diagnostic only because batch composition changes greedy trajectories.

## ResumeAware → FirstTokenGate

- ABC P95 TTFT change: -17.51% (positive is an improvement).
- ABC decode throughput retained: 93.72%; ABC makespan change: 6.08%.
- Memory-time change: 14.37 MiB*s; gated memory is 14.06% of historical ASAP (383.61 MiB*s).

## Gated A/B/C real trace structure

- rep0 admit/first/finish steps: A 73/74/119; B 75/76/121; C 77/78/123.
- rep1 admit/first/finish steps: A 75/76/121; B 77/78/123; C 79/80/125.
- rep2 admit/first/finish steps: A 72/73/118; B 74/75/120; C 76/77/122.

- Full-request serialization: **NO**. In every gated repetition B was admitted after A's first token but before A finished, and C likewise after B's first token but before B finished.
- OnDemand retained real `WAITING_FOR_REMOTE_KVS` trace events; its external recovery acts as implicit admission backpressure, but its median ABC P95 TTFT was higher than the gated policy in this multi-token workload.

## Result

**Result C.** The first-token gate maintains its scheduling invariants and overlapping decode, but does not provide a useful multi-token TTFT tradeoff under the full six-session workload.

## Measurement notes

- Scheduler timestamps retain raw `time.monotonic()` values and are aligned to the saved workload t0; they are not rebased to the first server trace event.
- Positive request-arrival delay can exceed 50 ms while the engine is busy; it is retained as a server-intake diagnostic. Only a material negative offset would invalidate the shared monotonic time coordinate.
- Native source-token attribution is endpoint-authoritative: the runtime response records external tokens loaded; vLLM's official external source counter does not include prefetch-only endpoint loads. Local-cache-hit source counters verify subsequent real-resume reuse.
