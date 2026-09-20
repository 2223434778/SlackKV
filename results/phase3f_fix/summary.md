# Phase 3F-Fix — in-engine FirstTokenGate

## Scope and validation

Phase 3F-Fix corrects the incomplete Phase 3F result. The measured gate is
inside the vLLM scheduler; the previous client-side `threading.Event` gate is
not used. The pinned vLLM runtime is 0.23.0 at source commit
`0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`, with the Phase 3B
connector-first implementation plus the small scheduler gate/trace patch in
`vllm_phase3f_fix.patch`. The SlackKV base is
`bc29bf1167f4d6e5b52474a3a60c7bb11b14073b`.

The workload is unchanged: Qwen2.5-1.5B-Instruct, GPU0 RTX A6000,
`max_tokens=1`, A–F 4096-token prefixes, deadlines 300/320/340/650/900/1300
ms, `p_safe=30 ms`, and `resume_guard=110 ms`. Because `max_tokens=1`, first
token release is approximately completion release in this experiment; this is
not evidence about overlapping multi-token decode.

The 3-policy × 3-repetition matrix completed. The A/B/C integration pre-gate
also passed before the full matrix: real integer scheduler step IDs showed
A admitted before B before C, B after A's `FIRST_TOKEN`, C after B's
`FIRST_TOKEN`, and maximum gate ownership of one.

## Real scheduler trace

The trace contains scheduler-generated `RESUME_ARRIVED`,
`WAITING_FOR_REMOTE_KVS`, `REMOTE_KV_READY`, `GATE_BLOCKED`,
`RESUME_ADMITTED`, `FIRST_MODEL_SCHEDULED`, `FIRST_TOKEN`, and
`REQUEST_FINISHED` events. Median A/B/C steps across repetitions were:

| policy | session | arrival | remote wait | remote ready | admitted | first model | first token |
|---|---:|---:|---:|---:|---:|---:|---:|
| OnDemand-Trace | A | 63 | 64 | 70 | 70 | 70 | 71 |
| OnDemand-Trace | B | 70 | 71 | 76 | 76 | 76 | 77 |
| OnDemand-Trace | C | 70 | 71 | 78 | 78 | 78 | 79 |
| ResumeAware-Trace | A | 84 | — | — | 85 | 85 | 86 |
| ResumeAware-Trace | B | 85 | — | — | 86 | 86 | 87 |
| ResumeAware-Trace | C | 85 | — | — | 86 | 86 | 87 |
| InEngineFirstTokenGate | A | 84 | — | — | 85 | 85 | 86 |
| InEngineFirstTokenGate | B | 85 | — | — | 87 | 87 | 88 |
| InEngineFirstTokenGate | C | 85 | — | — | 89 | 89 | 90 |

OnDemand traces contain real remote-KV waiting and ready transitions and its
admissions are naturally staggered. ResumeAware requests are locally ready and
admit A/B/C in nearly adjacent scheduler steps. The in-engine gate blocks B/C
while the current owner is pre-first-token, then admits them in deterministic
A → B → C order.

## Median results

| policy | ABC pairwise overlap (ms) | max pre-first-token concurrency | concurrency area (ms) | ABC P95 effective (ms) | overall P95 (ms) | corrected GPU memory-time (MiB·s) |
|---|---:|---:|---:|---:|---:|---:|
| OnDemand-Trace | 0.00 | 1 | 159.81 | 104.93 | 104.63 | 0.00 |
| ResumeAware-Trace | 94.14 | 3 | 223.46 | 127.20 | 124.34 | 24.85 |
| InEngineFirstTokenGate-ResumeAware | 0.00 | 1 | 160.17 | 88.99 | 88.10 | 24.73 |

The gate reduced ABC pairwise overlap by 100% relative to ResumeAware and
reduced ABC P95 by 30.1% (127.20 → 88.99 ms). Median gate-blocked time was
48.87 ms. Corrected GPU residency uses actual scheduler admission after
promotion completion, so waiting is charged rather than hidden. Its memory-time
was 6.4% of the historical Phase 3B ASAP reference (383.61 MiB·s).

All nine runs had zero promotion misses, zero blackout conflicts, zero long
prefetches, zero native-prefetch model forwards/generated tokens, and zero
mixed prefetch/model steps. Cache, generation, trace, and gate-invariant
validation passed for every run.

## Interpretation

1. **In-engine implementation:** YES. Ownership is assigned only when the
   scheduler admits a marked request; release occurs on actual generated
   output in `update_from_output()`. Ordinary unmarked requests and native
   prefetch-only requests are not gated. The client Event gate was removed.
2. **Gate correctness:** PASS. Maximum pre-first-token gate owners was exactly
   one in every measured gated run, and A/B/C admission order was preserved.
3. **Contention:** PASS. Pairwise overlap fell to zero and the concurrency
   area fell from 223.46 to 160.17 ms.
4. **End-to-end effect:** PASS under the predeclared Result A rule. The
   contention reduction more than offset gate waiting in this frozen
   `max_tokens=1` workload; gated ABC and overall P95 both improved over
   ordinary ResumeAware. OnDemand remained competitive (104.63 ms vs 88.10 ms
   for the gate), but was slower in this rerun.
5. **Implicit backpressure:** SUPPORTED. OnDemand contains actual
   `WAITING_FOR_REMOTE_KVS` → `REMOTE_KV_READY` transitions and staggered
   admissions, while local-ready ResumeAware admits A/B/C together. This is a
   trace-based classification, not an inference from latency alone.

The previous Phase 3F client-side Event implementation is exploratory only:
`Event.set()` could wake multiple waiters and violated the intended invariant.
It is not used as an algorithm baseline here.

## Limitations and status

This remains a research-only scheduler patch. With `max_tokens=1`, first-token
release approximately serializes measured completions; a future multi-token
experiment is required to test normal post-first-token decode overlap. No
CUDA kernel is preempted, and the Phase 3B connector-first promotion path is
otherwise reused unchanged.

Phase 3F-Fix result: **Result A** — a correct in-engine first-token gate
materially reduces local-ready resume burst contention while retaining low
GPU residency.

Execution status: **COMPLETE**.
