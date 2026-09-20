# Phase 3E — gated resume admission

Phase 3B connector-first vLLM implementation was reused unchanged. Frozen `p_safe=30 ms`, `resume_guard_ms=110 ms`, and `resume_spacing_ms=40 ms`.

| Policy | Promotion misses | Resume overlaps | ABC overlaps | P95 effective (ms) | ABC P95 (ms) | Memory-time (MiB·s) |
|---|---:|---:|---:|---:|---:|---:|
| OnDemand | 0.00 | 2.00 | 2.00 | 105.69 | 106.14 | 0.00 |
| ResumeAware | 0.00 | 2.00 | 2.00 | 122.72 | 125.78 | 24.51 |
| Gated-ResumeAware | 0.00 | 2.00 | 2.00 | 127.82 | 130.97 | 24.39 |

ResumeAware planned A/B/C sends: 300/320/340 ms. Gated planned sends: 300/340/380 ms. Gating changed intentional delays to 0/20/40 ms and reduced resume overlap from 2 to 2. ABC P95 changed from 125.78 to 130.97 ms; overall P95 changed from 122.72 to 127.82 ms, versus OnDemand 105.69 ms. Gated memory-time is 24.39 MiB·s. All cache, generation, zero-forward, and mixed-step invariants passed.

**Result B.** Phase 3E execution status: COMPLETE

Correction: the measured ABC overlap count remained 2 for both ResumeAware and Gated-ResumeAware; gating therefore did not reduce overlap in this run. Result B is based on the measured end-to-end outcome.
