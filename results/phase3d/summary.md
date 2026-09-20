# Phase 3D — resume-aware native promotion

Phase 3B connector-first vLLM code was reused unchanged. Frozen parameters: `p_safe=30 ms`, `resume_guard_ms=110 ms`; no sweeps or recalibration.

| Policy | Misses | Blackout conflicts | Long prefetches | P95 effective (ms) | Idle memory-time (MiB·s) |
|---|---:|---:|---:|---:|---:|
| OnDemand | 0.00 | 0.00 | 0.00 | 109.03 | 0.00 |
| QueueJIT-Safe30 | 1.00 | 1.00 | 1.00 | 109.52 | 9.65 |
| ResumeAware-QueueJIT-Safe30 | 0.00 | 0.00 | 0.00 | 124.47 | 24.43 |

ResumeAware moved C ahead of the merged A/B/C blackout: C native client E2E was 108.93–112.52 ms for QueueJIT-Safe30 versus 17.10–18.28 ms for ResumeAware. Blackout conflicts fell from 1 to 0, promotion misses from 1 to 0, and long prefetches from 1 to 0. ResumeAware uses 24.43 MiB·s, or 6.4% of historical Phase 3B ASAP (383.61 MiB·s). Its P95 is 124.47 ms versus 109.52 ms for QueueJIT-Safe30 and 109.03 ms for OnDemand, so it does not beat OnDemand end-to-end. All cache, generation, zero-forward and mixed-step invariants passed in the measured runs.

**Result A.** Phase 3D execution status: COMPLETE
