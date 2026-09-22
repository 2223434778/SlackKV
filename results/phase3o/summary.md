# Phase 3O — asynchronous native promotion end to end

- Async internal-ready: 9/9; local cache success: 9/9; remote waits: 0.
- Async max outstanding HTTP prefetch calls: 2; maximum overlapping T0→T6 intervals: 1.

| Policy | ABC P95 effective resume | ABC P95 completion | ABC decode tok/s | X slowdown | Combined tok/s | Makespan | Promotion memory-time |
|---|---:|---:|---:|---:|---:|---:|---:|
| OnDemand-K2 | 106.51 | 587.90 | 172.43 | 1.056 | 114.22 | 624.14 | 0.00 |
| Blocking70-K2 | 58.32 | 530.61 | 183.39 | 1.067 | 113.01 | 561.78 | 52.29 |
| Async40-K2 | 60.64 | 536.32 | 183.62 | 1.061 | 113.67 | 567.52 | 30.93 |

- Async vs Blocking: effective resume 3.97%; memory-time -40.85%; X slowdown -0.58%; combined throughput 0.58%.
- Result A; next direction `FULL_ABCDEF_E2E`.

T6 rather than HTTP client return defines GPU-ready residency. Per-session local cache success is validated by the run-global exact 12288 local-token source-counter delta plus each real scheduler trace showing no remote-KV wait; source counters are not request-labeled under concurrent K2 serving.
