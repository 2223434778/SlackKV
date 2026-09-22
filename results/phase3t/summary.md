# Phase 3T — 12-session residency reservation

## Result

**Result A.** Stage A passed and all three Stage B runs were valid. The finite two-lease ledger deliberately promoted 6/12 sessions per run and fell back to OnDemand for the other 6/12.

| Metric | Result |
|---|---:|
| Reservation promotions | 18 (6/run) |
| Reservation skips | 18 (6/run) |
| Promoted local | 18/18 |
| Promoted remote waits | 0 |
| Internal-ready misses | 0 |
| Promotion useful rate | 100.0% |
| Wasted promotion | 0 MiB |
| Maximum reserved blocks | 512 / 512 |
| Lease acquire/release/leak | 18/18/0 |
| Source tokens | 73728 local + 73728 external = 147456 |
| Overall P95 resume (median run) | 83.48 ms |
| Promoted P95 (median run) | 58.06 ms |
| Skipped P95 (median run) | 83.96 ms |

Every run naturally selected A/B, E/F, and I/J through EDF QueueJIT plus the ledger; C/D, G/H, and K/L were skipped. The ledger never exceeded 512 blocks. Every admitted promotion acquired and released exactly one lease, survived to the resume allocation boundary, and avoided `WAITING_FOR_REMOTE_KVS`. There were no lease leaks. K2 and background-X validation passed 3/3; X generated 256 tokens and remained active through L's first token.

Target-to-issue was 0.079/0.174/0.189 ms median/P95/max. Issue-to-T6 was 25.200/31.105/33.195 ms, and target-to-T6 was 25.279/31.286/33.334 ms. Removing the synchronous Prometheus decision restored sub-millisecond controller latency.

Mechanistically, Phase 3R-FIX promoted all 12 after an instantaneous-capacity query and later lost seven promoted prefixes. Phase 3T reserves only two future prefixes per cluster, physically leases them, and intentionally routes the remaining sessions through OnDemand. This reduced issued promotion traffic while making every issued promotion useful.

Promotion memory-time totaled 280.21 MiB*s across Stage B. Promoted P95 was lower than skipped P95, but residency correctness—not latency—is the primary success criterion.

Next direction: `RESERVATION_FULL_BASELINE_MATRIX`. It was not started.
