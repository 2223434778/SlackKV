# Phase 1E — JIT safety margins

## Setup

Phase 1E reuses the Phase 1D deterministic per-event multiplicative Gaussian prediction noise. It evaluates 56 KiB/token, context ≤131072, closed-loop TraceLab session replay, M=32, and seeds 0–4. FixedTTL-1 admits using the predicted pause; QueueJIT schedules against the maximum of predicted deadline minus margin and the physical D2H+H2D lower bound, while stall and memory use the actual deadline.

## Median metrics over five seeds

| sigma | margin_ms | miss_rate | P95 stall (ms) | normalized memory-time |
|---:|---:|---:|---:|---:|
| 0.10 | 0 | 0.1703 | 693.9 | 0.0520 |
| 0.10 | 100 | 0.1196 | 593.9 | 0.0535 |
| 0.10 | 250 | 0.0826 | 443.9 | 0.0562 |
| 0.10 | 500 | 0.0603 | 193.9 | 0.0614 |
| 0.10 | 1000 | 0.0397 | 0.0 | 0.0714 |
| 0.25 | 0 | 0.1684 | 1354.7 | 0.1082 |
| 0.25 | 100 | 0.1434 | 1254.7 | 0.1100 |
| 0.25 | 250 | 0.1174 | 1104.7 | 0.1130 |
| 0.25 | 500 | 0.0842 | 854.7 | 0.1184 |
| 0.25 | 1000 | 0.0589 | 354.7 | 0.1287 |
| 0.50 | 0 | 0.1699 | 3294.9 | 0.2395 |
| 0.50 | 100 | 0.1630 | 3194.9 | 0.2409 |
| 0.50 | 250 | 0.1434 | 3044.9 | 0.2432 |
| 0.50 | 500 | 0.1156 | 2794.9 | 0.2477 |
| 0.50 | 1000 | 0.0871 | 2294.9 | 0.2570 |

## Interpretation

For sigma=0.10, best practical margin: **250 ms** — miss rate 8.26%, P95 stall 443.9 ms, normalized memory-time 0.0562 (51.5% miss-rate reduction versus 0 ms).

For sigma=0.25, best practical margin: **1000 ms** — miss rate 5.89%, P95 stall 354.7 ms, normalized memory-time 0.1287 (65.0% miss-rate reduction versus 0 ms).

For sigma=0.50, best practical margin: **1000 ms** — miss rate 8.71%, P95 stall 2294.9 ms, normalized memory-time 0.2570 (48.7% miss-rate reduction versus 0 ms).

**Result B: Safety margin helps, but prediction error is expensive in GPU memory.** A 250 ms margin is effective at low noise, but the medium/high-noise cases require 1000 ms and high noise still retains a multi-second P95 stall. No further tuning was performed.

Phase 1E execution status: COMPLETE
