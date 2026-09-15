# Phase 1F — adaptive safety margins

## Objective

Test whether a margin proportional to prediction uncertainty and predicted pause improves the FixedTTL-1 QueueJIT stall–memory tradeoff over a fixed 500 ms margin.

## Setup

Frozen Phase 1E workload: 56 KiB/token, context <=131072, closed-loop TraceLab session replay, M=32, seeds 0–4. Admission is FixedTTL-1 using predicted pause >=1.0 s. Predictions use the Phase 1D/1E deterministic per-event noise and are shared across strategies. Safe deadlines use the physical D2H+H2D lower bound; stall uses actual deadlines.

## Median results over five seeds

| sigma | strategy | miss rate | P95 stall (ms) | normalized memory-time |
|---:|---|---:|---:|---:|
| 0.10 | NoMargin | 0.1703 | 693.9 | 0.0520 |
| 0.10 | Fixed500ms | 0.0603 | 193.9 | 0.0614 |
| 0.10 | Adaptive-0.5 | 0.1018 | 230.5 | 0.0816 |
| 0.10 | Adaptive-1.0 | 0.0479 | 0.0 | 0.1189 |
| 0.25 | NoMargin | 0.1684 | 1354.7 | 0.1082 |
| 0.25 | Fixed500ms | 0.0842 | 854.7 | 0.1184 |
| 0.25 | Adaptive-0.5 | 0.0970 | 319.1 | 0.1733 |
| 0.25 | Adaptive-1.0 | 0.0287 | 0.0 | 0.2755 |
| 0.50 | NoMargin | 0.1699 | 3294.9 | 0.2395 |
| 0.50 | Fixed500ms | 0.1156 | 2794.9 | 0.2477 |
| 0.50 | Adaptive-0.5 | 0.0826 | 370.9 | 0.3328 |
| 0.50 | Adaptive-1.0 | 0.0091 | 0.0 | 0.5190 |

## Interpretation

At sigma=0.10, **Fixed500ms** is best: 6.03% miss rate, 193.9-ms P95 stall, and 0.0614 normalized memory-time. Adaptive-0.5 is worse in both P95 stall and memory; Adaptive-1.0 reaches 0-ms P95 only at 0.1189 memory-time.

At sigma=0.25, **Fixed500ms** is the lower-memory operating point: 8.42% miss rate, 854.7-ms P95 stall, and 0.1184 memory-time. Adaptive-0.5 lowers P95 to 319.1 ms, but raises memory-time by 46.3%; Adaptive-1.0 needs 0.2755 memory-time.

At sigma=0.50, **Adaptive-0.5** is the least-cost adaptive tradeoff: 8.26% miss rate, 370.9-ms P95 stall, and 0.3328 memory-time, versus Fixed500ms at 11.56%, 2794.9 ms, and 0.2477. The 34.4% memory-time increase remains material; Adaptive-1.0 costs 0.5190.

Adaptive-0.5 needs 32.9%, 46.3%, and 34.4% more memory-time than Fixed500ms at sigma=0.10, 0.25, and 0.50 respectively, and is worse at sigma=0.10. Adaptive-1.0 increases memory-time by 93.6%, 132.6%, and 109.5%.

**Result C: This simple adaptive rule is not useful as a consistent replacement for Fixed500ms.** It improves high-noise stalls only by giving back substantially more local GPU memory, while harming the low-noise tradeoff. No k values beyond the requested two were tested.

## Anomalies and limitations

This remains an oracle-trace simulator except for injected deadline noise; it models one serialized inbound H2D resource and no real KV tensors or inference runtime.

Phase 1F execution status: COMPLETE
