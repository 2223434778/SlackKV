# Phase 3L — native promotion service curve

Valid promotions: 20/20.

| Active decode n | Client P50 (ms) | Client P95 (ms) | Client max (ms) | Server P50 (ms) | Copy P50 (ms) | Noncopy P50 (ms) | Recommended budget (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 20.923 | 21.467 | 21.595 | 14.624 | 5.507 | 14.743 | 35.000 |
| 1 | 57.087 | 58.909 | 58.997 | 50.453 | 5.360 | 51.755 | 70.000 |
| 2 | 51.391 | 52.389 | 52.437 | 44.914 | 5.407 | 44.739 | 65.000 |
| 3 | 53.533 | 59.798 | 60.934 | 46.078 | 5.586 | 47.948 | 75.000 |

P50 deltas: Δ01=36.165 ms, Δ12=-5.697 ms, Δ23=2.143 ms.
Copy-time load dependence: NO.
Non-copy overhead load dependence: YES.
Service curve: NON_MONOTONIC.
Decode count useful predictor: NO.
Recommended model: RICHER_RUNTIME_SIGNAL.
Next direction: STEP_TIME_AWARE_MODEL.

Endpoint response fields were persisted directly from the native runtime. Each promotion used one fixed 4096-token prefix after the active background first-token barrier; native promotions reported zero model forwards, zero generated tokens, and zero mixed steps.

Result C.

Phase 3L complete; Phase 3M was not started.
