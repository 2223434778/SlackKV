# Phase 3J summary

Phase 3J keeps admission at FirstTokenGate K=2 and changes only the promotion service budget. No vLLM scheduler algorithm changes were made; Phase 3H K-capacity gating was reused unchanged.

## Calibration

L0 idle client E2E P50/P95/max: 20.57/20.91/21.84 ms.
L1 one-active-decode client E2E P50/P95/max: 53.88/56.58/57.10 ms.
L0 server E2E P50/P95/max: 14.85/14.86/16.38 ms.
L1 server E2E P50/P95/max: 46.54/49.96/50.23 ms.
L0 copy P50/max: 5.42/6.74 ms; L1 copy P50/max: 5.52/6.96 ms.
Busy/idle median slowdown: 2.62x.
Frozen p_busy_ms: 70 ms.

## Validation

|Metric|Median|
|---|---:|
|promotion misses|0.00|
|blackout conflicts|0.00|
|budget overruns|0.00|
|max pre-first concurrency|2.00|
|ABC P95 TTFT (ms)|58.71|
|ABC decode tok/s|184.18|
|memory-time (MiB*s)|42.75|
|X slowdown|1.06|

All five runs had clean promotion: YES.
K2 reached pre-first concurrency exactly 2 in all runs: YES.
C was gate-blocked in all runs: NO.
X remained active through promotion and past C first token in all validation runs: YES.

Result D: promotion is clean, but the required C gate-blocking condition was not exercised in every run; do not treat this as a completed K2 admission validation.

Next direction: RETRY_BUSY_K_COMPARISON only after the K2 exercise condition is made observable; no dynamic K or Phase 3K work is started.
