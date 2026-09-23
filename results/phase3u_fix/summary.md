# Phase 3U-FIX summary

Previous Phase 3U status: INVALID. Its Blocking70 resumes were not coupled to native-prefetch completion; old raw CSVs are preserved and not used here.

## Validation

X-only controls: 3/3 (median decode 3697.17 ms; median TPOT 14.50 ms); Stage A: 4/4; Stage B: 12/12; real resumes: 144/144; source conservation: 12/12; K2: 12/12; background: 12/12.

Blocking70 wait helper: 12 calls/run; measured waits may be zero when prefetch completed before the deadline. Every measured resume was checked against its own client-return timestamp with 0.5-ms tolerance.

## Median across three fresh measured runs

| Policy | AL P95 resume (ms) | Combined tok/s | X slowdown | Local % | Total H2D MiB | Duplicate H2D MiB | Promotion useful |
|---|---:|---:|---:|---:|---:|---:|---:|
| OnDemand-K2 | 111.63 | 160.87 | 1.054 | 0.0% | 1344.0 | 0.0 | NA |
| Blocking70-K2 | 104.35 | 154.28 | 1.099 | 33.3% | 2240.0 | 896.0 | 0.3333333333333333 |
| NaiveAsync40-K2 | 101.40 | 153.54 | 1.104 | 33.3% | 2240.0 | 896.0 | 0.3333333333333333 |
| ReservationLedger2-Async40-K2 | 88.98 | 156.74 | 1.082 | 50.0% | 1344.0 | 0.0 | 1.0 |

Blocking70 actual waiting: measured wait median 0.010 ms, P95 0.012 ms, max 0.015 ms, sum 0.360 ms. A zero wait is valid when that session's native prefetch returned before its deadline.
Physical lease high-water mark: Stage A 512 blocks (end-live 0); measured max 512 blocks.

## Comparisons

ReservationLedger vs OnDemand: P95 resume change -20.3%; combined throughput change -2.6%; X slowdown difference +0.028; total H2D change 0.0%.
ReservationLedger vs NaiveAsync40: P95 resume change -12.3%; duplicate H2D removed 896.0 MiB; combined throughput change 2.1%; X slowdown difference -0.022.
ReservationLedger vs Blocking70: P95 resume change -14.7%; duplicate H2D change -896.0 MiB; combined throughput change 1.6%.
Native timing medians (target→issue / issue→T6 / target→T6, ms): Blocking70 0.06481644231826067 / 24.871189147233963 / 24.948645499534905; NaiveAsync40 0.05382194649428129 / 26.73984842840582 / 26.78629942238331; Reservation 0.07469707634299994 / 29.42958998028189 / 29.51083250809461.

Reservation usefulness: 18/18; remote waits 0; internal-ready misses 0; lease leaks 0; maximum reserved blocks 512.

X slowdown uses the median of three fresh X-only 256-token decode durations. No Phase 3U numbers were reused. Three repeats are descriptive mechanism evidence, not a population-level inference.

Result: VALID_BASELINE_MATRIX. Next direction: TRACE_DRIVEN_RESERVATION.
