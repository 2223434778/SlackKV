# Phase 3U — full twelve-session baseline matrix

## Objective
Compare OnDemand, blocking prefetch, naive async prefetch, and ReservationLedger under the identical A–L workload.

## Setup
vLLM 0.23.0; PyTorch 2.11.0+cu126; RTX A6000; Qwen2.5-1.5B-Instruct; K=2; GPU KV 1 GiB; CPU external KV 2 GiB. Phase 3S lease patch reused unchanged.

## Results
| Policy | AL P95 resume (ms) | Combined tok/s | X slowdown | Local % | Useful % | Total H2D MiB | Duplicate H2D MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| OnDemand-K2 | 115.11 | 162.40 | 1.064 | 0.00 | NA | 1344.00 | 0.00 |
| Blocking70-K2 | 96.41 | 156.08 | 1.107 | 33.33 | 33.33 | 2240.00 | 896.00 |
| NaiveAsync40-K2 | 115.93 | 152.11 | 1.136 | 33.33 | 33.33 | 2240.00 | 896.00 |
| ReservationLedger2-Async40-K2 | 89.04 | 157.94 | 1.094 | 50.00 | 100.00 | 1344.00 | 0.00 |

ReservationLedger issued 18 promotions and skipped 18 sessions across measured runs; all promotions were useful, with zero remote waits, internal-ready misses, or lease leaks. NaiveAsync reproduced residency failure: only 4/12 promotions per run were useful and the rest later restored externally. Blocking70 likewise did not preserve unleased promoted state. Source conservation and K2 validation passed in every run.

Reservation vs OnDemand: P95 -22.65%, combined throughput -2.75%, X slowdown difference +0.030, total H2D +0.00%.
Reservation vs NaiveAsync removed 896.00 MiB median duplicate H2D per run.

## Conclusion
Result A. Next direction: TRACE_DRIVEN_RESERVATION. ReservationLedger is on the observed latency/traffic Pareto frontier: it has the lowest median AL P95 and minimum H2D traffic, without a material combined-throughput or X-interference penalty.

## Scope
Phase 3U is an internal mechanism baseline matrix. It is not yet the final external SOTA comparison against MORI / Continuum / CacheWise / KVFlow-style systems.
