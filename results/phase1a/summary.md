# Phase 1A — Queue-Aware JIT KV Promotion Simulator

## Hypothesis

Oracle resume slack can schedule promotions just in time, but independent per-session JIT creates avoidable deadline collisions on a finite inbound transfer resource. Queue-aware JIT should reduce those collisions while retaining much less KV early than eager prefetching.

## Calibration

The simulator loads P50 latency programmatically from `results/phase0/summary_metrics.csv` at Phase 0 commit `0337837b43ddf78821839d86be6a89ae2a7c164a`. CPU→GPU0 promotion service times are 10.330 ms (256 MiB), 40.858 ms (1 GiB), and 164.427 ms (4 GiB). The optional peer source uses GPU1→GPU0 P50s of 10.183, 40.722, and 162.877 ms, respectively. No bandwidth constant is hard-coded.

## Policies

- **OnDemand:** queue eligibility is the actual resume deadline; it is the no-prefetch baseline.
- **ASAP-EDF:** a released job is immediately eligible; the single resource always serves the pending earliest deadline first.
- **NaiveJIT:** each job becomes eligible at `max(release, deadline − service_time)` and EDF resolves simultaneous ready jobs; it ignores cumulative demand.
- **SlackKV-QueueJIT:** for released unfinished jobs in deadline order, computes the minimum `deadline − now − prefix_service`; it idles only for that positive safe slack (or until another release), otherwise starts the earliest deadline job.

The simulator models one serialized, non-preemptive inbound promotion resource to GPU0. It is intentionally not a claim about every CUDA copy-engine topology.

## Controlled burst results

The controlled family uses N=4/8/16/32 jobs, five deterministic seeds, release jitter within 20 ms, heterogeneous 0.1–10 s resume gaps, and 256 MiB/1 GiB/4 GiB sizes. At the high-contention N=32 point, values below are medians over five seeds.

| Policy | Miss rate | P95 stall (ms) | Prefetch VRAM-time (GB·s) |
| --- | ---: | ---: | ---: |
| OnDemand | 1.0000 | 284.87 | 0.000 |
| ASAP-EDF | 0.2500 | 194.85 | 59.230 |
| NaiveJIT | 0.5313 | 148.08 | 0.000 |
| SlackKV-QueueJIT | 0.1250 | 57.06 | 0.956 |

Against NaiveJIT, QueueJIT reduced N=32 miss rate by 76.47% and P95 stall by 61.47%. Against ASAP-EDF, QueueJIT reduced prefetch VRAM-time by 98.39% while also lowering P95 stall by 70.72% for this burst configuration.

## Stochastic results

The stream uses Poisson arrivals, 5,000 jobs per load/seed, three seeds, a log-uniform oracle resume gap from 0.1 to 10 s, and nominal load `rho = lambda E[p]` of 0.25/0.50/0.75/0.90. At CPU-source rho=0.75, medians over the three seeds were:

| Policy | Miss rate | P95 stall (ms) | Prefetch VRAM-time (GB·s) | Prefetch / KeepReference |
| --- | ---: | ---: | ---: | ---: |
| OnDemand | 1.0000 | 674.50 | 0.00 | 0.0000 |
| ASAP-EDF | 0.0500 | 0.004 | 11061.85 | 0.8860 |
| NaiveJIT | 0.7538 | 564.13 | 0.00 | 0.0000 |
| SlackKV-QueueJIT | 0.4036 | 414.85 | 115.76 | 0.0094 |

Thus QueueJIT reduced rho=0.75 NaiveJIT miss rate by 46.46% and P95 stall by 26.46%, but did not retain ASAP-EDF's near-zero P95 stall. Its peak incremental early-prefetched memory at this point was 9.93 GB (exceeds the hypothetical 8 GB budget but not 16/24 GB); ASAP-EDF's peak was 70.87 GB (exceeds all three hypothetical budgets).

## Memory-time tradeoff

`KeepReference` retains every session through its entire pause and has zero resume stall by construction. QueueJIT's normalized prefetch VRAM-time at rho=0.75 was 0.0094 versus ASAP-EDF's 0.8860, but the stochastic tail-stall tradeoff is unfavorable relative to ASAP-EDF. The N=32 controlled burst shows the intended favorable region; the stochastic stream exposes that it is not universal.

## Queue-awareness result

QueueJIT consistently improved on NaiveJIT in the tested stream: at rho=0.25/0.50/0.75/0.90, median miss-rate reductions were 76.57%/66.88%/46.46%/24.50%, respectively. At rho=0.90 the result falls just below the 25% miss-rate threshold, while rho=0.50 and rho=0.75 meet it comfortably.

## Sensitivity

At rho=0.75 with the fixed seed-101 sensitivity workload, 1.15× service time produced QueueJIT/NaiveJIT miss rates of 0.4438/0.7640 (41.91% reduction) and P95 stalls of 607.86/762.76 ms (20.31% reduction). Across 1.0×/1.05×/1.15×/1.30×, QueueJIT preserved a 44.57%/43.54%/41.91%/37.60% miss-rate reduction versus NaiveJIT. These multipliers are sensitivity analysis, not measured inference behavior.

## Peer ablation

At rho=0.75, seed 101, peer-source QueueJIT had miss rate 0.4196, P95 stall 525.86 ms, and 102.11 GB·s prefetch VRAM-time. The matched CPU-source sensitivity run had 0.4214, 521.49 ms, and 105.12 GB·s. The difference is not material, consistent with the Phase 0 SYS-topology calibration.

## Success criteria

- **Criterion A: PASS.** N=32 burst gives 76.47% miss-rate reduction versus NaiveJIT; the stochastic rho=0.75 point also gives 46.46% miss-rate and 26.46% P95-stall reduction.
- **Criterion B: PASS.** In the meaningful high-contention N=32 burst, QueueJIT reduces ASAP-EDF prefetch VRAM-time by 98.39% while its P95 stall is lower, not more than 10% worse. This criterion does not hold for the stochastic stream, where QueueJIT's P95 stall is substantially higher than ASAP-EDF's.

## Limitations

Phase 1A is a simulator with oracle resume deadlines and one serialized inbound transfer resource. It does not model a real LLM, real agent traces, full vLLM KV block behavior, or prediction error. It does not claim peer GPU memory is universally superior to CPU memory.

Phase 1A execution status: COMPLETE
