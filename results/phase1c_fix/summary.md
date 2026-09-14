# Phase 1C-Fix: robust Pareto evaluation

## Purpose and preserved experiment

This fixes Phase 1C's directional transfer calibration and evaluation logic only.  It reuses the already-completed seed-level closed-loop replay in `aggregate_metrics.csv`: no sessions, offsets, policies, footprints, or event selections were re-sampled.  The primary workload is 56 KiB/token with the retained 131,072-token context cap, with the original stress sensitivity retained in the source results.

GPU0-to-pinned-CPU D2H and pinned-CPU-to-GPU0 H2D are separately calibrated from Phase 0 commit `0337837b43ddf78821839d86be6a89ae2a7c164a`.  At 256 MiB/1 GiB/4 GiB their D2H/H2D P50 times are 9.911/10.330, 41.488/40.858, and 162.655/164.427 ms, respectively.  The largest retained primary KV (7.52 GB decimal) maps to 320.764 ms D2H and 286.436 ms H2D.

## Evaluation

- **C0: PASS.** Every SlackRatio admission was checked per event: no isolated event with slack ratio below one was admitted.  `c0_violations.csv` is empty.
- **C1: PASS.** This is now a 10-seed primary closed-loop evaluation, not a single-seed existential check.  SlackRatio-4 passes at M=32 (median miss 3.09%, P95 0 ms, normalized local-memory time 0.256; 8/10 passing seeds), while SlackRatio-3 and SlackRatio-4 pass at M=64 (8/10 and 9/10 seeds).
- **C2: PASS.** At M=32, SlackRatio-2 is not dominated by the FixedTTL Pareto frontier and strictly dominates FixedTTL-0.5 in all three median dimensions: normalized local-memory time 0.2462 versus 0.2515, P95 stall 138.8 versus 636.1 ms, and miss rate 5.70% versus 6.75%.  Its P95 improvement is 78.2%, well above the 10% noise guard.  FixedTTL-2 remains a lower-stall point at a modestly higher memory-time cost, so this is a frontier expansion rather than a claim that SlackRatio is universally best.

## M=32 primary medians

| Policy | Miss rate | P95 stall | Normalized local-memory time | Interpretation |
|---|---:|---:|---:|---|
| OffloadAll-QueueJIT | 68.52% | 5543.1 ms | 0.2707 | Admission-free reference |
| FixedTTL-0.5 | 6.75% | 636.1 ms | 0.2515 | TTL frontier point dominated by SlackRatio-2 |
| FixedTTL-2 | 2.18% | 0.0 ms | 0.2526 | Low-stall TTL frontier point |
| SlackRatio-2 | 5.70% | 138.8 ms | 0.2462 | C2-qualifying transfer-normalized point |
| SlackRatio-4 | 3.09% | 0.0 ms | 0.2556 | C1-qualifying robust point |

## Outcome

**A = Strong GO.** C0, robust C1, and strict C2 all pass.  The result establishes that selective offload is useful and that a transfer-normalized SlackRatio point can extend the same-concurrency FixedTTL Pareto frontier under this simulator.  It does not establish that SlackRatio dominates every TTL operating point.

## Reproducibility and limitations

Regenerate the aggregation and plots with:

```bash
/home/a6000/anaconda3/bin/python experiments/phase1c_fix/analyze_phase1c_fix.py
/home/a6000/anaconda3/bin/python scripts/plot_phase1c_fix.py
```

This remains an oracle-deadline simulator with immediate, non-contended D2H, one serialized inbound H2D abstraction, hypothetical KV footprints, and no real KV tensors, LLM serving, or learned predictor.

Phase 1C-Fix execution status: COMPLETE
