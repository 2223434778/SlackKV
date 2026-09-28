# Phase 4G-A: 32-boundary multi-turn lifecycle evaluation

Final result: **EXPERIMENT_INVALID_MATRIX**. The 16-run performance matrix was stopped at the predeclared background hard gate; no A/D policy medians or H1/H2/H3 claims are reported.

## Workload and lifecycle

Frozen Phase4C workload SHA256: 9b381336a70e73b0efbddbb79d2c88608c19663247e819c1e7a364efd2971105. Eight sessions, 40 turns and 32 causal boundaries per run; 705728 reusable tokens and 44108 aligned blocks. The 19297.25-MiB one-full-crossing sum is a scale reference, not a lower bound on transfer.
Natural connector offload readiness was observable by exact block lookup. No boundary prefix was pre-materialized. In the final Stage L smoke, A and D each completed 40/40 turns and 32/32 boundaries; both had 10 source-ready and 22 source-not-ready boundaries. D skipped unready promotion candidates rather than fabricating external KV.
Stage L X-tail margins: A 5795.88 ms, D 5890.16 ms; both exceeded the frozen 3000-ms gate.

## Stopped measured matrix

Pre-matrix X-only: 4/4 valid. Measured: 2/16 valid (Rep1 A and D). The next run, Rep2 D, completed all 40 requests and 32 boundaries with K2=2, 0 OOM/crash/leaks, but X-tail margin was -25286.80 ms.
X finished 34479.33 ms after t0; the last Agent first token arrived 59766.12 ms after t0. X therefore ended 25286.80 ms before the last Agent first token. This is RUN_INVALID_BACKGROUND under the frozen protocol, not a 40-request completion failure.
The driver immediately stopped. No later repetitions or post-matrix X-only controls ran; partial results were not aggregated. Frozen X=2048, workload, deadlines, K2, PSA-1, and service budget were not changed.

## Interpretation

The causal multi-turn harness passed its lifecycle smoke but the frozen 2048-token background did not reliably span the full first-token contention window during the measured matrix. Consequently this phase cannot decide whether larger multi-turn sampling reveals latency benefit, whether compute queueing dominates, or whether H1/H2/H3 hold. The invalid outcome is preserved rather than treating a background-short run as a performance sample.
