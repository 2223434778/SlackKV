# Phase 4E-A — frozen hybrid performance matrix

**EXPERIMENT_INVALID_MATRIX**. The predeclared background-X validity gate failed in Stage B repetition 1, position 3. The driver stopped immediately; no incomplete matrix is interpreted as a performance result.

## Objective and frozen setup

This was the first controlled four-policy ablation on the frozen hybrid trace-driven workload: real coding-agent token/KV structure plus empirical TraceToolProxy timing, not real request-timestamp-faithful Agent execution. The eight first-boundary jobs contain 168,016 reusable tokens and 10,501 blocks (minimum H2D 4,594.1875 MiB). Runtime remained Qwen2.5-1.5B-Instruct, vLLM 0.23.0, FP16, GPU0, 32K, 2 GiB GPU KV (4,681 blocks), 8 GiB external KV, K2, P_SAFE=102 ms, 110-ms resume guard, and 3,510-block logical budget. No vLLM source mechanism was changed.

## Execution and failure

- X-only baseline: 4/4 valid, median decode 3633.51 ms and median TPOT 14.249 ms.
- Stage A semantic smoke: 4/4 valid in frozen A→B→C→D order. All smoke runs completed 8/8 resumes, source conservation and K2, with X active through the last Agent first token.
- Stage B: only 3/16 scheduled runs started; 2 valid. Repetition 1 positions 1 (OnDemand) and 2 (GuardedJIT) passed. Position 3 (GuardedJIT+Lease) completed 8/8 requests with exact output counts, source conservation, K2 maximum 2, zero lease leaks, zero OOM and no server crash, but failed background validity.
- In that failing run, X finished at t0+4521.29 ms. The last Agent FIRST_TOKEN (E03) occurred at t0+4664.73 ms, **143.44 ms after X ended**. This is a direct comparison of the saved monotonic-t0 scheduler/client events, not a parsing artifact.
- The driver did not start position 4 or repetitions 2–4. It did not retry just one policy, change X length, alter workload/deadlines, or recombine partial data.

## Diagnostic observations, not performance claims

The complete failure record is `failures/measured_rep1_pos3_GuardedJITplusLease-Oracle-K2/`. The invalid C run had local/external source tokens 61888/106128, three capacity admits and five rejects, and 0 logical/physical lease leaks. A single request (E04) had an output SHA different from its Phase 4C solo reference across multiple policies, while output lengths were exact; cross-policy batch-sensitive sequence identity is diagnostic, not a token-count failure. Because Stage B is incomplete, there is **no** policy median table, paired delta, H1/H2/H3 verdict, P95 superiority, H2D ablation conclusion, or statistical claim. `policy_summary.csv` marks medians NA rather than fabricating values.

## Next-step status

Phase 4E-A is not a valid completed performance matrix. No Phase 4E-B or all-boundary replay was started.
