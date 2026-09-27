# Phase 4E-A-FIX — incomplete frozen hybrid matrix

**EXPERIMENT_INVALID_MATRIX.** Phase 4E-A remains separately invalid: its fixed 256-token X ended 143.44 ms before the last Agent FIRST_TOKEN in one measured run. Phase 4E-A-FIX used a frozen 512-token X and recomputed X-dependent metrics from actual output counts. The original Phase 4E-A performance samples were not reused.

The workload and mechanism were unchanged: the Phase 4C first-boundary hybrid trace-derived workload (real coding-agent token/KV structure with an empirical TraceToolProxy pause, not real request-timestamp-faithful execution), eight sessions, 168,016 cached-source tokens, 10,501 blocks, and 4,594.1875 MiB minimum KV H2D. The vLLM source diff fingerprint remained `e4f53d8f99230d71ba7798cc192a873d954777a20a6e0932c027b662188d3f1f`; GPU KV had 4,681 blocks, logical budget 3,510 blocks, P_SAFE 102 ms, and K2.

## Background protocol

The four fresh X-only controls generated 512/512 tokens each; median decode duration was 7,447.455 ms and median TPOT was 14.574 ms. Stage A passed all four policy semantic smokes. X-tail margins (OnDemand / GuardedJIT / GuardedJIT+Lease / FullSlackKV) were 4,117.254 / 4,350.551 / 4,242.032 / 4,232.082 ms. The predeclared minimum 1,000-ms Stage A background protocol gate passed.

The first 15 Stage B runs were valid; their minimum X-tail margin was 3,935.807 ms. Thus the observed final-run failure is **not** another measured X-tail-margin failure.

## Matrix stopping event

The frozen counterbalanced matrix stopped at repetition 4, position 4, **GuardedJIT+Lease-Oracle-K2**. Its 30-second client request watchdog expired. The server trace contains eight `RESUME_ARRIVED`, zero `RESUME_ADMITTED`, zero Agent `FIRST_TOKEN`, and zero `REQUEST_FINISHED`. Eight guarded promotion decisions were logged: three admitted and five capacity-rejected. Three physical leases were acquired and none was released before the forced shutdown. The server remained alive until shutdown; the log shows no OOM or server ERROR. This is a request-progress failure under this ablation, not a completed latency sample. The exact cause is not established by this phase.

Across the first 15 valid runs, 120/120 Agent requests finished with exact output-token counts, source conservation passed 15/15, and no crash, OOM, logical/physical leak, or unsafe release was recorded. E04's output SHA differed from the Phase 4C solo reference once per valid run (15 total); output count remained correct and the cross-schedule hash is diagnostic only. The failed run cannot be counted toward those correctness totals.

## Scientific disposition

Only **15/16** measured runs completed, so no four-policy medians, paired comparisons, component-ablation winner, or H1/H2/H3 classification is reported. `policy_summary.csv` contains explicit NA values and `paired_deltas.csv` contains no partial comparisons. The raw repetition-level diagnostics are retained solely for failure analysis. The previously frozen performance criteria are **NOT_EVALUATED**.

The limitation is specific: extending X solved the known background-duration protocol defect, but the 16-run performance matrix encountered a separate non-progress case in the no-logical-reservation leased ablation. No vLLM mechanism, workload, deadline, P_SAFE, budget, K2, or policy was retuned; no later phase was started.
