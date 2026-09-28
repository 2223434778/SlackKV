# Phase 4G-A-FIX — final status

Final result: **EXPERIMENT_INVALID_MATRIX**. The sustained fixed-work background protocol passed, but the measured matrix stopped on a non-progress failure in repetition 8, position 2 (`FullSlackKV+PSA1-K2`). No partial performance aggregation or policy comparison was performed.

## Frozen protocol and completed gates

- Start commit: `1d9fa6ade07698071a7b74cae7241781b943e10b`.
- Frozen Phase 4C workload SHA256: `9b381336a70e73b0efbddbb79d2c88608c19663247e819c1e7a364efd2971105`.
- No vLLM source/mechanism change was made; the Phase 4G-A source-diff fingerprint remained `f0f8e327cc7ff4eb1797852a24e56e0e7f4a77b62e0f9439e88405fd5b470ae4`.
- The background was four serialized 2,048-token chunks using the same 9,500-token prompt, for 8,192 generated tokens total. All four fixed-protocol calibration runs passed the request-finish coverage gate. All four pre-matrix background-only controls produced 8,192 tokens, with maximum inter-chunk idle gaps of 19.7–22.1 ms (below the frozen 50-ms limit).
- The four lifecycle smoke runs passed: 40/40 turns and 32/32 boundaries per run, with X overlapping through the final agent request finish.
- Measured repetitions 1–7 passed in the frozen alternating order: 14/14 runs, each completing 40/40 turns and 32/32 boundaries with all 8,192 background tokens and inter-chunk gaps below 50 ms.

## Why the matrix stopped

The next run, repetition 8 / position 2 (`FullSlackKV+PSA1-K2`), did not make progress before the 30-second request watchdog. At capture, the server was alive, with no OOM and no engine error. The event trace contained 17 `RESUME_ARRIVED`, 11 `RESUME_ADMITTED`, and only 9 distinct `FIRST_TOKEN` / `REQUEST_FINISHED` events. It also contained 2,572,201 `GATE_BLOCKED` events.

The last captured gate event showed zero running requests, zero waiting requests, and zero scheduled requests, yet the K2 state still reported two pre-first-token owners. Those owner IDs mapped to E03:turn:1 and E06:turn:1, both of which had `FIRST_TOKEN` and `REQUEST_FINISHED` events. This is strong evidence of a stuck/stale K2 admission state (or its lifecycle bookkeeping) in this run. The compact trace does not prove which transition failed to clear it, so the exact implementation root cause remains unconfirmed. This is a serving/controller progress failure, not evidence of bad trace data or an insufficient background duration: all prior measured runs passed the background coverage gate.

The matrix was stopped as required. Repetition 8 / position 1 (`OnDemand-K2`) was not run, and the four post-matrix background-only controls were not run. Thus there is no complete before/after eight-run background baseline. No P50/P95 policy medians, paired deltas, throughput comparison, or performance conclusion is valid from this partial matrix.

## Result and artifacts

Classification: **`EXPERIMENT_INVALID_MATRIX` (`RUN_INVALID_NONPROGRESS`)**.

The failure details are in `failures/measured_rep8_d/failure_diagnostics.json`; run order and completed-run ledger are in `run_order.json` and `progress.json`. Raw high-volume server output was kept outside the repository; only compact diagnostics/logs are retained. The experiment was not resumed, and no later phase was started.
