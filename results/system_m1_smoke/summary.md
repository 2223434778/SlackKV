# M1 authorized GPU smoke — 2026-10-13

Objective: validate actual M1 restore/end lifecycle, then a frozen dynamic-prefix comparison.
Human approved NEW 0.5 GPUh after efb74ae delivery. Used 0.13476591384806347 GPUh, all setup/failures/exit included; prior 4.191988019727527, continuous total4.326753933575591. No new GPU work after this cohort.

Config: Qwen2.5-1.5B revision989aa798, FP16 TP1, compiled, GPU KV512MiB/actual18720tokens, host224MiB/512blocks for M1/Fair/Full; true NoHost no connector. Two independent-prefix agents, serial scripted pressure, closed-loop0.15s gaps, no forced GPU clear. Correctness batch-invariant separate from performance.

Three failed attempts retained: missing ninja PATH, missing inherited scheduler dependencies, INFO state logger not visible. Last failure completed13 model requests and cold match but did not meet full state gate, so remainsFAILED. PATH and logger fixed; dependencies supplied with algorithms off. Content4 PASS: real host store/restore, exact16-token independent cold match with zero cold hits, end-generation denied-write fallback, final512free slots. No arbitrary GPU cancellation/forced overlap proof.

| Run | Long E2E ms | Short E2E ms | Median ms | First request ms |
|---|---:|---:|---:|---:|
| M1 1 |5990.15|1751.11|3870.63|1124.36|
| Fair 1 |5670.84|1857.23|3764.04|454.22|
| Full 1 |5332.72|1773.41|3553.07|458.14|
| NoHost 1 |6565.68|1960.40|4263.04|1113.95|
| Fair 2 |5665.51|1851.28|3758.39|451.85|
| M1 2 |5313.94|1750.79|3532.36|462.91|
| NoHost 2 |5895.40|1954.56|3924.98|450.03|

M1/Fair median +2.83%/-6.01%; M1/NoHost -9.20%/-10.00%. First real request effect is retained, not removed. M1second vs singleFull -0.58% is a descriptive unmatched-repeat comparison, no superiority claim. Two repeats are not significance.

Mechanism: long target/retained512 → short joins64, long448 → short ends, goal512 while actual448 → next ordinary request actually stores512. Long3 cached coverage8192 vsFair4096, equal local2304, host5888 vs1792; first pair TTFT42.47 vs186.12ms. Expansion works, overall benefit mixed.

Whole protocol D2H/H2D MiB: M1504/512.3125, Fair140/231.875, Full336/485.1875. At session end M1D2H280; the final terminal new-generation diagnostic adds224MiB despite no later reuse. All traffic retained; this suggests avoiding useless terminal writes as a future admission improvement, not implemented now. Host physical pool remains224MiB, no RSS/active GPU memory saving claim.

All7performance runs match108544input/192output tokens; global source counters sum exactly108544. H2D bytes = external tokens×28672 for all connector runs. NoHost transfer counters absent(null), external source0 observed. Pressure requests E2E ~825–836ms, sequential; no concurrent-background fairness evidence.

Raw324files validated by SHA/length and compressed-only independent analysis. Commands in launch.json/commands.sh; analysis_verified.json, native_trace_audit.json and source snapshots. Provenance caveats in provenance_notes.json. Semaphore cleanup warning preserved; all owned groups exited and both GPUs19MiB/0% at final check.

Decision: M1 usable for the tested content/lifecycle path, not yet a stable win over strong nativeFull. Stop this smoke; next evaluate equal preregistered first-request warmup and terminal admission/shared-prefix coverage before M2/M3. Detailed report: docs/SLACKKV_SYSTEM_M1_GPU_REPORT_20261013.md.
