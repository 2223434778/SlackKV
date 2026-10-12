# Matched 32B prefix-admission screen

Baseline b9685ae, official model revision5ede1c97, FP16 TP2 compiled defaults.
Natural measured GPU pool70000tokens/17.08984GiB; host64GiB for Full/Fair, no connector/pool for NoHost.
Synthetic21sessions×3turns,28672+turn×512 input,128outputs,5s tool gap,3sequential BG8192/128,seed914477. No GPU reset. Not production/task-quality evidence.

| Metric | NoHost | Full | Fair |
|---|---:|---:|---:|
| Session E2E median s |969.1974|992.4999|809.8916|
| Resume P95 s |360.7371|369.0514|337.6716|
| BG makespan s |753.9746|772.3367|654.7160|
| Output tokens/s |7.4121|7.2409|9.0341|
| Compute prompt tokens |1863168|1863168|1338880|
| CPU restored tokens |0|0|524288|
| D2H GiB |0|448.875|64|
| H2D GiB |0|0|128|

Frozen primary gate PASS vs both baselines: Fair E2E−16.44/−18.40%, BG−13.16/−15.23%, all21sessions improved. P95−6.39/−8.50% only. Third BG first-token-to-response-end mean interval+16.94/+14.70%; total E2E faster. Same Full/Fair64GiB physical pool, no measured host RSS saving. Single triple/seed/order, no statistical significance or SOTA claim.

Oracle:2meaningful inputs,6batch-invariant calls; partial12480-prefix vs independentcold identical16tokens with correct distinct markers/source. Not all lifecycle/quality cases. Three performance runs66requests each preserve all logicalinput/output work, pool and execution configuration.

Raw-byte verification82files/84782502decompressedbytes PASS, compressed-only independent recomputation PASS. Lossless.xz/.gz; originals remain on server. raw_manifest.json, independent_recheck.json,comparison.json,budget.json,provenance/,commands.sh.
Downloads CPU-only: interrupted/failed attempts retained, third verified24files. No weights committed. Initial teardown uncertainty charged0.472485733GPUh upper bound, not observed busy time.
Continuous total4.1919880197/authorized4.5GPUh, large-model addition2.9261669850. Cap extension only finishes frozen triple; no extra experiments. Final GPUs19MiB/0%, owned groups empty.

DecisionA: retain capacity-conditioned static-prefix candidate; next reverse-order/seed and nonuniform/longer-output matched validation before dynamic policy. Do not claim generic deployment value. Full report docs/AUTONOMOUS_RESEARCH_LARGE_MODEL_20261012.md.
