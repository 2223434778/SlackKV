# Actual GPU prefix cost and host-checkpoint mechanism screen

[Report](../../docs/AUTONOMOUS_RESEARCH_HOST_PREFIX_20261010.md),
[cost protocol](../../docs/HOST_PREFIX_COST_PROTOCOL.md),
[mechanism protocol](../../docs/HOST_PREFIX_CHECKPOINT_PROTOCOL.md).
Compiled native OnDemand vs same stack with max_offload_tokens9360.
Synthetic single growing path, host256MiB, forced GPU cold; natural85%GPU allocation,
actual pool1,355,712tokens identical in4performance runs. Same12inputs per run,
96Agent outputs/1536BG outputs; all work completed. Not a production trace/SOTA result.

Paired orderfull1/frontier1 andfrontier2/full2: ResumeP95-17.86/-19.88%, BGmedian
-8.08/-9.15%; D2H-94.40%, total transfers-67.54%. H2D increases8.01× as usable
restoration rises5840→46800tokens. Warmup/source accounting, input identities,
output counts, actualH2D and finalrunning/waiting0 independently verified.
Only5resumes per run, two pairs; no statistical significance or natural-budget
generalization. No overallSessionE2E/SLO/fairness proof; fixed cap fails a two-path
CPU fairness counterexample. Preserve as mechanism PoC, not deployable pool policy.

Calibration36calls,9strict input groups: cumulativeGPU hits379248 verified.
Validoracle2:9360external/0local restore versus0external/0local cold,16/16tokens
equal under separatebatch-invariant mode. oracle1INVALID (salt field misplaced),
calibration1FAILED (empty reset body mishandled); both costs/raws retained.
Known fresh zero-transfer warmup explains absent initial Prometheus labels;
ordinary missing counters remainUNKNOWN. Analysis failure retained locally.

GPU cost0.13880706816885827GPUh,8runs including failure/invalid correctness,
startup/compile/warmup/cleanup; oneGPU only. Final bothGPUs19MiB/0%.
Conservative authorization continuity includesRound3/Round4:0.8855139290GPUh total
of4h; budget.json preserves family subtotals and the current runner enforces them.
Server originals retained; compact raw gzip, source snapshots, environment and
execution ledger committed. Runner legacy snapshots hash-verified against ledger.

```sh
python -m experiments.host_prefix.analyze_gpu --directory results/host_prefix_cost_20261010
python -m pytest -q tests/test_host_prefix.py tests/test_opportunity_decision.py
```

Full GPU replay commands are in the protocol and each execution.json.
