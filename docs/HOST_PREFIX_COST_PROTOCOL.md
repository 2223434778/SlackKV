# GPU cost calibration (frozen before execution)

2026-10-10. This is a synthetic diagnostic, not public trace replay or candidate
speedup. Need: cost-tail assumes length-squared recomputation matters; measure
actual missing-suffix cost before interpreting its structural result.

GPU0 A6000; cachedQwen2.5-1.5B FP16 TP1, same pinned round4 runtime; compiled default,
prefix cache and native CPU connector; max_offload_tokens=0, no JIT/pin/gate/credit/
epochs/backfill. Natural85% GPU memory utilization, not2GiB pressure. All requests
serial, no background. CPU cache storage is disabled per request to isolate GPU
prefix/recompute. Thus no host eviction advantage is being claimed.

Fixed lengths8192/16384/28672; three prompt seeds per length; prefix fractions0,
0.5,0.875,1. Identical full prompts across states;16 generated tokens each.
Reset GPU cache before each state, warm prefix with1 output then measure full
prompt. All startup/compilation/warmup/reset/cleanup included in GPU ledger.
36 measured calls; complete outputs, input hashes, metrics and timing saved.
Verify actual prefix-hit counters; full-prefix last-token recompute is expected.
Full-state responses need not numerically match non-batch-invariant paths; this
does not test generation quality or host KV content correctness.

Primary interpretation: measuredTTFT(L,k), counter-validated cached tokens;
compare linear missing-token and quadratic attention proxies as descriptive
fit only. No expectation of statistically significant speedup or algorithmGO.
If partial-prefix hits cannot be validated, mark calibration invalid. If cost
proxy is not materially better, do not trust cost-tail rankings as GPU work saved.
Old32K model limits applicability; not a faithful mapping of long Claude traces.

One run<=30min, current exploration<=4GPUh, one GPU used; startup and cleanup count.
Retry environment/API mistakes within budget with failure preserved. No larger
model download or server configuration change.

```sh
CUDA_VISIBLE_DEVICES=0 python -m experiments.host_prefix.calibrate --name calibration1 --source /media/a6000/D/ZhaoBinHe/slackkv-autonomous-round4-source
```
