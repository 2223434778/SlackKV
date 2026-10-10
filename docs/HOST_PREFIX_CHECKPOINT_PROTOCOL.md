# Host prefix checkpoint PoC: mechanism screen

2026-10-10, frozen before GPU execution. CPU cost-tail failed every capacity gate.
Different hypothesis supported by reference results: refusing suffix stores after
the usable prefix frontier stops write churn while maintaining prefix recovery.
Reuse existing native max_offload_tokens interface; no new kernel/scheduler.

Synthetic diagnostic only: one28672-token path grows64tokens each of6 calls;
each call generates16tokens. Before each call, reset inactive GPU cache, then
start8192-token background with256outputs and release agent after its first token.
Join background before next reset; all background work completed. This explicit
barrier is not a natural tool trace; episode makespan is not generic Agent E2E.

CPU host256MiB, much smaller than one context; deliberate host overflow regime,
not A6000 physicalRAM limit. GPU uses natural85% allocation, FP16 cached1.5B,
compiled default. Full OnDemand stores all input. Frontier caps stores at physical
CPU block capacity:floor(256MiB/(16*28672))*16=9360tokens. Cap is fixed by resource,
not fitted to outcomes. Same hashes, output counts, arrival rule, model/source/env.
No GPU pins, priority, extra leases, reservations or cancellation semantics changed.
GPU-cold intervention makes CPU reuse visible; cannot extrapolate to GPU-hot use.

Two pairs, reversed second order:full1/frontier1,frontier2/full2. Each<=30min,
all startup/compile/priming/cleanup in same host_prefix_cost_20261010 GPU ledger,
including failed calibration1. Total exploration<=4GPUh.

Screen gate on both matched pairs: D2H>=50% reduction, totalH2D+D2H>=40% reduction,
resumeP95 not worse>5%, backgroundmedian not worse>5%; all12inputs and prescribed
outputs match. Aggregate source counters must prove CPU recovery (no local Agent
hits after resets), measuredH2D bytes match external tokens*28672. Startup warmth
and first cold Agent calls retained, resumeP95 over5continuations; no statistical
significance or production algorithmGO from2pairs.

If native writes do not churn or frontier loses restores/harms background, reject
this mechanism. If resource gate passes, retain PoC and next study shared/adaptive
frontiers against capacity-conditioned admission/RR tail, under natural aggregate
working-set pressure. Fixed one-path cap is not claimed as an independent novel
paper algorithm. Multi-session quota fairness is untested; do not deploy this cap.

Separate correctness mode: batch-invariant1, one growing input with9360-token CPU
restore compared token-for-token to salted independent cold prefill; source
counters must confirm9360external/0local for restore and0external/0local for cold.
Timing excluded from performance pairs. Native manager refcount/cancel differential
tests remain the lifecycle check; this is not complete KV quality/preemption proof.

```sh
python -m experiments.host_prefix.calibrate --mode full --name full1 --source /media/a6000/D/ZhaoBinHe/slackkv-autonomous-round4-source
python -m experiments.host_prefix.calibrate --mode frontier --name frontier1 --source /media/a6000/D/ZhaoBinHe/slackkv-autonomous-round4-source
python -m experiments.host_prefix.calibrate --mode frontier --name frontier2 --source /media/a6000/D/ZhaoBinHe/slackkv-autonomous-round4-source
python -m experiments.host_prefix.calibrate --mode full --name full2 --source /media/a6000/D/ZhaoBinHe/slackkv-autonomous-round4-source
```
